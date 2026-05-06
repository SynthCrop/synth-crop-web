"""Build deploy/artifacts/ from upstream RF + TabSyn outputs.

The Streamlit app reads only files written into deploy/artifacts/.

Required input:
    --source-csv   preprocessed dataset (CSV or parquet) with LDD-coded labels

Optional input:
    --synth-csv    TabSyn synthetic CSV / parquet
    --meta-json    grid metadata JSON (CRS, transform, height, width, classes)
    --preds-dir    directory holding `preds_<year>.parquet` (LDD-coded)
    --rf-model     pretrained RF joblib (skip training)

Each path can also be supplied via environment variable:
    SYNTH_CROP_SOURCE, SYNTH_CROP_SYNTH, SYNTH_CROP_META,
    SYNTH_CROP_PREDS, SYNTH_CROP_RF_MODEL.

Schema notes:
  - Source uses LDD numeric labels (2101, 2204, ..., 9999); this script
    remaps to dense 0..14 ordered by sorted LDD code (matches `lib/palette.py`).
  - Feature columns use `ndwi 10..ndwi 12` (renamed upstream from internal
    `mndwi` aliases).

Usage:
    python prepare_artifacts.py --source-csv path/to/dataset.parquet
    python prepare_artifacts.py --tiny           # 10K-row run
    python prepare_artifacts.py --rows 100000
    python prepare_artifacts.py --no-synth --no-meta
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "deploy" / "artifacts"

# ---- 15 feature columns (ndwi rename from internal mndwi happens upstream) ----
FEATURE_COLS = [
    "ndvi 10", "evi 10", "ndwi 10", "mtci 10", "swir 10",
    "ndvi 11", "evi 11", "ndwi 11", "mtci 11", "swir 11",
    "ndvi 12", "evi 12", "ndwi 12", "mtci 12", "swir 12",
]

# ---- 15-class dense order — matches deploy/lib/palette.py ----
CLASSES = [
    "Rice", "Cassava", "Pineapple", "Para rubber", "Oil palm",
    "Durian", "Mango", "Jackfruit", "Coconut", "Mangosteen",
    "Longan", "Rambutan", "Langsat", "Reservoir", "Others",
]
N_CLASSES = len(CLASSES)

# ---- LDD numeric label -> dense 0..14 (palette index) ----
LABEL_MAP_LDD_TO_DENSE = {
    2101: 0,   # Rice (Paddy)
    2204: 1,   # Cassava
    2205: 2,   # Pineapple
    2302: 3,   # Para rubber
    2303: 4,   # Oil palm
    2403: 5,   # Durian
    2407: 6,   # Mango
    2416: 7,   # Jackfruit
    2405: 8,   # Coconut
    2419: 9,   # Mangosteen
    2413: 10,  # Longan
    2404: 11,  # Rambutan
    2420: 12,  # Langsat
    4201: 13,  # Reservoir
    9999: 14,  # Others
}
DENSE_TO_LDD = {v: k for k, v in LABEL_MAP_LDD_TO_DENSE.items()}

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features="log2",
    class_weight="balanced_subsample",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

SEED = 42
CHUNK_SIZE = 200_000


def log(msg: str) -> None:
    print(f"[prep] {msg}", flush=True)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURE_COLS)
    df["label"] = df["label"].astype(np.int32)
    for c in FEATURE_COLS:
        df[c] = df[c].astype(np.float32)
    if "year" in df.columns:
        df["year"] = df["year"].astype(np.int16)
    for c in ("row", "col"):
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    return df


def _remap_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """LDD numeric -> dense 0..14. Drops rows with unmapped labels."""
    n0 = len(df)
    df[label_col] = df[label_col].map(LABEL_MAP_LDD_TO_DENSE)
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(np.int32)
    n1 = len(df)
    if n0 != n1:
        log(f"  remapped LDD->dense: dropped {n0-n1:,} rows with unknown labels")
    return df


def load_source(path: Path) -> pd.DataFrame:
    log(f"loading source: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    wanted = FEATURE_COLS + ["label", "year", "row", "col"]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        schema_cols = pq.ParquetFile(path).schema_arrow.names
        keep = [c for c in wanted if c in schema_cols]
        df = pd.read_parquet(path, columns=keep)
    else:
        wanted_set = set(wanted)
        df = pd.read_csv(path, encoding="utf-8-sig",
                         usecols=lambda c: c in wanted_set)
    df = _downcast(df)
    log(f"  loaded shape: {df.shape}")
    return df


def stratified_subset(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = df["label"].value_counts()
    n_classes = len(counts)
    per_class = max(total // n_classes, 1)
    parts = []
    for cid, cnt in counts.items():
        idx = df.index[df["label"] == cid].to_numpy()
        take = min(per_class, int(cnt))
        parts.append(rng.choice(idx, take, replace=False))
    keep = np.concatenate(parts)
    rng.shuffle(keep)
    return df.loc[keep].reset_index(drop=True)


def fit_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    log(f"fitting RF: train={X.shape}  params={RF_PARAMS}")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    log(f"  fit: {time.time() - t0:.1f}s")
    return clf


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(N_CLASSES))
    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "kappa":        float(cohen_kappa_score(y_true, y_pred)),
        "f1_weighted":  float(f1_score(y_true, y_pred, average="weighted",
                                       labels=labels, zero_division=0)),
        "f1_macro":     float(f1_score(y_true, y_pred, average="macro",
                                       labels=labels, zero_division=0)),
        "precision_w":  float(precision_score(y_true, y_pred, average="weighted",
                                              labels=labels, zero_division=0)),
        "recall_w":     float(recall_score(y_true, y_pred, average="weighted",
                                           labels=labels, zero_division=0)),
        "f1_per_class": [float(v) for v in f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0)],
    }


def maybe_load_synth(path: Path | None, target_n: int) -> pd.DataFrame | None:
    if path is None or not path.exists():
        log("synth: skipped (path missing or --no-synth)")
        return None
    log(f"loading synth: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # TabSyn synth CSV may be header-less (matches data/<DATASET>/<DATASET>.csv schema)
        # Use first row to detect.
        head = pd.read_csv(path, nrows=1, encoding="utf-8-sig", header=None)
        try:
            float(head.iloc[0, 0])
            df = pd.read_csv(path, header=None, encoding="utf-8-sig")
            df.columns = FEATURE_COLS + ["label"]
        except (ValueError, TypeError):
            df = pd.read_csv(path, encoding="utf-8-sig")
    # synth CSV may use internal 'mndwi' col names — rename to demo schema 'ndwi'
    df = df.rename(columns={c: c.replace("mndwi", "ndwi") for c in df.columns if "mndwi" in c})
    df = df[[c for c in FEATURE_COLS + ["label"] if c in df.columns]]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log(f"  [warn] synth CSV missing cols: {missing} — skipping synth")
        return None
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # synth labels may be float; round + clip + remap
    df["label"] = df["label"].astype(float).round().astype(int)
    valid_ldd = list(LABEL_MAP_LDD_TO_DENSE.keys())
    df["label"] = df["label"].clip(min(valid_ldd), max(valid_ldd))
    # snap to nearest valid LDD class — synth may emit out-of-set values
    arr = np.array(valid_ldd)
    df["label"] = df["label"].apply(lambda v: int(arr[np.argmin(np.abs(arr - v))]))
    df = _remap_labels(df)

    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=SEED).reset_index(drop=True)
    for c in FEATURE_COLS:
        df[c] = df[c].astype(np.float32)
    log(f"  synth shape (post-remap, sampled): {df.shape}")
    return df


def maybe_smote_synth(df_real: pd.DataFrame, target_per_class: int) -> pd.DataFrame | None:
    """Run SMOTE on dense-labelled real sample. Returns synth-only rows (no real)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log("smote: skipped (imbalanced-learn not installed — `pip install imbalanced-learn`)")
        return None
    counts = df_real["label"].value_counts().sort_index()
    minority = [int(c) for c, n in counts.items() if n < target_per_class]
    if not minority:
        log(f"smote: skipped (no class < {target_per_class:,} in sample)")
        return None
    sampling_strategy = {c: target_per_class for c in minority}
    k_eff = max(1, min(5, min(int(counts[c]) for c in minority) - 1))
    log(f"smote: lifting {len(minority)} classes -> {target_per_class:,} each  (k_neighbors={k_eff})")
    X = df_real[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df_real["label"].to_numpy(dtype=np.int32)
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_eff,
                  random_state=SEED)
    X_aug, y_aug = smote.fit_resample(X, y)
    n_syn = len(X_aug) - len(X)
    log(f"smote: generated {n_syn:,} synthetic rows")
    syn = pd.DataFrame(X_aug[len(X):], columns=FEATURE_COLS)
    syn["label"] = y_aug[len(X):].astype(np.int32)
    return syn


def maybe_load_meta(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        log("meta: skipped (path missing or --no-meta)")
        return None
    log(f"loading meta: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifacts(
    df_sample: pd.DataFrame,
    clf: RandomForestClassifier,
    metrics: dict,
    cm: np.ndarray,
    df_synth: pd.DataFrame | None,
    meta: dict | None,
    df_smote: pd.DataFrame | None = None,
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    log("writing dataset.parquet")
    df_sample.to_parquet(ARTIFACTS / "dataset.parquet",
                         compression="zstd", index=False)

    log("writing class_counts.parquet")
    if "year" in df_sample.columns:
        cc = (df_sample.groupby(["year", "label"]).size()
              .rename("count").reset_index())
    else:
        cc = (df_sample.groupby("label").size()
              .rename("count").reset_index())
    cc["class"] = cc["label"].map(lambda i: CLASSES[i] if 0 <= i < N_CLASSES else "?")
    cc.to_parquet(ARTIFACTS / "class_counts.parquet", index=False)

    log("writing feature_stats.parquet")
    fs = (df_sample.groupby("label")[FEATURE_COLS]
          .agg(["mean", "std", "min", "max"]))
    fs.columns = [f"{a}__{b}" for a, b in fs.columns]
    fs = fs.reset_index()
    fs["class"] = fs["label"].map(lambda i: CLASSES[i] if 0 <= i < N_CLASSES else "?")
    fs.to_parquet(ARTIFACTS / "feature_stats.parquet", index=False)

    log("writing corr_real.npy")
    np.save(ARTIFACTS / "corr_real.npy",
            df_sample[FEATURE_COLS].corr().to_numpy().astype(np.float32))

    log("writing metrics.json")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "rf_params":   {k: (v if v != -1 else "-1") for k, v in RF_PARAMS.items()},
        "n_classes":   N_CLASSES,
        "classes":     CLASSES,
        "feature_cols": FEATURE_COLS,
        "label_map_ldd_to_dense": {str(k): v for k, v in LABEL_MAP_LDD_TO_DENSE.items()},
        "metrics":     metrics,
        "seed":        SEED,
    }, indent=2, default=str), encoding="utf-8")

    log("writing confusion.npy")
    np.save(ARTIFACTS / "confusion.npy", cm.astype(np.int64))

    log("writing feature_importance.parquet")
    fi = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": clf.feature_importances_.astype(np.float32),
    }).sort_values("importance", ascending=False)
    fi.to_parquet(ARTIFACTS / "feature_importance.parquet", index=False)

    log("writing rf_model.joblib")
    joblib.dump(clf, ARTIFACTS / "rf_model.joblib", compress=("xz", 9))

    if df_synth is not None:
        log("writing synth_tabsyn.parquet")
        df_synth.to_parquet(ARTIFACTS / "synth_tabsyn.parquet",
                            compression="zstd", index=False)
        log("writing corr_syn.npy")
        np.save(ARTIFACTS / "corr_syn.npy",
                df_synth[FEATURE_COLS].corr().to_numpy().astype(np.float32))

    if df_smote is not None:
        log("writing synth_smote.parquet")
        df_smote.to_parquet(ARTIFACTS / "synth_smote.parquet",
                            compression="zstd", index=False)
        log("writing corr_smote.npy")
        np.save(ARTIFACTS / "corr_smote.npy",
                df_smote[FEATURE_COLS].corr().to_numpy().astype(np.float32))

    if meta is not None:
        log("writing grid_meta.json")
        (ARTIFACTS / "grid_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")


def maybe_copy_rasters(seg_dir: Path | None) -> int:
    if seg_dir is None or not seg_dir.exists():
        return 0
    n = 0
    for tif in sorted(seg_dir.glob("preds_*.tif")):
        dst = ARTIFACTS / tif.name
        shutil.copy(tif, dst)
        log(f"  copied raster: {tif.name}  ({dst.stat().st_size/1e6:.1f} MB)")
        n += 1
    return n


def _env_path(name: str) -> Path | None:
    val = os.environ.get(name)
    return Path(val) if val else None


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--source-csv", type=Path,
                   default=_env_path("SYNTH_CROP_SOURCE"),
                   help="preprocessed dataset (CSV or parquet) with LDD-coded labels. "
                        "Required (or set SYNTH_CROP_SOURCE).")
    p.add_argument("--synth-csv", type=Path,
                   default=_env_path("SYNTH_CROP_SYNTH"),
                   help="TabSyn synthetic CSV or parquet (or set SYNTH_CROP_SYNTH).")
    p.add_argument("--meta-json", type=Path,
                   default=_env_path("SYNTH_CROP_META"),
                   help="grid metadata JSON (or set SYNTH_CROP_META).")
    p.add_argument("--seg-dir", type=Path, default=None,
                   help="directory containing preds_<year>.tif rasters.")
    p.add_argument("--rf-model", type=Path,
                   default=_env_path("SYNTH_CROP_RF_MODEL"),
                   help="pretrained RF joblib (skip training). Must use dense 0..14 labels.")
    p.add_argument("--rows", type=int, default=50_000,
                   help="stratified sample size.")
    p.add_argument("--tiny", action="store_true",
                   help="quick run: 10K rows, skip synth + meta.")
    p.add_argument("--no-synth", action="store_true", help="skip synth load.")
    p.add_argument("--no-meta",  action="store_true", help="skip meta load.")
    p.add_argument("--preds-dir", type=Path,
                   default=_env_path("SYNTH_CROP_PREDS"),
                   help="directory with preds_<year>.parquet (LDD-coded). "
                        "Copied + remapped to dense (or set SYNTH_CROP_PREDS).")
    p.add_argument("--with-smote", action="store_true",
                   help="also generate SMOTE synth (requires imbalanced-learn).")
    p.add_argument("--smote-target", type=int, default=10_000,
                   help="lift each minority to this count via SMOTE.")
    args = p.parse_args()

    if args.source_csv is None:
        log("ERROR: --source-csv not provided and SYNTH_CROP_SOURCE not set.")
        return 2
    src = args.source_csv.resolve()
    if not src.exists():
        log(f"ERROR: source file not found: {src}")
        return 1

    rows = 10_000 if args.tiny else args.rows

    df = load_source(src)
    df = _remap_labels(df)
    log(f"after remap: {df.shape}")

    df_sample = stratified_subset(df, total=rows, seed=SEED)
    log(f"sampled to {len(df_sample):,} rows; class counts:")
    for cid, n in df_sample["label"].value_counts().sort_index().items():
        name = CLASSES[cid] if 0 <= cid < N_CLASSES else f"cid={cid}"
        log(f"    {cid:2d}  {name:12s}  {n:>6,}")

    X = df_sample[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df_sample["label"].to_numpy(dtype=np.int32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f"train={X_tr.shape}  test={X_te.shape}")

    if args.rf_model is not None and args.rf_model.exists():
        log(f"loading pretrained RF: {args.rf_model}")
        clf = joblib.load(args.rf_model)
    else:
        clf = fit_rf(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    metrics = metric_block(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(N_CLASSES)))

    log(f"acc={metrics['accuracy']:.4f}  kappa={metrics['kappa']:.4f}  "
        f"f1_w={metrics['f1_weighted']:.4f}  f1_m={metrics['f1_macro']:.4f}")

    df_synth = None if (args.tiny or args.no_synth) else maybe_load_synth(args.synth_csv, rows)
    meta     = None if (args.tiny or args.no_meta)  else maybe_load_meta(args.meta_json)
    df_smote = maybe_smote_synth(df_sample, args.smote_target) if args.with_smote else None

    write_artifacts(df_sample, clf, metrics, cm, df_synth, meta, df_smote)

    if args.seg_dir:
        n_tif = maybe_copy_rasters(args.seg_dir)
        log(f"copied {n_tif} segmentation rasters")

    # ---- copy + remap preds_<year>.parquet ----
    if args.preds_dir and args.preds_dir.exists():
        n_pred = 0
        for src_pred in sorted(args.preds_dir.glob("preds_*.parquet")):
            dfp = pd.read_parquet(src_pred)
            for col in ("label_true", "label_pred"):
                if col in dfp.columns:
                    dfp[col] = dfp[col].map(LABEL_MAP_LDD_TO_DENSE)
            dfp = dfp.dropna(subset=["label_true", "label_pred"]).copy()
            dfp["label_true"] = dfp["label_true"].astype(np.int8)
            dfp["label_pred"] = dfp["label_pred"].astype(np.int8)
            dfp["row"] = dfp["row"].astype(np.int32)
            dfp["col"] = dfp["col"].astype(np.int32)
            dst = ARTIFACTS / src_pred.name
            dfp.to_parquet(dst, compression="zstd", index=False)
            agree = float((dfp["label_true"] == dfp["label_pred"]).mean())
            log(f"  preds {src_pred.name}: {len(dfp):>8,} rows  agree={agree:.4f}  "
                f"{dst.stat().st_size/1e6:.2f} MB")
            n_pred += 1
        log(f"copied {n_pred} preds_<year>.parquet (LDD->dense remapped)")

    written = sorted(ARTIFACTS.glob("*"))
    total_mb = sum(f.stat().st_size for f in written if f.is_file()) / 1e6
    log(f"\ndone. {len(written)} files, {total_mb:.1f} MB")
    for f in written:
        if f.is_file():
            log(f"  {f.name:32s}  {f.stat().st_size / 1e6:>7.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
