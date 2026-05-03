"""Build deploy/artifacts/ from raw dataset (+ optional synth + meta).

The Streamlit app reads ONLY files written here. Heavy raw data
(3 GB CSV, S2 rasters) stays out of the repo and out of the cloud container.

Pipeline:
    1. Load preprocessed CSV (cid 0..14 label, 15 feature columns).
    2. Stratified sample (50k rows by default, 10k under --tiny).
    3. Stratified 80/20 split.
    4. Train RF on train split (params from gis-workflow notebook).
    5. Evaluate on test → metrics, per-class F1, confusion.
    6. Compute feature_stats, class_counts, correlation.
    7. (Optional) load synth CSV → match-size sample + corr.
    8. (Optional) copy grid_meta.json.
    9. Write all artifacts.

Usage:
    python prepare_artifacts.py
    python prepare_artifacts.py --tiny
    python prepare_artifacts.py --source-csv /path/to/dataset.csv \\
                                --synth-csv /path/to/tabsyn.csv \\
                                --meta-json /path/to/meta.json
"""
from __future__ import annotations
import argparse
import json
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

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "deploy" / "artifacts"

FEATURE_COLS = [
    "ndvi 10", "evi 10", "ndwi 10", "mtci 10", "swir 10",
    "ndvi 11", "evi 11", "ndwi 11", "mtci 11", "swir 11",
    "ndvi 12", "evi 12", "ndwi 12", "mtci 12", "swir 12",
]

CLASSES = [
    "Rice", "Cassava", "Pineapple", "Para rubber", "Oil palm",
    "Durian", "Mango", "Jackfruit", "Coconut", "Mangosteen",
    "Longan", "Rambutan", "Langsat", "Reservoir", "Others",
]
N_CLASSES = len(CLASSES)

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features="log2",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

SEED = 42


def log(msg: str) -> None:
    print(f"[prep] {msg}", flush=True)


CHUNK_SIZE = 200_000


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


def load_and_sample(path: Path, total: int, seed: int) -> pd.DataFrame:
    """Stream the CSV in chunks; reservoir-sample per class up to a budget.

    Memory stays bounded (a few hundred MB). Two passes:
      1. count per-class rows (single label column)
      2. reservoir-sample each class to its budget
    """
    log(f"loading {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = _downcast(df)
        log(f"  parquet shape after NaN drop: {df.shape}")
        return _stratified_subset(df, total, seed)

    wanted = set(FEATURE_COLS + ["label", "year", "row", "col"])

    # ---- Pass 1: per-class counts ----
    log("  pass 1/2: counting classes (chunked)")
    t0 = time.time()
    counts: dict[int, int] = {}
    rows_total = 0
    for chunk in pd.read_csv(path, encoding="utf-8-sig",
                             usecols=["label"], chunksize=CHUNK_SIZE):
        rows_total += len(chunk)
        vc = chunk["label"].dropna().astype(np.int32).value_counts()
        for cid, n in vc.items():
            counts[int(cid)] = counts.get(int(cid), 0) + int(n)
    log(f"    counted {rows_total:,} rows across {len(counts)} classes  "
        f"({time.time() - t0:.1f}s)")

    n_classes = len(counts)
    per_class = max(total // n_classes, 1)
    budget = {cid: min(per_class, c) for cid, c in counts.items()}
    log(f"    per-class budget: {per_class:,}")

    # ---- Pass 2: per-row Bernoulli sampling, oversample 1.5x then trim ----
    log("  pass 2/2: Bernoulli-sampling per class")
    t0 = time.time()
    probs = {cid: min(1.0, budget[cid] / max(counts[cid], 1) * 1.5)
             for cid in budget}
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig",
                             usecols=lambda c: c in wanted,
                             chunksize=CHUNK_SIZE):
        chunk = _downcast(chunk)
        if len(chunk) == 0:
            continue
        p = chunk["label"].astype(int).map(probs).fillna(0.0).to_numpy()
        keep = rng.random(len(chunk)) < p
        if keep.any():
            pieces.append(chunk.loc[keep].copy())

    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    # Trim each class to exact budget (oversample buffer absorbs variance).
    parts = []
    for cid, sub in df.groupby("label"):
        cap = budget.get(int(cid), 0)
        if len(sub) > cap:
            sub = sub.sample(n=cap, random_state=seed)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    log(f"    sampled {len(df):,} rows  ({time.time() - t0:.1f}s)")
    return df


def _stratified_subset(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
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
    log(f"fitting RF: train={X.shape}")
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


def write_artifacts(
    df_sample: pd.DataFrame,
    clf: RandomForestClassifier,
    metrics: dict,
    cm: np.ndarray,
    df_synth: pd.DataFrame | None,
    meta: dict | None,
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
    cc.to_parquet(ARTIFACTS / "class_counts.parquet", index=False)

    log("writing feature_stats.parquet")
    fs = (df_sample.groupby("label")[FEATURE_COLS]
          .agg(["mean", "std", "min", "max"]))
    fs.columns = [f"{a}__{b}" for a, b in fs.columns]
    fs.reset_index().to_parquet(ARTIFACTS / "feature_stats.parquet", index=False)

    log("writing corr_real.npy")
    np.save(ARTIFACTS / "corr_real.npy",
            df_sample[FEATURE_COLS].corr().to_numpy().astype(np.float32))

    log("writing metrics.json")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "rf_params":  {k: (v if v != -1 else "-1") for k, v in RF_PARAMS.items()},
        "n_classes":  N_CLASSES,
        "classes":    CLASSES,
        "feature_cols": FEATURE_COLS,
        "metrics":    metrics,
        "seed":       SEED,
    }, indent=2, default=str))

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

    if meta is not None:
        log("writing grid_meta.json")
        (ARTIFACTS / "grid_meta.json").write_text(
            json.dumps(meta, indent=2, default=str))


def maybe_load_synth(path: Path | None, target_n: int) -> pd.DataFrame | None:
    if path is None or not path.exists():
        log("synth: skipped (no --synth-csv or file missing)")
        return None
    log(f"loading synth: {path}")
    wanted = set(FEATURE_COLS + ["label"])
    df = pd.read_csv(path, encoding="utf-8-sig",
                     usecols=lambda c: c in wanted)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=SEED).reset_index(drop=True)
    for c in FEATURE_COLS:
        df[c] = df[c].astype(np.float32)
    if "label" in df.columns:
        df["label"] = df["label"].astype(np.int32)
    log(f"  synth shape: {df.shape}")
    return df


def maybe_load_meta(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        log("meta: skipped (no --meta-json or file missing)")
        return None
    log(f"loading meta: {path}")
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-csv", type=Path,
                   default=Path("../2020_2024_dataset.csv"),
                   help="preprocessed dataset (CSV or parquet)")
    p.add_argument("--synth-csv", type=Path,
                   default=Path("../synthetic/crop_full/tabsyn.csv"),
                   help="TabSyn synthetic CSV (optional)")
    p.add_argument("--meta-json", type=Path,
                   default=Path("../2020_2024_dataset_meta.json"),
                   help="grid metadata from gis-workflow (optional)")
    p.add_argument("--rf-model", type=Path, default=None,
                   help="pretrained RF joblib to copy in (skip training)")
    p.add_argument("--rows", type=int, default=50_000,
                   help="stratified sample size for the web artifact")
    p.add_argument("--tiny", action="store_true",
                   help="smoke-test: 10k rows, skip synth + meta")
    args = p.parse_args()

    src = args.source_csv.resolve()
    if not src.exists():
        log(f"ERROR: source CSV not found: {src}")
        return 1

    rows = 10_000 if args.tiny else args.rows

    df_sample = load_and_sample(src, total=rows, seed=SEED)
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

    df_synth = None if args.tiny else maybe_load_synth(args.synth_csv, rows)
    meta     = None if args.tiny else maybe_load_meta(args.meta_json)

    write_artifacts(df_sample, clf, metrics, cm, df_synth, meta)

    written = sorted(ARTIFACTS.glob("*"))
    total_mb = sum(f.stat().st_size for f in written if f.is_file()) / 1e6
    log(f"\ndone. {len(written)} files, {total_mb:.1f} MB")
    for f in written:
        if f.is_file():
            log(f"  {f.name:32s}  {f.stat().st_size / 1e6:>7.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
