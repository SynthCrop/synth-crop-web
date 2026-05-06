"""Cached loaders for parquet, raster, and JSON artifacts.

The web app reads only files in `deploy/artifacts/` produced by
`prepare_artifacts.py` and `prepare_basemap.py`.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

ARTIFACTS_SUBDIR = "artifacts"

REQUIRED_FILES = [
    "dataset.parquet",
    "class_counts.parquet",
    "feature_stats.parquet",
    "corr_real.npy",
    "metrics.json",
    "confusion.npy",
    "feature_importance.parquet",
]

# rf_model.joblib is optional. The trained RF cascade is large; the deployed
# app reads pre-computed `preds_<year>.parquet` outputs instead of running
# live inference.
OPTIONAL_FILES = [
    "rf_model.joblib",
    "synth_tabsyn.parquet",
    "corr_syn.npy",
    "synth_smote.parquet",
    "corr_smote.npy",
    "grid_meta.json",
    # SMOTE cascade artifacts (notebooks/crop_smote_rf.ipynb)
    "metrics_smote.json",
    "confusion_smote.npy",
    "feature_importance_smote.parquet",
    # TabSyn cascade artifacts (notebooks/crop_tabsyn.ipynb)
    "tabsyn_metrics.json",
    "tabsyn_confusion.npy",
    "tabsyn_feature_importance.parquet",
    # 3-way comparison summary
    "three_way_compare.csv",
]

EXPECTED_FILES = REQUIRED_FILES + OPTIONAL_FILES


def artifacts_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / ARTIFACTS_SUBDIR


def artifact_status() -> dict[str, Any]:
    """Quick scan of artifacts/. Used by the home page banner.

    `missing` lists only REQUIRED files. Optional files are reported in
    `optional_missing` and never trigger the banner.
    """
    d = artifacts_dir()
    files = {name: (d / name).exists() for name in EXPECTED_FILES}
    missing = [n for n in REQUIRED_FILES if not files[n]]
    optional_missing = [n for n in OPTIONAL_FILES if not files[n]]

    rows = 0
    years: list[int] = []
    if files["dataset.parquet"]:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(d / "dataset.parquet")
            rows = pf.metadata.num_rows
            schema = pf.schema_arrow
            if "year" in schema.names:
                table = pf.read(columns=["year"])
                years = sorted(set(table.column("year").to_pylist()))
        except Exception:
            pass

    pred_tif_years = sorted({int(p.stem.split("_")[1])
                              for p in d.glob("preds_*.tif")
                              if p.stem.split("_")[1].isdigit()})
    pred_pq_years  = sorted({int(p.stem.split("_")[1])
                              for p in d.glob("preds_*.parquet")
                              if p.stem.split("_")[1].isdigit()})
    pred_years = sorted(set(pred_tif_years) | set(pred_pq_years))

    return {
        "files": files,
        "missing": missing,
        "optional_missing": optional_missing,
        "rows": rows,
        "years": years,
        "n_rasters": len(pred_tif_years),
        "n_pred_pq": len(pred_pq_years),
        "n_pred_years": len(pred_years),
        "pred_years": pred_years,
    }


@st.cache_data(show_spinner=False)
def load_parquet(name: str):
    import pandas as pd
    return pd.read_parquet(artifacts_dir() / name)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((artifacts_dir() / name).read_text())


@st.cache_data(show_spinner=False)
def load_npy(name: str):
    import numpy as np
    return np.load(artifacts_dir() / name)


@lru_cache(maxsize=8)
def list_pred_rasters() -> list[Path]:
    return sorted(artifacts_dir().glob("preds_*.tif"))


@lru_cache(maxsize=8)
def list_pred_parquets() -> list[Path]:
    return sorted(artifacts_dir().glob("preds_*.parquet"))


@st.cache_data(show_spinner=False)
def load_pred_parquet(name: str):
    import pandas as pd
    return pd.read_parquet(artifacts_dir() / name)
