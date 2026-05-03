"""Cached loaders for parquet / raster / json artifacts.

The web app must never touch raw rasters or the full 3 GB CSV — only the
files in deploy/artifacts/ produced by prepare_artifacts.py.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

ARTIFACTS_SUBDIR = "artifacts"

EXPECTED_FILES = [
    "dataset.parquet",
    "synth_tabsyn.parquet",
    "class_counts.parquet",
    "feature_stats.parquet",
    "metrics.json",
    "confusion.npy",
    "feature_importance.parquet",
    "rf_model.joblib",
    "grid_meta.json",
]


def artifacts_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / ARTIFACTS_SUBDIR


def artifact_status() -> dict[str, Any]:
    """Quick scan of artifacts/. Used by the home page banner."""
    d = artifacts_dir()
    files = {name: (d / name).exists() for name in EXPECTED_FILES}
    missing = [n for n, ok in files.items() if not ok]

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

    n_rasters = len(list(d.glob("preds_*.tif")))

    return {
        "files": files,
        "missing": missing,
        "rows": rows,
        "years": years,
        "n_rasters": n_rasters,
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


@st.cache_resource(show_spinner=False)
def load_rf_model():
    import joblib
    return joblib.load(artifacts_dir() / "rf_model.joblib")


@lru_cache(maxsize=8)
def list_pred_rasters() -> list[Path]:
    return sorted(artifacts_dir().glob("preds_*.tif"))
