"""Crop classification — TabSyn x Random Forest. Streamlit entry point."""
from __future__ import annotations
from pathlib import Path

import streamlit as st

from lib.io import artifacts_dir, artifact_status

st.set_page_config(
    page_title="Crop classification — TabSyn x Random Forest",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(":seedling: Crop classification — TabSyn x Random Forest")

st.markdown(
    """
    **Pages**

    - :bar_chart: **Dataset** — class distribution, feature stats, correlations
    - :earth_africa: **Temporal Change** — year-to-year class transitions, Sankey flows, change maps
    - :sparkles: **Segmentation** — predicted-class rasters layered on a basemap
    - :dart: **Predict** — interactive classifier (single pixel + batch CSV)
    - :clipboard: **Model Card** — metrics, per-class F1, hyperparameters
    """
)

status = artifact_status()

c1, c2, c3, c4 = st.columns(4)
c1.metric("rows in dataset", f"{status['rows']:,}" if status["rows"] else "—")
c2.metric("years",            ", ".join(map(str, status["years"])) if status["years"] else "—")
c3.metric("classes (palette)", "15")
c4.metric("pred rasters",     str(status["n_rasters"]))

if status["missing"]:
    missing_names = ", ".join(status["missing"])
    st.error(
        f"`{artifacts_dir()}/{missing_names}` missing — "
        "run `prepare_artifacts.py` first"
    )
    st.info(
        "Run `python prepare_artifacts.py` to populate `deploy/artifacts/`."
    )

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))
    for name, ok in status["files"].items():
        icon = ":white_check_mark:" if ok else ":x:"
        st.write(f"{icon} `{name}`")

st.caption("navigate via the sidebar →")
