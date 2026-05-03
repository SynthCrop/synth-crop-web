"""Page 1 — Dataset: distribution, feature stats, real-vs-synth correlations."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status, load_parquet, load_npy
from lib.palette import CLASSES, N_CLASSES, PALETTE_HEX
from lib.plots import class_bar

st.title(":bar_chart: Dataset")

status = artifact_status()
if status["missing"]:
    st.error("artifacts missing — run `prepare_artifacts.py`")
    st.stop()

# TODO: implement
# - load class_counts.parquet, plot real vs synth bar
# - load feature_stats.parquet, render describe table
# - violin per class via st.selectbox
# - corr_real.npy, corr_syn.npy → side-by-side heatmaps

st.info("Dataset page — to be implemented after artifacts exist.")
