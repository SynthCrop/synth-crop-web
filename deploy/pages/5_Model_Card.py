"""Page 5 — Model Card: hyperparameters, metrics, per-class F1, confusion."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status, load_json, load_npy, load_parquet
from lib.plots import per_class_f1_bar, confusion_heatmap

st.title(":clipboard: Model Card")

status = artifact_status()
if status["missing"]:
    st.error("artifacts missing — run `prepare_artifacts.py`")
    st.stop()

# TODO:
# - hyperparams table (RF + TabSyn) from metrics.json
# - TRTR vs TSTR vs TRTR+synth weighted-F1, macro-F1, accuracy
# - per-class F1 bar (real vs synth-trained)
# - confusion heatmap
# - feature importance bar

metrics = load_json("metrics.json")
st.json(metrics)

if status["files"].get("confusion.npy"):
    cm = load_npy("confusion.npy")
    st.plotly_chart(confusion_heatmap(cm), use_container_width=True)

st.info("Model Card page — full layout to be implemented.")
