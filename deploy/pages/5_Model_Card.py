"""Page 5 — Model Card: hyperparameters, metrics, per-class F1, confusion."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.io import (
    artifact_status, artifacts_dir,
    load_json, load_npy, load_parquet,
)
from lib.palette import CLASSES, N_CLASSES, PALETTE_HEX

st.set_page_config(page_title="Model Card", layout="wide")
st.title(":clipboard: Model Card")

status = artifact_status()
need = ["metrics.json", "confusion.npy", "feature_importance.parquet"]
missing = [n for n in need if not status["files"].get(n)]
if missing:
    st.error(f"missing: {missing} — run `python prepare_artifacts.py`")
    st.stop()

m = load_json("metrics.json")
cm = load_npy("confusion.npy")
fi = load_parquet("feature_importance.parquet")
metrics = m["metrics"]

# ---- Headline tiles ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("accuracy",     f"{metrics['accuracy']:.4f}")
c2.metric("kappa",        f"{metrics['kappa']:.4f}")
c3.metric("F1 weighted",  f"{metrics['f1_weighted']:.4f}")
c4.metric("F1 macro",     f"{metrics['f1_macro']:.4f}")

st.markdown("---")

# ---- Hyperparameter table ---------------------------------------------------
st.subheader("Hyperparameters")
hp = pd.DataFrame(
    [(k, str(v)) for k, v in m["rf_params"].items()],
    columns=["param", "value"],
)
st.dataframe(hp, use_container_width=True, hide_index=True)
st.caption(f"seed = {m.get('seed', 42)} • classes = {m['n_classes']}")

# ---- Per-class F1 -----------------------------------------------------------
st.subheader("Per-class F1")
f1_pc = metrics["f1_per_class"]
fig = go.Figure(data=[
    go.Bar(x=CLASSES, y=f1_pc, marker_color=PALETTE_HEX,
           text=[f"{v:.2f}" for v in f1_pc], textposition="outside"),
])
fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-30, height=380,
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- Confusion matrix -------------------------------------------------------
st.subheader("Confusion matrix")
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
mode = st.radio("mode", ["counts", "row-normalized"], horizontal=True)
matrix = cm if mode == "counts" else cm_norm
fig2 = go.Figure(
    data=go.Heatmap(z=matrix, x=CLASSES, y=CLASSES,
                    colorscale="Blues",
                    text=cm.astype(int), texttemplate="%{text}",
                    textfont={"size": 9}),
    layout=dict(height=560, xaxis_tickangle=-30,
                yaxis_autorange="reversed",
                margin=dict(l=10, r=10, t=10, b=10)),
)
st.plotly_chart(fig2, use_container_width=True)

# ---- Feature importance -----------------------------------------------------
st.subheader("Feature importance")
fi_sorted = fi.sort_values("importance", ascending=True)
fig3 = go.Figure(data=[
    go.Bar(x=fi_sorted["importance"], y=fi_sorted["feature"],
           orientation="h", marker_color="#2ca02c",
           text=[f"{v:.3f}" for v in fi_sorted["importance"]],
           textposition="outside"),
])
fig3.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig3, use_container_width=True)

with st.expander("raw metrics.json"):
    st.json(m)

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))
