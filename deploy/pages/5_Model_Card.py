"""Page 5 — Model Card: hyperparameters, metrics, per-class F1, confusion.

Detects up to three trained variants and lets the user switch between them:

  - **baseline RF**            → `metrics.json`, `confusion.npy`,
                                 `feature_importance.parquet`
  - **SMOTE-augmented cascade** → `metrics_smote.json`, `confusion_smote.npy`,
                                  `feature_importance_smote.parquet`
  - **TabSyn-augmented cascade** → `tabsyn_metrics.json`, `tabsyn_confusion.npy`,
                                   `tabsyn_feature_importance.parquet`

When `three_way_compare.csv` is also present, a side-by-side comparison is shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.io import (
    artifact_status, artifacts_dir,
    load_json, load_npy, load_parquet,
)
from lib.palette import CLASSES, N_CLASSES, PALETTE_HEX, LDD_CODES

st.set_page_config(page_title="Model Card", layout="wide")
st.title(":clipboard: Model Card")
st.caption(
    "Hyperparameters, metrics, per-class F1, confusion matrix, and feature "
    "importance for the trained classifier(s). Switch between variants when "
    "more than one set of artifacts is available."
)

# ---- model registry --------------------------------------------------------
MODELS = {
    "baseline": {
        "label":     "Baseline RF (real only)",
        "metrics":   "metrics.json",
        "confusion": "confusion.npy",
        "fi":        "feature_importance.parquet",
    },
    "smote": {
        "label":     "SMOTE-augmented cascade",
        "metrics":   "metrics_smote.json",
        "confusion": "confusion_smote.npy",
        "fi":        "feature_importance_smote.parquet",
    },
    "tabsyn": {
        "label":     "TabSyn-augmented cascade",
        "metrics":   "tabsyn_metrics.json",
        "confusion": "tabsyn_confusion.npy",
        "fi":        "tabsyn_feature_importance.parquet",
    },
}

status = artifact_status()
available = [k for k, m in MODELS.items()
             if status["files"].get(m["metrics"])
             and status["files"].get(m["confusion"])
             and status["files"].get(m["fi"])]

if not available:
    st.error(
        "No model card artifacts found. Need at least `metrics.json`, "
        "`confusion.npy`, and `feature_importance.parquet`. Run "
        "`python prepare_artifacts.py` for the baseline RF, or run the "
        "`notebooks/crop_smote_rf.ipynb` / `notebooks/crop_tabsyn.ipynb` "
        "notebooks for the augmented cascades."
    )
    st.stop()

picked_key = st.radio(
    "Variant",
    available,
    format_func=lambda k: MODELS[k]["label"],
    horizontal=True,
    index=0,
)
picked = MODELS[picked_key]

m  = load_json(picked["metrics"])
cm = load_npy(picked["confusion"])
fi = load_parquet(picked["fi"])


def pick_metrics(raw: dict) -> tuple[dict, str]:
    """Return (metric_block, label). Handles two metrics layouts:
       - prepare_artifacts.py: {"metrics": {acc, kappa, ...}}
       - notebook export:      {"pixel_test": {acc, kappa, ...}, "parcel_test": {...}}
    """
    if "metrics" in raw and isinstance(raw["metrics"], dict):
        return raw["metrics"], "test"
    if "pixel_test" in raw:
        return raw["pixel_test"], "pixel test"
    return raw, "test"


metrics, metric_label = pick_metrics(m)


def metric_get(d: dict, *names: str, default=float("nan")) -> float:
    for n in names:
        if n in d:
            return float(d[n])
    return default


accuracy = metric_get(metrics, "accuracy", "acc")
kappa    = metric_get(metrics, "kappa")
f1_w     = metric_get(metrics, "f1_weighted")
f1_m     = metric_get(metrics, "f1_macro")

# ---- headline tiles --------------------------------------------------------
h1, h2, h3, h4 = st.columns(4)
h1.metric(f"{metric_label} accuracy", f"{accuracy:.4f}")
h2.metric("Cohen's κ",                f"{kappa:.4f}")
h3.metric("F1 weighted",              f"{f1_w:.4f}")
h4.metric("F1 macro",                 f"{f1_m:.4f}")

if "parcel_test" in m:
    p = m["parcel_test"]
    pa, pb, pc, pd_ = st.columns(4)
    pa.metric("parcel accuracy",   f"{metric_get(p, 'accuracy', 'acc'):.4f}")
    pb.metric("parcel κ",          f"{metric_get(p, 'kappa'):.4f}")
    pc.metric("parcel F1 weighted", f"{metric_get(p, 'f1_weighted'):.4f}")
    pd_.metric("parcel F1 macro",  f"{metric_get(p, 'f1_macro'):.4f}")
    st.caption("Parcel-level = majority vote per LDD polygon, then evaluate.")

st.markdown("---")

# ---- hyperparameters -------------------------------------------------------
st.subheader("Hyperparameters")
rf_params    = m.get("rf_params") or {}
rf_params_s2 = m.get("rf_params_s2")

if rf_params_s2:
    cols = st.columns(2)
    for col, title, params in [
        (cols[0], "Stage-1 (15 classes)",        rf_params),
        (cols[1], "Stage-2 (orchard specialist)", rf_params_s2),
    ]:
        col.markdown(f"**{title}**")
        col.dataframe(
            pd.DataFrame([(k, str(v)) for k, v in params.items()],
                          columns=["param", "value"]),
            use_container_width=True, hide_index=True,
        )
else:
    st.dataframe(
        pd.DataFrame([(k, str(v)) for k, v in rf_params.items()],
                      columns=["param", "value"]),
        use_container_width=True, hide_index=True,
    )

caption_bits: list[str] = []
if "model_label" in m:        caption_bits.append(m["model_label"])
if "n_train" in m:            caption_bits.append(f"n_train = {m['n_train']:,}")
if "n_val"   in m:            caption_bits.append(f"n_val = {m['n_val']:,}")
if "n_test"  in m:            caption_bits.append(f"n_test = {m['n_test']:,}")
if "smote_target_per_class" in m:
    caption_bits.append(f"SMOTE target/class = {m['smote_target_per_class']:,}")
if "smote_k_neighbors" in m:
    caption_bits.append(f"k_neighbors = {m['smote_k_neighbors']}")
caption_bits.append(f"classes = {N_CLASSES}")
st.caption(" • ".join(caption_bits))

# ---- per-class F1 ----------------------------------------------------------
st.subheader("Per-class F1")
st.caption("Bars colored by class palette. LDD code shown beneath the class name.")

f1_pc_raw = metrics.get("f1_per_class", [])
if isinstance(f1_pc_raw, dict):
    f1_pc = [float(f1_pc_raw.get(str(LDD_CODES[i]),
                                  f1_pc_raw.get(LDD_CODES[i], float("nan"))))
             for i in range(N_CLASSES)]
elif isinstance(f1_pc_raw, list) and len(f1_pc_raw) == N_CLASSES:
    f1_pc = [float(v) for v in f1_pc_raw]
else:
    f1_pc = [float("nan")] * N_CLASSES

x_labels = [f"{CLASSES[i]}<br><span style='color:#888;font-size:11px'>LDD {LDD_CODES[i]}</span>"
            for i in range(N_CLASSES)]
fig = go.Figure(data=[
    go.Bar(x=x_labels, y=f1_pc, marker_color=PALETTE_HEX,
           text=[f"{v:.2f}" if not np.isnan(v) else "—" for v in f1_pc],
           textposition="outside",
           hovertemplate="%{x}<br>F1 = %{y:.3f}<extra></extra>"),
])
fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-30, height=420,
                  margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="F1")
st.plotly_chart(fig, use_container_width=True)

# ---- confusion matrix ------------------------------------------------------
st.subheader("Confusion matrix")
st.caption(
    "Rows = ground truth; columns = predicted. Diagonal = correct. "
    "Toggle row-normalized to compare per-class recall regardless of class size."
)

mode = st.radio("scale", ["counts", "row-normalized"], horizontal=True,
                 key=f"cm_mode_{picked_key}")
row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
matrix = cm if mode == "counts" else cm / row_sums

text_counts = cm.astype(int)
fig2 = go.Figure(
    data=go.Heatmap(
        z=matrix, x=CLASSES, y=CLASSES,
        colorscale="Blues",
        text=text_counts, texttemplate="%{text:,}", textfont={"size": 9},
        hovertemplate="true: %{y}<br>pred: %{x}<br>"
                      "count: %{text:,}<extra></extra>",
    ),
    layout=dict(height=600, xaxis_tickangle=-30,
                yaxis_autorange="reversed", xaxis_title="predicted",
                yaxis_title="true",
                margin=dict(l=10, r=10, t=10, b=10)),
)
st.plotly_chart(fig2, use_container_width=True)

# ---- feature importance ----------------------------------------------------
st.subheader("Feature importance (RF stage-1)")
st.caption("Mean decrease in impurity, sorted ascending. Higher = bigger contribution to split decisions.")
fi_sorted = fi.sort_values("importance", ascending=True)
fig3 = go.Figure(data=[
    go.Bar(x=fi_sorted["importance"], y=fi_sorted["feature"],
           orientation="h", marker_color="#2ca02c",
           text=[f"{v:.3f}" for v in fi_sorted["importance"]],
           textposition="outside",
           hovertemplate="%{y}<br>importance = %{x:.4f}<extra></extra>"),
])
fig3.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title="importance")
st.plotly_chart(fig3, use_container_width=True)

# ---- 3-way comparison ------------------------------------------------------
if status["files"].get("three_way_compare.csv") or len(available) > 1:
    st.markdown("---")
    st.subheader("Side-by-side comparison")

    rows = []
    for key in available:
        info = MODELS[key]
        raw = load_json(info["metrics"])
        pix, _ = pick_metrics(raw)
        par = raw.get("parcel_test", {})
        rows.append({
            "method":             info["label"],
            "pixel acc":          metric_get(pix, "accuracy", "acc"),
            "pixel κ":            metric_get(pix, "kappa"),
            "pixel F1 weighted":  metric_get(pix, "f1_weighted"),
            "pixel F1 macro":     metric_get(pix, "f1_macro"),
            "parcel acc":         metric_get(par, "accuracy", "acc"),
            "parcel κ":           metric_get(par, "kappa"),
            "parcel F1 weighted": metric_get(par, "f1_weighted"),
            "parcel F1 macro":    metric_get(par, "f1_macro"),
        })
    cmp_df = pd.DataFrame(rows).set_index("method")
    st.dataframe(
        cmp_df.style.format("{:.4f}"),
        use_container_width=True,
    )

    # per-class F1 comparison
    f1_rows = []
    for key in available:
        info = MODELS[key]
        raw = load_json(info["metrics"])
        pix, _ = pick_metrics(raw)
        f1_raw = pix.get("f1_per_class", [])
        if isinstance(f1_raw, dict):
            vals = [float(f1_raw.get(str(LDD_CODES[i]),
                                      f1_raw.get(LDD_CODES[i], float("nan"))))
                    for i in range(N_CLASSES)]
        elif isinstance(f1_raw, list) and len(f1_raw) == N_CLASSES:
            vals = [float(v) for v in f1_raw]
        else:
            vals = [float("nan")] * N_CLASSES
        for i, v in enumerate(vals):
            f1_rows.append({"class": CLASSES[i], "method": info["label"], "F1": v})
    f1_cmp = pd.DataFrame(f1_rows)
    fig_cmp = go.Figure()
    method_colors = {
        "Baseline RF (real only)":      "#7f7f7f",
        "SMOTE-augmented cascade":      "#ffb347",
        "TabSyn-augmented cascade":     "#1f77b4",
    }
    for method, sub in f1_cmp.groupby("method"):
        fig_cmp.add_trace(go.Bar(
            name=method, x=sub["class"], y=sub["F1"],
            marker_color=method_colors.get(method, "#2ca02c"),
            hovertemplate="%{x}<br>" + method + ": %{y:.3f}<extra></extra>",
        ))
    fig_cmp.update_layout(
        barmode="group", height=420, xaxis_tickangle=-30,
        yaxis_title="F1", yaxis_range=[0, 1.05],
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    if status["files"].get("three_way_compare.csv"):
        with st.expander("`three_way_compare.csv` (raw)"):
            st.dataframe(
                pd.read_csv(artifacts_dir() / "three_way_compare.csv"),
                use_container_width=True, hide_index=True,
            )

with st.expander(f"raw {picked['metrics']}"):
    st.json(m)

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))
