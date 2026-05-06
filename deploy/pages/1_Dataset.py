"""Page 1 — Dataset: distribution, feature stats, real-vs-synth correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.io import (
    artifact_status, artifacts_dir,
    load_parquet, load_npy,
)
from lib.palette import CLASSES, N_CLASSES, PALETTE_HEX, LDD_CODES, to_dense

st.set_page_config(page_title="Dataset", layout="wide")
st.title(":bar_chart: Dataset")
st.caption(
    "Sample preview, geospatial metadata, class balance, per-feature statistics, "
    "and pairwise correlation. All numbers come from `deploy/artifacts/*` "
    "(written by `prepare_artifacts.py`)."
)

status = artifact_status()
if not status["files"].get("dataset.parquet"):
    st.error("`dataset.parquet` missing — run `python prepare_artifacts.py`")
    st.stop()

@st.cache_data(show_spinner=False)
def _dense(name: str) -> pd.DataFrame:
    df = load_parquet(name)
    if "label" not in df.columns:
        return df
    df = df.copy()
    df["label"] = to_dense(df["label"].to_numpy())
    return df[df["label"] >= 0].reset_index(drop=True)


df = _dense("dataset.parquet")
class_counts_raw = load_parquet("class_counts.parquet")
feature_stats    = load_parquet("feature_stats.parquet")
corr_real        = load_npy("corr_real.npy")
df_synth = _dense("synth_tabsyn.parquet") if status["files"].get("synth_tabsyn.parquet", False) else None
corr_syn = load_npy("corr_syn.npy") if status["files"].get("corr_syn.npy", False) else None

# Geospatial metadata for the Rayong AOI.
crs    = "EPSG:4326 (WGS 84) — display ; native EPSG:32647 (UTM 47N)"
bounds = {"West": 100.8405, "East": 101.8619, "North": 13.5616, "South": 12.5604}

raster_meta = {
    "10 m":  {"resolution": "10 m",  "width_px": 10980, "height_px": 10980, "bands": 6,  "dtype": "Float32"},
    "20 m":  {"resolution": "20 m",  "width_px":  5490, "height_px":  5490, "bands": 14, "dtype": "Float32"},
    "60 m":  {"resolution": "60 m",  "width_px":  1830, "height_px":  1830, "bands": 15, "dtype": "Float32"},
}

vector_meta = {"geometry": "Polygon", "feature_layers": 13, "landuse_polygons": 41571}

FEATURE_COLS = [c for c in df.columns
                if c not in ("label", "year", "row", "col", "parcel_id", "n_valid_months")]

# ---- header tiles ----------------------------------------------------------
h1, h2, h3, h4 = st.columns(4)
h1.metric("rows (sampled)",   f"{len(df):,}")
h2.metric("classes",          f"{df['label'].nunique()} / {N_CLASSES}")
h3.metric("years",            ", ".join(map(str, sorted(df["year"].unique())))
                              if "year" in df.columns else "—")
h4.metric("synth rows",       f"{len(df_synth):,}" if df_synth is not None else "—")

st.markdown("---")

# ---- geospatial metadata ---------------------------------------------------
st.subheader("Geospatial metadata — Rayong AOI")

g1, g2 = st.columns([1, 2])
with g1:
    st.markdown("**Coordinate Reference System**")
    st.code(crs, language=None)
with g2:
    st.markdown("**Bounding box**")
    st.dataframe(pd.DataFrame([bounds]),
                 hide_index=True, use_container_width=True)

st.markdown("**Sentinel-2 raster details** (native S2 grid, three resolutions)")
res_choice = st.selectbox("Resolution", list(raster_meta.keys()), index=0)
m = raster_meta[res_choice]
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("resolution", m["resolution"])
r2.metric("width (px)", f"{m['width_px']:,}")
r3.metric("height (px)", f"{m['height_px']:,}")
r4.metric("bands", str(m["bands"]))
r5.metric("dtype", m["dtype"])

st.markdown("**LDD landuse shapefile**")
v1, v2, v3 = st.columns(3)
v1.metric("geometry",          vector_meta["geometry"])
v2.metric("feature layers",    str(vector_meta["feature_layers"]))
v3.metric("landuse polygons",  f"{vector_meta['landuse_polygons']:,}")

st.markdown("---")

# ---- class distribution ----------------------------------------------------
st.subheader("Class distribution")

# Source counts come from class_counts.parquet (full labelled pixel cube).
# Sampled counts come from dataset.parquet (the stratified subset).
# Synth counts come from synth_tabsyn.parquet when present.


def _per_class_counts(label_series: pd.Series) -> pd.Series:
    return (label_series.value_counts()
            .reindex(range(N_CLASSES), fill_value=0)
            .sort_index())


# Source: aggregate class_counts.parquet across years (LDD-coded)
src_label_dense = to_dense(class_counts_raw["label"].to_numpy())
src_counts_per = (
    pd.DataFrame({"label": src_label_dense,
                  "count": class_counts_raw["count"].to_numpy()})
    .query("label >= 0")
    .groupby("label")["count"].sum()
    .reindex(range(N_CLASSES), fill_value=0)
)

sampled_counts = _per_class_counts(df["label"])
synth_counts   = _per_class_counts(df_synth["label"]) if (df_synth is not None and "label" in df_synth.columns) else None

scope = st.radio(
    "Show",
    ["Source population", "Sampled subset", "Source vs sampled",
     *(["Sampled vs synth"] if synth_counts is not None else [])],
    horizontal=True,
    help=(
        "Source = full labelled pixel cube (`class_counts.parquet`). "
        "Sampled = stratified subset (`dataset.parquet`). "
        "Synth = TabSyn samples."
    ),
)

frames = []
if scope == "Source population":
    frames.append(("source", src_counts_per))
elif scope == "Sampled subset":
    frames.append(("sampled (dataset.parquet)", sampled_counts))
elif scope == "Source vs sampled":
    frames.append(("source", src_counts_per))
    frames.append(("sampled (dataset.parquet)", sampled_counts))
elif scope == "Sampled vs synth":
    frames.append(("sampled (dataset.parquet)", sampled_counts))
    frames.append(("synth", synth_counts))

bar_rows = []
for label, series in frames:
    for cid in range(N_CLASSES):
        bar_rows.append({
            "class":    CLASSES[cid],
            "LDD code": LDD_CODES[cid],
            "count":    int(series.iloc[cid]),
            "source":   label,
        })
plot_df = pd.DataFrame(bar_rows)

fig = px.bar(
    plot_df, x="class", y="count", color="source", barmode="group",
    log_y=True,
    color_discrete_map={
        "source":              "#2ca02c",
        "sampled (dataset.parquet)":  "#ffb347",
        "synth":               "#1f77b4",
    },
    category_orders={"class": CLASSES},
    hover_data={"LDD code": True, "count": ":,", "source": True},
)
fig.update_layout(height=400, xaxis_tickangle=-30,
                  margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

c_total_src = int(src_counts_per.sum())
c_total_smp = int(sampled_counts.sum())
maj_src = int(src_counts_per.max())
maj_smp = int(sampled_counts.max())
st.caption(
    f"**Source total:** {c_total_src:,} pixels (majority class: "
    f"{CLASSES[int(src_counts_per.idxmax())]} = {maj_src:,}).  "
    f"**Sampled subset:** {c_total_smp:,} rows "
    f"(majority class: {CLASSES[int(sampled_counts.idxmax())]} = {maj_smp:,}). "
    "Sampled subset is proportional to the source — class imbalance is preserved, "
    "not flattened."
)

# ---- feature stats ---------------------------------------------------------
st.subheader("Per-feature summary stats — sampled rows")
st.caption("describe() on the 15 monthly index features (Oct/Nov/Dec 2020 NDVI, EVI, NDWI, MTCI, SWIR).")
desc = df[FEATURE_COLS].describe().T
st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

# ---- per-class violin -----------------------------------------------------
st.subheader("Per-class feature distribution")
st.caption("Per-class violin distribution for the selected feature. Highlights which crops the feature separates.")
feat = st.selectbox("Feature", FEATURE_COLS, index=0)
df_plot = df[[feat, "label"]].copy()
df_plot["class"] = df_plot["label"].map(lambda c: CLASSES[c])
fig2 = px.violin(
    df_plot, x="class", y=feat, color="class",
    color_discrete_sequence=PALETTE_HEX,
    category_orders={"class": CLASSES},
    box=True, points=False,
)
fig2.update_layout(height=420, showlegend=False, xaxis_tickangle=-30,
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

# ---- correlation heatmaps -------------------------------------------------
st.subheader("Feature correlation")
st.caption("Pearson correlation across the 15 features. Red = positive, blue = negative.")


def corr_fig(matrix: np.ndarray, title: str) -> go.Figure:
    return go.Figure(
        data=go.Heatmap(z=matrix, x=FEATURE_COLS, y=FEATURE_COLS,
                        zmin=-1, zmax=1, colorscale="RdBu_r",
                        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>"),
        layout=dict(title=dict(text=title, x=0.02, xanchor="left"),
                    height=520, xaxis_tickangle=-45,
                    yaxis_autorange="reversed",
                    margin=dict(l=10, r=10, t=40, b=10)),
    )


if corr_syn is not None:
    if corr_syn.shape != corr_real.shape:
        # Synth correlation may include extra delta features. Crop to the
        # leading block matching the real correlation, which holds the same
        # base features in the same order.
        n = corr_real.shape[0]
        if corr_syn.shape[0] >= n and corr_syn.shape[1] >= n:
            corr_syn = corr_syn[:n, :n]
        else:
            corr_syn = None
if corr_syn is not None:
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(corr_fig(corr_real, "real"), use_container_width=True)
    cc2.plotly_chart(corr_fig(corr_syn,  "synthetic"), use_container_width=True)
    delta = corr_syn - corr_real
    st.caption(f"max |Δ| between real and synth correlation: **{np.abs(delta).max():.3f}** "
               f"· mean |Δ|: **{np.abs(delta).mean():.3f}** (lower = synth preserves real structure)")
else:
    st.plotly_chart(corr_fig(corr_real, "real"), use_container_width=True)
    st.caption("synthetic correlation will appear once `synth_tabsyn.parquet` exists.")

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))
