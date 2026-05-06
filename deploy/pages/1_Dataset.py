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
from lib.palette import CLASSES, N_CLASSES, PALETTE_HEX

st.set_page_config(page_title="Dataset", layout="wide")
st.title(":bar_chart: Dataset")

status = artifact_status()
if not status["files"].get("dataset.parquet"):
    st.error("`dataset.parquet` missing — run `python prepare_artifacts.py`")
    st.stop()

df = load_parquet("dataset.parquet")
class_counts = load_parquet("class_counts.parquet")
feature_stats = load_parquet("feature_stats.parquet")
corr_real = load_npy("corr_real.npy")
synth_present = status["files"].get("synth_tabsyn.parquet", False)
df_synth = load_parquet("synth_tabsyn.parquet") if synth_present else None
corr_syn = load_npy("corr_syn.npy") if status["files"].get("corr_syn.npy", False) else None

#Manually entered metadata - replace with reading from files if available
crs = "EPSG:4326 (WGS 84)"
bounds = {"West": 100.8405, "East": 101.8619, "North": 13.5616, "South": 12.5604}

raster_meta = {
    "10 m": {
        "Resolution": "10 m",
        "Width (pixels)": 10980,
        "Height (pixels)": 10980,
        "Band Count": 6,
        "Data Type": "Float32"
    },
    "20 m": {
        "Resolution": "20 m",
        "Width (pixels)": 5490,
        "Height (pixels)": 5490,
        "Band Count": 14,
        "Data Type": "Float32"
    },
    "60 m": {
        "Resolution": "60 m",
        "Width (pixels)": 1830,
        "Height (pixels)": 1830,
        "Band Count": 15,
        "Data Type": "Float32"
    }
}

vector_meta = {
    "Geometry Type": "Polygon",
    "Feature Count": 13,
    "Landuse Count": 41571
}


FEATURE_COLS = [c for c in df.columns
                if c not in ("label", "year", "row", "col")]

c1, c2, c3, c4 = st.columns(4)
c1.metric("rows", f"{len(df):,}")
c2.metric("classes", str(df["label"].nunique()))
c3.metric("years", ", ".join(map(str, sorted(df["year"].unique())))
          if "year" in df.columns else "—")
c4.metric("synth rows", f"{len(df_synth):,}" if df_synth is not None else "—")

st.markdown("---")

st.subheader("Geospatial Metadata")

sc1, sc2 = st.columns([1, 2])
with sc1:
    st.metric("**Coordinate Reference System (CRS)**", crs)
with sc2:
    st.write("**Bounding Box (Extent):**")
    bounds_df = pd.DataFrame([bounds])
    st.dataframe(bounds_df, hide_index=True, use_container_width=True)

st.markdown("---")

sc1, sc2 = st.columns([1, 2])
with sc1:
    st.markdown("**Raster Details**")
with sc2:
    view_mode = st.selectbox("Select Resolution", options=list(raster_meta.keys()))    

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Resolution", f"{raster_meta[view_mode]['Resolution']}")
c2.metric("Width (pixels)", str(raster_meta[view_mode]['Width (pixels)']))
c3.metric("Height (pixels)", str(raster_meta[view_mode]['Height (pixels)']))
c4.metric("Band Count", f"{raster_meta[view_mode]['Band Count']}")
c5.metric("Data Type", f"{raster_meta[view_mode]['Data Type']}")

st.markdown("---")

st.markdown("**Shapefile Details**")
c1, c2, c3 = st.columns(3)
c1.metric("Geometry", f"{vector_meta['Geometry Type']}")
c2.metric("Feature Count", str(vector_meta['Feature Count']))
c3.metric("Landuse Count", str(vector_meta['Landuse Count']))

st.markdown("---")

st.subheader("Class distribution")

real_counts = (df["label"].value_counts()
               .reindex(range(N_CLASSES), fill_value=0)
               .reset_index())
real_counts.columns = ["label", "count"]
real_counts["class"] = real_counts["label"].map(lambda c: CLASSES[c])
real_counts["source"] = "real"

frames = [real_counts]
if df_synth is not None and "label" in df_synth.columns:
    sc = (df_synth["label"].value_counts()
          .reindex(range(N_CLASSES), fill_value=0)
          .reset_index())
    sc.columns = ["label", "count"]
    sc["class"] = sc["label"].map(lambda c: CLASSES[c])
    sc["source"] = "synth"
    frames.append(sc)
plot_df = pd.concat(frames, ignore_index=True)

fig = px.bar(plot_df, x="class", y="count", color="source",
             barmode="group", log_y=True,
             color_discrete_map={"real": "#2ca02c", "synth": "#1f77b4"},
             category_orders={"class": CLASSES})
fig.update_layout(height=380, xaxis_tickangle=-30,
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-feature summary stats")
desc = df[FEATURE_COLS].describe().T
st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

st.subheader("Per-class feature distribution")
feat = st.selectbox("feature", FEATURE_COLS, index=0)
df_plot = df[[feat, "label"]].copy()
df_plot["class"] = df_plot["label"].map(lambda c: CLASSES[c])
fig2 = px.violin(df_plot, x="class", y=feat, color="class",
                 color_discrete_sequence=PALETTE_HEX,
                 category_orders={"class": CLASSES},
                 box=True, points=False)
fig2.update_layout(height=420, showlegend=False, xaxis_tickangle=-30,
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature correlation")

def corr_fig(matrix: np.ndarray, title: str) -> go.Figure:
    return go.Figure(
        data=go.Heatmap(z=matrix, x=FEATURE_COLS, y=FEATURE_COLS,
                        zmin=-1, zmax=1, colorscale="RdBu_r"),
        layout=dict(title=title, height=520,
                    xaxis_tickangle=-45, yaxis_autorange="reversed",
                    margin=dict(l=10, r=10, t=40, b=10)),
    )

if corr_syn is not None:
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(corr_fig(corr_real, "real"), use_container_width=True)
    cc2.plotly_chart(corr_fig(corr_syn,  "synthetic"), use_container_width=True)
else:
    st.plotly_chart(corr_fig(corr_real, "real"), use_container_width=True)
    st.caption("synthetic correlation will appear once `synth_tabsyn.parquet` exists")

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))