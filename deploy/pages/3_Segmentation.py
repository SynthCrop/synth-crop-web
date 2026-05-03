"""Page 3 — Segmentation: predicted-class rasters layered on a basemap."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status, list_pred_rasters

st.title(":sparkles: Segmentation")

status = artifact_status()
if status["missing"]:
    st.error("artifacts missing — run `prepare_artifacts.py`")
    st.stop()

# TODO:
# - pydeck basemap
# - toggle layers per year (preds_*.tif → BitmapLayer)
# - 15-class legend
# - click → popup (class, prob, time-series)

rasters = list_pred_rasters()
st.write(f"available prediction rasters: {len(rasters)}")
for r in rasters:
    st.write(f"- `{r.name}`")
st.info("Segmentation page — to be implemented.")
