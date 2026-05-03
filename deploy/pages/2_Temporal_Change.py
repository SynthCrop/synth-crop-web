"""Page 2 — Temporal Change: year-to-year class transitions, Sankey, change maps."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status

st.title(":earth_africa: Temporal Change")

status = artifact_status()
if status["missing"]:
    st.error("artifacts missing — run `prepare_artifacts.py`")
    st.stop()

# TODO:
# - year picker (2018 / 2020 / 2024)
# - Sankey: class flow year→year (inner-join on row,col)
# - per-class delta bar
# - change map: pixels where label changed

st.info("Temporal Change page — to be implemented.")
