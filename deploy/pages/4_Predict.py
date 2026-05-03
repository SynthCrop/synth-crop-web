"""Page 4 — Predict: single-pixel sliders + batch CSV upload."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status, load_rf_model
from lib.infer import FEATURE_COLS, predict_single, predict_batch, validate_batch
from lib.palette import CLASSES

st.title(":dart: Predict")

status = artifact_status()
if not status["files"].get("rf_model.joblib"):
    st.error("`rf_model.joblib` missing — run `prepare_artifacts.py`")
    st.stop()

tab_single, tab_batch = st.tabs(["Single pixel", "Batch CSV"])

with tab_single:
    st.write("Move sliders to feed all 15 features (5 indices × 3 months).")
    cols = st.columns(5)
    values = []
    for i, name in enumerate(FEATURE_COLS):
        with cols[i % 5]:
            values.append(st.slider(name, -1.0, 10.0, 0.0, 0.01))
    if st.button("predict", type="primary"):
        model = load_rf_model()
        pred, proba = predict_single(model, values)
        st.success(f"prediction: **{CLASSES[pred]}** (cid={pred})")
        top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
        st.write("top-3:")
        for cid, p in top3:
            st.progress(float(p), text=f"{CLASSES[cid]}: {p:.2%}")

with tab_batch:
    f = st.file_uploader("upload CSV with 15 feature columns", type=["csv"])
    if f is not None:
        import pandas as pd
        df = pd.read_csv(f)
        ok, err = validate_batch(df)
        if not ok:
            st.error(err)
        else:
            model = load_rf_model()
            preds = predict_batch(model, df)
            df["pred_cid"] = preds
            df["pred_class"] = [CLASSES[c] for c in preds]
            st.dataframe(df.head(50))
            st.download_button("download predictions",
                               df.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")
            st.bar_chart(df["pred_class"].value_counts())
