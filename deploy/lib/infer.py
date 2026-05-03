"""Random Forest inference helpers — single pixel + batch CSV."""
from __future__ import annotations
from typing import Iterable

FEATURE_COLS = [
    "ndvi 10", "evi 10", "ndwi 10", "mtci 10", "swir 10",
    "ndvi 11", "evi 11", "ndwi 11", "mtci 11", "swir 11",
    "ndvi 12", "evi 12", "ndwi 12", "mtci 12", "swir 12",
]


def validate_batch(df) -> tuple[bool, str]:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"
    if df[FEATURE_COLS].isna().any().any():
        return False, "NaN values present in feature columns"
    return True, ""


def predict_batch(model, df):
    return model.predict(df[FEATURE_COLS].to_numpy())


def predict_single(model, values: Iterable[float]):
    import numpy as np
    x = np.asarray(list(values), dtype="float32").reshape(1, -1)
    proba = model.predict_proba(x)[0]
    pred = int(proba.argmax())
    return pred, proba
