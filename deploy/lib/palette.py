"""15-class palette + LDD-code helpers shared across pages.

Class order follows the **sorted LDD numeric code** order (matches
`grid_meta.json`, `class_counts.parquet`, `confusion.npy`, and
`metrics.json` → `pixel_test.f1_per_class`):

    idx  code   name
     0   2101   Rice
     1   2204   Cassava
     2   2205   Pineapple
     3   2302   Para rubber
     4   2303   Oil palm
     5   2403   Durian
     6   2404   Rambutan
     7   2405   Coconut
     8   2407   Mango
     9   2413   Longan
    10   2416   Jackfruit
    11   2419   Mangosteen
    12   2420   Langsat
    13   4201   Reservoir
    14   9999   Others
"""
from __future__ import annotations
import numpy as np

CLASSES = [
    "Rice", "Cassava", "Pineapple", "Para rubber", "Oil palm",
    "Durian", "Rambutan", "Coconut", "Mango", "Longan",
    "Jackfruit", "Mangosteen", "Langsat", "Reservoir", "Others",
]
N_CLASSES = len(CLASSES)
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

LDD_CODES = [
    2101, 2204, 2205, 2302, 2303,
    2403, 2404, 2405, 2407, 2413,
    2416, 2419, 2420, 4201, 9999,
]
LDD_TO_DENSE = {c: i for i, c in enumerate(LDD_CODES)}
DENSE_TO_LDD = {i: c for c, i in LDD_TO_DENSE.items()}

PALETTE_HEX = [
    "#e6c229",  # Rice         — straw
    "#a0522d",  # Cassava      — sienna
    "#f4a300",  # Pineapple    — amber
    "#1f7a3a",  # Para rubber  — deep green
    "#bcae00",  # Oil palm     — olive
    "#c46a3e",  # Durian       — terracotta
    "#d62728",  # Rambutan     — red
    "#8b6f47",  # Coconut      — coconut brown
    "#ffb347",  # Mango        — mango
    "#d2b48c",  # Longan       — tan
    "#9acd32",  # Jackfruit    — yellow-green
    "#7b1e1e",  # Mangosteen   — dark red
    "#9467bd",  # Langsat      — purple
    "#1f77b4",  # Reservoir    — blue
    "#7f7f7f",  # Others       — gray
]
assert len(PALETTE_HEX) == N_CLASSES

NODATA_HEX = "#1a1a1a"


def palette_rgb255() -> list[tuple[int, int, int]]:
    return [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in PALETTE_HEX]


def class_color(cid: int) -> str:
    return PALETTE_HEX[cid] if 0 <= cid < N_CLASSES else NODATA_HEX


def to_dense(arr) -> np.ndarray:
    """Coerce label array to dense 0..14.

    Accepts either dense indices (already 0..14) or raw LDD codes
    (2101..9999). Unknown values become -1 (nodata). Returns int8.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return a.astype(np.int8)
    sample = a[~np.isnan(a)] if a.dtype.kind == "f" else a
    if sample.size and sample.max() < N_CLASSES and sample.min() >= 0:
        return a.astype(np.int8)
    out = np.full(a.shape, -1, dtype=np.int8)
    for ldd, dense in LDD_TO_DENSE.items():
        out[a == ldd] = dense
    return out


def discrete_colorscale() -> list[list]:
    """Plotly discrete colorscale for dense 0..N-1 categorical heatmaps.

    Pair with `zmin=-0.5, zmax=N_CLASSES-0.5` so each integer maps to one band.
    """
    cs = []
    for i, hex_ in enumerate(PALETTE_HEX):
        cs.append([i / N_CLASSES, hex_])
        cs.append([(i + 1) / N_CLASSES, hex_])
    return cs
