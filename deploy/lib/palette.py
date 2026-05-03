"""15-class color palette used across all pages."""
from __future__ import annotations

CLASSES = [
    "Rice", "Cassava", "Pineapple", "Para rubber", "Oil palm",
    "Durian", "Mango", "Jackfruit", "Coconut", "Mangosteen",
    "Longan", "Rambutan", "Langsat", "Reservoir", "Others",
]
N_CLASSES = len(CLASSES)
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

PALETTE_HEX = [
    "#e6c229",  # Rice          — straw
    "#a0522d",  # Cassava       — sienna
    "#f4a300",  # Pineapple     — amber
    "#1f7a3a",  # Para rubber   — deep green
    "#bcae00",  # Oil palm      — olive
    "#c46a3e",  # Durian        — terracotta
    "#ffb347",  # Mango         — mango
    "#9acd32",  # Jackfruit     — yellow-green
    "#8b6f47",  # Coconut       — coconut brown
    "#7b1e1e",  # Mangosteen    — dark red
    "#d2b48c",  # Longan        — tan
    "#d62728",  # Rambutan      — red
    "#9467bd",  # Langsat       — purple
    "#1f77b4",  # Reservoir     — blue
    "#7f7f7f",  # Others        — gray
]
assert len(PALETTE_HEX) == N_CLASSES


def palette_rgb255() -> list[tuple[int, int, int]]:
    return [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in PALETTE_HEX]


def class_color(cid: int) -> str:
    return PALETTE_HEX[cid] if 0 <= cid < N_CLASSES else "#000000"
