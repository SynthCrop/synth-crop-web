"""Plotly chart helpers shared across pages."""
from __future__ import annotations
from typing import Sequence

from .palette import CLASSES, PALETTE_HEX


def class_bar(counts: Sequence[int], title: str = "Class distribution"):
    import plotly.graph_objects as go
    return go.Figure(
        data=[go.Bar(x=CLASSES, y=list(counts), marker_color=PALETTE_HEX)],
        layout=dict(title=title, yaxis_type="log",
                    xaxis_tickangle=-30, height=380),
    )


def confusion_heatmap(matrix):
    import plotly.graph_objects as go
    return go.Figure(
        data=go.Heatmap(z=matrix, x=CLASSES, y=CLASSES,
                        colorscale="Blues", showscale=True),
        layout=dict(title="Confusion matrix", height=520,
                    xaxis_tickangle=-30),
    )


def per_class_f1_bar(f1_scores: Sequence[float]):
    import plotly.graph_objects as go
    return go.Figure(
        data=[go.Bar(x=CLASSES, y=list(f1_scores), marker_color=PALETTE_HEX,
                     text=[f"{v:.2f}" for v in f1_scores], textposition="outside")],
        layout=dict(title="Per-class F1", yaxis_range=[0, 1.05],
                    xaxis_tickangle=-30, height=380),
    )
