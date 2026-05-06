"""Page 4 — Synthetic Data Lab.

The point of this project (synth-crop-web) is *augmenting minority crop classes
with synthetic samples*. This page exposes the synthetic side directly so it
can be inspected on its own terms (the Segmentation page already covers the
classifier output).

Content:
  - Class-balance lift (real vs SMOTE vs TabSyn).
  - Marginal feature distributions (real vs synth) for any feature × class.
  - 2D PCA projection of real and synth in shared feature space.
  - Per-feature 1-Wasserstein distance (real vs synth) per class.
  - Correlation drift (|corr_real − corr_synth|).
"""
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

st.set_page_config(page_title="Synthetic Data Lab", layout="wide")
st.title(":test_tube: Synthetic Data Lab")
st.caption(
    "Inspect synthetic samples generated for minority crops. "
    "Real = `dataset.parquet` (stratified subset). "
    "SMOTE = `synth_smote.parquet`. TabSyn = `synth_tabsyn.parquet`. "
    "Metrics computed on the 15 monthly Sentinel-2 features."
)

status = artifact_status()
if not status["files"].get("dataset.parquet"):
    st.error("`dataset.parquet` missing — run `python prepare_artifacts.py`")
    st.stop()

FEATURE_COLS = [
    "ndvi 10", "evi 10", "ndwi 10", "mtci 10", "swir 10",
    "ndvi 11", "evi 11", "ndwi 11", "mtci 11", "swir 11",
    "ndvi 12", "evi 12", "ndwi 12", "mtci 12", "swir 12",
]


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map upstream `mndwi *` columns onto the canonical `ndwi *` schema."""
    rename = {c: c.replace("mndwi", "ndwi")
              for c in df.columns if c.startswith("mndwi")}
    return df.rename(columns=rename) if rename else df


@st.cache_data(show_spinner=False)
def _load_real() -> pd.DataFrame:
    df = _normalize_features(load_parquet("dataset.parquet")).copy()
    df["label"] = to_dense(df["label"].to_numpy())
    keep = [c for c in FEATURE_COLS + ["label"] if c in df.columns]
    return df[df["label"] >= 0][keep].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_synth(name: str) -> pd.DataFrame | None:
    if not status["files"].get(name):
        return None
    df = _normalize_features(load_parquet(name)).copy()
    df["label"] = to_dense(df["label"].to_numpy())
    keep = [c for c in FEATURE_COLS + ["label"] if c in df.columns]
    return df[keep][df["label"] >= 0].reset_index(drop=True)


real  = _load_real()
smote = _load_synth("synth_smote.parquet")
tabsyn = _load_synth("synth_tabsyn.parquet")

# Use the intersection of FEATURE_COLS actually present across loaded sources,
# so downstream code never references a missing column.
_present = set(real.columns)
for d in (smote, tabsyn):
    if d is not None:
        _present &= set(d.columns)
FEATURE_COLS = [c for c in FEATURE_COLS if c in _present]

if smote is None and tabsyn is None:
    st.error(
        "No synthetic artifacts found. Run `prepare_artifacts.py --with-smote` "
        "and pass `--synth-csv` to populate `synth_smote.parquet` / "
        "`synth_tabsyn.parquet`."
    )
    st.stop()

# ---- header tiles ----------------------------------------------------------
def _classes_in(df: pd.DataFrame | None) -> set[int]:
    return set() if df is None else set(int(c) for c in df["label"].unique())


real_classes  = _classes_in(real)
smote_classes = _classes_in(smote)
tabsyn_classes = _classes_in(tabsyn)
augmented = sorted((smote_classes | tabsyn_classes))

h1, h2, h3, h4 = st.columns(4)
h1.metric("real rows",        f"{len(real):,}")
h2.metric("SMOTE rows",       f"{len(smote):,}" if smote is not None else "—")
h3.metric("TabSyn rows",      f"{len(tabsyn):,}" if tabsyn is not None else "—")
h4.metric("augmented classes", f"{len(augmented)} / {N_CLASSES}")

st.markdown("---")

# ---- tabs -----------------------------------------------------------------
tab_balance, tab_marg, tab_pca, tab_dist, tab_corr = st.tabs([
    ":bar_chart: Class lift",
    ":chart_with_upwards_trend: Marginal distributions",
    ":dna: PCA projection",
    ":triangular_ruler: Distance metrics",
    ":fire: Correlation drift",
])

# ---- tab 1 : class lift ---------------------------------------------------
with tab_balance:
    st.subheader("Per-class row count — real vs synth")
    st.caption(
        "Synthetic generators in this project target the minority crops "
        "(Rambutan, Coconut, Longan, Mangosteen, Langsat) and leave the "
        "well-represented classes unchanged."
    )

    def per_class_counts(df: pd.DataFrame | None) -> np.ndarray:
        if df is None:
            return np.zeros(N_CLASSES, dtype=int)
        return (df["label"].value_counts()
                .reindex(range(N_CLASSES), fill_value=0)
                .sort_index().to_numpy())

    real_n   = per_class_counts(real)
    smote_n  = per_class_counts(smote)
    tabsyn_n = per_class_counts(tabsyn)

    rows = []
    for cid in range(N_CLASSES):
        rows.append({"class": CLASSES[cid], "LDD": LDD_CODES[cid],
                     "source": "real",   "count": int(real_n[cid])})
        rows.append({"class": CLASSES[cid], "LDD": LDD_CODES[cid],
                     "source": "SMOTE",  "count": int(smote_n[cid])})
        rows.append({"class": CLASSES[cid], "LDD": LDD_CODES[cid],
                     "source": "TabSyn", "count": int(tabsyn_n[cid])})
    plot_df = pd.DataFrame(rows)
    fig = px.bar(plot_df, x="class", y="count", color="source", barmode="group",
                  log_y=True,
                  color_discrete_map={"real": "#2ca02c", "SMOTE": "#ffb347",
                                      "TabSyn": "#1f77b4"},
                  category_orders={"class": CLASSES},
                  hover_data={"LDD": True, "count": ":,"})
    fig.update_layout(height=420, xaxis_tickangle=-30,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- tab 2 : marginal distributions -------------------------------------
with tab_marg:
    st.subheader("Marginal feature distribution — real vs synth")
    st.caption("Pick a feature and class; bars are histograms (50 bins) over the "
               "feature for each source. If real and synth shapes overlap closely, "
               "the generator captures that feature's distribution.")

    c1, c2 = st.columns(2)
    feat = c1.selectbox("Feature", FEATURE_COLS, index=0)
    augmented_or_all = sorted(set(real_classes) | set(augmented))
    cls = c2.selectbox(
        "Class",
        augmented_or_all,
        format_func=lambda i: f"{CLASSES[i]}  (LDD {LDD_CODES[i]})",
        index=augmented_or_all.index(augmented[0]) if augmented else 0,
    )

    real_sel  = real[real["label"] == cls][feat].to_numpy()
    smote_sel = smote[smote["label"] == cls][feat].to_numpy() if smote is not None else None
    tabsyn_sel = tabsyn[tabsyn["label"] == cls][feat].to_numpy() if tabsyn is not None else None

    fig = go.Figure()
    if real_sel.size:
        fig.add_trace(go.Histogram(
            x=real_sel, nbinsx=50, name=f"real (n={real_sel.size:,})",
            marker_color="#2ca02c", opacity=0.55, histnorm="probability density",
        ))
    if smote_sel is not None and smote_sel.size:
        fig.add_trace(go.Histogram(
            x=smote_sel, nbinsx=50, name=f"SMOTE (n={smote_sel.size:,})",
            marker_color="#ffb347", opacity=0.55, histnorm="probability density",
        ))
    if tabsyn_sel is not None and tabsyn_sel.size:
        fig.add_trace(go.Histogram(
            x=tabsyn_sel, nbinsx=50, name=f"TabSyn (n={tabsyn_sel.size:,})",
            marker_color="#1f77b4", opacity=0.55, histnorm="probability density",
        ))
    fig.update_layout(barmode="overlay", height=380,
                       xaxis_title=feat, yaxis_title="density",
                       legend=dict(orientation="h", y=1.08),
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    def _row(name, x):
        if x is None or len(x) == 0:
            return {"source": name, "n": 0, "mean": float("nan"),
                    "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {"source": name, "n": len(x), "mean": float(np.mean(x)),
                "std": float(np.std(x)), "min": float(np.min(x)),
                "max": float(np.max(x))}

    stats = pd.DataFrame([_row("real", real_sel), _row("SMOTE", smote_sel),
                           _row("TabSyn", tabsyn_sel)])
    st.dataframe(stats.style.format(
        {"mean": "{:.3f}", "std": "{:.3f}", "min": "{:.3f}",
         "max": "{:.3f}", "n": "{:,}"}),
        hide_index=True, use_container_width=True)

# ---- tab 3 : PCA projection ---------------------------------------------
with tab_pca:
    st.subheader("PCA projection — shared feature space")
    st.caption(
        "PCA fitted on real samples; synth samples projected onto the same axes. "
        "Synthetic clouds should overlap real clouds for the same class."
    )

    cls_pca = st.selectbox(
        "Class to plot",
        augmented or list(real_classes),
        format_func=lambda i: f"{CLASSES[i]}  (LDD {LDD_CODES[i]})",
        key="pca_class",
    )
    max_per_source = st.slider("Max points per source", 200, 5000, 1500, step=100,
                                key="pca_max")

    def _sample(df, cid, n):
        sel = df[df["label"] == cid][FEATURE_COLS].to_numpy()
        if len(sel) == 0:
            return sel
        if len(sel) > n:
            idx = np.random.default_rng(42).choice(len(sel), n, replace=False)
            sel = sel[idx]
        return sel

    real_pts = _sample(real, cls_pca, max_per_source)
    if real_pts.size == 0:
        st.warning("No real samples for this class — pick another.")
    else:
        from sklearn.decomposition import PCA  # lazy import — heavy module
        pca = PCA(n_components=2, random_state=42).fit(real_pts)
        scatter_rows = []
        for name, df_, color in [("real", real, "#2ca02c"),
                                  ("SMOTE", smote, "#ffb347"),
                                  ("TabSyn", tabsyn, "#1f77b4")]:
            if df_ is None:
                continue
            pts = _sample(df_, cls_pca, max_per_source)
            if pts.size == 0:
                continue
            xy = pca.transform(pts)
            scatter_rows.append((name, xy, color))

        fig = go.Figure()
        for name, xy, color in scatter_rows:
            fig.add_trace(go.Scattergl(
                x=xy[:, 0], y=xy[:, 1], mode="markers",
                name=f"{name} (n={len(xy):,})",
                marker=dict(size=4, color=color, opacity=0.55),
                hovertemplate=f"{name}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>",
            ))
        ev1, ev2 = pca.explained_variance_ratio_[:2]
        fig.update_layout(
            height=520,
            xaxis_title=f"PC1 ({ev1*100:.1f}% var)",
            yaxis_title=f"PC2 ({ev2*100:.1f}% var)",
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Total variance captured by PC1+PC2: **{(ev1+ev2)*100:.1f}%**. "
            "Closer point clouds = the synth distribution matches real more closely."
        )

# ---- tab 4 : distance metrics -------------------------------------------
with tab_dist:
    st.subheader("Per-feature 1-Wasserstein distance — real vs synth")
    st.caption(
        "Computed per (class × feature). Lower = synth marginal closer to real. "
        "Wasserstein distance is in the same units as the feature."
    )

    src = st.radio("Synth source",
                    [s for s, df_ in [("SMOTE", smote), ("TabSyn", tabsyn)] if df_ is not None],
                    horizontal=True)
    synth_df = smote if src == "SMOTE" else tabsyn

    @st.cache_data(show_spinner=True)
    def wass_table(src_name: str) -> pd.DataFrame:
        from scipy.stats import wasserstein_distance  # lazy import
        df_synth = smote if src_name == "SMOTE" else tabsyn
        # pre-bucket per-class numpy arrays once per source
        real_groups  = {int(cid): g[FEATURE_COLS].to_numpy()
                         for cid, g in real.groupby("label")}
        synth_groups = {int(cid): g[FEATURE_COLS].to_numpy()
                         for cid, g in df_synth.groupby("label")}
        out = []
        for cid in sorted(set(real_groups) & set(synth_groups)):
            r_arr = real_groups[cid]
            s_arr = synth_groups[cid]
            row = {"class": CLASSES[cid], "LDD": LDD_CODES[cid],
                   "n_real": len(r_arr), "n_synth": len(s_arr)}
            ws = []
            for j, f in enumerate(FEATURE_COLS):
                w = (float(wasserstein_distance(r_arr[:, j], s_arr[:, j]))
                     if (r_arr.size and s_arr.size) else float("nan"))
                row[f] = w
                ws.append(w)
            row["mean W"] = float(np.nanmean(ws))
            out.append(row)
        return pd.DataFrame(out).sort_values("mean W")

    wass_df = wass_table(src)
    st.dataframe(
        wass_df.style.format(
            {**{f: "{:.3f}" for f in FEATURE_COLS},
             "mean W": "{:.3f}", "n_real": "{:,}", "n_synth": "{:,}"})
        .background_gradient(subset=FEATURE_COLS + ["mean W"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Lower row = synth distribution for that class is closer to real overall. "
        f"Source: **{src}**."
    )

# ---- tab 5 : correlation drift -------------------------------------------
with tab_corr:
    st.subheader("Correlation drift — |corr_real − corr_synth|")
    st.caption(
        "How well synth preserves the pairwise correlation structure of the real "
        "features. Cooler = closer to real; hotter = drift."
    )

    corr_real = load_npy("corr_real.npy") if status["files"].get("corr_real.npy") else None
    if corr_real is None:
        st.error("`corr_real.npy` missing.")
    else:
        N_REAL = corr_real.shape[0]
        cols = st.columns(2)
        for slot, name, fname in [(cols[0], "SMOTE", "corr_smote.npy"),
                                    (cols[1], "TabSyn", "corr_syn.npy")]:
            if not status["files"].get(fname):
                slot.info(f"`{fname}` missing — skip {name}.")
                continue
            arr = load_npy(fname)
            if arr.shape != corr_real.shape:
                # Synth correlation may include extra delta features. Crop to
                # the leading N_REAL × N_REAL block, which holds the same base
                # features in matching order.
                if arr.shape[0] >= N_REAL and arr.shape[1] >= N_REAL:
                    arr = arr[:N_REAL, :N_REAL]
                else:
                    slot.warning(
                        f"`{fname}` shape {arr.shape} smaller than "
                        f"`corr_real.npy` {corr_real.shape}; skipping {name}."
                    )
                    continue
            delta = np.abs(arr - corr_real)
            fig = go.Figure(data=go.Heatmap(
                z=delta, x=FEATURE_COLS, y=FEATURE_COLS,
                zmin=0, zmax=max(0.05, float(delta.max())),
                colorscale="OrRd",
                hovertemplate="%{x} × %{y}<br>|Δ| = %{z:.3f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"|Δ| {name}  (max={float(delta.max()):.3f}, "
                                f"mean={float(delta.mean()):.3f})",
                            x=0.02, xanchor="left"),
                height=480, xaxis_tickangle=-45,
                yaxis_autorange="reversed",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            slot.plotly_chart(fig, use_container_width=True)

with st.expander("artifact paths"):
    st.code(str(artifacts_dir()))
