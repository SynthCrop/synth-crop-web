"""Page 3 — Segmentation: parcel inspector on Sentinel-2 imagery.

Sentinel-2 TCI basemap with the RF cascade prediction layered on top, plus
clickable LDD landuse polygons. Clicking a parcel aggregates the per-pixel
predictions inside that parcel against the ground-truth label and reports:

    - parcel attributes from the LDD shapefile (class, code, area)
    - count of test pixels inside the parcel
    - per-pixel agreement vs the global agreement
    - majority true class vs majority predicted class
    - full per-class histogram (true and predicted)
"""
from __future__ import annotations
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
import streamlit as st
from shapely.geometry import Point
from shapely.vectorized import contains as poly_contains
from streamlit_folium import st_folium

from lib.io import (
    artifact_status, artifacts_dir, load_json,
    list_pred_parquets, load_pred_parquet,
)
from lib.palette import (
    CLASSES, N_CLASSES, PALETTE_HEX, LDD_CODES, to_dense,
)

st.set_page_config(page_title="Segmentation", layout="wide")
st.title(":sparkles: Segmentation — parcel inspector")
st.caption(
    "Sentinel-2 imagery with the RF cascade prediction overlaid. Click any "
    "landuse parcel to see how the model's per-pixel predictions inside that "
    "parcel compare to the ground-truth label."
)

# ---- artifact gate ---------------------------------------------------------
status = artifact_status()
preds_files = list_pred_parquets()
if not preds_files:
    st.error(
        "no `preds_<year>.parquet` files in `deploy/artifacts/`. Run "
        "`prepare_artifacts.py --preds-dir <dir>` after producing predictions."
    )
    st.stop()
if not status["files"].get("grid_meta.json"):
    st.error(
        "`grid_meta.json` missing — required for the raster ↔ parcel "
        "coordinate transform."
    )
    st.stop()

grid_meta = load_json("grid_meta.json")
GRID_CRS  = grid_meta["crs"]
PX_W, _, ORIGIN_X, _, PX_H, ORIGIN_Y = grid_meta["transform"]
H_GRID, W_GRID = int(grid_meta["height"]), int(grid_meta["width"])

# ---- year + display controls -----------------------------------------------
YEAR_TO_THAI = {"2018": "2561", "2020": "2563", "2024": "2567"}
preds_years   = sorted(int(p.stem.split("_")[1]) for p in preds_files)
year_options  = [str(y) for y in preds_years if str(y) in YEAR_TO_THAI]
if not year_options:
    st.error("no overlap between `preds_<year>.parquet` years and shapefiles.")
    st.stop()

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
year_label   = ctrl1.selectbox("Year", year_options, index=len(year_options) - 1)
thai_year    = YEAR_TO_THAI[year_label]
year_int     = int(year_label)
opacity_pred = ctrl2.slider("Prediction overlay", 0.0, 1.0, 0.55, 0.05,
                             help="Opacity of the RF prediction layer over the S2 imagery.")
ctrl3.caption(
    f"LDD landuse polygons (clickable) on Sentinel-2 TCI for "
    f"**{year_label} ({thai_year} BE)** with the RF prediction overlay."
)

# ---- parcels ---------------------------------------------------------------
shp_dir      = artifacts_dir() / year_label
shp_path     = shp_dir / f"LU_RYG_{thai_year}.shp"
parquet_4326 = shp_dir / "parquet" / f"LU_RYG_{thai_year}_v2.parquet"

KEEP_FIELDS = ["LU_DES_EN", "LU_CODE", "LUL2_CODE", "Area_Rai", "geometry"]
SIMPLIFY_TOL_M = 15.0


@st.cache_data(show_spinner=False)
def load_parcels(shp_p: str, pq_p: str, target_crs: str):
    """Return (gdf_4326 for the map, gdf_grid for pixel-aligned containment tests)."""
    pq = Path(pq_p)
    if pq.exists():
        gdf_4326 = gpd.read_parquet(pq)
    else:
        pq.parent.mkdir(parents=True, exist_ok=True)
        gdf = gpd.read_file(shp_p, engine="pyogrio")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32647")
        keep = [c for c in KEEP_FIELDS if c in gdf.columns]
        gdf = gdf[keep].copy()
        gdf_m = gdf if str(gdf.crs).endswith(":32647") else gdf.to_crs("EPSG:32647")
        gdf_m["geometry"] = gdf_m.geometry.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
        gdf_4326 = gdf_m.to_crs("EPSG:4326")
        gdf_4326.to_parquet(pq, index=False)
    gdf_grid = gdf_4326.to_crs(target_crs)
    return gdf_4326, gdf_grid


if not shp_path.exists() and not parquet_4326.exists():
    st.error(f"shapefile not found: `{shp_path}`")
    st.stop()

with st.spinner("loading parcels..."):
    gdf_4326, gdf_grid = load_parcels(str(shp_path), str(parquet_4326), GRID_CRS)


# ---- predictions -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_preds_dense(year: int) -> pd.DataFrame:
    df = load_pred_parquet(f"preds_{year}.parquet")
    out = df[["row", "col"]].copy()
    out["label_true"] = to_dense(df["label_true"].to_numpy())
    out["label_pred"] = to_dense(df["label_pred"].to_numpy())
    return out[(out["label_true"] >= 0) & (out["label_pred"] >= 0)].reset_index(drop=True)


with st.spinner("loading predictions..."):
    preds = load_preds_dense(year_int)

agree_global = float((preds["label_true"] == preds["label_pred"]).mean())

# ---- header tiles ----------------------------------------------------------
h1, h2, h3, h4 = st.columns(4)
h1.metric("year",             year_label)
h2.metric("parcels",          f"{len(gdf_grid):,}")
h3.metric("test pixels",      f"{len(preds):,}")
h4.metric("global agreement", f"{agree_global*100:.2f}%")

st.markdown("---")

# ---- coordinate helpers ---------------------------------------------------
TR_TO_GRID = pyproj.Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)


def latlng_to_grid_xy(lat: float, lng: float) -> tuple[float, float]:
    return TR_TO_GRID.transform(lng, lat)


def grid_xy_to_pixel(x: float, y: float) -> tuple[int, int]:
    col = int((x - ORIGIN_X) / PX_W)
    row = int((y - ORIGIN_Y) / PX_H)
    return row, col


# ---- map ------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    fopt1, fopt2 = st.columns([2, 1])
    with fopt1:
        max_polys = st.select_slider(
            "Max parcels rendered (largest by area)",
            options=[2000, 4000, 6000, 8000, 12000, 20000, len(gdf_4326)],
            value=6000,
            help="Cap how many parcels go to the map. Lower = faster render.",
        )
    with fopt2:
        show_basemap = st.checkbox("Show S2 basemap", value=True)

    if "Area_Rai" in gdf_4326.columns and len(gdf_4326) > max_polys:
        gdf_show_4326 = gdf_4326.nlargest(max_polys, "Area_Rai").reset_index(drop=True)
    else:
        gdf_show_4326 = gdf_4326

    minx, miny, maxx, maxy = gdf_4326.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    fmap = folium.Map(location=center, zoom_start=10, tiles=None,
                       control_scale=True)
    folium.TileLayer("cartodbpositron", name="Carto basemap").add_to(fmap)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri satellite", attr="Esri", overlay=False, control=True,
    ).add_to(fmap)

    bm_jpg       = artifacts_dir() / f"basemap_{year_int}.jpg"
    bm_json_path = artifacts_dir() / f"basemap_{year_int}.json"
    pr_png       = artifacts_dir() / f"preds_{year_int}.png"
    bm_meta = None
    if bm_jpg.exists() and bm_json_path.exists():
        bm_meta = load_json(f"basemap_{year_int}.json")
        bounds = [bm_meta["lnglat_bounds"]["sw"], bm_meta["lnglat_bounds"]["ne"]]
        if show_basemap:
            folium.raster_layers.ImageOverlay(
                name=f"Sentinel-2 TCI ({year_label})",
                image=str(bm_jpg), bounds=bounds, opacity=1.0,
                interactive=False, zindex=400,
            ).add_to(fmap)
        if pr_png.exists() and opacity_pred > 0:
            folium.raster_layers.ImageOverlay(
                name=f"RF prediction ({year_label})",
                image=str(pr_png), bounds=bounds, opacity=opacity_pred,
                interactive=False, zindex=500,
            ).add_to(fmap)
    else:
        st.info(
            "Sentinel-2 basemap not baked. Run `python prepare_basemap.py` "
            "to generate the imagery + colorized prediction overlay."
        )

    folium.GeoJson(
        gdf_show_4326,
        name=f"Parcels {year_label}",
        style_function=lambda f: {
            "fillColor": "#000000", "color": "#ffffff",
            "weight": 0.6, "fillOpacity": 0.0,
        },
        highlight_function=lambda f: {
            "fillOpacity": 0.30, "color": "#ff6600", "weight": 1.8,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=[c for c in ["LU_DES_EN", "LU_CODE", "Area_Rai"]
                    if c in gdf_show_4326.columns],
            aliases=["Class:", "LU code:", "Area (rai):"],
            sticky=True,
        ),
        smooth_factor=0.5,
    ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    map_state = st_folium(fmap, width=None, height=620,
                           returned_objects=["last_clicked"])

# ---- click -> parcel inspector --------------------------------------------
with right:
    st.markdown("**Inspector**")
    click = (map_state or {}).get("last_clicked")

    if not click:
        st.info("Click any parcel on the map to inspect.")
        st.stop()

    lat, lng = click["lat"], click["lng"]
    x, y = latlng_to_grid_xy(lat, lng)
    pt_grid = Point(x, y)

    cand_idx = list(gdf_grid.sindex.query(pt_grid, predicate="intersects"))
    if not cand_idx:
        st.warning(f"No parcel under ({lat:.5f}, {lng:.5f}).")
        st.stop()
    cand = gdf_grid.iloc[cand_idx]
    hits = cand[cand.geometry.contains(pt_grid)]
    if len(hits) == 0:
        st.warning("Click outside any parcel.")
        st.stop()

    parcel = hits.iloc[0]
    poly = parcel.geometry

    a1, a2 = st.columns(2)
    a1.metric("class (LU_DES_EN)", str(parcel.get("LU_DES_EN", "—")))
    a2.metric("LU_CODE",           str(parcel.get("LU_CODE", "—")))
    a3, a4 = st.columns(2)
    a3.metric("LUL2_CODE",         str(parcel.get("LUL2_CODE", "—")))
    a4.metric("area (rai)",        f"{float(parcel.get('Area_Rai', float('nan'))):,.2f}"
                                    if "Area_Rai" in parcel else "—")

    minx_p, miny_p, maxx_p, maxy_p = poly.bounds
    ra, ca = grid_xy_to_pixel(minx_p, maxy_p)
    rb, cb = grid_xy_to_pixel(maxx_p, miny_p)
    row_lo, row_hi = sorted((ra, rb))
    col_lo, col_hi = sorted((ca, cb))

    bbox = preds[
        (preds["row"] >= row_lo) & (preds["row"] <= row_hi)
        & (preds["col"] >= col_lo) & (preds["col"] <= col_hi)
    ]
    if len(bbox) == 0:
        st.info("No labelled test pixels in this parcel's bounding box.")
        st.stop()

    px_x = ORIGIN_X + (bbox["col"].to_numpy() + 0.5) * PX_W
    px_y = ORIGIN_Y + (bbox["row"].to_numpy() + 0.5) * PX_H
    inside = poly_contains(poly, px_x, px_y)
    inside_df = bbox.loc[inside].reset_index(drop=True)

    if len(inside_df) == 0:
        st.warning(
            f"Parcel intersects bbox of {len(bbox):,} pixels but none fall "
            "strictly inside the polygon (small / sliver parcel)."
        )
        st.stop()

    n = len(inside_df)
    agree_parcel = float((inside_df["label_true"] == inside_df["label_pred"]).mean())
    true_mode = int(inside_df["label_true"].mode().iat[0])
    pred_mode = int(inside_df["label_pred"].mode().iat[0])

    b1, b2, b3 = st.columns(3)
    b1.metric("pixels in parcel", f"{n:,}")
    b2.metric("agreement",        f"{agree_parcel*100:.2f}%")
    b3.metric("Δ vs global",      f"{(agree_parcel - agree_global)*100:+.2f} pp")

    def swatch(cid: int) -> str:
        return (
            f"<span style='display:inline-block; width:12px; height:12px; "
            f"background:{PALETTE_HEX[cid]}; border:1px solid #444; "
            f"border-radius:2px; margin-right:6px; vertical-align:middle;'></span>"
        )

    match_icon = "✅" if true_mode == pred_mode else "⚠️"
    st.markdown(
        f"<div style='font-size:14px; line-height:1.7; margin-top:6px;'>"
        f"<b>Majority true:</b> {swatch(true_mode)}{CLASSES[true_mode]}"
        f" <span style='color:#888'>· LDD {LDD_CODES[true_mode]}</span><br>"
        f"<b>Majority pred:</b> {swatch(pred_mode)}{CLASSES[pred_mode]}"
        f" <span style='color:#888'>· LDD {LDD_CODES[pred_mode]}</span> "
        f"&nbsp;{match_icon}"
        f"</div>",
        unsafe_allow_html=True,
    )

    true_n = (inside_df["label_true"].value_counts()
              .reindex(range(N_CLASSES), fill_value=0))
    pred_n = (inside_df["label_pred"].value_counts()
              .reindex(range(N_CLASSES), fill_value=0))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="true", x=CLASSES, y=true_n.to_numpy(),
                          marker_color="#2ca02c",
                          hovertemplate="%{x}<br>true: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(name="pred", x=CLASSES, y=pred_n.to_numpy(),
                          marker_color="#1f77b4",
                          hovertemplate="%{x}<br>pred: %{y}<extra></extra>"))
    fig.update_layout(barmode="group", height=320,
                       xaxis_tickangle=-30,
                       margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.1),
                       yaxis_title="pixels")
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ---- legend ---------------------------------------------------------------
st.markdown("---")
st.subheader("Class legend")
legend_html = (
    "<div style='display:flex; flex-wrap:wrap; gap:6px 14px; padding:8px 4px;'>"
    + "".join(
        f"<div style='display:flex; align-items:center; gap:6px; "
        f"font-size:13px; color:#e6e6e6;'>"
        f"<span style='display:inline-block; width:14px; height:14px; "
        f"background:{PALETTE_HEX[i]}; border:1px solid #333; border-radius:3px;'></span>"
        f"<span><b>{CLASSES[i]}</b> "
        f"<span style='color:#888'>· LDD {LDD_CODES[i]}</span></span></div>"
        for i in range(N_CLASSES)
    )
    + "</div>"
)
st.markdown(legend_html, unsafe_allow_html=True)
