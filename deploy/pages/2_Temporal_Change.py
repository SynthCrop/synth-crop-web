"""Page 2 — Temporal Change: per-year LDD landuse polygons over a basemap.

Color is keyed to LUL2_CODE (broad land-use category): A=agriculture, F=forest,
M=miscellaneous, U=urban, W=water. Tooltip shows LU_DES_EN, LU_CODE, area in rai.
"""
from __future__ import annotations
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from lib.io import artifact_status, artifacts_dir

st.set_page_config(page_title="Temporal Change", layout="wide")
st.title(":earth_africa: Temporal Change")
st.caption(
    "LDD landuse polygons for the Rayong AOI. Pick a year to see what was on the "
    "ground that season — colored by broad category (LUL2_CODE), tooltip shows the "
    "fine-grained class and area in rai."
)

status = artifact_status()
if status["missing"]:
    st.error("required artifacts missing — run `prepare_artifacts.py`")
    st.stop()

YEAR_TO_THAI = {"2018": "2561", "2020": "2563", "2024": "2567"}
year_label = st.selectbox("Year", list(YEAR_TO_THAI.keys()), index=0)
thai_year  = YEAR_TO_THAI[year_label]

file_prefix = f"LU_RYG_{thai_year}"
year_dir    = artifacts_dir() / year_label
shp_path    = year_dir / f"{file_prefix}.shp"
# v2 cache: simplified geometry, kept fields only — much smaller GeoJSON for folium.
parquet_path = year_dir / "parquet" / f"{file_prefix}_v2.parquet"

# Broad-category palette — LUL2 first character is the category root
CATEGORY_COLOR = {
    "A": "#2ca02c",  # agriculture
    "F": "#1f7a3a",  # forest
    "M": "#bcbd22",  # miscellaneous
    "U": "#9467bd",  # urban
    "W": "#1f77b4",  # water
}
CATEGORY_NAME = {
    "A": "Agriculture",
    "F": "Forest",
    "M": "Miscellaneous",
    "U": "Urban / built-up",
    "W": "Water",
}


KEEP_FIELDS = ["LU_DES_EN", "LU_CODE", "LUL2_CODE", "Area_Rai", "geometry"]
SIMPLIFY_TOL_M = 15.0  # in source UTM metres; ~1 px at zoom 10 in folium


@st.cache_data(show_spinner=False)
def load_and_reproject(shp_path: str, parquet_path: str) -> gpd.GeoDataFrame:
    """Load shapefile, simplify geometry, reproject to EPSG:4326, cache as parquet."""
    pq = Path(parquet_path)
    if pq.exists():
        return gpd.read_parquet(pq)

    pq.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(shp_path, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:32647")  # Rayong UTM 47N (default for LDD shp)
    keep = [c for c in KEEP_FIELDS if c in gdf.columns]
    gdf = gdf[keep].copy()
    # simplify in metric CRS so the tolerance is interpretable, then reproject
    if str(gdf.crs).endswith(":4326"):
        gdf_m = gdf.to_crs("EPSG:32647")
        gdf_m["geometry"] = gdf_m.geometry.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
        gdf = gdf_m.to_crs("EPSG:4326")
    else:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_parquet(pq, index=False)
    return gdf


if not shp_path.exists() and not parquet_path.exists():
    st.error(
        f"shapefile not found: `{shp_path}`. "
        f"Expected `{file_prefix}.shp` (+ .dbf/.shx/.prj) under `{year_dir}`."
    )
    st.stop()

with st.spinner("Loading polygons..."):
    gdf = load_and_reproject(str(shp_path), str(parquet_path))

# ---- header tiles ----------------------------------------------------------
total_rai = float(gdf["Area_Rai"].sum()) if "Area_Rai" in gdf.columns else float("nan")
n_classes_l3 = int(gdf["LU_CODE"].nunique()) if "LU_CODE" in gdf.columns else 0
n_categories = int(gdf["LUL2_CODE"].astype(str).str[0].nunique()) if "LUL2_CODE" in gdf.columns else 0
h1, h2, h3, h4 = st.columns(4)
h1.metric("year (CE / BE)", f"{year_label} / {thai_year}")
h2.metric("polygons",       f"{len(gdf):,}")
h3.metric("LU_CODE classes", f"{n_classes_l3:,}")
h4.metric("total area (rai)", f"{total_rai:,.0f}" if total_rai == total_rai else "—")

# ---- filters ---------------------------------------------------------------
fcol1, fcol2 = st.columns([2, 1])
with fcol1:
    categories_present = sorted({str(c)[0] for c in gdf.get("LUL2_CODE", []) if str(c)})
    default = [c for c in categories_present if c in CATEGORY_COLOR]
    chosen = st.multiselect(
        "Show categories",
        options=categories_present,
        default=default,
        format_func=lambda c: f"{c} — {CATEGORY_NAME.get(c, 'Unknown')}",
        help="LUL2_CODE first letter: A=Agri, F=Forest, M=Misc, U=Urban, W=Water.",
    )
with fcol2:
    max_polys = st.select_slider(
        "Max polygons (largest by area)",
        options=[2000, 4000, 6000, 8000, 12000, 20000, len(gdf)],
        value=6000,
        help="Limit the polygons rendered to the largest N by area. Lower = faster map.",
    )

if "LUL2_CODE" in gdf.columns and chosen:
    gdf = gdf[gdf["LUL2_CODE"].astype(str).str[0].isin(chosen)].reset_index(drop=True)

if "Area_Rai" in gdf.columns and len(gdf) > max_polys:
    gdf = gdf.nlargest(max_polys, "Area_Rai").reset_index(drop=True)

if len(gdf) == 0:
    st.warning("no polygons after filter")
    st.stop()

# ---- map -------------------------------------------------------------------
minx, miny, maxx, maxy = gdf.total_bounds
center = [(miny + maxy) / 2, (minx + maxx) / 2]

m = folium.Map(
    location=center, zoom_start=10, tiles="cartodbpositron",
    control_scale=True,
)


def style_fn(feature: dict) -> dict:
    code = str(feature["properties"].get("LUL2_CODE", ""))[:1]
    return {
        "fillColor": CATEGORY_COLOR.get(code, "#888888"),
        "color":     "#222222",
        "weight":    0.4,
        "fillOpacity": 0.6,
    }


tooltip_fields = [c for c in
                  ["LU_DES_EN", "LU_CODE", "LUL2_CODE", "Area_Rai"]
                  if c in gdf.columns]
tooltip_aliases = {
    "LU_DES_EN": "Class:",
    "LU_CODE":   "LU code:",
    "LUL2_CODE": "Category:",
    "Area_Rai":  "Area (rai):",
}

folium.GeoJson(
    gdf,
    name=f"Landuse {year_label}",
    style_function=style_fn,
    tooltip=folium.features.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=[tooltip_aliases[f] for f in tooltip_fields],
        sticky=True,
    ),
    smooth_factor=0.5,
).add_to(m)

# Inline legend
legend_html = (
    "<div style='position:fixed; bottom:18px; left:18px; z-index:9999; "
    "background:rgba(255,255,255,0.92); padding:8px 12px; border-radius:6px; "
    "border:1px solid #ccc; font-size:12px; color:#222;'>"
    "<b>Category (LUL2)</b><br>"
    + "".join(
        f"<div style='display:flex; align-items:center; gap:6px; margin:2px 0;'>"
        f"<span style='display:inline-block; width:12px; height:12px; "
        f"background:{CATEGORY_COLOR[c]}; border:1px solid #444;'></span>"
        f"<span>{c} — {CATEGORY_NAME[c]}</span></div>"
        for c in CATEGORY_COLOR
    )
    + "</div>"
)
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

st.write(f"### Interactive map — {year_label}")
st_folium(m, width=None, height=620, returned_objects=["last_clicked"])

# ---- per-category breakdown ----------------------------------------------
if "LUL2_CODE" in gdf.columns and "Area_Rai" in gdf.columns:
    st.subheader("Area by category (rai)")
    gdf["category"] = gdf["LUL2_CODE"].astype(str).str[0]
    summary = (gdf.groupby("category")["Area_Rai"].agg(["sum", "count"])
               .rename(columns={"sum": "rai", "count": "polygons"})
               .reset_index()
               .sort_values("rai", ascending=False))
    summary["category"] = summary["category"].map(
        lambda c: f"{c} — {CATEGORY_NAME.get(c, 'Unknown')}")
    st.dataframe(
        summary.style.format({"rai": "{:,.0f}", "polygons": "{:,}"}),
        use_container_width=True, hide_index=True,
    )
