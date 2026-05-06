"""Page 2 — Temporal Change: year-to-year class transitions, Sankey, change maps."""
from __future__ import annotations
import streamlit as st

from lib.io import artifact_status

import folium
from streamlit_folium import st_folium
import geopandas as gpd
import os

st.title(":earth_africa: Temporal Change")

status = artifact_status()
if status["missing"]:
    st.error("artifacts missing — run `prepare_artifacts.py`")
    st.stop()


selected_year = st.selectbox("Select Year", ["2018", "2020", "2024"])

year_to_thai = {
    "2018": "2561",
    "2020": "2563",
    "2024": "2567"
}
thai_year = year_to_thai[selected_year]

file_prefix = f"LU_RYG_{thai_year}"
LOCAL_FILE_PATH = f"deploy/artifacts/{selected_year}/shapefile/{file_prefix}.shp"
PARQUET_FILE_PATH = f"deploy/artifacts/{selected_year}/parquet/{file_prefix}_optimized.parquet"

@st.cache_data
def load_and_reproject_data(shp_path, parquet_path): 

    #if os.path.exists(parquet_path):
        gdf = gpd.read_parquet(parquet_path)
        #st.toast(f"Read parquet: {parquet_path}")
        return gdf
    #return

    #os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        
    #gdf = gpd.read_file(shp_path, engine="pyogrio")
        
    #if gdf.crs is None:
        #st.warning("Original Shapefile is missing CRS data. Assuming UTM Zone 47N (EPSG:32647).")
        #gdf = gdf.set_crs("EPSG:32647") 
            
    #if gdf.crs != "EPSG:4326":
        #gdf = gdf.to_crs("EPSG:4326")
            
    #gdf.to_parquet(parquet_path, index=False)
    #st.toast(f"Saved optimized data to {parquet_path}")
    #return gdf

try:
    with st.spinner("Preparing map..."):
        
        if os.path.exists(PARQUET_FILE_PATH):
            gdf = load_and_reproject_data(LOCAL_FILE_PATH, PARQUET_FILE_PATH)
        else:
            st.error(f"No file at directory path")
        
        # --- 1. Apply column renaming and mapping ---
        # (Checking if 'LU_CODE' exists just in case it was already renamed)
        if 'LU_CODE' in gdf.columns:
            gdf = gdf.rename(columns={
                'LU_CODE': 'Area Code',
                'LU_DES_EN': 'Name',
                'LUL1_CODE': 'Type',
                'Area_Sqm': 'Area (Sqm)'
            })
            
            land_use_mapping = {
                'A': 'Agriculture', 
                'F': 'Forest', 
                'M': 'Miscellaneous', 
                'U': 'Urban and Built-up', 
                'W': 'Water Body'
            }
            gdf['Type'] = gdf['Type'].replace(land_use_mapping)

        # --- 2. Define colors for each Type ---
        color_map = {
            'Agriculture': '#2ca02c',        # Green
            'Forest': '#006400',             # Dark Green
            'Miscellaneous': '#7f7f7f',      # Gray
            'Urban and Built-up': '#d62728', # Red
            'Water Body': '#1f77b4'          # Blue
        }
        
        bounds = gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        attribute_columns = gdf.columns.drop(['geometry', 'Area Code']).tolist()
        tooltip_fields = attribute_columns[:4] if attribute_columns else None

        # --- 3. Apply dynamic style function ---
        folium.GeoJson(
            gdf,
            name=f"Spatial Data {selected_year}",
            style_function=lambda feature: {
                # Look up the color based on the 'Type' property. Default to 'gray' if not found.
                'fillColor': color_map.get(feature['properties'].get('Type'), 'gray'),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=folium.features.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # --- 4. Create and add custom HTML Legend ---
        legend_html = '''
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 170px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 10px; border-radius: 5px; color: black;
            ">
            <b style="color: black;">Land Use Type</b><br>
        '''
        for label, color in color_map.items():
            # I added color: black to the text line as well just to be perfectly safe
            legend_html += f'<i style="background:{color};width:12px;height:12px;float:left;margin-right:8px;margin-top:4px;"></i><span style="color: black;">{label}</span><br>'
        legend_html += '</div>'
        
        # Inject the HTML into the map
        m.get_root().html.add_child(folium.Element(legend_html))

        st.write(f"### Interactive Map Preview: {selected_year}")
        
        map_data = st_folium(m, width=800, height=600, returned_objects=["last_clicked"])

        st.success(f"Load data from: {PARQUET_FILE_PATH} successfully!")

except FileNotFoundError:
    st.error(f"Could not find the file at `{LOCAL_FILE_PATH}`. Please double-check the path.")
except Exception as e:
    st.error(f"An error occurred: {e}")