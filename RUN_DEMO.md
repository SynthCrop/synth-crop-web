# Running the demo

Quickstart for getting a local Streamlit instance up against the deploy artifacts.

## 1. Build deploy artifacts

```bash
# any python env with pandas + scikit-learn + joblib + pyarrow
python prepare_artifacts.py --source-csv path/to/preprocess_dataset.parquet
python prepare_artifacts.py --source-csv ... --tiny           # 10K-row run
python prepare_artifacts.py --source-csv ... --rows 100000
python prepare_artifacts.py --source-csv ... --with-smote --smote-target 5000
```

Each input path can also be supplied via environment variable:

| Path | CLI flag | Env var |
|---|---|---|
| Source dataset (LDD-coded) | `--source-csv` | `SYNTH_CROP_SOURCE` |
| TabSyn synthetic | `--synth-csv`  | `SYNTH_CROP_SYNTH` |
| Grid metadata JSON         | `--meta-json`  | `SYNTH_CROP_META` |
| Per-year preds parquet dir | `--preds-dir`  | `SYNTH_CROP_PREDS` |
| Pretrained RF joblib       | `--rf-model`   | `SYNTH_CROP_RF_MODEL` |

Outputs land in `deploy/artifacts/`:

| File | Required | Notes |
|---|---|---|
| `dataset.parquet` | ✓ | Stratified subset with dense 0..14 labels |
| `class_counts.parquet` | ✓ | Per-(year, class) counts |
| `feature_stats.parquet` | ✓ | Per-class summary stats for the 15 features |
| `corr_real.npy` | ✓ | 15×15 feature correlation |
| `metrics.json` | ✓ | RF params + per-class F1 + LDD→dense map |
| `confusion.npy` | ✓ | 15×15 confusion matrix |
| `feature_importance.parquet` | ✓ | RF feature importances |
| `rf_model.joblib` | optional | Trained RF (used by tools, not the deployed UI) |
| `synth_tabsyn.parquet` | optional | TabSyn synthetic (post-remap) |
| `corr_syn.npy` | optional | 15×15 correlation on TabSyn output |
| `synth_smote.parquet` | optional | SMOTE-generated synth (`--with-smote`) |
| `corr_smote.npy` | optional | 15×15 correlation on SMOTE output |
| `grid_meta.json` | optional | AOI transform / CRS / years |
| `preds_<year>.parquet` | optional | Per-pixel prediction tables |
| `metrics_smote.json` | optional | Metrics for SMOTE-augmented cascade (written by `notebooks/crop_smote_rf.ipynb`) |
| `confusion_smote.npy` | optional | 15×15 confusion matrix for SMOTE-augmented cascade |
| `feature_importance_smote.parquet` | optional | RF stage-1 importances for SMOTE-augmented cascade |
| `tabsyn_metrics.json` | optional | Metrics for TabSyn-augmented cascade (written by `notebooks/crop_tabsyn.ipynb`) |
| `tabsyn_confusion.npy` | optional | 15×15 confusion matrix for TabSyn-augmented cascade |
| `tabsyn_feature_importance.parquet` | optional | RF stage-1 importances for TabSyn-augmented cascade |
| `three_way_compare.csv` | optional | Side-by-side comparison row per method |

## 2. (Optional) Bake Sentinel-2 basemaps

Generates the satellite raster + colorized prediction overlay used by the
Segmentation and Synth Lab map views.

```bash
python prepare_basemap.py --safe-root path/to/rayong_raster
# or:    SYNTH_CROP_SAFE_ROOT=path/to/rayong_raster python prepare_basemap.py
```

Writes `basemap_<year>.jpg`, `basemap_<year>.json`, and `preds_<year>.png`
into `deploy/artifacts/`.

## 3. Run the app

```bash
pip install -r deploy/requirements.txt
streamlit run deploy/streamlit_app.py
```

Open `http://localhost:8501`.

## 4. Cloud deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. On Streamlit Community Cloud, create a new app pointing at
   `deploy/streamlit_app.py` with `deploy/requirements.txt`.
3. Either un-ignore `deploy/artifacts/` (use Git LFS for files >50 MB) or fetch
   them at boot from a GitHub Release asset.

### Page 2 (Temporal Change) shapefile layout

```
deploy/artifacts/2018/LU_RYG_2561.{shp,shx,dbf,prj,cpg,sbn,sbx}
deploy/artifacts/2020/LU_RYG_2563.{...}
deploy/artifacts/2024/LU_RYG_2567.{...}
```

Page 2 caches the EPSG:4326-reprojected version under
`deploy/artifacts/<year>/parquet/<prefix>.parquet` on first load.

## LDD → dense label mapping

Upstream uses LDD numeric codes; pages use dense 0..14 (LDD-sorted order):

```python
LABEL_MAP_LDD_TO_DENSE = {
    2101: 0,   # Rice
    2204: 1,   # Cassava
    2205: 2,   # Pineapple
    2302: 3,   # Para rubber
    2303: 4,   # Oil palm
    2403: 5,   # Durian
    2404: 6,   # Rambutan
    2405: 7,   # Coconut
    2407: 8,   # Mango
    2413: 9,   # Longan
    2416: 10,  # Jackfruit
    2419: 11,  # Mangosteen
    2420: 12,  # Langsat
    4201: 13,  # Reservoir
    9999: 14,  # Others
}
```

Rows with labels outside this map are dropped during remap. The inverse
mapping is stored in `metrics.json` for traceability.
