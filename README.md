# synth-crop-web

Sentinel-2 + LDD landuse → 15-class crop classification, served as a Streamlit app.

The app reads only pre-baked artifacts; training stays in notebooks. Synthetic
augmentation uses **TabSyn** (latent diffusion over a VAE) and **SMOTE**; the
classifier is a 2-stage **Random Forest cascade** (stage-1 generalist plus a
tropical-orchard specialist). AOI: Rayong, Thailand. Years: 2018, 2020, 2024
(LDD calendars 2561 / 2563 / 2567 BE).

## Pages

| # | Page | Reads |
|---|---|---|
| 1 | **Dataset** — class distribution, per-feature stats, real-vs-synth correlation | `dataset.parquet`, `class_counts.parquet`, `feature_stats.parquet`, `corr_real.npy`, `corr_syn.npy`, `synth_tabsyn.parquet` |
| 2 | **Temporal Change** — per-year landuse polygons over a folium basemap; click any parcel to open a modal that compares the same point across all three years | `<year>/LU_RYG_<thai>.{shp,shx,dbf,prj,cpg}` for every available year |
| 3 | **Segmentation** — Sentinel-2 basemap with RF prediction overlay; click a parcel to compare per-pixel predictions to the ground truth | `preds_<year>.parquet`, `grid_meta.json`, `basemap_<year>.{jpg,json}`, `preds_<year>.png`, `<year>/LU_RYG_<thai>.shp` |
| 4 | **Synth Lab** — TabSyn / SMOTE quality (class lift, marginals, PCA, Wasserstein, correlation drift) | `dataset.parquet`, `synth_tabsyn.parquet`, `synth_smote.parquet`, `corr_real.npy`, `corr_syn.npy`, `corr_smote.npy` |
| 5 | **Model Card** — variant selector + side-by-side comparison (baseline / SMOTE / TabSyn): hyperparameters, per-class F1, confusion matrix, feature importance | `metrics.json`, `confusion.npy`, `feature_importance.parquet`, `metrics_smote.json` *(opt)*, `confusion_smote.npy` *(opt)*, `feature_importance_smote.parquet` *(opt)*, `tabsyn_metrics.json` *(opt)*, `tabsyn_confusion.npy` *(opt)*, `tabsyn_feature_importance.parquet` *(opt)*, `three_way_compare.csv` *(opt)* |

## Layout

```
notebooks/                       Sentinel-2 prep, generative models, RF cascade
  rayong_rf_cascade.ipynb        full pipeline (S2 -> labelled parquet -> 2-stage RF)
  crop_tabsyn.ipynb              TabSyn synth + RF comparison
  crop_smote_rf.ipynb            SMOTE baseline
  crop_vae_bgm.ipynb             VAE + Bayesian Gaussian Mixture
docs/
  WORKFLOW.md                    pipeline specification
  diagrams/                      architecture SVGs
prepare_artifacts.py             builds deploy/artifacts/ from notebook outputs
prepare_basemap.py               bakes Sentinel-2 basemaps + colorized preds rasters
deploy/
  streamlit_app.py
  pages/                         1_Dataset .. 5_Model_Card
  lib/                           io, palette
  requirements.txt
  artifacts/                     gitignored - built by prepare_artifacts.py
.streamlit/                      theme + server config
RUN_DEMO.md                      step-by-step run + cloud-deploy guide
```

## Quickstart

```bash
git clone <this-repo>
cd synth-crop-web
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r deploy/requirements.txt

# Build artifacts (point at your upstream RF + TabSyn outputs)
python prepare_artifacts.py --source-csv path/to/preprocess_dataset.parquet

# (Optional) bake Sentinel-2 basemaps for the map views
python prepare_basemap.py --safe-root path/to/rayong_raster

# Preview locally
streamlit run deploy/streamlit_app.py
```

Open `http://localhost:8501`. Each input path can also be supplied via an
environment variable:

| CLI flag        | Environment variable     |
|-----------------|--------------------------|
| `--source-csv`  | `SYNTH_CROP_SOURCE`      |
| `--synth-csv`   | `SYNTH_CROP_SYNTH`       |
| `--meta-json`   | `SYNTH_CROP_META`        |
| `--preds-dir`   | `SYNTH_CROP_PREDS`       |
| `--rf-model`    | `SYNTH_CROP_RF_MODEL`    |
| `--safe-root`   | `SYNTH_CROP_SAFE_ROOT`   |

## Cloud deploy

`RUN_DEMO.md` covers the deploy path. In short:

1. Trim `deploy/artifacts/` to just what the app reads.
2. Either un-ignore `deploy/artifacts/**` (use Git LFS for files larger than
   50 MB) or fetch artifacts at boot from a GitHub Release asset.
3. On Streamlit Community Cloud: New app → `deploy/streamlit_app.py` →
   `deploy/requirements.txt`.

## Pipeline

```
Sentinel-2 SAFE.zip --+
LDD shapefile         +-> rayong_rf_cascade.ipynb -> preprocess_dataset.parquet
                                                  -> rf_model.joblib (cascade)
                                                  -> preds_<year>.parquet

preprocess_dataset.parquet -> crop_tabsyn.ipynb   -> tabsyn synthetic
                           -> crop_smote_rf.ipynb -> smote rows

all of the above -> prepare_artifacts.py -> deploy/artifacts/ <- streamlit app
```

Class order (dense index : LDD code → name) — LDD-sorted to match
`class_counts.parquet`, `confusion.npy`, and `metrics.json`:

```
0: 2101 Rice         5: 2403 Durian        10: 2416 Jackfruit
1: 2204 Cassava      6: 2404 Rambutan      11: 2419 Mangosteen
2: 2205 Pineapple    7: 2405 Coconut       12: 2420 Langsat
3: 2302 Para rubber  8: 2407 Mango         13: 4201 Reservoir
4: 2303 Oil palm     9: 2413 Longan        14: 9999 Others
```

## Dev

```bash
pip install -r requirements-dev.txt
```

Notebooks are not exercised by the deployed app; they feed `prepare_artifacts.py`.
