# Notebooks

Research notebooks. Each generative + classification model gets its own notebook; preprocessing is shared.

| Notebook | Purpose |
|---|---|
| `gis-workflow-project.ipynb` | Sentinel-2 → 15-class crop CSV/parquet preprocessing (legacy) |
| `rayong_rf_cascade.ipynb` | **Rayong full pipeline**: S2 → labelled parquet → 2-stage RF cascade (stage-1 + tropical-orchard specialist) |
| `crop_smote_rf.ipynb` | RF + SMOTE minority oversampling (comparison baseline) |
| `crop_tabsyn.ipynb` | TabSyn (latent diffusion over VAE) — minority synth + RF cascade compare |
| `crop_vae_bgm.ipynb` | VAE + Bayesian Gaussian Mixture synthetic data |
| `crop_gan.ipynb` | (planned) GAN-based tabular synthesizer |
| `crop_diffusion.ipynb` | (planned) standalone TabDDPM-style diffusion |
| `crop_llm.ipynb` | (planned) LLM-based tabular generation |

These are NOT run by the deployed web app — they produce the inputs that `prepare_artifacts.py` consumes.

## Heavy data layout (local, gitignored)

```
../full_dataset/
  Landuse_ryg/                Thai LDD shapefiles (LU_RYG_25{61,63,67}.shp)
  rayong_raster/              Sentinel-2 SAFE.zip per year/month
../baseline_dataset.parquet   per-pixel labelled features (~7 M rows × 21 cols)
../rf_model.joblib            cascade bundle (clf + clf_orchard + metadata)
../synthetic/<name>/          tabsyn / vae-bgm / smote / etc synthetic CSVs
```

## Architecture (rayong_rf_cascade.ipynb)

- **15 classes**: 14 LDD-coded crop targets + OTHER (9999)
- **25 features**: 5 indices (NDVI / EVI / MNDWI / MTCI / SWIR) × 3 months (Oct/Nov/Dec) + 10 temporal deltas
- **Drop logic**: urban + composite + mixed-plant + abandoned + aquaculture-pond + eucalyptus
- **Adaptive parcel buffer**: -10m if area ≥ 400m² else keep original (preserves rare-class small parcels)
- **Two-pass rasterise**: target overrides OTHER at conflict pixels
- **Pixel-stratified 80/10/10 split** (NOT SGKF — SGKF collapses minorities)
- **Stage-1 RF**: 15 classes, 300 trees, max_depth=36, balanced_subsample
- **Stage-2 specialist**: 8 tropical-orchard classes, refine stage-1 orchard predictions
- **Eval**: pixel-level + parcel-vote (mode of pixel preds per parcel) on val + test

See `../docs/WORKFLOW.md` for full spec + `../docs/diagrams/` for architecture SVGs.
