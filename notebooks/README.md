# Notebooks

Research notebooks. Each generative model gets its own notebook; preprocessing
is shared.

| Notebook | Purpose |
|---|---|
| `gis-workflow-project.ipynb` | Sentinel-2 → 15-class crop CSV/parquet preprocessing |
| `crop_vae_bgm.ipynb` | VAE + Bayesian Gaussian Mixture synthetic data |
| `crop_tabsyn.ipynb` | TabSyn (latent diffusion over VAE) — paper baseline |
| `crop_gan.ipynb` | (planned) GAN-based tabular synthesizer |
| `crop_diffusion.ipynb` | (planned) standalone TabDDPM-style diffusion |
| `crop_llm.ipynb` | (planned) LLM-based tabular generation |

These are NOT run by the deployed web app — they produce the inputs that
`prepare_artifacts.py` consumes.

## Heavy data layout (local, gitignored)

```
../full_dataset/
  Landuse_ryg/                Thai LDD shapefiles
  rayong_raster/              Sentinel-2 SAFE.zip per year/month
../2020_2024_dataset.parquet  primary preprocessed dataset
../random_forest_model.joblib trained RF
../synthetic/crop_full/       TabSyn / VAE-BGM / etc synthetic CSVs
```
