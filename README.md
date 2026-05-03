# Crop Classification Web App

Sentinel-2 → 15-class crop classification, deployed as a Streamlit app on GitHub.

The app loads pre-baked artifacts only — heavy training stays in notebooks. Multiple
generative models live in this repo (TabSyn, VAE-BGM, GAN, LLM, Diffusion). The
deployed app currently uses TabSyn synthetic data plus a Random Forest baseline.

## Pages

- **Dataset** — class distribution, per-feature stats, real-vs-synthetic correlations.
- **Temporal Change** — year-to-year class transitions, Sankey flows, change maps.
- **Segmentation** — predicted-class rasters layered on a basemap.
- **Predict** — interactive classifier (single pixel + batch CSV).
- **Model Card** — metrics, per-class F1, hyperparameters.

## Layout

```
notebooks/             research notebooks (preprocess + each gen model)
models/                gen-model code, importable from notebooks
prepare_artifacts.py   builds deploy/artifacts/ from notebooks' outputs
deploy/                Streamlit app (reads artifacts only)
  streamlit_app.py
  pages/
  lib/
  artifacts/           gitignored, built locally
.streamlit/            theme + server config
```

## Quickstart

```bash
git clone <this-repo>
cd crop-classification-web
python -m venv .venv && source .venv/bin/activate
pip install -r deploy/requirements.txt

# build artifacts (needs the raw data + trained models, not in repo)
python prepare_artifacts.py

# preview locally
streamlit run deploy/streamlit_app.py
```

## Deploy

1. Push to `main`.
2. Streamlit Community Cloud → New app → point at `deploy/streamlit_app.py`.
3. Re-runs auto on every push.

## Status

Scaffold only. See `deployment_design.pdf` for the full design.
