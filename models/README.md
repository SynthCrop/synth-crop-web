# Generative model registry

Each subdir = one generative model. Public surface (planned):

```python
from models.tabsyn import load, sample
model = load(ckpt_dir="...")
df = sample(model, n=10_000)
```

| Model | Status | Notes |
|---|---|---|
| `tabsyn/` | active | latent diffusion over VAE; arXiv:2310.09656 |
| `vae_bgm/` | active | VAE + Bayesian Gaussian Mixture |
| `gan/` | planned | CTGAN / TVAE candidate |
| `diffusion/` | planned | TabDDPM-style |
| `llm/` | planned | LLM tabular gen |

Adding a new model: drop a folder, expose `load()` + `sample(n)`, update this
table. The Predict page does not invoke these at runtime — they are used only
by notebooks and `prepare_artifacts.py`.
