"""Build deploy/artifacts/ from the raw dataset + trained models.

Run locally before pushing — outputs are gitignored. The Streamlit app reads
ONLY files written here; it never touches raw rasters or the full 3 GB CSV.

Outputs (planned):
    dataset.parquet              50k stratified sample of 2020_2024_dataset.parquet
    synth_tabsyn.parquet         matched-size sample from TabSyn synthetic CSV
    class_counts.parquet         per-year × class counts (Sankey, bars)
    feature_stats.parquet        mean/std/quantiles per class
    corr_real.npy / corr_syn.npy 15×15 correlation matrices
    metrics.json                 acc, kappa, weighted-F1, macro-F1, per-class F1
    confusion.npy                15×15 confusion matrix
    feature_importance.parquet   RF feature importance
    preds_{2018,2020,2024}.tif   uint8 class raster, 3× downsampled COG
    preds_{year}_thumb.png       low-bandwidth fallback PNG
    rf_model.joblib              copied from training output
    grid_meta.json               copied from preprocessing output

Use --tiny for a smoke-test run (10k rows, no rasters) used by CI.
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "deploy" / "artifacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true",
                        help="smoke-test mode (small sample, skip rasters)")
    parser.add_argument("--source-csv", type=Path,
                        default=Path("../2020_2024_dataset.csv"),
                        help="full preprocessed dataset")
    parser.add_argument("--synth-csv", type=Path,
                        default=Path("../synthetic/crop_full/tabsyn.csv"),
                        help="TabSyn synthetic dataset")
    parser.add_argument("--rf-model", type=Path,
                        default=Path("../random_forest_model.joblib"))
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"output dir: {ARTIFACTS}")
    print(f"tiny mode : {args.tiny}")
    # TODO:
    # 1. sample real dataset (stratified, 50k rows; 10k if --tiny)
    # 2. sample TabSyn synth to matching size
    # 3. compute class_counts, feature_stats, correlations
    # 4. load RF model, compute metrics on held-out test, save confusion + importance
    # 5. paint predictions onto AOI grid → preds_{year}.tif (skip if --tiny)
    # 6. copy rf_model.joblib + grid_meta.json
    raise NotImplementedError("scaffold only — implement next")


if __name__ == "__main__":
    main()
