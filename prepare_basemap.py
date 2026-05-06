"""Bake Sentinel-2 TCI basemaps and colorized prediction rasters for the web app.

For each year that has both:
  - a SAFE zip in `<safe-root>/<year>/<month>/*.SAFE.zip`
  - a `preds_<year>.parquet` in `deploy/artifacts/`

extracts the TCI (true-color) image at 10 m, windows it to the AOI grid from
`deploy/artifacts/grid_meta.json`, downsamples for display, and writes:

    deploy/artifacts/basemap_<year>.jpg         (Sentinel-2 RGB)
    deploy/artifacts/basemap_<year>.json        (lat/lng bounds + grid info)
    deploy/artifacts/preds_<year>.png           (15-class RGBA, transparent nodata)

Both PNGs are co-registered to the same lat/lng bounding box so they can be
stacked as `folium.raster_layers.ImageOverlay` on the Segmentation and Synth
Lab pages.

Usage:
    python prepare_basemap.py --safe-root /path/to/rayong_raster
    python prepare_basemap.py --safe-root /path/to/rayong_raster --month Dec --max-side 1800

The `--safe-root` argument can also be supplied via the `SYNTH_CROP_SAFE_ROOT`
environment variable.
"""
from __future__ import annotations
import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from PIL import Image
from rasterio.windows import Window, from_bounds

ROOT      = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "deploy" / "artifacts"

# 15-class palette (must match deploy/lib/palette.py LDD-sorted order)
PALETTE_HEX = [
    "#e6c229", "#a0522d", "#f4a300", "#1f7a3a", "#bcae00",
    "#c46a3e", "#d62728", "#8b6f47", "#ffb347", "#d2b48c",
    "#9acd32", "#7b1e1e", "#9467bd", "#1f77b4", "#7f7f7f",
]
LDD_TO_DENSE = {
    2101: 0, 2204: 1, 2205: 2, 2302: 3, 2303: 4,
    2403: 5, 2404: 6, 2405: 7, 2407: 8, 2413: 9,
    2416: 10, 2419: 11, 2420: 12, 4201: 13, 9999: 14,
}
N_CLASSES = 15


def log(msg: str) -> None:
    import sys
    try:
        print(f"[basemap] {msg}", flush=True)
    except UnicodeEncodeError:
        sys.stdout.write(("[basemap] " + msg).encode("ascii", "replace").decode("ascii") + "\n")
        sys.stdout.flush()


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


def find_tci(safe_zip: Path) -> str | None:
    """Return rasterio path to TCI_10m.jp2 inside SAFE zip, or None."""
    with zipfile.ZipFile(safe_zip) as z:
        for n in z.namelist():
            if n.endswith("_TCI_10m.jp2"):
                return f"zip+file://{safe_zip}!{n}"
    return None


def pick_safe_zip(year_dir: Path, month: str | None) -> Path | None:
    """Pick the first .SAFE.zip under <year>/<month> (or any month if --month=any)."""
    if month and month.lower() != "any":
        candidates = sorted((year_dir / month).glob("*.SAFE.zip"))
    else:
        candidates = sorted(year_dir.rglob("*.SAFE.zip"))
    return candidates[0] if candidates else None


def grid_window(src_transform, grid_meta: dict) -> Window:
    """Window into the SAFE TCI raster (in source pixel coords) matching the AOI grid."""
    a, _, ox, _, e, oy = grid_meta["transform"]
    H, W = grid_meta["height"], grid_meta["width"]
    minx = ox
    maxx = ox + W * a
    maxy = oy
    miny = oy + H * e        # e is negative for north-up
    return from_bounds(minx, miny, maxx, maxy, transform=src_transform)


def downsample_uint8(arr: np.ndarray, stride: int) -> np.ndarray:
    """arr: (H, W) or (C, H, W) uint8 — top-left pixel per tile (categorical-safe)."""
    if stride <= 1:
        return arr
    if arr.ndim == 2:
        H, W = arr.shape
        return arr[: (H // stride) * stride : stride,
                   : (W // stride) * stride : stride]
    C, H, W = arr.shape
    return arr[:, : (H // stride) * stride : stride,
                  : (W // stride) * stride : stride]


def utm_corners_to_lnglat(xs: list[float], ys: list[float],
                           grid_crs: str) -> list[tuple[float, float]]:
    """Project (x,y) corners from grid CRS to (lng, lat) for folium."""
    tr = pyproj.Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)
    return [tr.transform(x, y) for x, y in zip(xs, ys)]


def colorize_preds(df: pd.DataFrame, H: int, W: int) -> np.ndarray:
    """Build (H, W, 4) RGBA array with 15-class colors, alpha=0 for nodata."""
    palette_rgb = np.array([hex_to_rgb(h) for h in PALETTE_HEX], dtype=np.uint8)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    label_true = df["label_true"].to_numpy()
    if label_true.dtype.kind != "i" or (label_true.max() if len(label_true) else 0) >= N_CLASSES:
        label_pred = df["label_pred"].map(LDD_TO_DENSE).to_numpy()
    else:
        label_pred = df["label_pred"].to_numpy()
    rs = df["row"].to_numpy(dtype=np.int32)
    cs = df["col"].to_numpy(dtype=np.int32)
    valid = (label_pred >= 0) & (label_pred < N_CLASSES) & (rs >= 0) & (rs < H) & (cs >= 0) & (cs < W)
    rs, cs, lp = rs[valid], cs[valid], label_pred[valid].astype(np.int32)
    rgba[rs, cs, :3] = palette_rgb[lp]
    rgba[rs, cs, 3]  = 220                          # ~85% opaque pred pixels
    return rgba


def process_year(year: int, safe_root: Path, month: str | None,
                  grid_meta: dict, max_side: int, jpeg_quality: int) -> bool:
    log(f"=== year {year} ===")
    year_dir = safe_root / str(year)
    if not year_dir.exists():
        log(f"  no SAFE dir at {year_dir} — skip")
        return False
    safe_zip = pick_safe_zip(year_dir, month)
    if safe_zip is None:
        log(f"  no .SAFE.zip under {year_dir}/{month or '*'} — skip")
        return False
    log(f"  SAFE: {safe_zip.name}")

    tci_path = find_tci(safe_zip)
    if tci_path is None:
        log(f"  no TCI_10m.jp2 inside zip — skip")
        return False

    H_grid, W_grid = grid_meta["height"], grid_meta["width"]
    grid_crs = grid_meta["crs"]
    a, _, ox, _, e, oy = grid_meta["transform"]

    # ---- read + window TCI ----
    with rasterio.open(tci_path) as src:
        if str(src.crs) != grid_crs:
            log(f"  [warn] TCI CRS {src.crs} != grid CRS {grid_crs} — output may be misaligned")
        win = grid_window(src.transform, grid_meta).round_offsets().round_lengths()
        # clip the window to the source raster extent (window may overrun)
        win_h = int(min(win.height, src.height - win.row_off))
        win_w = int(min(win.width,  src.width  - win.col_off))
        win = Window(int(win.col_off), int(win.row_off), win_w, win_h)
        log(f"  read window: row_off={win.row_off} col_off={win.col_off} "
            f"h={win.height} w={win.width}")
        rgb = src.read(window=win)                  # (3, H, W) uint8

    # if window smaller than grid (tile doesn't fully cover AOI), pad black
    if rgb.shape[1] != H_grid or rgb.shape[2] != W_grid:
        log(f"  [info] padding TCI {rgb.shape[1:]} -> ({H_grid}, {W_grid})")
        full = np.zeros((3, H_grid, W_grid), dtype=np.uint8)
        full[:, : rgb.shape[1], : rgb.shape[2]] = rgb
        rgb = full

    # ---- downsample for display ----
    stride = max(1, max(H_grid, W_grid) // max_side)
    rgb_ds = downsample_uint8(rgb, stride)
    Hd, Wd = rgb_ds.shape[1], rgb_ds.shape[2]
    log(f"  downsample stride={stride} -> {Hd} x {Wd}")

    # ---- save basemap ----
    out_jpg = ARTIFACTS / f"basemap_{year}.jpg"
    Image.fromarray(np.transpose(rgb_ds, (1, 2, 0))).save(
        out_jpg, "JPEG", quality=jpeg_quality, optimize=True,
    )
    log(f"  wrote {out_jpg.name}  ({out_jpg.stat().st_size/1e6:.2f} MB)")

    # ---- bounds in lat/lng for folium ----
    minx, maxx = ox, ox + W_grid * a
    miny, maxy = oy + H_grid * e, oy
    corners_xy = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    corners_lnglat = utm_corners_to_lnglat(
        [c[0] for c in corners_xy], [c[1] for c in corners_xy], grid_crs,
    )
    sw = (min(c[1] for c in corners_lnglat), min(c[0] for c in corners_lnglat))
    ne = (max(c[1] for c in corners_lnglat), max(c[0] for c in corners_lnglat))
    bounds_json = {
        "year": year,
        "grid_crs": grid_crs,
        "grid_bounds_xy": [minx, miny, maxx, maxy],
        "lnglat_bounds":  {"sw": list(sw), "ne": list(ne)},
        "image_size":     {"height": Hd, "width": Wd},
        "stride":         stride,
        "source_safe":    safe_zip.name,
    }
    out_json = ARTIFACTS / f"basemap_{year}.json"
    out_json.write_text(json.dumps(bounds_json, indent=2), encoding="utf-8")
    log(f"  wrote {out_json.name}")

    # ---- preds RGBA ----
    pred_pq = ARTIFACTS / f"preds_{year}.parquet"
    if pred_pq.exists():
        df = pd.read_parquet(pred_pq)
        rgba = colorize_preds(df, H_grid, W_grid)
        rgba_ds = rgba[: (H_grid // stride) * stride : stride,
                        : (W_grid // stride) * stride : stride]
        out_png = ARTIFACTS / f"preds_{year}.png"
        Image.fromarray(rgba_ds, mode="RGBA").save(out_png, "PNG", optimize=True)
        log(f"  wrote {out_png.name}  ({out_png.stat().st_size/1e6:.2f} MB)")
    else:
        log(f"  no {pred_pq.name} — skipping preds PNG")

    return True


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    default_safe = os.environ.get("SYNTH_CROP_SAFE_ROOT")
    p.add_argument("--safe-root", type=Path,
                    default=Path(default_safe) if default_safe else None,
                    help="root directory containing <year>/<month>/*.SAFE.zip "
                         "(or set SYNTH_CROP_SAFE_ROOT).")
    p.add_argument("--month", type=str, default="Dec",
                    help="month subdir to pull SAFE archives from "
                         "('any' to scan all months).")
    p.add_argument("--years", type=int, nargs="+", default=None,
                    help="years to process (default: all dirs under safe-root).")
    p.add_argument("--max-side", type=int, default=1800,
                    help="downsample so the longer side is approximately this "
                         "many display pixels.")
    p.add_argument("--jpeg-quality", type=int, default=82,
                    help="JPEG quality for basemap (60-90 typical).")
    args = p.parse_args()

    if args.safe_root is None:
        log("ERROR: --safe-root not provided and SYNTH_CROP_SAFE_ROOT not set.")
        return 2
    if not args.safe_root.exists():
        log(f"ERROR: safe-root not found: {args.safe_root}")
        return 1

    grid_meta_path = ARTIFACTS / "grid_meta.json"
    if not grid_meta_path.exists():
        log(f"ERROR: {grid_meta_path} missing — run prepare_artifacts.py first")
        return 1
    grid_meta = json.loads(grid_meta_path.read_text(encoding="utf-8"))

    if args.years is None:
        years = sorted(int(d.name) for d in args.safe_root.iterdir()
                        if d.is_dir() and d.name.isdigit())
    else:
        years = args.years
    log(f"years to process: {years}")

    n_ok = 0
    for y in years:
        if process_year(y, args.safe_root, args.month, grid_meta,
                         args.max_side, args.jpeg_quality):
            n_ok += 1
    log(f"\ndone — {n_ok}/{len(years)} years processed")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
