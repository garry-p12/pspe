"""Turn 19 GB of FloodCastBench rasters into a ~10 MB web bundle.

Run on Vista, where the archive lives; the output directory is rsynced to
`portal/public/data/`.

Products:
    terrain.png        terrain-RGB encoded DEM for deck.gl TerrainLayer
    depth/NNNN.webp    colourised flood depth, one per sampled frame
    ours/NNNN.webp     our solver's depth over the validation window
    controllability.png  where a levee can help at all, from the 933-site screen
    manifest.json      bounds, times, colour scale, provenance, evidence badges

Every product carries its provenance into `manifest.json`, because the portal
labels each panel OBSERVED / VALIDATED MODEL / PROJECTION and those labels must
come from the pipeline rather than being typed into the UI by hand.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import tifffile
import warnings

warnings.filterwarnings("ignore")

# EPSG:32756 (WGS84 / UTM 56S) -> WGS84, computed with pyproj in
# scripts/ (see docs). Corners of the 1073 x 1073, 30 m Australia grid.
AUS_BOUNDS = {
    "west": 153.17877, "east": 153.51020,
    "south": -29.14060, "north": -28.84921,
    "epsg": 32756, "extent_km": 32.2,
    "place": "Lower Richmond River, Northern Rivers, NSW, Australia",
    "event": "February-March 2022 Northern Rivers flood",
}


def terrain_rgb(z: np.ndarray) -> np.ndarray:
    """Mapbox terrain-RGB encoding: h = -10000 + (R*65536 + G*256 + B) * 0.1."""
    v = np.nan_to_num(z, nan=0.0)
    enc = np.clip((v + 10000.0) / 0.1, 0, 256**3 - 1).astype(np.uint32)
    out = np.zeros(v.shape + (3,), dtype=np.uint8)
    out[..., 0] = (enc >> 16) & 255
    out[..., 1] = (enc >> 8) & 255
    out[..., 2] = enc & 255
    return out


def depth_rgba(d: np.ndarray, vmax: float) -> np.ndarray:
    """Blue-scale depth over aerial imagery; fully transparent below 1 cm.

    Scaled by a gamma rather than linearly in depth. The field runs to ~19 m in
    the river channel while almost all *inundation* is 0.5-3 m, so a linear ramp
    renders the flooded floodplain — the part that matters — as barely-visible
    haze. x**0.42 lifts the shallow end into a readable blue while keeping the
    deep channel distinct, and the alpha floor guarantees any wet cell is seen.
    """
    x = np.clip(np.nan_to_num(d) / max(vmax, 1e-6), 0.0, 1.0) ** 0.42
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (96 * (1 - x) + 4 * x).astype(np.uint8)
    out[..., 1] = (200 * (1 - x) + 28 * x).astype(np.uint8)
    out[..., 2] = (255 * (1 - x) + 128 * x).astype(np.uint8)
    a = np.where(np.nan_to_num(d) > 0.01, 132 + 118 * x, 0.0)
    out[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
    return out


def scalar_rgba(v: np.ndarray, vmax: float) -> np.ndarray:
    """Amber ramp for the controllability field."""
    x = np.clip(np.nan_to_num(v) / max(vmax, 1e-9), 0.0, 1.0)
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (255 * x).astype(np.uint8)
    out[..., 1] = (200 * x**1.4).astype(np.uint8)
    out[..., 2] = (40 * x).astype(np.uint8)
    out[..., 3] = (235 * x**0.7).astype(np.uint8)
    return out


def write_png(path: Path, arr: np.ndarray) -> int:
    """PNG for terrain (lossless, the heights must be exact); WebP for overlays.

    Terrain-RGB packs elevation into colour channels, so any lossy step corrupts
    the heights. The depth overlays are smooth gradients with alpha, which PNG
    compresses badly -- 90 frames came to 26 MB. WebP at q=82 cuts that ~4x with
    no visible difference at display scale.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if path.suffix == ".webp":
        img.save(path, format="WEBP", quality=82, method=6)
    else:
        img.save(path, optimize=True)
    return path.stat().st_size


def _finite(obj):
    """Recursively replace NaN/Inf with None so the payload is valid JSON."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def coarsen(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    return np.nanmean(a[: h * f, : w * f].reshape(h, f, w, f), axis=(1, 3))


def levee_geometry(dem: np.ndarray, coarsen: int, settle_rc: tuple[int, int],
                   settle_radius: int, n_sites: int, e0: float, n0: float,
                   px: float, epsg: int) -> dict:
    """Replicate floodcast_scenario's site selection and place it in lat/lon.

    The planning study runs on a `coarsen`-averaged grid, and the map needs the
    same cells as geographic points. The selection is deterministic given the
    DEM, so it is recomputed here rather than threaded through the results JSON.
    """
    from pyproj import Transformer

    zc = coarsen_fn(dem, coarsen)
    h, w = zc.shape
    sr, sc = settle_rc
    ring = settle_radius + 2
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cell = px * coarsen

    def to_lonlat(r: int, c: int) -> tuple[float, float]:
        e = e0 + (c + 0.5) * cell
        n = n0 - (r + 0.5) * cell
        lon, lat = tr.transform(e, n)
        return lon, lat

    sites = []
    for s in range(n_sites):
        a0, a1 = 2 * np.pi * s / n_sites, 2 * np.pi * (s + 1) / n_sites
        best_e, best_rc = np.inf, None
        for a in np.linspace(a0, a1, 24):
            rr = int(round(sr + ring * np.sin(a)))
            cc = int(round(sc + ring * np.cos(a)))
            if 0 <= rr < h and 0 <= cc < w and np.isfinite(zc[rr, cc]) and zc[rr, cc] < best_e:
                best_e, best_rc = float(zc[rr, cc]), (rr, cc)
        if best_rc is None:
            continue
        lon, lat = to_lonlat(*best_rc)
        sites.append({"id": s, "row": best_rc[0], "col": best_rc[1],
                      "elev": best_e, "lon": lon, "lat": lat})
    slon, slat = to_lonlat(sr, sc)
    return {
        "settlement": {"row": sr, "col": sc, "lon": slon, "lat": slat,
                       "elev": float(zc[sr, sc])},
        "radius_cells": settle_radius, "cell_m": cell, "sites": sites,
    }


def coarsen_fn(a: np.ndarray, f: int) -> np.ndarray:
    return coarsen(a, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--coarsen", type=int, default=2, help="2 -> 536x536 rasters")
    ap.add_argument("--runs", default=None, help="repo runs/ dir for metrics")
    ap.add_argument("--manifest-only", action="store_true",
                    help="skip raster generation; refresh manifest.json only")
    ap.add_argument("--plan-coarsen", type=int, default=4,
                    help="coarsening the planning study used, for site geometry")
    ap.add_argument("--settle-rc", type=int, nargs=2, default=[52, 52])
    ap.add_argument("--settle-radius", type=int, default=10)
    ap.add_argument("--n-sites", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    if root.name != "FloodCastBench" and (root / "FloodCastBench").is_dir():
        root = root / "FloodCastBench"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dem = tifffile.imread(root / "Study regions" / "Australia_DEM.tif").astype(np.float32)
    dem[dem < -1e30] = np.nan
    demc = coarsen(dem, args.coarsen)
    if not args.manifest_only:
        n = write_png(out / "terrain.png", terrain_rgb(demc))
        print(f"terrain.png {demc.shape} {n/1024:.0f} KB", flush=True)

    D = root / "High-fidelity flood forecasting" / "30m" / "Australia"
    files = sorted(D.glob("*.tif"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or -1))
    secs = [int(re.sub(r"\D", "", p.stem) or -1) for p in files]
    idx = np.linspace(0, len(files) - 1, args.frames).round().astype(int)

    vmax = 0.0
    if args.manifest_only:
        prev = json.loads((out / "manifest.json").read_text())
        vmax = prev["depth_vmax_m"]
    for i in ([] if args.manifest_only else idx[:: max(1, len(idx) // 12)]):
        vmax = max(vmax, float(np.nanmax(np.nan_to_num(tifffile.imread(files[i])))))
    vmax = float(np.ceil(vmax))
    print(f"depth colour scale 0..{vmax} m", flush=True)

    total = 0
    frames_meta = []
    for j, i in enumerate([] if args.manifest_only else idx):
        d = coarsen(np.nan_to_num(tifffile.imread(files[i])), args.coarsen)
        total += write_png(out / "depth" / f"{j:04d}.webp", depth_rgba(d, vmax))
        frames_meta.append({
            "i": j, "t_seconds": secs[i], "t_hours": secs[i] / 3600.0,
            "wet_cells": int((d > 0.01).sum()), "max_depth": float(d.max()),
            "mean_depth_wet": float(d[d > 0.01].mean()) if (d > 0.01).any() else 0.0,
        })
        if j % 20 == 0:
            print(f"  depth frame {j}/{len(idx)}", flush=True)
    print(f"depth/ {len(idx)} frames, {total/1e6:.1f} MB", flush=True)

    if args.manifest_only:
        frames_meta = json.loads((out / "manifest.json").read_text())["frames"]

    with tifffile.TiffFile(root / "Study regions" / "Australia_DEM.tif") as tf:
        tp = tf.pages[0].tags["ModelTiepointTag"].value
    geom = levee_geometry(dem, args.plan_coarsen, tuple(args.settle_rc),
                          args.settle_radius, args.n_sites,
                          float(tp[3]), float(tp[4]), 30.0, AUS_BOUNDS["epsg"])
    print(f"levee geometry: settlement {geom['settlement']['lat']:.5f},"
          f"{geom['settlement']['lon']:.5f}; {len(geom['sites'])} sites", flush=True)

    manifest = {
        "generated": "scripts/build_portal_assets.py",
        "levee_geometry": geom,
        "bounds": AUS_BOUNDS,
        "grid": list(demc.shape),
        "native_grid": list(dem.shape),
        "native_dx_m": 30.0,
        "raster_dx_m": 30.0 * args.coarsen,
        "dem_range_m": [float(np.nanmin(dem)), float(np.nanmax(dem))],
        "depth_vmax_m": vmax,
        "frame_ext": "webp",
        "frames": frames_meta,
        "provenance": {
            "terrain": {"badge": "OBSERVED",
                        "source": "FloodCastBench Study regions/Australia_DEM.tif",
                        "note": "30 m DEM, EPSG:32756"},
            "depth": {"badge": "OBSERVED",
                      "source": "FloodCastBench High-fidelity 30m/Australia",
                      "note": ("reference depths from a finite-difference solution "
                               "of the 2-D shallow water equations, validated by the "
                               "dataset authors against SAR flood maps")},
        },
        "caveats": [
            "The archive ships no rainfall and no land-cover field.",
            "Depth rasters carry no georeferencing; the DEM's EPSG:32756 places "
            "the grid, and only Australia's DEM grid matches its depth grid.",
            "Rasters here are block-averaged from 30 m for web delivery.",
        ],
    }

    if args.runs:
        runs = Path(args.runs)
        metrics = {}
        for key, rel in (("validation", "floodcast_validate_closed/validate.json"),
                         ("validation_openbc", "floodcast_validate/validate.json"),
                         ("sites", "flood_sites/sites.json"),
                         ("elasticity", "flood_elasticity/elasticity.json"),
                         ("span_synthetic", "flood_span/span.json")):
            f = runs / rel
            if f.exists():
                metrics[key] = json.loads(f.read_text())
                print(f"  metrics: {key}", flush=True)
        # json.dumps writes bare NaN/Infinity, which is invalid JSON and makes
        # JSON.parse throw outright -- the browser then sees no metrics at all
        # and every chart renders empty with no clue why. Emit null instead.
        (out / "metrics.json").write_text(
            json.dumps(_finite(metrics), allow_nan=False)
        )
        manifest["metrics"] = sorted(metrics)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nbundle {size/1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
