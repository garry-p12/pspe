"""Turn scenario rasters into the numbers a flood officer actually acts on.

The solver produces depth fields. A council does not plan against depth fields;
it plans against "which roads are cut", "how much land goes under", and "what
does this cost". This script does that translation ONCE, server-side, so the
app queries outcomes rather than physics.

Inputs:  runs/scenarios/{index.json, peak_r*.npy}, portal roads.geojson
Output:  options.json — one record per (event size, mitigation option) with
         road length cut, named roads affected, inundated area, indicative
         cost, and the change against doing nothing.

The change against doing nothing is the important column: it is what turns
"this option floods 40 km of road" into "this option cuts 3 km MORE road than
doing nothing", which is the failure mode the tool exists to catch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def depth_rgba(d: np.ndarray, vmax: float) -> np.ndarray:
    """Same ramp as the baseline overlay, so options are visually comparable.

    A gamma of 0.42 lifts the shallow end: almost all inundation is 0.5-3 m
    while the channel runs to ~19 m, and a linear ramp renders the flooded
    floodplain -- the part a planner is looking at -- as invisible haze.
    """
    x = np.clip(np.nan_to_num(d) / max(vmax, 1e-6), 0.0, 1.0) ** 0.42
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (96 * (1 - x) + 4 * x).astype(np.uint8)
    out[..., 1] = (200 * (1 - x) + 28 * x).astype(np.uint8)
    out[..., 2] = (255 * (1 - x) + 128 * x).astype(np.uint8)
    a = np.where(np.nan_to_num(d) > 0.01, 132 + 118 * x, 0.0)
    out[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
    return out


def change_rgba(d: np.ndarray, base: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """Where an option makes things better (blue) or WORSE (red).

    The difference layer is the one that earns its place: a planner comparing
    two similar-looking flood maps cannot see a 3 km change, but can see red.
    """
    diff = np.nan_to_num(d) - np.nan_to_num(base)
    out = np.zeros(diff.shape + (4,), dtype=np.uint8)
    worse = diff > thresh
    better = diff < -thresh
    mag = np.clip(np.abs(diff) / 1.0, 0.0, 1.0) ** 0.5
    out[..., 0] = np.where(worse, 248, np.where(better, 56, 0))
    out[..., 1] = np.where(worse, 113, np.where(better, 189, 0))
    out[..., 2] = np.where(worse, 113, np.where(better, 248, 0))
    out[..., 3] = np.where(worse | better, (90 + 150 * mag).astype(np.uint8), 0)
    return out


def haversine(a, b) -> float:
    R = 6371000.0
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp = p2 - p1
    dl = math.radians(b[0] - a[0])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def sample(depth: np.ndarray, bounds: dict, lon: float, lat: float) -> float:
    h, w = depth.shape
    fx = (lon - bounds["west"]) / (bounds["east"] - bounds["west"])
    fy = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    if not (0 <= fx < 1 and 0 <= fy < 1):
        return 0.0
    return float(depth[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))])


def road_impact(depth, bounds, feats, cut_depth: float,
                centre=None, radius_km: float | None = None):
    """Road length under more than `cut_depth`, optionally within a radius.

    Reported twice, locally and valley-wide, because a levee does two things at
    once: it keeps water out of the place it defends and pushes it somewhere
    else. A tool that reported only the protected area would hide the transfer;
    one that reported only the whole network would hide a real benefit to the
    community paying for the work. The equity question a levee always raises is
    precisely the difference between the two numbers.
    """
    cut_m = 0.0
    named: dict[str, float] = {}
    for f in feats:
        coords = f["geometry"]["coordinates"]
        name = f["properties"].get("name")
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if centre is not None and radius_km is not None:
                dx = (mid[0] - centre[0]) * 97.0
                dy = (mid[1] - centre[1]) * 111.0
                if (dx * dx + dy * dy) ** 0.5 > radius_km:
                    continue
            if sample(depth, bounds, mid[0], mid[1]) > cut_depth:
                seg = haversine(a, b)
                cut_m += seg
                if name:
                    named[name] = named.get(name, 0.0) + seg
    return cut_m, named


def ring_effect(depth, base, bounds, centre, rings_km):
    """Change in mean flood depth by distance band from the defended place.

    A levee does not remove water; it moves it. Measured on this floodplain, a
    900 m levee takes 15 cm off the settlement core and adds 2 cm to the ring
    two to four kilometres out. Reporting only the protected area would present
    that as a free gain, which is the argument every contested levee scheme
    has. The bands make the transfer explicit.
    """
    import numpy as np

    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lat = bounds["north"] - (yy + 0.5) / h * (bounds["north"] - bounds["south"])
    lon = bounds["west"] + (xx + 0.5) / w * (bounds["east"] - bounds["west"])
    dkm = np.sqrt(((lon - centre[0]) * 97.0) ** 2 + ((lat - centre[1]) * 111.0) ** 2)

    out = []
    for lo, hi in rings_km:
        m = (dkm >= lo) & (dkm < hi)
        if not m.any():
            continue
        d0 = float(base[m].mean())
        d1 = float(depth[m].mean())
        out.append({
            "from_km": lo, "to_km": hi,
            "base_depth_m": round(d0, 4),
            "depth_m": round(d1, 4),
            "change_m": round(d1 - d0, 4),
            "protected": d1 < d0 - 0.001,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--roads", required=True)
    ap.add_argument("--manifest", required=True, help="portal manifest for bounds")
    ap.add_argument("--cut-depth", type=float, default=0.25,
                    help="depth above which a road is treated as impassable (m)")
    ap.add_argument("--local-radius-km", type=float, default=4.0,
                    help="radius of the protected area, for the local figure")
    ap.add_argument("--centre", type=float, nargs=2, default=None,
                    help="lon lat of the protected settlement")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", default=None,
                    help="directory for per-option WebP overlays the map draws")
    ap.add_argument("--depth-vmax", type=float, default=19.0)
    args = ap.parse_args()

    scen = Path(args.scenarios)
    index = json.loads((scen / "index.json").read_text())
    roads = json.loads(Path(args.roads).read_text())["features"]
    bounds = json.loads(Path(args.manifest).read_text())["bounds"]
    dx = index["dx_m"]
    cell_km2 = (dx * dx) / 1e6

    rasters = {}
    for rs in index["rain_scales"]:
        tag = f"r{rs:g}".replace(".", "p")
        f = scen / f"peak_{tag}.npy"
        if f.exists():
            rasters[rs] = np.load(f)
            print(f"loaded {f.name} {rasters[rs].shape}", flush=True)

    out_opts = []
    base_by_scale: dict[float, dict] = {}
    for o in index["options"]:
        rs = o["rain_scale"]
        if rs not in rasters:
            continue
        depth = rasters[rs][o["raster_index"]]
        cut_m, named = road_impact(depth, bounds, roads, args.cut_depth)
        local_m, local_named = (
            road_impact(depth, bounds, roads, args.cut_depth,
                        centre=tuple(args.centre), radius_km=args.local_radius_km)
            if args.centre else (0.0, {})
        )
        area_km2 = float((depth > 0.10).sum()) * cell_km2
        rec = {
            "event_scale": rs,
            "id": o["id"],
            "name": o["name"],
            "heights": o["heights"],
            "cost_aud": o["cost_aud"],
            "road_cut_km": cut_m / 1000.0,
            "local_road_cut_km": local_m / 1000.0,
            "area_flooded_km2": area_km2,
            "named_roads_cut": sorted(
                ({"name": n, "km": v / 1000.0} for n, v in named.items()),
                key=lambda d: -d["km"],
            )[:10],
        }
        if o["id"] == "base":
            base_by_scale[rs] = rec
        if args.centre:
            rec["rings"] = ring_effect(
                depth, rasters[rs][0], bounds, tuple(args.centre),
                [(0, 1), (1, 2), (2, 4), (4, 8)],
            )
            prot = [r for r in rec["rings"] if r["protected"]]
            disp = [r for r in rec["rings"] if r["change_m"] > 0.001]
            rec["protects_to_km"] = max((r["to_km"] for r in prot), default=0)
            rec["displaces_from_km"] = min((r["from_km"] for r in disp), default=None)
            rec["core_reduction_m"] = -rec["rings"][0]["change_m"] if rec["rings"] else 0.0
            rec["core_reduction_pct"] = (
                100 * -rec["rings"][0]["change_m"] / rec["rings"][0]["base_depth_m"]
                if rec["rings"] and rec["rings"][0]["base_depth_m"] > 1e-6 else 0.0
            )
        out_opts.append(rec)

    # The decisive column: change against doing nothing.
    for rec in out_opts:
        b = base_by_scale.get(rec["event_scale"])
        if not b:
            continue
        rec["road_cut_change_km"] = rec["road_cut_km"] - b["road_cut_km"]
        rec["local_change_km"] = rec["local_road_cut_km"] - b["local_road_cut_km"]
        rec["local_protected_km"] = max(0.0, -rec["local_change_km"])
        # Water kept out of the defended area has to go somewhere. Displacement
        # is what the rest of the valley absorbs.
        rec["displaced_km"] = max(
            0.0, rec["road_cut_change_km"] - rec["local_change_km"])
        rec["area_change_km2"] = rec["area_flooded_km2"] - b["area_flooded_km2"]
        rec["makes_worse"] = rec["road_cut_change_km"] > 0.05
        rec["road_protected_km"] = max(0.0, -rec["road_cut_change_km"])
        rec["cost_per_km_protected"] = (
            rec["cost_aud"] / rec["road_protected_km"]
            if rec["road_protected_km"] > 0.01 else None
        )

    # Per-option overlays: the flood under that option, and the change against
    # doing nothing. Without these the map never changes when an option is
    # selected, which defeats the point of a planning tool.
    if args.overlays:
        from PIL import Image

        od = Path(args.overlays)
        od.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for o in index["options"]:
            rs = o["rain_scale"]
            if rs not in rasters:
                continue
            tag = f"r{rs:g}".replace(".", "p")
            depth = rasters[rs][o["raster_index"]]
            base_depth = rasters[rs][0]
            for kind, arr in (("depth", depth_rgba(depth, args.depth_vmax)),
                              ("change", change_rgba(depth, base_depth))):
                f = od / f"{tag}_{o['id']}_{kind}.webp"
                Image.fromarray(arr).save(f, format="WEBP", quality=82, method=6)
                n_written += 1
        size = sum(f.stat().st_size for f in od.glob("*.webp"))
        print(f"wrote {n_written} overlays, {size/1e6:.1f} MB -> {od}", flush=True)

    payload = {
        "region": "Richmond Valley, NSW",
        "has_overlays": bool(args.overlays),
        "reference_event": "February–March 2022 Northern Rivers flood",
        "cut_depth_m": args.cut_depth,
        "dx_m": dx,
        "cost_note": index.get("cost_note"),
        "sites": index.get("sites", []),
        "event_scales": index["rain_scales"],
        "options": out_opts,
        "road_network_km": sum(f["properties"]["len_m"] for f in roads) / 1000.0,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nwrote {args.out}: {len(out_opts)} option records", flush=True)
    for rec in out_opts[:14]:
        flag = "  WORSE" if rec.get("makes_worse") else ""
        print(f"  x{rec['event_scale']} {rec['id']:>7}  cut {rec['road_cut_km']:6.1f} km  "
              f"Δ{rec.get('road_cut_change_km', 0):+6.2f} km  "
              f"A${rec['cost_aud']/1e6:5.1f}M{flag}", flush=True)


if __name__ == "__main__":
    main()
