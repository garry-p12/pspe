"""Place the surveyed Lismore floor levels, then audit our terrain against them.

`fetch_lismore_floor_levels.py` produces 2,924 council-surveyed properties with
floor, gate and road levels in metres AHD, and no coordinates: the source
documents are address lists. This places them and then uses them for the thing
they are most valuable for, which is not what it looks like.

**Placing them.** NSW Spatial Services publishes the authoritative address point
for every address in the state through an open ArcGIS service, no key and no
bulk download: 10,825 points over the five Lismore suburbs, with the address
string in the same form the council writes it. Matching on the full address
places 83.1%; falling back to number-and-street inside the same box, which the
box itself disambiguates, reaches **85.9%**. OpenStreetMap was tried first and
reached 15%: OSM has the footprints (3,029 buildings here) but not the
addresses -- 2 of the 155 council properties on Casino Street carry one.

The 14% that never match are not all failures. The council list was compiled
from 1996 and last updated in 2018; the NSW address file is current; and
Lismore ran a voluntary house buyback after the 2022 flood. Some of these
addresses no longer exist.

**What they are actually for.** 2,512 surveyed ground elevations, inside a box
the solver already runs on, from an instrument rather than from another model.
That makes them a check on the Copernicus GLO-30 terrain everything in this
project stands on, and the check does not pass:

    DEM minus surveyed gate level, n = 2,512
      median +2.21 m   mean +2.55 m   RMSE 3.21 m
      above the surveyed ground at 97.2% of points
      more than 1 m above at 89.7%, more than 3 m above at 28.2%

GLO-30 is a Digital SURFACE Model. It includes canopy and buildings, and in a
leafy river town that is most of a storey. The bias rises with building density
-- +2.04 m median where OSM maps no building within one cell, +3.38 m where it
maps four or more -- but it does not fall to zero anywhere, and it is the same
on the floodplain (+2.21 m) as above it (+2.24 m), so it is not a floodplain
artefact.

**It is mostly not the datum.** AHD and the Copernicus EGM2008 differ, and in
northern Australia AHD sits roughly half a metre above the geoid, which would
be a CONSTANT. Fitted and scored out of fold, a constant takes the error from
3.21 m to 1.98 m and no further; eight neighbouring surveyed points take it to
**1.16 m**. Most of the bias is spatial, so most of it is surface, and the
project does not have to take a literature number for the datum to say so.

1.16 m is a floor, not a fix. It is what calibrating a 30 m surface model
against thousands of surveyed points buys, and it is still large next to the
difference between a flood that enters a house and one that does not. The real
fix is a bare-earth LiDAR DTM; NSW has 1 m LiDAR under CC-BY through ELVIS, but
it is an order-and-download service rather than an API, so it is a deliberate
step and not something this script can do.

Usage:
    python3 scripts/locate_and_validate_floor_levels.py --out data/lismore
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ADDR_URL = ("https://portal.spatial.nsw.gov.au/server/rest/services/"
            "NSW_Geocoded_Addressing_Theme/FeatureServer/1/query")
PAGE = 2000


def norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())).strip()


def fetch_address_points(box: dict) -> list[dict]:
    """Every NSW address point in the box, paged past the 2,000 record cap."""
    feats: list[dict] = []
    offset = 0
    while True:
        q = urllib.parse.urlencode({
            "f": "json", "where": "1=1", "outFields": "address,housenumber",
            "geometry": json.dumps({**box, "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "true",
            "resultOffset": offset, "resultRecordCount": PAGE,
        })
        req = urllib.request.Request(f"{ADDR_URL}?{q}",
                                     headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
        if "error" in d:
            raise RuntimeError(d["error"])
        got = d.get("features", [])
        feats += got
        if len(got) < PAGE:
            return feats
        offset += PAGE


def locate(rows: list[dict], pts: list[dict]) -> tuple[list[dict], int, int]:
    """Council rows with coordinates, by exact address then by street."""
    full: dict[str, list] = defaultdict(list)
    part: dict[str, list] = defaultdict(list)
    for f in pts:
        a = norm(f["attributes"]["address"])
        full[a].append(f)
        w = a.split()
        # Drop the trailing suburb -- one word or two -- and let the bounding
        # box do the disambiguating that the suburb name would have done.
        for k in (1, 2):
            if len(w) > k + 1:
                part[" ".join(w[:-k])].append(f)

    out, exact, loose = [], 0, 0
    for r in rows:
        nums = [r["number"], r["number"].split("/")[-1]]
        cands, how = None, None
        for n in nums:
            cands = full.get(norm(f'{n} {r["street"]} {r["suburb"]}'))
            if cands:
                how = "address"
                break
        if not cands:
            for n in nums:
                cands = part.get(norm(f'{n} {r["street"]}'))
                if cands:
                    how = "street"
                    break
        if not cands:
            continue
        exact += how == "address"
        loose += how == "street"
        g = cands[0]["geometry"]
        out.append({**r, "lon": g["x"], "lat": g["y"], "match": how,
                    "candidates": len(cands)})
    return out, exact, loose


def rmse(v) -> float:
    v = list(v)
    return math.sqrt(sum(x * x for x in v) / len(v))


def audit(located: list[dict], dx: float = 30.0, folds: int = 5, seed: int = 0) -> dict:
    """The DEM against the survey, and how far a fitted correction gets."""
    import numpy as np                                    # noqa: PLC0415
    from pspe.simulate.real.anywhere import build_domain  # noqa: PLC0415

    lons = [p["lon"] for p in located]
    lats = [p["lat"] for p in located]
    pad = 0.004
    dom = build_domain(min(lons) - pad, min(lats) - pad,
                       max(lons) + pad, max(lats) + pad, dx=dx, name="lismore-audit")
    b, z = dom.bounds, np.asarray(dom.z, dtype=np.float64)
    H, W = z.shape

    pts = []
    for p in located:
        col = int(round((p["lon"] - b["west"]) / (b["east"] - b["west"]) * (W - 1)))
        row = int(round((b["north"] - p["lat"]) / (b["north"] - b["south"]) * (H - 1)))
        if not (0 <= col < W and 0 <= row < H):
            continue
        g = float(z[row, col])
        if not math.isfinite(g):
            continue
        pts.append({**p, "dem_m": g,
                    "bias": g - float(p["gate_m_ahd"]),
                    "x": p["lon"] * 111320.0 * math.cos(math.radians(p["lat"])),
                    "y": p["lat"] * 110540.0})

    bias = [p["bias"] for p in pts]
    n = len(bias)
    report = {
        "grid": list(z.shape), "dx_m": dom.dx, "n": n,
        "raw": {
            "median": round(st.median(bias), 2), "mean": round(sum(bias) / n, 2),
            "sd": round(st.pstdev(bias), 2), "rmse": round(rmse(bias), 2),
            "frac_dem_above_ground": round(sum(1 for x in bias if x > 0) / n, 3),
            "frac_above_1m": round(sum(1 for x in bias if x > 1) / n, 3),
            "frac_above_3m": round(sum(1 for x in bias if x > 3) / n, 3),
        },
    }

    rng = random.Random(seed)
    fold = [[] for _ in range(folds)]
    for i, p in enumerate(sorted(pts, key=lambda _: rng.random())):
        fold[i % folds].append(p)

    def cv(k_neighbours: int | None) -> dict:
        res = []
        for k in range(folds):
            test = fold[k]
            train = [p for j in range(folds) if j != k for p in fold[j]]
            if k_neighbours is None:
                c = st.median(p["bias"] for p in train)
                res += [p["bias"] - c for p in test]
                continue
            for q in test:
                near = sorted(((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2, p["bias"])
                              for p in train)[:k_neighbours]
                wsum = vsum = 0.0
                for d2, v in near:
                    w = 1.0 / max(d2, 1.0)
                    wsum += w
                    vsum += w * v
                res.append(q["bias"] - vsum / wsum)
        a = sorted(abs(x) for x in res)
        return {"rmse": round(rmse(res), 2), "mean": round(sum(res) / len(res), 2),
                "p95_abs": round(a[int(0.95 * len(a))], 2)}

    report["corrected_out_of_fold"] = {
        "constant": cv(None),
        "idw_8": cv(8),
        "idw_24": cv(24),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/lismore")
    ap.add_argument("--no-audit", action="store_true",
                    help="place the properties but skip the terrain audit")
    args = ap.parse_args()
    out = Path(args.out)

    rows = list(csv.DictReader((out / "floor_levels.csv").open()))
    lo = {"xmin": 153.245, "ymin": -28.845, "xmax": 153.315, "ymax": -28.775}
    print(f"{len(rows)} council properties; fetching NSW address points…")
    pts = fetch_address_points(lo)
    print(f"  {len(pts)} address points")

    located, exact, loose = locate(rows, pts)
    n = len(rows)
    print(f"placed {len(located)}/{n} = {100*len(located)/n:.1f}% "
          f"({exact} by full address, {loose} by number and street)")

    path = out / "floor_levels_located.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(located[0].keys()))
        w.writeheader()
        w.writerows(located)
    print(f"-> {path}")

    if args.no_audit:
        return
    rep = audit(located)
    (out / "terrain_audit.json").write_text(json.dumps(rep, indent=2) + "\n")
    r = rep["raw"]
    print(f"\nCopernicus GLO-30 at {rep['dx_m']:.0f} m against {rep['n']} surveyed gate levels")
    print(f"  median {r['median']:+.2f} m   mean {r['mean']:+.2f}   RMSE {r['rmse']:.2f}")
    print(f"  DEM above the surveyed ground at {100*r['frac_dem_above_ground']:.1f}% of points")
    print("\n  out-of-fold, after fitting a correction to the survey:")
    for k, v in rep["corrected_out_of_fold"].items():
        print(f"    {k:10s} RMSE {v['rmse']:.2f} m   mean {v['mean']:+.2f}   p95 |err| {v['p95_abs']:.2f}")


if __name__ == "__main__":
    main()
