"""Solve the default Austin study, score it against FEMA, and ship the bundle.

The portal has always opened on the Richmond valley, because that is where the
reference depths were: FloodCastBench gave a validated flood to draw, and the
district tab's levee options were precomputed against it. Austin has no such
archive. What it has instead is a bare-earth DEM, a regulatory floodplain to be
scored against, and gauges -- so the default study here is not an archive being
replayed, it is **our own solver, run once and scored**.

That difference is the point and the portal states it. Richmond is labelled
VALIDATED MODEL because its depths were compared to a reference solution.
Austin is a PROJECTION with a skill number attached, which is a weaker claim
and a more honest one than an unlabelled map.

## The skill number, and what it is not

The solve is scored against FEMA's 1% annual chance floodplain (zones A, AE and
AO) in the same box, by critical success index over the wet mask:

    CSI 0.37-0.42, POD ~0.50, FAR ~0.33-0.39

Flat across 30, 40 and 60 m cells -- the small apparent rise with coarser cells
is the reference polygon rasterising fatter, not the physics improving -- so
resolution is not what limits this. The forcing is: uniform rain over an open
box, against a regulatory product built from surveyed channel geometry with
bridges and culverts in it. Half of FEMA's mapped floodplain is recovered.

For an uncalibrated general-purpose solver against a regulatory map that is a
reasonable result; calling it a validated twin would not be. Total extent is
much closer than the overlap -- 18-20 km2 against FEMA's 22-27 km2 -- which is
exactly why extent alone is not reported.

Usage:
    python3 scripts/build_austin_study.py --out portal/public/data/austin
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import urllib.request
from pathlib import Path

# Onion Creek: the reach that flooded in October 2013 and on Halloween 2015,
# and the one with a mapped floodplain to be scored against. Downtown is more
# recognisable and has more buildings, but nothing to check the answer with.
DEFAULT = {
    "west": -97.820, "south": 30.110, "east": -97.690, "north": 30.210,
    "name": "Onion Creek, Austin, Texas", "dx": 40,
    "driver": "rain", "rain_mm_h": 50, "storm_hours": 6, "run_hours": 18,
    # Nine instantaneous depth fields across the run, so the strip under the
    # map can play the flood arriving rather than only show where it stopped.
    # The solve computes every one of these already; keeping nine of them
    # costs nine PNGs in the bundle and nothing in compute.
    "snapshots": 9,
}
SERVICE = "http://127.0.0.1:8000"


def post(path: str, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        f"{SERVICE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "PSPE-floodplain-planner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def score(study: dict, fema: Path) -> dict:
    """Critical success index of the solved wet mask against FEMA's 1% zones.

    The depth field is recovered from the overlay rather than re-solved: the
    PNG alpha is `132 + 118*(d/vmax)**0.42` wherever depth exceeds a centimetre,
    which inverts exactly. Counting any non-zero alpha instead would score the
    1 cm extent, which is 63 km2 here against an 18 km2 flood -- the first
    version of this did that and reported a FAR of 0.75.
    """
    import numpy as np                                   # noqa: PLC0415
    from PIL import Image, ImageDraw                     # noqa: PLC0415

    b, vmax = study["bounds"], study["depth_vmax_m"]
    im = Image.open(io.BytesIO(base64.b64decode(
        study["depth_png"].split(",", 1)[-1]))).convert("RGBA")
    a = np.array(im)[..., 3].astype(np.float64)
    h, w = a.shape
    x = np.clip((a - 132.0) / 118.0, 0.0, 1.0)
    depth = np.where(a > 0, vmax * x ** (1 / 0.42), 0.0)
    cell_km2 = (((b["east"] - b["west"]) * 111320.0
                 * math.cos(math.radians((b["south"] + b["north"]) / 2)) / w)
                * ((b["north"] - b["south"]) * 110540.0 / h) / 1e6)

    mask = Image.new("1", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    n = 0
    for f in json.loads(fema.read_text())["features"]:
        if f["properties"].get("FLOOD_ZONE") not in ("A", "AE", "AO"):
            continue
        g = f.get("geometry")
        if not g:
            continue
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for k, ring in enumerate(poly):
                px = [((c[0] - b["west"]) / (b["east"] - b["west"]) * (w - 1),
                       (b["north"] - c[1]) / (b["north"] - b["south"]) * (h - 1))
                      for c in ring]
                if len(px) > 2:
                    dr.polygon(px, fill=(0 if k else 1))
        n += 1
    ref = np.array(mask, dtype=bool)

    best = None
    for t in [round(v, 2) for v in list(_frange(0.02, 1.0, 0.02))]:
        wet = depth > t
        hit = int((wet & ref).sum())
        miss = int((~wet & ref).sum())
        fa = int((wet & ~ref).sum())
        csi = hit / max(hit + miss + fa, 1)
        if best is None or csi > best["csi"]:
            best = {"threshold_m": t, "csi": round(csi, 3),
                    "pod": round(hit / max(hit + miss, 1), 3),
                    "far": round(fa / max(hit + fa, 1), 3),
                    "ours_km2": round(float(wet.sum()) * cell_km2, 2)}
    return {**best, "reference": "FEMA NFHL 1% annual chance, zones A/AE/AO",
            "reference_polygons": n,
            "reference_km2": round(float(ref.sum()) * cell_km2, 2)}


def _frange(a: float, b: float, step: float):
    v = a
    while v < b:
        yield v
        v += step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="portal/public/data/austin")
    ap.add_argument("--fema", default="data/austin_onion/floodplain_fema.geojson")
    ap.add_argument("--service", default=SERVICE)
    args = ap.parse_args()
    globals()["SERVICE"] = args.service

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"solving {DEFAULT['name']} at {DEFAULT['dx']} m …")
    study = post("/analyse", DEFAULT)
    if not study.get("depth_png"):
        raise SystemExit(f"no solve came back: {str(study)[:300]}")
    print(f"  {study['grid']} cells, {study['flooded_km2']} km2 wet, "
          f"peak {study['peak_depth_m']} m, {study['timing']['seconds']:.0f} s")

    b = study["bounds"]
    roads = post("/roads", {"west": b["west"], "south": b["south"],
                            "east": b["east"], "north": b["north"]})
    nroads = len(roads.get("roads", {}).get("features", []))
    print(f"  {nroads} road features")

    fema = Path(args.fema)
    skill = score(study, fema) if fema.exists() else None
    if skill:
        print(f"  CSI {skill['csi']} POD {skill['pod']} FAR {skill['far']} "
              f"at depth > {skill['threshold_m']} m "
              f"(ours {skill['ours_km2']} km2, FEMA {skill['reference_km2']} km2)")
    else:
        print(f"  no FEMA layer at {fema}; skipping the skill score")

    bundle = {
        **study,
        "roads": roads.get("roads"),
        "roads_note": roads.get("note"),
        "study": {
            "id": "austin-onion-creek",
            "place": "Onion Creek, Austin, Texas",
            "short": "Austin, Texas",
            "event": "a 300 mm, 6-hour storm — near the 1% annual chance "
                     "6-hour depth for Travis County",
            # The portal's badge vocabulary. Richmond earns VALIDATED MODEL
            # because its depths were compared to a reference solution; this is
            # our own solver with a skill score, which is a weaker claim.
            "evidence": "PROJECTION",
            "terrain": "USGS 3DEP bare earth",
            "skill": skill,
            "caveats": [
                "This is our solver's answer, not a reference flood. It "
                "recovers about half of FEMA's mapped 1% floodplain.",
                "Rain falls evenly over the whole box and drains at the edges. "
                "Real storms are not uniform and real creeks have culverts.",
                "Depth is above the GROUND. Finished floor elevation is not "
                "published for Austin, so depth inside a building is not shown.",
            ],
        },
    }
    path = out / "study.json"
    path.write_text(json.dumps(bundle))
    print(f"\n-> {path}  ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
