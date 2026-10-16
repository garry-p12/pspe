#!/usr/bin/env python
"""Ship the widened Austin study box, once OpenStreetMap is answering again.

## Why this script exists rather than a second `build_austin_study.py` run

The box was widened from 12.5 to 14.6 km so that USGS 08159000 -- the outlet
gauge -- falls 2 km inside the grid instead of 0.13 km outside it. That is what
lets `service/shadow.py` mark the model against an instrument it was never
told about.

The rebuild solved correctly and then could not fetch roads: every Overpass
mirror was returning HTTP 500. The service refused rather than inventing a
network, which is right, but it left a bundle with 0 road features where the
shipped one has 3,624 -- and the Roads tab needs them. So the working bundle
was restored and the new solve was kept aside.

This script finishes the move: it checks Overpass, fetches roads for the WIDER
box, and only then swaps the bundle in.

## Why the old roads are not simply copied across

They cover the old box. The new box extends 2 km further east, and that strip
contains the outlet gauge and the roads around it. Splicing the old network
into the new bundle would ship a road layer that stops 2 km short of the
domain edge while claiming to cover it -- which is the class of error this
project keeps cataloguing, not one to add to the list.

    python3 scripts/finish_box_move.py --check      # is Overpass back?
    python3 scripts/finish_box_move.py              # fetch and ship
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "portal" / "public" / "data" / "austin" / "study.json"
NEW_SOLVE = Path("/tmp/study_newbox_noroads.json")
SERVICE = "http://127.0.0.1:8000"

#: The widened box. Must match DEFAULT in build_austin_study.py.
BOX = {"west": -97.820, "south": 30.110, "east": -97.6678, "north": 30.210}


def fetch_roads(timeout: float = 240.0) -> dict:
    req = urllib.request.Request(
        f"{SERVICE}/roads",
        data=json.dumps(BOX).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report whether Overpass is answering, change nothing")
    ap.add_argument("--wait", type=int, default=0,
                    help="retry every N seconds until it answers")
    args = ap.parse_args()

    while True:
        try:
            got = fetch_roads()
        except Exception as exc:                                # noqa: BLE001
            got = {"ok": False, "reason": f"request failed: {str(exc)[:120]}"}
        n = len((got.get("roads") or {}).get("features", []))
        if got.get("ok") and n:
            break
        msg = got.get("reason") or f"{n} features returned"
        print(f"OpenStreetMap not usable yet: {str(msg)[:150]}")
        if args.check or not args.wait:
            return 1
        time.sleep(args.wait)

    print(f"roads: {n} features for the {BOX['east']} box")
    if args.check:
        return 0

    if not NEW_SOLVE.exists():
        print(f"missing {NEW_SOLVE}; re-run build_austin_study.py first",
              file=sys.stderr)
        return 2

    new = json.loads(NEW_SOLVE.read_text())
    if new["bounds"]["east"] != BOX["east"]:
        print(f"the saved solve is for east={new['bounds']['east']}, not "
              f"{BOX['east']}; re-run build_austin_study.py", file=sys.stderr)
        return 2

    new["roads"] = got["roads"]
    new["roads_note"] = None

    backup = BUNDLE.with_suffix(".json.prebox")
    if not backup.exists():
        shutil.copy(BUNDLE, backup)
        print(f"kept the 12.5 km bundle at {backup.name}")
    BUNDLE.write_text(json.dumps(new))
    print(f"-> {BUNDLE}  box {new['extent_km']} km, "
          f"{new['flooded_km2']} km2 wet, {n} road features")
    print("\nnext: node portal/scripts/verify-austin-default.mjs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
