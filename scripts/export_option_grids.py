"""Per-option depth grids for the district planner.

The district's option list is ranked and selectable, and selecting a row moved
the levee walls but not the water: the map drew the same baseline event under
every plan. The fields were not missing -- `build_scenario_library.py` solved
all 37 options in one batch and saved every peak field -- they had simply never
been published to the portal (defect 36, which the Anywhere tab had in its own
form: there the field was computed per request and thrown away before the
response).

This publishes them in the encoding the portal already reads for the baseline
(`inspect.json` + `peak_depth.u8`), so the client swaps one array and the water
body, the paint on the terrain, and the click-anywhere depth query all follow
from the same grid rather than from three sources that can disagree.

    python3 scripts/export_option_grids.py

NOT WIRED TO THE PORTAL, AND THE CHECK AT THE END IS WHY. The district's
published baseline is the FloodCastBench reference solution -- `provenance.depth`
in the manifest badges it OBSERVED, and the frames are that dataset's own 30 m
run -- while this library is OUR solver on the same terrain. They disagree by a
mean of 4.1 of 255 over 58% of cells (45k wet cells against 151k). So swapping
one of these grids in under a selection would make "do nothing" the reference
field and "with levees" ours: the picture would change because the model
changed, not only because the plan did, which is defect 21 wearing different
clothes.

Drawing per-option water in the district therefore needs a decision, not a
patch: either the whole comparison comes from our library (baseline included,
and the reference becomes the validation layer it already has elsewhere), or the
district keeps the reference field and option selection keeps moving only the
walls and the metrics. The Anywhere tab has no such question -- every field
there, baseline and options alike, comes from the same solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="runs/scenarios_v2",
                    help="output of build_scenario_library.py")
    ap.add_argument("--raster", default="peak_r1.npy")
    ap.add_argument("--out", default="portal/public/data/opt")
    args = ap.parse_args()

    lib = Path(args.library)
    index = json.loads((lib / "index.json").read_text())
    peak = np.load(lib / args.raster)

    # The quantisation has to match the baseline the client already has, or a
    # chosen option would be drawn on a different scale from the do-nothing
    # field it is being compared against.
    data = Path("portal/public/data")
    meta = json.loads((data / "inspect.json").read_text())
    vmax = float(meta["depth_vmax_m"])
    grid = tuple(meta["grid"])
    if peak.shape[1:] != grid:
        raise SystemExit(f"library raster is {peak.shape[1:]}, portal grid is {grid}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for o in index["options"]:
        j = o["raster_index"]
        u8 = (np.clip(np.nan_to_num(peak[j]) / vmax, 0.0, 1.0) * 255).astype(np.uint8)
        (out / f"{o['id']}.u8").write_bytes(u8.tobytes())
        written[o["id"]] = {"wet_10cm": int((u8 / 255 * vmax > 0.1).sum()),
                            "peak_m": round(float(peak[j].max()), 2)}

    (out / "index.json").write_text(json.dumps(
        {"grid": list(grid), "depth_vmax_m": vmax,
         "note": "peak depth per option, quantised as portal/public/data/peak_depth.u8",
         "options": written}, indent=1))

    # The published baseline was written by a different script from the same
    # solve. If the two disagree, one of them is not the field it claims to be.
    base = np.frombuffer((data / "peak_depth.u8").read_bytes(), dtype=np.uint8)
    mine = np.frombuffer((out / "base.u8").read_bytes(), dtype=np.uint8)
    diff = np.abs(base.astype(int) - mine.astype(int))
    print(f"{len(written)} options -> {out}")
    print(f"base.u8 vs published peak_depth.u8: max {diff.max()}, mean {diff.mean():.4f}, "
          f"cells differing {int((diff > 0).sum())} of {diff.size}")
    for k in ("base", "s0", "s3", "all"):
        if k in written:
            print(f"  {k:6s} wet {written[k]['wet_10cm']:6d} cells, peak {written[k]['peak_m']} m")


if __name__ == "__main__":
    main()
