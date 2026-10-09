"""Independent expectations for `verify-section-math.mjs`.

The point of that harness is that `src/lib/section.ts` is checked against
figures computed *somewhere else*, from the same solved profile. Asserting
TypeScript against literals copied out of the TypeScript's own output proves
only that it still does what it did.

So this is a second implementation, in a second language, of the structural
quantities: where the water is, which run is the main one, where the channel
is, and what the floodplain panels either side of it are. It deliberately
does NOT reimplement the crest search or the costing — those are checked in
the harness as properties ("the crest it returns holds", "raising a crest
cannot start overtopping", "the trapezoid is dearer than the flat rate it
replaced"), which is a stronger statement than a number and does not go stale
when the fixture changes.

    python3 scripts/section_baseline.py scripts/fixtures/section_onion_creek.json

Writes `<fixture>.expected.json` beside the fixture. The harness reads it and
refuses to run without it, so a regenerated fixture cannot quietly leave the
expectations describing a different river.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The spec this re-implements, from `src/lib/section.ts`:
#
#   A channel cell is one deeper than `max(median + DEEPER_BY_M,
#   maxDepth / 2)` of the reach it is in — the reach's OWN median, because an
#   absolute threshold called a 2.49 m backwater a channel on one river and
#   missed a real one on another (§5.14d). Strictly deeper, not at-or-deeper.
#   Channel cells no more than BELT_GAP_SAMPLES apart are one belt, so the
#   braided threads of a river are not counted as several rivers.
#
# Implemented here from that description rather than ported from the
# TypeScript, which is the only way the comparison means anything.
DEEPER_BY_M = 2.0
BELT_GAP_SAMPLES = 3


def sample_spacing(res: dict) -> float:
    n = max(int(res["samples"]) - 1, 1)
    return float(res["length_m"]) / n


def reaches(profile: list[dict], min_samples: int = 2) -> list[dict]:
    """Contiguous wet runs, longest first."""
    runs, cur = [], []
    for i, q in enumerate(profile):
        if q.get("water_m") is not None:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < min_samples:
            continue
        waters = [profile[i]["water_m"] for i in r]
        depths = [float(profile[i].get("depth_m") or 0.0) for i in r]
        out.append({
            "i0": r[0], "i1": r[-1],
            "m0": profile[r[0]]["m"], "m1": profile[r[-1]]["m"],
            "samples": len(r),
            "maxWaterM": max(waters),
            "maxDepthM": max(depths),
        })
    out.sort(key=lambda r: -r["samples"])
    return out


def channel_belts(profile: list[dict], reach: dict) -> list[dict]:
    """Where the river is, inside one wet run.

    Against the reach's own median depth: the channel is where the water is
    much deeper than the floodplain around it, which is a statement about
    this reach and not about a number in metres.
    """
    idx = list(range(reach["i0"], reach["i1"] + 1))
    depths = [float(profile[i].get("depth_m") or 0.0) for i in idx]
    if not depths:
        return []
    mid = sorted(depths)[len(depths) // 2]
    cut = max(mid + DEEPER_BY_M, reach["maxDepthM"] / 2.0)
    deep = [i for i in idx if float(profile[i].get("depth_m") or 0.0) > cut]
    if not deep:
        return []
    belts, start, prev = [], deep[0], deep[0]
    for i in deep[1:]:
        if i - prev > BELT_GAP_SAMPLES:
            belts.append({"i0": start, "i1": prev})
            start = i
        prev = i
    belts.append({"i0": start, "i1": prev})
    return belts


def panels(profile: list[dict], reach: dict, belts: list[dict]) -> list[dict]:
    """The floodplain either side of the channel, widest first."""
    out, cursor = [], reach["i0"]
    for b in sorted(belts, key=lambda x: x["i0"]):
        if b["i0"] - 1 >= cursor:
            out.append({"i0": cursor, "i1": b["i0"] - 1})
        cursor = b["i1"] + 1
    if reach["i1"] >= cursor:
        out.append({"i0": cursor, "i1": reach["i1"]})
    out = [p for p in out if p["i1"] > p["i0"]]
    out.sort(key=lambda p: -(p["i1"] - p["i0"]))
    return out


def stage_of(profile: list[dict], span: dict) -> float:
    ws = [profile[i]["water_m"] for i in range(span["i0"], span["i1"] + 1)
          if profile[i].get("water_m") is not None]
    return max(ws) if ws else float("-inf")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-4].strip(), file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    res = json.loads(path.read_text())
    prof = res["profile"]

    all_runs = reaches(prof, 1)
    kept = reaches(prof, 2)
    if not kept:
        print("the fixture has no wet reach; nothing to check against",
              file=sys.stderr)
        return 1
    main_reach = kept[0]
    belts = channel_belts(prof, main_reach)
    pans = panels(prof, main_reach, belts)

    out = {
        "fixture": path.name,
        "note": "Computed by scripts/section_baseline.py from the fixture "
                "alone. Do not copy these from the TypeScript's output.",
        "spacing_m": round(sample_spacing(res), 4),
        "runs_unfiltered": len(all_runs),
        "runs_2plus": len(kept),
        "main": {
            "m0": main_reach["m0"], "m1": main_reach["m1"],
            "samples": main_reach["samples"],
            "maxWaterM": round(main_reach["maxWaterM"], 4),
            "maxDepthM": round(main_reach["maxDepthM"], 4),
        },
        "belts": [{"i0": b["i0"], "i1": b["i1"],
                   "m0": prof[b["i0"]]["m"], "m1": prof[b["i1"]]["m"],
                   "samples": b["i1"] - b["i0"] + 1} for b in belts],
        "panels": [{"i0": p["i0"], "i1": p["i1"],
                    "m0": prof[p["i0"]]["m"], "m1": prof[p["i1"]]["m"],
                    "stage_m": round(stage_of(prof, p), 4)} for p in pans],
        "whole_line_peak_m": res.get("peak_water_m"),
    }
    dest = path.with_suffix(".expected.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"→ {dest}")
    print(f"  {out['runs_2plus']} reaches, main {out['main']['samples']} samples "
          f"at stage {out['main']['maxWaterM']} m")
    print(f"  {len(out['belts'])} channel belt(s), {len(out['panels'])} panel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
