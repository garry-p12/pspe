"""Does the brief guard actually refuse a brief that misdescribes the plan?

`route_brief.checked` is the Explain stage's equivalent of
`margins.conformal_quantile` returning None: it withholds rather than ships
something it cannot stand behind. A guard that has never been shown to fire is
indistinguishable from no guard, so this breaks the renderer in each of the ways
it could realistically drift and checks that each one is caught and named.

    python3 eval/run_brief_guard.py

Both branches are needed. The no-route branch is where the guard was weakest:
it checked only the departure call, and so scored 1.000 on a brief whose own
sentences disagreed -- "every way through is under more water than 25 cm"
beside "0 km of 212 km is impassable", because the cut length was being read at
t = 0, when every run is still dry. The weakest part of a guard is wherever it
stops looking, which is why the drift list below covers each branch's claims
rather than only the ones that were easy to check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.explain import route_brief as rb  # noqa: E402

# Each case rewrites the shipped brief the way a drifting renderer would, and
# names the claim that should be flagged. Cases whose text is absent from a
# given payload are skipped and reported as skipped, so a payload that does not
# exercise a branch cannot be mistaken for one that passed it.
DRIFTS: list[tuple[str, str, str, str]] = [
    # a route was found
    ("distance", "24.8 km", "31.4 km", "km"),
    ("duration", "Drive 22 minutes", "Drive 35 minutes", "minutes"),
    ("departure call", "Leave now.", "Leaving now does not get through; leave at 1h00.", "decision"),
    ("deadline", "Deadline 3h30 from now", "Deadline 7h00 from now", "deadline"),
    ("water on the route", "No standing water on this route",
     "Deepest water 40 cm on Invented Road, 5 minutes in", "deepest"),
    # no route was found -- the branch that was passing a self-contradiction
    ("cut length at t=0", "Up to 117 km", "0 km", "cut_km"),
    ("invented cut road", "Cut: Casino Street", "Cut: Invented Street", "cut_roads"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", nargs="+",
                    default=["eval/fixtures/route_found.json",
                             "eval/fixtures/route_none.json"],
                    help="saved /route responses; defaults to the two "
                         "committed fixtures, one per branch")
    args = ap.parse_args()

    bad, exercised = 0, 0
    for path in args.payload:
        out = json.loads(Path(path).read_text())
        if not out.get("ok"):
            raise SystemExit(f"{path}: not a completed route response "
                             f"({out.get('reason')})")
        ctx = rb.context_from(out)
        text, report = rb.checked(out)
        branch = "route found" if ctx["ok"] else "no route"
        print(f"=== {Path(path).name}  [{branch}] ===")
        print(f"as shipped: faithfulness {report['faithfulness']}, "
              f"withheld={report['withheld']}")
        print(f"  {text}")
        if text is None or report["withheld"]:
            raise SystemExit("FAIL: the guard withheld the brief the renderer "
                             "itself produced")

        for name, find, replace, expect in DRIFTS:
            if find not in text:
                continue
            exercised += 1
            broken = text.replace(find, replace, 1)
            score, flagged = rb.faithfulness(broken, ctx)
            caught = score < 1.0 and expect in flagged
            if not caught:
                bad += 1
            print(f"  {name:20} score {score:.3f}  flagged {flagged}  "
                  f"{'caught' if caught else 'MISSED'}")
        print()

    print(f"{exercised} drifts exercised across {len(args.payload)} payloads")
    print("PASS — every drift is caught and named" if bad == 0
          else f"FAIL — {bad} drifts went unflagged")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
