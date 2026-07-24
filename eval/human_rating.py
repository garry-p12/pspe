#!/usr/bin/env python
"""Local human-rating harness for explanation briefs.

A free substitute for a paid crowdsourced study: colleagues or labmates rate
briefs on correctness, completeness and actionability, one brief at a time, from
the terminal. Ratings append to `ratings.jsonl` next to the briefs file.

    python eval/human_rating.py --briefs runs/explain/briefs.jsonl --rater alex
    python eval/human_rating.py --briefs runs/explain/briefs.jsonl --summarise

Raters do not see the reference brief or the model's own faithfulness score
before rating - showing either would anchor the ratings and make the
correlation with F(b) meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.utils import project_path  # noqa: E402

DIMENSIONS = {
    "correctness": "Does the brief describe an intervention consistent with the situation?",
    "completeness": "Does it cover the actuators, the cost and the confidence?",
    "actionability": "Could an operator act on this brief without further information?",
}


def rate(briefs_path: Path, rater: str, limit: int | None) -> int:
    records = [json.loads(line) for line in briefs_path.read_text().splitlines() if line]
    if limit:
        records = records[:limit]
    out_path = briefs_path.parent / "ratings.jsonl"

    print(f"Rating {len(records)} briefs as {rater!r}. Scores are 1-5; blank to skip.\n")
    written = 0
    with out_path.open("a") as handle:
        for record in records:
            print("-" * 72)
            print(record["generated"])
            print("-" * 72)
            scores: dict[str, int] = {}
            for name, question in DIMENSIONS.items():
                while True:
                    raw = input(f"  {name:<14} (1-5)  {question}\n  > ").strip()
                    if raw == "":
                        break
                    if raw.isdigit() and 1 <= int(raw) <= 5:
                        scores[name] = int(raw)
                        break
                    print("  enter a digit 1-5, or blank to skip")
            if scores:
                handle.write(
                    json.dumps({"id": record["id"], "rater": rater, "scores": scores}) + "\n"
                )
                written += 1
            print()
    print(f"wrote {written} ratings to {out_path}")
    return 0


def summarise(briefs_path: Path) -> int:
    ratings_path = briefs_path.parent / "ratings.jsonl"
    if not ratings_path.exists():
        print(f"no ratings at {ratings_path}; run without --summarise first.")
        return 1

    briefs = {
        json.loads(line)["id"]: json.loads(line)
        for line in briefs_path.read_text().splitlines() if line
    }
    ratings = [json.loads(line) for line in ratings_path.read_text().splitlines() if line]

    print(f"{len(ratings)} ratings from {len({r['rater'] for r in ratings})} rater(s)\n")
    for name in DIMENSIONS:
        values = [r["scores"][name] for r in ratings if name in r["scores"]]
        if values:
            spread = statistics.pstdev(values) if len(values) > 1 else 0.0
            print(f"  {name:<14} mean {statistics.mean(values):.2f}  sd {spread:.2f}  n={len(values)}")

    # Correlation between human correctness and the automatic faithfulness score
    # is the number that says whether F(b) is measuring anything a reader cares about.
    pairs = [
        (briefs[r["id"]]["faithfulness"], r["scores"]["correctness"])
        for r in ratings
        if "correctness" in r["scores"] and r["id"] in briefs
    ]
    if len(pairs) > 2:
        auto, human = zip(*pairs)
        if statistics.pstdev(auto) > 0 and statistics.pstdev(human) > 0:
            corr = statistics.correlation(auto, human)
            print(f"\n  corr(F(b), human correctness) = {corr:+.3f}  over {len(pairs)} briefs")
        else:
            print("\n  correlation undefined: one of the series is constant")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", default="runs/explain/briefs.jsonl")
    parser.add_argument("--rater", default="anonymous")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()

    path = project_path(args.briefs)
    if not path.exists():
        print(f"no briefs at {path}. Run scripts/train_explain.py first.")
        return 1
    return summarise(path) if args.summarise else rate(path, args.rater, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
