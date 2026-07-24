#!/usr/bin/env python
"""Collect every `summary.json` under `runs/` into one results table.

    python eval/report.py
    python eval/report.py --root runs/ablations --out runs/report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import collect, markdown_table, to_rows  # noqa: E402
from pspe.utils import project_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = project_path(args.root)
    if not root.exists():
        print(f"no runs at {root}. Train something first (see `make train-simulate`).")
        return 1

    summaries = collect(root)
    if not summaries:
        print(f"no summary.json files under {root}.")
        return 1

    table = markdown_table(to_rows(summaries))
    print(table)
    if args.out:
        out = project_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
