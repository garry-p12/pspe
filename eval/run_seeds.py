#!/usr/bin/env python
"""Phase F: the results table over N seeds, with error bars.

    python eval/run_seeds.py                                   # 5 seeds, quick budgets
    python eval/run_seeds.py --seeds 0 1 2 3 4 --full          # paper-scale
    python eval/run_seeds.py --only adaptive baselines         # just the Phase 2 comparison
    python eval/run_seeds.py --aggregate-only                  # re-table what already ran

Every number in this repo was a single seed, which makes none of them
reportable: a 3% gap between adaptive and fixed alpha means nothing without the
seed-to-seed spread to compare it against. This runs `eval/run_ablations.py`
once per seed in its own process (so the seed is set before any module touches
the RNG) and collapses the per-seed `results.json` files into one mean +/- std
table.

Two things it deliberately does not do:

* it does not regenerate the PDE dataset per seed. `ensure_dataset` caches by
  (testbed, grid), so all seeds train on the same trajectories and the spread
  measures training stochasticity — initialisation, minibatch order, policy
  sampling, evaluation rollouts — not dataset resampling. That is the variance
  an ablation comparison needs; dataset variance is a separate, larger study.
* it does not average across testbeds. Runs are matched by name, and `dar` and
  `rdf` numbers are not commensurable.

Per-seed directories are reused when they already hold a `results.json`, so an
interrupted sweep resumes instead of restarting; pass `--force` to rerun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import aggregate_seeds, markdown_table  # noqa: E402
from pspe.utils import project_path  # noqa: E402

ABLATIONS = Path(__file__).resolve().parent / "run_ablations.py"


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}"


def load_rows(path: Path) -> list[dict]:
    """Per-seed result rows, from either results layout this repo writes.

    `eval/run_ablations.py` writes a bare list of rows. `eval/run_transfer.py`
    writes an object whose `pairs` key holds the same shape alongside the square
    matrix, so both can be aggregated by the same code rather than growing a
    second aggregator that drifts from this one.
    """
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
        return payload["pairs"]
    raise SystemExit(f"{path}: expected a list of rows or an object with a 'pairs' list")


def run_seed(seed: int, args: argparse.Namespace, root: Path) -> list[dict]:
    """One full ablation sweep at `seed`, in a fresh process."""
    out = seed_dir(root, seed)
    results = out / args.results_file
    if results.exists() and not args.force:
        print(f"[seed {seed}] reusing {results}")
        return load_rows(results)

    cmd = [
        sys.executable, str(ABLATIONS),
        "--seed", str(seed),
        "--out", str(out),
        "--testbed", args.testbed,
        "--grid", str(args.grid),
        "--device", args.device,
    ]
    if args.full:
        cmd.append("--full")
    if args.only:
        cmd += ["--only", *args.only]

    print(f"[seed {seed}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"seed {seed} failed (exit {proc.returncode}); nothing aggregated")
    return load_rows(results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="paper-scale budgets")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=["physics", "adaptive", "perception", "faithful", "transfer",
                 "baselines", "operators"],
    )
    parser.add_argument("--out", default="runs/seeds")
    parser.add_argument("--force", action="store_true", help="rerun seeds that already have results")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training; table whatever seed dirs already exist")
    parser.add_argument("--results-file", default="results.json",
                        help="per-seed results filename; transfer sweeps write transfer_matrix.json")
    parser.add_argument("--key", default="run",
                        help="row-identity column. The ablation runner labels rows "
                             "'run'; the baseline runners label them 'arm'. Rows are "
                             "matched across seeds by this, so the wrong key silently "
                             "merges every arm into one meaningless average")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit(f"duplicate seeds in {args.seeds}: they would be averaged as independent runs")

    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, list[dict]] = {}
    for seed in args.seeds:
        if args.aggregate_only:
            results = seed_dir(root, seed) / args.results_file
            if not results.exists():
                print(f"[seed {seed}] no results at {results}; skipping")
                continue
            per_seed[seed] = load_rows(results)
        else:
            per_seed[seed] = run_seed(seed, args, root)

    if not per_seed:
        print(f"no per-seed results under {root}. Run without --aggregate-only first.")
        return 1
    if len(per_seed) < 2:
        print(f"warning: {len(per_seed)} seed(s) — the table will carry means with no spread.")

    missing = [s for s, rows in per_seed.items()
               if rows and not any(args.key in row for row in rows)]
    if missing:
        available = sorted({k for rows in per_seed.values() for row in rows for k in row})
        raise SystemExit(
            f"no '{args.key}' column in seeds {missing}; columns present: {available}. "
            f"Pass --key with one of them."
        )
    rows, stats = aggregate_seeds(per_seed, key=args.key)
    table = markdown_table(rows)
    header = (
        f"# {args.testbed} over {len(per_seed)} seeds "
        f"({', '.join(str(s) for s in sorted(per_seed))}), "
        f"{'full' if args.full else 'quick'} budgets, grid {args.grid}\n\n"
        "Cells are mean ± sample std across seeds.\n\n"
    )
    (root / "results_seeds.md").write_text(header + table + "\n")
    (root / "results_seeds.json").write_text(json.dumps(
        {"testbed": args.testbed, "grid": args.grid, "full": args.full,
         "seeds": sorted(per_seed), "stats": stats},
        indent=2, default=float,
    ))
    print("\n" + table)
    print(f"\nwritten to {root / 'results_seeds.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
