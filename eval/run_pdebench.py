#!/usr/bin/env python
"""Train the surrogates on a real PDEBench file and compare to published numbers.

    python scripts/download_pdebench.py 2d_reacdiff      # once
    python eval/run_pdebench.py --file data/pdebench/2D_diff-react_NA_NA.h5 --testbed rdf

Trains FNO / DeepONet / GNOT on the PDEBench data under the identical loop used
for the synthetic testbeds, and prints relative L2 next to PDEBench's published
FNO forecast error for that family. This is the Section 7.2 "surrogate accuracy
vs baselines on a real dataset" comparison.

Caveat printed in the output: PDEBench reports normalised RMSE, this repo
reports relative L2. They are close but not identical, so the published column
is an anchor for order-of-magnitude agreement, not an exact match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.simulate import SimulateTrainConfig, SimulateTrainer, make_surrogate  # noqa: E402
from pspe.simulate.pdebench import PDEBenchConfig, load_pdebench  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

# Published PDEBench FNO forecast error (normalised RMSE), from Takamoto et al.
# (2022), Table 5/6. Anchors only — see the metric caveat above.
PUBLISHED_FNO = {
    "2d_reacdiff": 0.12,      # 2D diffusion-reaction
    "2d_shallowwater": 0.0044,  # 2D shallow-water (radial dam break)
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="path to a PDEBench .h5")
    parser.add_argument("--testbed", default="rdf", choices=["dar", "swe", "rdf"],
                        help="which testbed's channel count / task the file maps to")
    parser.add_argument("--benchmark", default=None,
                        help="key in PUBLISHED_FNO for the reference number")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--operators", nargs="+", default=["fno", "deeponet", "gnot"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/pdebench")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    data = load_pdebench(PDEBenchConfig(path=args.file, grid=args.grid,
                                        n_samples=args.n_samples))
    n_channels = data["states"].shape[2]
    print(f"loaded PDEBench: states {data['states'].shape}, {n_channels} channel(s)")

    published = PUBLISHED_FNO.get(args.benchmark or "")
    rows = []
    for op in args.operators:
        model = make_surrogate(op, n_channels, grid=args.grid)
        trainer = SimulateTrainer(
            model,
            SimulateTrainConfig(testbed=args.testbed, surrogate=op, epochs=args.epochs,
                                grid=args.grid, eval_vs_data=True),
            data, RunLogger(root / op, use_tensorboard=False), device,
        )
        summary = trainer.train()
        rows.append({
            "operator": op,
            "rel L2 (1 step)": round(summary["rel_l2_1step"], 4),
            "rel L2 (rollout)": round(summary["rel_l2_final"], 4),
            "params": int(summary["params/total"]),
            "wall (s)": round(summary["wall_clock_s"], 1),
            "PDEBench FNO (nRMSE)": published if op == "fno" and published else "",
        })

    table = markdown_table(rows)
    (root / "pdebench_results.md").write_text(table + "\n")
    (root / "pdebench_results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    if published:
        print(f"\nReference: PDEBench published FNO nRMSE = {published} for "
              f"'{args.benchmark}'. Note: nRMSE vs our relative L2 — an "
              "order-of-magnitude anchor, not an identical metric.")
    print(f"\nwritten to {root / 'pdebench_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
