#!/usr/bin/env python
"""Resolution-generalization eval (paper Section 7.4).

Train a surrogate at one resolution, evaluate it at higher ones against the
numerical solver run at those resolutions. A discretization-invariant operator
(FNO) should degrade gracefully; a model that secretly memorised a fixed grid
would fall off a cliff.

    python eval/run_resolution.py                         # train 32, eval 32/48/64
    python eval/run_resolution.py --train-grid 64 --eval-grids 64 96 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
    resolution_generalization,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--train-grid", type=int, default=32)
    parser.add_argument("--eval-grids", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/resolution")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    data = ensure_dataset(args.testbed, grid=args.train_grid, n_trajectories=96)
    testbed = make_testbed(args.testbed, grid=args.train_grid)
    model = make_surrogate("fno", testbed.n_channels, grid=args.train_grid)
    trainer = SimulateTrainer(
        model,
        SimulateTrainConfig(testbed=args.testbed, surrogate="fno", epochs=args.epochs,
                            grid=args.train_grid),
        data, RunLogger(root / "train", use_tensorboard=False), device,
    )
    trainer.train()

    report = resolution_generalization(
        trainer.model, args.testbed, args.train_grid, args.eval_grids,
        steps=args.steps, device=device,
    )

    print(f"\n{args.testbed}: FNO trained at {args.train_grid}^2, "
          f"evaluated at unseen resolutions (rel L2, {args.steps}-step rollout):\n")
    print(f"{'grid':>6}{'cells':>10}{'rel L2 final':>14}{'rel L2 mean':>13}   note")
    for grid in args.eval_grids:
        r = report[grid]
        note = "(train grid)" if r["is_train_grid"] else ""
        print(f"{grid:>6}{grid*grid:>10}{r['rel_l2_final']:>14.4f}"
              f"{r['rel_l2_mean']:>13.4f}   {note}")

    (root / f"{args.testbed}_resolution.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    print(f"\nwritten to {root / f'{args.testbed}_resolution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
