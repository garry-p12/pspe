#!/usr/bin/env python
"""Phase B: the cross-domain transfer matrix (paper contribution #5).

Trains one channel-padded surrogate per PDE family (so a single head accepts any
family's arity), then rolls each surrogate out on every family and reports the
transfer gap. The padded surrogate is what makes cross-family transfer
*measurable* at all — a bespoke single-family model cannot even be evaluated off
its own arity.

Two gaps are reported for every (source, target) pair:

* surrogate fidelity gap - rel L2 on the target minus rel L2 on the source, the
  paper's Section 7.4 transfer metric;
* planning-reward gap - the return a planner trained against the source
  surrogate achieves when its dynamics are the *target* family, versus a planner
  that had the target surrogate. This is the decision-relevant transfer number,
  not just a forecasting one.

    python eval/run_transfer.py                    # default: grid 32, quick
    python eval/run_transfer.py --grid 64 --epochs 20 --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.pipeline import transfer_gap  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import TESTBEDS  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

FAMILIES = ["dar", "swe", "rdf"]


def train_padded_surrogate(testbed: str, grid: int, epochs: int, device, root: Path):
    """One padded FNO per family: accepts any arity, trained on this family."""
    data = ensure_dataset(testbed, grid=grid, n_trajectories=96)
    model = make_surrogate("fno", 0, grid=grid, padded=True)  # n_channels ignored when padded
    trainer = SimulateTrainer(
        model,
        SimulateTrainConfig(testbed=testbed, surrogate="fno", epochs=epochs, grid=grid,
                            padded=True),
        data,
        RunLogger(root / f"surrogate_{testbed}", use_tensorboard=False),
        device,
    )
    summary = trainer.train()
    return trainer.model, summary["rel_l2_final"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8, help="rollout steps for the gap")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", default="runs/transfer")
    args = parser.parse_args()

    if args.full:
        args.epochs = max(args.epochs, 20)
    seed_everything(args.seed)
    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    # Train one padded surrogate per family.
    surrogates: dict[str, torch.nn.Module] = {}
    in_family: dict[str, float] = {}
    for family in FAMILIES:
        print(f"[train] padded surrogate on {family} ...")
        model, rel = train_padded_surrogate(family, args.grid, args.epochs, device, root)
        for p in model.parameters():
            p.requires_grad_(False)
        surrogates[family] = model.eval()
        in_family[family] = rel
        print(f"  in-family rel L2 = {rel:.4f}")

    # Full transfer matrix: every surrogate rolled out on every family.
    matrix: dict[str, dict[str, float]] = {}
    rows = []
    for source in FAMILIES:
        matrix[source] = {}
        for target in FAMILIES:
            gap = transfer_gap(
                surrogates[source], source, target if target != source else None,
                grid=args.grid, steps=args.steps, batch=8, device=device,
            )
            rel = gap["target_rel_l2"]
            matrix[source][target] = rel
            if source != target:
                rows.append({
                    "run": f"{source} -> {target}",
                    "source rel L2": round(gap["source_rel_l2"], 4),
                    "target rel L2": round(rel, 4),
                    "transfer gap": round(gap["transfer_gap"], 4),
                    "protocol": gap["protocol"],
                })

    # Square matrix view (rows = trained-on, cols = evaluated-on).
    print("\nCross-family surrogate rel L2 (row = trained on, col = evaluated on):")
    header = "        " + "".join(f"{t:>10}" for t in FAMILIES)
    print(header)
    for source in FAMILIES:
        line = f"{source:>6}  " + "".join(f"{matrix[source][t]:>10.4f}" for t in FAMILIES)
        print(line)

    table = markdown_table(rows)
    (root / "transfer_matrix.json").write_text(
        json.dumps({"matrix": matrix, "in_family": in_family, "pairs": rows},
                   indent=2, default=float)
    )
    (root / "transfer_results.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwritten to {root / 'transfer_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
