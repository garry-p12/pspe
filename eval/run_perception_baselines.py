#!/usr/bin/env python
"""Section 7.2 perception baselines: what does the frozen VLM + LoRA design buy?

    python eval/run_perception_baselines.py                       # stub backbone, CPU-friendly
    python eval/run_perception_baselines.py --backbone google/siglip-base-patch16-224 --device cuda

Three arms over one dataset and one decoder, so the encoder is the only thing
that differs:

* **pspe**  - frozen backbone + LoRA adapters + contrastive term. The method.
* **probe** - the same frozen backbone with no adapters: decoder-only training.
  Isolates what LoRA contributes, which the freeze-vs-finetune ablation cannot
  (both of its arms have adapters).
* **cnn**   - a conv encoder trained from scratch on the regression loss alone,
  no freezing, no adapters, no contrastive term. The Section 7.2 baseline: if
  this matches the method, the architecture claim is unsupported on this task.

The zero-shot arm the proposal also lists (prompt a frozen VLM for the field
statistic) is not here: it needs a real instruction-tuned VLM and a text
protocol, so it belongs with the Phase C run, not with an offline stub.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.perceive import (  # noqa: E402
    PerceiveConfig,
    PerceiveTrainConfig,
    PerceiveTrainer,
    PerceptionDataConfig,
)
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

ARMS = {
    # name:  (config overrides,                          train overrides)
    "pspe":  (dict(use_lora=True),                       dict(freeze_encoder=True)),
    "probe": (dict(use_lora=False),                      dict(freeze_encoder=True, w_contrastive=0.0)),
    "cnn":   (dict(backbone="cnn", use_lora=False),      dict(freeze_encoder=False, w_contrastive=0.0)),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--backbone", default="tiny",
                        help="'tiny' stub, or an HF vision model id for the pspe/probe arms")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/perception_baselines")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, (model_over, train_over) in ARMS.items():
        # Reseed per arm: the arms must differ by architecture, not by which
        # random draw of initial weights and batch order each happened to get.
        seed_everything(args.seed)

        backbone = model_over.get("backbone", args.backbone)
        cfg = PerceiveConfig(
            **{**dict(backbone=backbone, image_size=args.grid, out_grid=args.grid),
               **{k: v for k, v in model_over.items() if k != "backbone"}}
        )
        trainer = PerceiveTrainer(
            cfg,
            PerceptionDataConfig(testbed=args.testbed, grid=args.grid,
                                 image_size=args.grid, seed=args.seed),
            PerceiveTrainConfig(**{**dict(epochs=args.epochs, batch=args.batch,
                                          seed=args.seed), **train_over}),
            RunLogger(root / name, use_tensorboard=False),
            device,
        )
        summary = trainer.train()
        rows.append({
            "arm": name,
            "backbone": backbone,
            "field rel L2": round(summary["val/loss/regression"], 5),
            "mse": round(summary["val/loss/mse"], 6),
            "retrieval acc": round(summary.get("val/metric/retrieval_acc", float("nan")), 4),
            "trainable": summary["params/trainable"],
            "stub backbone": summary.get("backbone_is_stub"),
            "wall (s)": round(summary["wall_clock_s"], 1),
        })
        print(f"[{name}] rel L2 {rows[-1]['field rel L2']}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
