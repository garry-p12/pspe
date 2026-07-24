#!/usr/bin/env python
"""Phase 1 comparison: FNO vs DeepONet vs GNOT on one testbed.

    python baselines/run_operator_baselines.py testbed=dar
    python baselines/run_operator_baselines.py testbed=rdf train.epochs=30

Every model trains under the identical loop, loss and data, so the reported
relative L2 differences are the operators' and nothing else's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

OPERATORS = ["fno", "deeponet", "gnot"]


@hydra.main(version_base=None, config_path="../configs", config_name="simulate")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    grid = cfg.train.grid

    data = ensure_dataset(cfg.testbed, grid=grid, seed=cfg.seed)
    testbed = make_testbed(cfg.testbed, grid=grid)
    root = project_path("runs", "baselines", "operators")

    rows = []
    for name in OPERATORS:
        kwargs = OmegaConf.to_container(cfg.model, resolve=True) if name == "fno" else {}
        model = make_surrogate(name, testbed.n_channels, grid=grid, **kwargs)
        trainer = SimulateTrainer(
            model,
            SimulateTrainConfig(
                testbed=cfg.testbed, surrogate=name, seed=cfg.seed,
                **OmegaConf.to_container(cfg.train, resolve=True),
            ),
            data,
            RunLogger(root / f"{cfg.testbed}_{name}"),
            device,
        )
        summary = trainer.train()
        rows.append({
            "run": name,
            "rel L2 (1 step)": summary["rel_l2_1step"],
            "rel L2 (rollout)": summary["rel_l2_final"],
            "params": summary["params/total"],
            "wall clock (s)": summary["wall_clock_s"],
            "peak memory (MB)": summary["peak_memory_mb"],
        })

    table = markdown_table(rows)
    out = root / f"{cfg.testbed}_results.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
