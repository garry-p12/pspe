#!/usr/bin/env python
"""Phase 1: train a neural-operator surrogate.

    python scripts/train_simulate.py testbed=dar
    python scripts/train_simulate.py testbed=rdf surrogate=gnot
    python scripts/train_simulate.py train.use_physics=false     # ablation
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from pspe.simulate import SimulateTrainConfig, SimulateTrainer, ensure_dataset, make_surrogate
from pspe.simulate.solvers import make_testbed
from pspe.utils import RunLogger, get_device, project_path, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="simulate")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    data = ensure_dataset(cfg.testbed, grid=cfg.train.grid, seed=cfg.seed)
    testbed = make_testbed(cfg.testbed, grid=cfg.train.grid)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    if cfg.surrogate in ("deeponet", "gnot"):
        model_kwargs = {}  # those constructors take a different hyperparameter set
    model = make_surrogate(
        cfg.surrogate, testbed.n_channels, grid=cfg.train.grid,
        padded=cfg.train.padded, **model_kwargs
    )

    train_cfg = SimulateTrainConfig(
        testbed=cfg.testbed, surrogate=cfg.surrogate, seed=cfg.seed,
        log_dir=str(project_path(cfg.output.log_dir)),
        **OmegaConf.to_container(cfg.train, resolve=True),
    )
    logger = RunLogger(project_path(cfg.output.log_dir) / f"{cfg.testbed}_{cfg.surrogate}")
    trainer = SimulateTrainer(model, train_cfg, data, logger, device)

    summary = trainer.train()
    trainer.save(project_path(cfg.output.checkpoint))
    logger.close()

    print(json.dumps(summary, indent=2))
    print(
        f"[{cfg.testbed}/{cfg.surrogate}] relative L2 @ {int(summary['eval_horizon'])} steps: "
        f"{summary['rel_l2_final']:.4f} (1-step {summary['rel_l2_1step']:.4f})"
    )


if __name__ == "__main__":
    main()
