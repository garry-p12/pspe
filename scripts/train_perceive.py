#!/usr/bin/env python
"""Phase 3: LoRA-adapt a frozen perception backbone to recover physical fields.

    python scripts/train_perceive.py
    python scripts/train_perceive.py model.backbone=Qwen/Qwen2-VL-2B-Instruct
    python scripts/train_perceive.py data.source=real
    python scripts/train_perceive.py train.freeze_encoder=false   # ablation
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from pspe.perceive import (
    PerceiveConfig,
    PerceiveTrainConfig,
    PerceiveTrainer,
    PerceptionDataConfig,
)
from pspe.utils import RunLogger, get_device, project_path, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="perceive")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    model_cfg = PerceiveConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    data_cfg = PerceptionDataConfig(
        seed=cfg.seed, **OmegaConf.to_container(cfg.data, resolve=True)
    )
    train_cfg = PerceiveTrainConfig(
        seed=cfg.seed, log_dir=str(project_path(cfg.output.log_dir)),
        **OmegaConf.to_container(cfg.train, resolve=True),
    )
    logger = RunLogger(project_path(cfg.output.log_dir))
    trainer = PerceiveTrainer(model_cfg, data_cfg, train_cfg, logger, device)

    summary = trainer.train()
    trainer.save(project_path(cfg.output.checkpoint))
    logger.close()

    print(json.dumps(summary, indent=2))
    print(
        f"[perceive/{cfg.model.backbone}] held-out relative L2 "
        f"{summary['val/loss/regression']:.4f} | retrieval acc "
        f"{summary['val/metric/retrieval_acc']:.2f} | trainable "
        f"{summary['params/trainable_fraction'] * 100:.2f}% of parameters"
    )


if __name__ == "__main__":
    main()
