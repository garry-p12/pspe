#!/usr/bin/env python
"""Phase 4: train the faithfulness-regularised explainer.

    python scripts/train_explain.py
    python scripts/train_explain.py model.backbone=Qwen/Qwen2.5-1.5B-Instruct
    python scripts/train_explain.py train.use_faithfulness=false   # ablation

Briefs for the human-rating harness land in `<log_dir>/briefs.jsonl`.
"""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pspe.envs import make_env
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer
from pspe.plan import GaussianFieldPolicy
from pspe.utils import RunLogger, get_device, project_path, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="explain")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    env = make_env(
        testbed=cfg.testbed, dynamics="truth", grid=cfg.env.grid,
        horizon=cfg.env.horizon, n_actuators=cfg.env.n_actuators,
        device=device, batched=True,
    )
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)

    checkpoint = cfg.policy_checkpoint and project_path(cfg.policy_checkpoint)
    if checkpoint and checkpoint.exists():
        policy.load_state_dict(torch.load(checkpoint, map_location=device)["policy"])
        print(f"loaded planner from {checkpoint}")
    else:
        print(
            "no planner checkpoint; explaining an untrained policy. The "
            "faithfulness objective is still well defined - it measures agreement "
            "with whatever the planner intends - but the briefs describe a random plan."
        )

    trainer = ExplainTrainer(
        env,
        policy,
        ExplainConfig(**OmegaConf.to_container(cfg.model, resolve=True)),
        ExplainTrainConfig(
            seed=cfg.seed, log_dir=str(project_path(cfg.output.log_dir)),
            **OmegaConf.to_container(cfg.train, resolve=True),
        ),
        RunLogger(project_path(cfg.output.log_dir)),
        device,
    )

    summary = trainer.train()
    trainer.save(project_path(cfg.output.checkpoint))
    trainer.logger.close()

    print(json.dumps(summary, indent=2))
    print(
        f"[explain/{cfg.model.backbone}] F(b) {summary['eval/faithfulness']:.3f} "
        f"(reference ceiling {summary['eval/faithfulness_reference']:.3f}) | "
        f"briefs -> {summary['eval/briefs_path']}"
    )


if __name__ == "__main__":
    main()
