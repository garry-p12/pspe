#!/usr/bin/env python
"""Phase 2: train the constrained hybrid-gradient planner.

    python scripts/train_plan.py testbed=dar
    python scripts/train_plan.py planner.adaptive_alpha=false   # ablation
    python scripts/train_plan.py dynamics=truth                 # no surrogate

Training rolls the differentiable dynamics; evaluation always uses the
numerical solver, so the reported return and violation rate are not
self-graded by the surrogate.
"""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from pspe.envs import make_env
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig
from pspe.simulate.trainer import load_surrogate
from pspe.utils import RunLogger, get_device, project_path, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="plan")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    surrogate = None
    dynamics = cfg.dynamics
    if dynamics == "surrogate":
        checkpoint = project_path(cfg.surrogate_checkpoint)
        if checkpoint.exists():
            surrogate = load_surrogate(checkpoint, device).eval()
            for p in surrogate.parameters():
                p.requires_grad_(False)
        else:
            print(
                f"no surrogate at {checkpoint}; falling back to dynamics=truth. "
                "Run scripts/train_simulate.py first for the surrogate-based planner."
            )
            dynamics = "truth"

    env_kwargs = dict(
        testbed=cfg.testbed, grid=cfg.env.grid, horizon=cfg.env.horizon,
        n_actuators=cfg.env.n_actuators, device=device, batched=True,
    )
    train_env = make_env(dynamics=dynamics, surrogate=surrogate, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)

    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    planner_cfg = PlannerConfig(
        horizon=cfg.env.horizon, seed=cfg.seed,
        log_dir=str(project_path(cfg.output.log_dir)),
        **OmegaConf.to_container(cfg.planner, resolve=True),
    )
    logger = RunLogger(project_path(cfg.output.log_dir) / cfg.testbed)
    trainer = HybridPlannerTrainer(
        train_env, policy, cfg=planner_cfg, eval_env=eval_env, logger=logger, device=device
    )

    summary = trainer.train()
    trainer.save(project_path(cfg.output.checkpoint))
    logger.close()

    print(json.dumps(summary, indent=2))
    print(
        f"[{cfg.testbed}] return {summary['return']:+.3f} | "
        f"episode cost {summary['episode_cost']:.3f} / limit {summary['cost_limit']:.2f} | "
        f"violation rate {summary['violation_rate']:.2f} | "
        f"alpha {summary['final_alpha']:.2f} | samples {summary['samples']}"
    )


if __name__ == "__main__":
    main()
