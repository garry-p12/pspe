#!/usr/bin/env python
"""Phase 5: train (or load) all four modules and run the end-to-end loop.

    python scripts/train_e2e.py
    python scripts/train_e2e.py testbed=rdf ablations.use_physics=false
    python scripts/train_e2e.py stages.simulate=false      # reuse a checkpoint

Writes `runs/e2e/results.json` with every Section-7 metric measured on one
trajectory, plus wall-clock and peak memory per stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import hydra
import torch
from omegaconf import DictConfig

from pspe.envs import make_env
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer
from pspe.perceive import (
    PerceiveConfig,
    PerceiveTrainConfig,
    PerceiveTrainer,
    PerceptionDataConfig,
)
from pspe.pipeline import PSPEPipeline, transfer_gap
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig
from pspe.simulate import (
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    load_surrogate,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed
from pspe.utils import RunLogger, get_device, project_path, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="e2e")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    log_root = project_path(cfg.output.log_dir)
    results: dict[str, object] = {"testbed": cfg.testbed, "ablations": dict(cfg.ablations)}

    testbed = make_testbed(cfg.testbed, grid=cfg.env.grid)

    # ---- Simulate ---------------------------------------------------------- #
    surrogate_path = project_path(cfg.checkpoints.surrogate)
    if cfg.stages.simulate:
        data = ensure_dataset(cfg.testbed, grid=cfg.env.grid, seed=cfg.seed)
        model = make_surrogate("fno", testbed.n_channels, grid=cfg.env.grid)
        sim_trainer = SimulateTrainer(
            model,
            SimulateTrainConfig(
                testbed=cfg.testbed, epochs=cfg.budgets.simulate_epochs,
                grid=cfg.env.grid, use_physics=cfg.ablations.use_physics, seed=cfg.seed,
            ),
            data,
            RunLogger(log_root / "simulate"),
            device,
        )
        results["simulate"] = sim_trainer.train()
        sim_trainer.save(surrogate_path)
    surrogate = load_surrogate(surrogate_path, device).eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # ---- Perceive ---------------------------------------------------------- #
    perceive_cfg = PerceiveConfig(
        image_size=cfg.env.grid, out_grid=cfg.env.grid, out_channels=1
    )
    perceive_trainer = PerceiveTrainer(
        perceive_cfg,
        PerceptionDataConfig(testbed=cfg.testbed, grid=cfg.env.grid,
                             image_size=cfg.env.grid, seed=cfg.seed),
        PerceiveTrainConfig(
            epochs=cfg.budgets.perceive_epochs,
            freeze_encoder=cfg.ablations.freeze_encoder, seed=cfg.seed,
        ),
        RunLogger(log_root / "perceive"),
        device,
    )
    if cfg.stages.perceive:
        results["perceive"] = perceive_trainer.train()
        perceive_trainer.save(project_path(cfg.checkpoints.perceive))
    else:
        blob = torch.load(project_path(cfg.checkpoints.perceive), map_location=device,
                          weights_only=False)
        perceive_trainer.model.load_state_dict(blob["model"])

    # ---- Plan -------------------------------------------------------------- #
    env_kwargs = dict(
        testbed=cfg.testbed, grid=cfg.env.grid, horizon=cfg.env.horizon,
        n_actuators=cfg.env.n_actuators, device=device, batched=True,
    )
    train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(testbed.n_channels, train_env.action_dim)

    planner = HybridPlannerTrainer(
        train_env, policy,
        cfg=PlannerConfig(
            iterations=cfg.budgets.plan_iterations, horizon=cfg.env.horizon,
            adaptive_alpha=cfg.ablations.adaptive_alpha, seed=cfg.seed,
        ),
        eval_env=eval_env,
        logger=RunLogger(log_root / "plan"),
        device=device,
    )
    if cfg.stages.plan:
        results["plan"] = planner.train()
        planner.save(project_path(cfg.checkpoints.policy))
    else:
        policy.load_state_dict(
            torch.load(project_path(cfg.checkpoints.policy), map_location=device)["policy"]
        )

    # ---- Explain ----------------------------------------------------------- #
    explain_trainer = ExplainTrainer(
        eval_env, policy, ExplainConfig(),
        ExplainTrainConfig(
            iterations=cfg.budgets.explain_iterations,
            use_faithfulness=cfg.ablations.use_faithfulness, seed=cfg.seed,
        ),
        RunLogger(log_root / "explain"),
        device,
    )
    if cfg.stages.explain:
        results["explain"] = explain_trainer.train()
        explain_trainer.save(project_path(cfg.checkpoints.explain))

    # ---- End-to-end loop --------------------------------------------------- #
    pipeline = PSPEPipeline(
        env=make_env(dynamics="truth", **env_kwargs),
        policy=policy,
        perceive=perceive_trainer.model,
        explain=explain_trainer.model,
        device=device,
    )
    report = pipeline.rollout(batch=4, generator=torch.Generator().manual_seed(cfg.seed))
    results["e2e"] = report.summary(eval_env.task.cost_limit)
    results["e2e_steps"] = [asdict(r) for r in report.records]
    # Within-family regime shift: the only transfer protocol a single-headed
    # surrogate can be scored on, since the testbeds differ in channel count.
    results["transfer"] = transfer_gap(
        surrogate, cfg.testbed, grid=cfg.env.grid, device=device
    )

    out_path = project_path(cfg.output.results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))

    e2e = results["e2e"]
    print(json.dumps(e2e, indent=2))
    print(f"\nfirst brief: {report.records[0].brief}")
    print(f"full results -> {out_path}")


if __name__ == "__main__":
    main()
