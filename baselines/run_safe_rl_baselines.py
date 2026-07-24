#!/usr/bin/env python
"""Phase 2 comparison: the hybrid planner vs the four constrained-RL baselines.

    python baselines/run_safe_rl_baselines.py testbed=dar
    python baselines/run_safe_rl_baselines.py planner.iterations=400

Reports cumulative reward, constraint-violation rate and sample count for each
algorithm. All five run on identical environments, reward and cost functionals.
The baselines are model-free; the hybrid planner may additionally use the
differentiable surrogate, which is the point of the sample-efficiency column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent  # noqa: E402
from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import load_dataset  # noqa: E402
from pspe.simulate.trainer import load_surrogate  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="plan")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    root = project_path("runs", "baselines", "safe_rl")

    env_kwargs = dict(
        testbed=cfg.testbed, grid=cfg.env.grid, horizon=cfg.env.horizon,
        n_actuators=cfg.env.n_actuators, device=device, batched=True,
    )
    rows = []

    # -- the four baselines -------------------------------------------------- #
    for name in ALGORITHMS:
        env = make_env(dynamics="truth", **env_kwargs)
        agent = make_agent(
            name, env,
            cfg=SafeRLConfig(
                iterations=cfg.planner.iterations, horizon=cfg.env.horizon,
                seed=cfg.seed, log_dir=str(root),
            ),
            device=device,
        )
        summary = agent.train()
        rows.append({
            "run": name,
            "return": summary["return"],
            "episode cost": summary["episode_cost"],
            "violation rate": summary["violation_rate"],
            # Model-free: every rollout step is a real environment interaction.
            "real env samples": summary["samples"],
            "surrogate samples": 0,
            "wall clock (s)": summary["wall_clock_s"],
        })

    # -- the proposal's planner ---------------------------------------------- #
    surrogate = None
    dynamics = "truth"
    surrogate_transitions = 0
    checkpoint = project_path(cfg.surrogate_checkpoint)
    if checkpoint.exists():
        surrogate = load_surrogate(checkpoint, device).eval()
        for p in surrogate.parameters():
            p.requires_grad_(False)
        dynamics = "surrogate"
        # The planner's real sample cost is the data the surrogate was fitted
        # on, not the rollouts it then takes inside the surrogate for free.
        try:
            blob = load_dataset(cfg.testbed, grid=cfg.env.grid)
            surrogate_transitions = int(
                blob["controls"].shape[0] * blob["controls"].shape[1]
            )
        except FileNotFoundError:
            print("surrogate present but its training set is missing; "
                  "real-sample accounting will report 0")
    else:
        print(f"no surrogate at {checkpoint}; the planner differentiates the solver instead.")

    train_env = make_env(dynamics=dynamics, surrogate=surrogate, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    trainer = HybridPlannerTrainer(
        train_env, policy,
        cfg=PlannerConfig(
            horizon=cfg.env.horizon, seed=cfg.seed, log_dir=str(root),
            **OmegaConf.to_container(cfg.planner, resolve=True),
        ),
        eval_env=eval_env, logger=RunLogger(root / "pspe_hybrid"), device=device,
        surrogate_train_transitions=surrogate_transitions,
    )
    summary = trainer.train()
    rows.append({
        "run": f"pspe_hybrid ({dynamics})",
        "return": summary["return"],
        "episode cost": summary["episode_cost"],
        "violation rate": summary["violation_rate"],
        "real env samples": summary["samples_real_env"],
        "surrogate samples": summary["samples_surrogate"],
        "wall clock (s)": summary["wall_clock_s"],
    })

    table = markdown_table(rows)
    out = root / f"{cfg.testbed}_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
