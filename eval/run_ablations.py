#!/usr/bin/env python
"""Phase 5 ablation runner: one command, the full results table.

    python eval/run_ablations.py                          # default: dar, quick budgets
    python eval/run_ablations.py --testbed rdf --full     # paper-scale budgets
    python eval/run_ablations.py --only physics adaptive

Ablations, straight from the proposal:
    physics     physics-informed surrogate loss on/off
    adaptive    adaptive vs fixed pathwise/likelihood mixing coefficient
    perception  frozen vs fine-tuned perception encoder
    faithful    faithfulness loss on/off
    transfer    cross-PDE-family transfer of the surrogate
    baselines   CPO / PID-Lagrangian PPO / Sauté RL / primal-dual NPG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent  # noqa: E402
from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer  # noqa: E402
from pspe.perceive import (  # noqa: E402
    PerceiveConfig,
    PerceiveTrainConfig,
    PerceiveTrainer,
    PerceptionDataConfig,
)
from pspe.pipeline import transfer_gap  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

QUICK = {
    "simulate_epochs": 3, "plan_iterations": 30, "perceive_epochs": 3,
    "explain_iterations": 30, "baseline_iterations": 30, "n_trajectories": 64,
}
FULL = {
    "simulate_epochs": 20, "plan_iterations": 200, "perceive_epochs": 10,
    "explain_iterations": 200, "baseline_iterations": 200, "n_trajectories": 256,
}


def ensure_data(testbed: str, grid: int, n_trajectories: int) -> dict:
    return ensure_dataset(testbed, grid=grid, n_trajectories=n_trajectories)


def run_simulate(testbed: str, grid: int, budget: dict, device, root: Path,
                 use_physics: bool, surrogate: str = "fno") -> tuple[dict, torch.nn.Module]:
    data = ensure_data(testbed, grid, budget["n_trajectories"])
    spec = make_testbed(testbed, grid=grid)
    model = make_surrogate(surrogate, spec.n_channels, grid=grid)
    tag = f"simulate/{testbed}_{surrogate}_physics-{'on' if use_physics else 'off'}"
    trainer = SimulateTrainer(
        model,
        SimulateTrainConfig(
            testbed=testbed, surrogate=surrogate, epochs=budget["simulate_epochs"],
            grid=grid, use_physics=use_physics,
        ),
        data, RunLogger(root / tag), device,
    )
    return trainer.train(), trainer.model


def run_plan(testbed: str, grid: int, budget: dict, device, root: Path,
             surrogate: torch.nn.Module, adaptive: bool) -> dict:
    env_kwargs = dict(testbed=testbed, grid=grid, horizon=12, n_actuators=9,
                      device=device, batched=True)
    train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    tag = f"plan/{testbed}_alpha-{'adaptive' if adaptive else 'fixed'}"
    trainer = HybridPlannerTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=budget["plan_iterations"], adaptive_alpha=adaptive),
        eval_env=eval_env, logger=RunLogger(root / tag), device=device,
    )
    return trainer.train()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="paper-scale budgets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=["physics", "adaptive", "perception", "faithful", "transfer",
                 "baselines", "operators"],
    )
    parser.add_argument("--out", default="runs/ablations")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    budget = FULL if args.full else QUICK
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only) if args.only else {
        "physics", "adaptive", "perception", "faithful", "transfer", "baselines", "operators"
    }
    rows: list[dict] = []
    surrogate_for_plan: torch.nn.Module | None = None

    # --- Ablation 1: physics-informed loss on/off --------------------------- #
    if "physics" in selected or "adaptive" in selected or "transfer" in selected:
        for use_physics in (True, False):
            if "physics" not in selected and not use_physics:
                continue
            summary, model = run_simulate(
                args.testbed, args.grid, budget, device, root, use_physics
            )
            if use_physics:
                surrogate_for_plan = model
            rows.append({"run": f"simulate physics={'on' if use_physics else 'off'}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 2: operator baselines ------------------------------------- #
    if "operators" in selected:
        for name in ("deeponet", "gnot"):
            summary, _ = run_simulate(
                args.testbed, args.grid, budget, device, root, True, surrogate=name
            )
            rows.append({"run": f"simulate {name}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 3: adaptive vs fixed mixing ------------------------------- #
    if "adaptive" in selected:
        assert surrogate_for_plan is not None
        for adaptive in (True, False):
            summary = run_plan(args.testbed, args.grid, budget, device, root,
                               surrogate_for_plan, adaptive)
            rows.append({"run": f"plan alpha={'adaptive' if adaptive else 'fixed'}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 4: safe-RL baselines -------------------------------------- #
    if "baselines" in selected:
        for name in ALGORITHMS:
            env = make_env(testbed=args.testbed, grid=args.grid, horizon=12,
                           n_actuators=9, device=device, batched=True, dynamics="truth")
            agent = make_agent(
                name, env,
                cfg=SafeRLConfig(iterations=budget["baseline_iterations"],
                                 log_dir=str(root / "baselines")),
                device=device,
            )
            summary = agent.train()
            rows.append({"run": f"baseline {name}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 5: frozen vs fine-tuned perception encoder ---------------- #
    if "perception" in selected:
        for frozen in (True, False):
            trainer = PerceiveTrainer(
                PerceiveConfig(image_size=args.grid, out_grid=args.grid),
                PerceptionDataConfig(testbed=args.testbed, grid=args.grid,
                                     image_size=args.grid),
                PerceiveTrainConfig(epochs=budget["perceive_epochs"], freeze_encoder=frozen),
                RunLogger(root / f"perceive/{'frozen' if frozen else 'finetuned'}"),
                device,
            )
            summary = trainer.train()
            rows.append({"run": f"perceive encoder={'frozen' if frozen else 'finetuned'}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 6: faithfulness loss on/off ------------------------------- #
    if "faithful" in selected:
        for use_faith in (True, False):
            env = make_env(testbed=args.testbed, grid=args.grid, horizon=12,
                           n_actuators=9, device=device, batched=True, dynamics="truth")
            policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
            trainer = ExplainTrainer(
                env, policy, ExplainConfig(),
                ExplainTrainConfig(iterations=budget["explain_iterations"],
                                   use_faithfulness=use_faith),
                RunLogger(root / f"explain/{'on' if use_faith else 'off'}"),
                device,
            )
            summary = trainer.train()
            rows.append({"run": f"explain faithfulness={'on' if use_faith else 'off'}",
                         **{k: v for k, v in summary.items() if isinstance(v, (int, float))}})

    # --- Ablation 7: cross-domain transfer ---------------------------------- #
    # Within-family regime shift always applies; the cross-family swap is only
    # scored where the two testbeds share a channel count.
    if "transfer" in selected and surrogate_for_plan is not None:
        gap = transfer_gap(surrogate_for_plan, args.testbed, grid=args.grid, device=device)
        rows.append({"run": f"transfer {args.testbed} regime-shift",
                     **{k: v for k, v in gap.items() if isinstance(v, (int, float))}})
        for target in (t for t in ("dar", "swe", "rdf") if t != args.testbed):
            gap = transfer_gap(surrogate_for_plan, args.testbed, target,
                               grid=args.grid, device=device)
            if "note" in gap:
                print(f"skipping transfer {args.testbed}->{target}: {gap['note']}")
                continue
            rows.append({"run": f"transfer {args.testbed}->{target}",
                         **{k: v for k, v in gap.items() if isinstance(v, (int, float))}})

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print(table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
