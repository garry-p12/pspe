#!/usr/bin/env python
"""Theorem 1 / Eq. 8 tested as the paper's Section 6 specifies.

    python eval/run_alpha_rule.py --seed 0 --out runs/alpha_rule/seed_0

Three mixing rules for the hybrid gradient, one surrogate, identical seeds:

    fixed         alpha = 0.5 throughout
    variance      alpha* = (V_L - Cov) / (V_p + V_L - 2 Cov)   — what shipped
    eq8           alpha* = (V_L - Cov) / (B^2 + V_p + V_L - 2 Cov), with B^2
                  the *measured* squared pathwise bias, from differentiating
                  through the true solver and the surrogate from the same states

The earlier ablation compared only the first two and found a tail effect, not
a mean effect. Eq. 8 is the paper's actual claim; this is the first time it is
run. All three arms use probe + margin so the constraint is enforced the same
way, and the Eq. 8 arm's extra truth rollouts are counted in its real samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

ARMS = {
    "fixed":    dict(adaptive_alpha=False, eq8_alpha=False),
    "variance": dict(adaptive_alpha=True,  eq8_alpha=False),
    "eq8":      dict(adaptive_alpha=True,  eq8_alpha=True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--probe-every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/alpha_rule")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    sim = SimulateTrainer(
        make_surrogate("fno", spec.n_channels, grid=args.grid),
        SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                            epochs=args.epochs, grid=args.grid),
        data, RunLogger(root / "surrogate", use_tensorboard=False), device,
    )
    sim.train()
    surrogate = sim.model
    for p in surrogate.parameters():
        p.requires_grad_(False)
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    rows = []
    for name, overrides in ARMS.items():
        seed_everything(args.seed)
        train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
        eval_env = make_env(dynamics="truth", **env_kwargs)
        policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
        trainer = HybridPlannerTrainer(
            train_env, policy,
            cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                              seed=args.seed, real_cost_every=args.probe_every,
                              cost_margin_k=2.0, **overrides),
            eval_env=eval_env,
            logger=RunLogger(root / name, use_tensorboard=False),
            device=device, surrogate_train_transitions=fit_transitions,
        )
        s = trainer.train()
        rows.append({
            "arm": name,
            "return": round(s["return"], 4),
            "violating_evals": round(s.get("eval/violating_eval_fraction", 0.0), 4),
            "worst_cost": round(s.get("eval/cost_max_over_run", 0.0), 4),
            "final_alpha": round(s["final_alpha"], 4),
            "pathwise_bias_sq": round(s.get("alpha/final_pathwise_bias_sq", 0.0), 5),
            "real_samples": s["samples_real_env"],
        })
        print(f"[{name}] return {rows[-1]['return']} alpha {rows[-1]['final_alpha']} "
              f"B^2 {rows[-1]['pathwise_bias_sq']} real {rows[-1]['real_samples']}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
