#!/usr/bin/env python
"""Does closing the surrogate/reality cost gap fix the constraint violation?

    python eval/run_constraint_fix.py --seed 0 --out runs/constraint_fix/seed_0

The failure: the dual controller is driven by cost measured on *surrogate*
rollouts while violation is realised under the *true* dynamics. On `dar` the
planner exceeded a 0.936 limit on 7.3% of evaluations while every model-free
baseline exceeded it on none.

Four arms, chosen so the two mechanisms can be separated rather than confounded:

* **baseline**    - the current planner. The failure, reproduced.
* **probe**       - dual driven by real-environment cost every N iterations,
                    with the surrogate's cost bias-corrected in between.
* **margin**      - no probe; plan against a limit tightened by a fixed margin.
                    Tests whether conservatism alone is enough, without paying
                    any real samples.
* **probe+margin** - both.

The comparison is not only "did violations stop". A planner can satisfy any
constraint by refusing to act, so the table reports return alongside violation,
and the real-sample cost each arm pays: the whole claim against model-free
baselines is sample efficiency, and probe transitions are real interaction.
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

def arms(probe_every: int, margin_k: float) -> dict[str, dict]:
    return {
        "baseline":     dict(real_cost_every=0,           cost_margin_k=0.0),
        "probe":        dict(real_cost_every=probe_every, cost_margin_k=0.0),
        "margin":       dict(real_cost_every=0,           cost_margin_k=margin_k),
        "probe+margin": dict(real_cost_every=probe_every, cost_margin_k=margin_k),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="surrogate epochs")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-every", type=int, default=20,
                        help="iterations between real-environment cost probes. Must be "
                             "< --iterations or the probe arms never fire and every arm "
                             "silently collapses onto the baseline")
    parser.add_argument("--margin-k", type=float, default=2.0)
    parser.add_argument("--out", default="runs/constraint_fix")
    args = parser.parse_args()

    if args.probe_every >= args.iterations:
        raise SystemExit(
            f"--probe-every {args.probe_every} >= --iterations {args.iterations}: "
            "the probe would never fire and all four arms would be identical."
        )

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    # One surrogate, shared by every arm: the arms must differ in how the dual
    # is driven, not in how good their model of the world happens to be.
    seed_everything(args.seed)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    surrogate = make_surrogate("fno", spec.n_channels, grid=args.grid)
    sim = SimulateTrainer(
        surrogate,
        SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                            epochs=args.epochs, grid=args.grid),
        data, RunLogger(root / "surrogate", use_tensorboard=False), device,
    )
    sim_summary = sim.train()
    surrogate = sim.model
    for p in surrogate.parameters():
        p.requires_grad_(False)
    print(f"surrogate rel L2 {sim_summary['rel_l2_final']:.4f}")

    # Real transitions spent fitting the surrogate — the denominator of the
    # sample-efficiency claim, and what the probe arms add to.
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    rows = []
    for name, overrides in arms(args.probe_every, args.margin_k).items():
        seed_everything(args.seed)
        train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
        eval_env = make_env(dynamics="truth", **env_kwargs)
        policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
        trainer = HybridPlannerTrainer(
            train_env, policy,
            cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                              seed=args.seed, **overrides),
            eval_env=eval_env,
            logger=RunLogger(root / name, use_tensorboard=False),
            device=device,
            surrogate_train_transitions=fit_transitions,
        )
        s = trainer.train()
        rows.append({
            "arm": name,
            "return": round(s["return"], 4),
            "episode_cost": round(s["episode_cost"], 4),
            "cost_limit": s["cost_limit"],
            "violating_evals": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
            "worst_cost": round(s.get("eval/cost_max_over_run", float("nan")), 4),
            "cost_bias": round(s.get("dual/final_cost_bias", 0.0), 4),
            "effective_limit": round(s.get("dual/effective_limit", s["cost_limit"]), 4),
            "real_samples": s["samples_real_env"],
            "probe_samples": s.get("samples_real_probe", 0),
        })
        print(f"[{name}] return {rows[-1]['return']} "
              f"violating {rows[-1]['violating_evals']} "
              f"real samples {rows[-1]['real_samples']}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
