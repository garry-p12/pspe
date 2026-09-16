#!/usr/bin/env python
"""The comparison the paper calls central to its claim: joint vs disaggregated.

    python eval/run_joint.py --seed 0 --out runs/joint/seed_0

Section 6 of the paper: "construct a disaggregated baseline — the best
standalone predictor and planner, each trained to its own convergence and
wired together only at inference — and compare it against jointly-trained PSPE
on identical testbeds and metrics." This is that comparison, for the
Simulate + Plan pair, with the arms chosen so the answer cannot be gamed:

    disaggregated   surrogate fitted to data, frozen; planner trained on it.
                    Everything measured so far in this repo is this arm.
    joint           the surrogate keeps learning from the planning loss
                    (beta > 0) with a data anchor and probe + margin on.
    joint-unanchored the same without the data term — the failure mode,
                    run on purpose so its signature is in the table.

For every arm: return, violating evaluations, real samples, and the
surrogate's held-out one-step rel L2 before and after — the capture diagnostic.
A joint arm that raises return while raising held-out error has not learned
better dynamics; it has learned to flatter the policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import (  # noqa: E402
    GaussianFieldPolicy,
    HybridPlannerTrainer,
    JointConfig,
    JointPlannerTrainer,
    PlannerConfig,
)
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr-surrogate", type=float, default=1e-4)
    parser.add_argument("--probe-every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/joint")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    # One pretrained surrogate, deep-copied per arm so every arm starts from
    # the same weights and differs only in what happens next.
    seed_everything(args.seed)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    sim = SimulateTrainer(
        make_surrogate("fno", spec.n_channels, grid=args.grid),
        SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                            epochs=args.epochs, grid=args.grid),
        data, RunLogger(root / "surrogate", use_tensorboard=False), device,
    )
    sim_summary = sim.train()
    base_surrogate = sim.model.cpu()
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    plan_cfg = dict(iterations=args.iterations, horizon=args.horizon, seed=args.seed,
                    real_cost_every=args.probe_every, cost_margin_k=2.0)

    arms = {
        "disaggregated":    None,
        "joint":            JointConfig(beta=args.beta, lr_surrogate=args.lr_surrogate, anchor=True),
        "joint-unanchored": JointConfig(beta=args.beta, lr_surrogate=args.lr_surrogate, anchor=False),
    }

    rows = []
    for name, joint in arms.items():
        seed_everything(args.seed)
        surrogate = copy.deepcopy(base_surrogate).to(device)
        train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
        eval_env = make_env(dynamics="truth", **env_kwargs)
        policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
        logger = RunLogger(root / name, use_tensorboard=False)

        if joint is None:
            for p in surrogate.parameters():
                p.requires_grad_(False)
            trainer = HybridPlannerTrainer(
                train_env, policy, cfg=PlannerConfig(**plan_cfg), eval_env=eval_env,
                logger=logger, device=device, surrogate_train_transitions=fit_transitions,
            )
            # Same held-out diagnostic as the joint arms, for a like-for-like column.
            probe = JointPlannerTrainer(
                train_env, policy, cfg=PlannerConfig(**plan_cfg), eval_env=eval_env,
                logger=RunLogger(root / f"{name}_probe", use_tensorboard=False),
                device=device, surrogate=surrogate, dataset=data, joint=JointConfig(beta=0.0),
            )
            before = probe.surrogate_rel_l2()
            s = trainer.train()
            after = probe.surrogate_rel_l2()
            s.update({"joint/surrogate_rel_l2_before": before,
                      "joint/surrogate_rel_l2_after": after,
                      "joint/surrogate_drift": after - before})
        else:
            trainer = JointPlannerTrainer(
                train_env, policy, cfg=PlannerConfig(**plan_cfg), eval_env=eval_env,
                logger=logger, device=device, surrogate_train_transitions=fit_transitions,
                surrogate=surrogate, dataset=data, joint=joint,
            )
            s = trainer.train()

        rows.append({
            "arm": name,
            "return": round(s["return"], 4),
            "violating_evals": round(s.get("eval/violating_eval_fraction", 0.0), 4),
            "worst_cost": round(s.get("eval/cost_max_over_run", 0.0), 4),
            "surrogate relL2 before": round(s["joint/surrogate_rel_l2_before"], 4),
            "surrogate relL2 after": round(s["joint/surrogate_rel_l2_after"], 4),
            "surrogate drift": round(s["joint/surrogate_drift"], 4),
            "real_samples": s["samples_real_env"],
        })
        print(f"[{name}] return {rows[-1]['return']} violating {rows[-1]['violating_evals']} "
              f"surrogate {rows[-1]['surrogate relL2 before']} -> {rows[-1]['surrogate relL2 after']}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + f"\n\npretrained surrogate rel L2 (rollout): "
                                     f"{sim_summary['rel_l2_final']:.4f}\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
