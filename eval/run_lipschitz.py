#!/usr/bin/env python
"""Assumption 1 and Proposition 1, measured.

    python eval/run_lipschitz.py --seed 0 --out runs/lipschitz/seed_0

The paper assumes the surrogate is 1-Lipschitz (enforced by spectral
normalisation) and derives from that a horizon-independent bound on the return
bias of planning through it:  |J_G - J_R| <= gamma / (1 - gamma)^2 * L * eps.

Two arms — the default FNO and the Lipschitz-mode FNO — and for each:

    rel L2          what the constraint costs in accuracy
    L_G             the Lipschitz constant of the *trained* network, by power
                    iteration on its Jacobian at sampled states
    eps             the calibration error sup||G - F||, as the max one-step
                    error over sampled (state, action) pairs
    return bias     |J_G(pi) - J_R(pi)| for a fixed stochastic policy, rolled
                    through the surrogate and through the true solver
    bound           gamma / (1 - gamma)^2 * eps, i.e. Prop. 1 with L = 1

Prop. 1 holds if bias <= bound. If L_G > 1 the bound's premise fails and the
recursion in the proof compounds geometrically instead of linearly, which is
the case the default architecture is in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


@torch.no_grad()
def rollout_return(env, policy, episodes: int, horizon: int, gamma: float, gen) -> float:
    state = env.reset(episodes, gen)
    total = torch.zeros(episodes, device=state.device)
    for t in range(horizon):
        action, _ = policy.sample(state)
        state, reward, _, _ = env.step(action)
        total += (gamma ** t) * reward
    return float(total.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/lipschitz")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)

    rows = []
    for lipschitz in (False, True):
        name = "lipschitz" if lipschitz else "default"
        seed_everything(args.seed)
        model = make_surrogate("fno", spec.n_channels, grid=args.grid, lipschitz=lipschitz)
        trainer = SimulateTrainer(
            model,
            SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                                epochs=args.epochs, grid=args.grid),
            data, RunLogger(root / name, use_tensorboard=False), device,
        )
        summary = trainer.train()
        model = trainer.model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # -- L_G on sampled states -------------------------------------------- #
        gen = torch.Generator().manual_seed(args.seed + 1)
        testbed = make_testbed(args.testbed, grid=args.grid, device=device)
        u0 = testbed.initial_condition(8, gen).to(device)
        ctrl = torch.zeros(8, 1, args.grid, args.grid, device=device)
        lip = model.estimate_lipschitz(u0, ctrl, iters=30)

        # -- eps: max one-step error over sampled (state, action) ------------- #
        env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                          n_actuators=9, device=device, batched=True)
        truth = make_env(dynamics="truth", **env_kwargs)
        surr = make_env(dynamics="surrogate", surrogate=model, **env_kwargs)
        policy = GaussianFieldPolicy(truth.obs_shape[0], truth.action_dim).to(device)

        with torch.no_grad():
            state = truth.reset(args.episodes, torch.Generator().manual_seed(args.seed + 2))
            errs = []
            for _ in range(args.horizon):
                action, _ = policy.sample(state)
                truth.state = state.clone()
                surr.state = state.clone()
                s_true, *_ = truth.step(action)
                s_surr, *_ = surr.step(action)
                errs.append((s_surr - s_true).flatten(1).norm(dim=1))
                state = s_true
            errs = torch.stack(errs)
        eps_max, eps_mean = float(errs.max()), float(errs.mean())

        # -- return bias vs the bound ----------------------------------------- #
        g = torch.Generator().manual_seed(args.seed + 3)
        j_true = rollout_return(truth, policy, args.episodes, args.horizon, args.gamma, g)
        g = torch.Generator().manual_seed(args.seed + 3)
        j_surr = rollout_return(surr, policy, args.episodes, args.horizon, args.gamma, g)
        bias = abs(j_surr - j_true)
        # Paper's bound: infinite-horizon sum, gamma / (1 - gamma)^2 * L * eps.
        # At gamma = 0.98 that prefactor is 2450, which makes the bound vacuous
        # by orders of magnitude. The same proof truncated at the rollout
        # horizon gives eps * sum_{t<H} gamma^t t — the number that is actually
        # informative about a horizon-H planner. Both reported, L = 1.
        bound_inf = args.gamma / (1 - args.gamma) ** 2 * eps_max
        bound_h = eps_max * sum(args.gamma ** t * t for t in range(args.horizon))

        rows.append({
            "arm": name,
            "rel L2": round(summary["rel_l2_final"], 4),
            "L_G": round(lip, 3),
            "L_G <= 1": lip <= 1.0,
            "eps max": round(eps_max, 4),
            "eps mean": round(eps_mean, 4),
            "return bias": round(bias, 4),
            "Prop1 bound, inf-horizon": round(bound_inf, 2),
            "Prop1 bound, H-horizon": round(bound_h, 3),
            "bias <= H-bound": bias <= bound_h,
        })
        print(f"[{name}] rel L2 {rows[-1]['rel L2']} L_G {lip:.3f} eps {eps_max:.4f} "
              f"bias {bias:.4f} bound_H {bound_h:.3f} bound_inf {bound_inf:.1f}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
