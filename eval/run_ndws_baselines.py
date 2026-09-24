#!/usr/bin/env python
"""Constrained RL baselines on the wildfire task, against PSPE and the heuristic.

    python eval/run_ndws_baselines.py --seed 0 --unet-ckpt runs/ndws_planning/seed_0/unet.pt

The gap this closes: PSPE's result on observed fire data is measured against
forecast-then-greedy, the doctrine operations actually use. The obvious
question is what a standard constrained-RL method does on the same fires, and
until now that comparison did not exist.

Every arm plans or learns against the *same* frozen next-day U-Net, under the
same daily crew budget, and is scored by the same function on the same
held-out fires. What differs is only how the intervention is chosen:

    greedy            forecast one day, treat the riskiest cells, repeat
    PPO-Lagrangian    model-free constrained RL, trained across fires
    CPO               trust-region constrained RL
    Sauté RL          budget folded into the state
    primal-dual NPG   natural gradient with a dual
    PSPE per-instance optimise this fire's three-day plan through the surrogate

Note on sample efficiency: on the synthetic testbeds the baselines interact
with true dynamics while PSPE acts inside a surrogate, so the sample-count
column means something. Here there is no separate true environment — the
surrogate is the best model of the fire anyone has — so every arm draws from
the same place and this comparison is about **decision quality only**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent  # noqa: E402
from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws_planning import (  # noqa: E402
    per_instance_plan,
    rollout,
    score_policy,
    train_unet,
)
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.simulate.real import NDWSConfig, driver_stats, load_ndws, normalize_drivers  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


class DetachedFireEnv(FireEnv):
    """FireEnv that hands out detached transitions.

    The model-free agents never differentiate through dynamics, but FireEnv's
    step keeps the surrogate's graph alive for the pathwise planner. Left
    attached, a horizon-length rollout accumulates the whole graph for nothing.
    """

    def step(self, action: torch.Tensor):
        with torch.no_grad():
            state, reward, cost, done = super().step(action)
        return state.detach(), reward.detach(), cost.detach(), done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--patches", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.03)
    parser.add_argument("--block", type=float, default=0.9)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--plan-steps", type=int, default=50)
    parser.add_argument("--algos", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--fair-init", action="store_true",
                        help="start the baselines' policy at an action that already MEETS the "
                             "budget, which is where PSPE's projected optimiser starts. Without "
                             "it a randomly initialised policy emits intensity ((0+1)/2)^2 = 0.25, "
                             "i.e. 25%% treated against a 3%% limit, and the agent has to walk 64 "
                             "dimensions down to the budget on scalar cost feedback alone. "
                             "Comparing against that is comparing against an initialisation.")
    parser.add_argument("--baseline-ki", type=float, default=0.5,
                        help="integral gain of the baselines' dual. 0.5, not the stock 0.05: "
                             "that change is what fixed our own planner's dual on rdf and it "
                             "took PPO-Lagrangian from 1.8%% to 0%% violations there, so the "
                             "baselines get it too")
    parser.add_argument("--unet-ckpt", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/ndws_baselines")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, grid=args.grid))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, grid=args.grid))
    dstats = driver_stats(train)
    train["drivers"] = normalize_drivers(train["drivers"], dstats)
    val["drivers"] = normalize_drivers(val["drivers"], dstats)

    if args.unet_ckpt:
        unet = make_fire_model("unet", 1 + train["drivers"].shape[1], grid=args.grid,
                               width=args.width).to(device)
        unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device)); unet.eval()
    else:
        unet, _ = train_unet(train, val, args, device)
        torch.save(unet.state_dict(), root / "unet.pt")

    task = FireTask(cost_limit=args.budget, block=args.block)
    env_kw = dict(grid=args.grid, patches=args.patches, horizon=args.horizon, device=device)
    train_env = DetachedFireEnv(unet, train, task, **env_kw)
    test_env = FireEnv(unet, val, task, **env_kw)          # scoring needs no grad either way
    held = torch.arange(test_env.n)
    k = args.patches * args.patches
    per_patch_budget = args.budget * k

    rows = []

    # --- the two reference points, recomputed here so every number in this
    #     table comes from one evaluation pass on one set of fires ---
    def none(state, t=0):
        return -torch.ones(state.shape[0], k, device=state.device)

    def greedy(state, t=0):
        p = test_env.predict(state)
        risk = torch.nn.functional.adaptive_avg_pool2d(p[:, None], args.patches).flatten(1)
        u = torch.zeros_like(risk)
        top = risk.topk(max(1, int(round(per_patch_budget))), dim=1).indices
        u.scatter_(1, top, 1.0)
        u = u * min(1.0, per_patch_budget / max(1, int(round(per_patch_budget))))
        return FireEnv.action_for(u)

    for name, act in (("none", none), ("greedy heuristic", greedy)):
        rows.append({"method": name, **score_policy(test_env, act, held, args.horizon),
                     "trained": "no"})
        print(f"[{name}] {rows[-1]}", flush=True)

    # --- constrained RL baselines ---
    capped_agents = []
    for algo in args.algos:
        seed_everything(args.seed)
        agent = make_agent(
            algo, train_env,
            cfg=SafeRLConfig(iterations=args.iterations, horizon=args.horizon,
                             batch=16, seed=args.seed, lagrangian_ki=args.baseline_ki,
                             log_dir=str(root / algo)),
            device=device,
        )
        if args.fair_init:
            # tanh(b)^-> intensity: solve ((tanh(b)+1)/2)^2 = budget for b.
            import math
            target = math.sqrt(max(args.budget, 1e-6))
            bias = math.atanh(max(-0.999, min(0.999, 2 * target - 1)))
            with torch.no_grad():
                agent.policy.mean_head.bias.fill_(bias)
                agent.policy.mean_head.weight.mul_(0.1)   # start near-uniform, as PSPE does
        summary = agent.train()
        agent.policy.eval()

        def act(state, t=0, ag=agent, name=algo):
            # Sauté conditions on remaining budget; at evaluation it starts full
            # and the scorer replays one plan per day, so pass the full budget.
            if name == "saute":
                budget = torch.full((state.shape[0],), task.cost_limit, device=state.device)
                state = ag.observe(state, budget)
            return ag.policy.sample(state, deterministic=True)[0]

        rows.append({"method": algo + (" (fair init)" if args.fair_init else ""),
                     **score_policy(test_env, act, held, args.horizon),
                     "trained": "across fires",
                     "train return": round(summary.get("return", float("nan")), 3)})
        capped_agents.append((algo, act))
        print(f"[{algo}] {rows[-1]}", flush=True)

    # --- PSPE, decision-time planning through the same surrogate ---
    plan_cache: dict = {}

    def per_instance(state, t=0):
        if t == 0:
            plan_cache["plan"] = per_instance_plan(test_env, state, args.horizon,
                                                   steps=args.plan_steps)
        return plan_cache["plan"][:, t]

    rows.append({"method": "PSPE per-instance", **score_policy(test_env, per_instance, held,
                                                               args.horizon),
                 "trained": "per fire, at decision time"})
    print(f"[PSPE] {rows[-1]}", flush=True)

    # --- matched-budget pass -------------------------------------------------
    # A method that ignores the budget reduces more burn for free, so the raw
    # table cannot be read as a comparison. Rescale every plan onto the budget
    # and score again: this is the only apples-to-apples column.
    def budget_capped(act):
        def wrapped(state, t=0):
            u = FireEnv.intensity(act(state, t))
            mean = u.mean(1, keepdim=True).clamp(min=1e-9)
            u = u * (task.cost_limit / mean).clamp(max=1.0)
            return FireEnv.action_for(u)
        return wrapped

    capped_rows = []
    for name, act in [("greedy heuristic", greedy)] + capped_agents + [("PSPE per-instance", per_instance)]:
        capped_rows.append({"method": name + " @ budget",
                            **score_policy(test_env, budget_capped(act), held, args.horizon),
                            "trained": "matched budget"})
        print(f"[matched] {capped_rows[-1]}", flush=True)
    rows.extend(capped_rows)
    for r in rows:
        for key, v in list(r.items()):
            if isinstance(v, float):
                r[key] = round(v, 4)
    table = markdown_table(rows)
    (root / "results.md").write_text(
        table + f"\n\nbudget {args.budget}; horizon {args.horizon}; block {args.block}\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
