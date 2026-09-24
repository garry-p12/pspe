#!/usr/bin/env python
"""Constrained firebreak planning on observed wildfire data.

    python eval/run_ndws_planning.py --seed 0 --out runs/ndws_planning/seed_0

The first Plan result on real data. Trains the next-day U-Net on NDWS (the
same recipe as eval/run_ndws.py), freezes it, and plans firebreaks through it
with the PSPE v2 planner under a crew budget. Scored on held-out real patches
against three controls that need no learning:

    none      no treatment
    random    budget spread over random patches
    greedy    budget spent on the highest predicted-risk patches (one forecast,
              then act), the operational heuristic

and against the planner without the budget dual (`unconstrained`), which shows
what the budget costs.

Metrics per policy on the held-out patches, all in-model (see FireEnv):
expected burned fraction (population-weighted and not), treated fraction,
fraction of patches over budget. Plus the surrogate's forecast AUC-PR on the
same patches, the only number here that is checked against what the fire
actually did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws import average_precision, batched  # noqa: E402
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate.real import NDWSConfig, driver_stats, fire_stats, load_ndws, normalize_drivers  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


def train_unet(train, val, args, device) -> tuple[torch.nn.Module, float]:
    """The eval/run_ndws.py recipe, selected on validation AUC-PR."""
    seed_everything(args.seed)
    in_channels = 1 + train["drivers"].shape[1]
    model = make_fire_model("unet", in_channels, grid=args.grid, width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    prevalence = max(fire_stats(train)["fire_prevalence"], 1e-6)
    pos_weight = torch.tensor(max((1.0 / prevalence) ** 0.5, 1.0), device=device)
    n = train["states"].shape[0]
    best = {"aucpr": -1.0, "state": None}

    def evaluate(data):
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for begin in range(0, data["states"].shape[0], 64):
                idx = np.arange(begin, min(begin + 64, data["states"].shape[0]))
                x, target, valid = batched(data, idx, device)
                p = torch.sigmoid(model(x)[:, 0])
                scores.append(p[valid].cpu().numpy()); labels.append((target[valid] > 0.5).float().cpu().numpy())
        model.train()
        return np.concatenate(scores), np.concatenate(labels)

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        for begin in range(0, n, args.batch):
            x, target, valid = batched(train, perm[begin:begin + args.batch], device)
            loss_map = F.binary_cross_entropy_with_logits(
                model(x)[:, 0], (target > 0.5).float(), pos_weight=pos_weight, reduction="none")
            loss = (loss_map * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        s, l = evaluate(val)
        aucpr = average_precision(s, l)
        if aucpr > best["aucpr"]:
            best = {"aucpr": aucpr, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        print(f"[unet] epoch {epoch + 1}/{args.epochs} val AUC-PR {aucpr:.4f}")
    model.load_state_dict(best["state"])
    return model.eval(), best["aucpr"]


def rollout(env: FireEnv, state: torch.Tensor, actions, horizon: int, intensities: bool = False):
    """Autoregressive H-day rollout. `actions(state, t)` gives day t's action
    (or intensities in [0, 1] when `intensities=True`).
    Returns final-day ignition probability (B, H, W) and per-day treated fractions (B, T)."""
    treated = []
    for t in range(horizon):
        a = actions(state, t)
        b = env.treatment_from_intensity(a) if intensities else env.treatment(a)
        p = env.predict(state, b)
        treated.append(b.flatten(1).mean(1))
        state = state.clone(); state[:, 0] = p
    return p, torch.stack(treated, 1)


def project_budget(u: torch.Tensor, limit: float, iters: int = 20) -> torch.Tensor:
    """Euclidean projection of intensities (…, K) onto {0 <= u <= 1, mean(u) <= limit}:
    subtract a per-instance threshold tau, clamp, with tau found by bisection."""
    u = u.clamp(0, 1)
    over = u.mean(-1, keepdim=True) > limit
    lo = torch.zeros_like(u[..., :1]); hi = torch.ones_like(u[..., :1])
    for _ in range(iters):
        tau = (lo + hi) / 2
        m = (u - tau).clamp(0, 1).mean(-1, keepdim=True)
        hi = torch.where(m > limit, hi, tau); lo = torch.where(m > limit, tau, lo)
    return torch.where(over, (u - hi).clamp(0, 1), u)


def per_instance_plan(env: FireEnv, state: torch.Tensor, horizon: int, steps: int = 100,
                      lr: float = 0.05) -> torch.Tensor:
    """Decision-time planning: optimise this fire's break intensities for every
    day jointly, through the autoregressive surrogate rollout (pure pathwise
    gradient). Projected gradient descent in intensity space: start with the
    budget spread uniformly, normalised gradient steps, exact projection onto
    the per-day budget after every step. No penalty, no Adam: a penalty plus
    Adam froze the optimiser at a tenth of the budget (seed 0, first attempt).
    Returns (B, T, K) actions."""
    k = env.action_dim
    u = torch.full((state.shape[0], horizon, k), env.task.cost_limit, device=state.device)
    w = 1 + env.task.population_weight * env.population(state)
    with torch.enable_grad():
        for _ in range(steps):
            u = u.detach().requires_grad_(True)
            p, _ = rollout(env, state, lambda st, t: u[:, t], horizon, intensities=True)
            burned = (p * w).flatten(1).mean(1).sum()
            (g,) = torch.autograd.grad(burned, u)
            g = g / (g.flatten(1).norm(dim=1)[:, None, None] + 1e-12)   # per-instance normalised
            u = project_budget(u.detach() - lr * g * (k ** 0.5), env.task.cost_limit)
    return FireEnv.action_for(u.detach())


@torch.no_grad()
def score_policy(env: FireEnv, actions, idx: torch.Tensor, horizon: int, batch: int = 64) -> dict[str, float]:
    """Outcomes of `actions(state, t)` over held-out patches, rolled out `horizon` days.
    Outcome = final-day expected burned fraction; budget is per day."""
    burned, weighted, treated, over, none, learned = [], [], [], [], [], []
    for begin in range(0, idx.numel(), batch):
        state = env.reset(idx=idx[begin:begin + batch])
        w = 1 + env.task.population_weight * env.population(state)
        p, tr = rollout(env, state, actions, horizon)
        p0, _ = rollout(env, state, lambda st, t: -torch.ones(st.shape[0], env.action_dim, device=st.device), horizon)
        burned.append(p.flatten(1).mean(1)); none.append(p0.flatten(1).mean(1))
        weighted.append((p * w).flatten(1).mean(1))
        treated.append(tr.mean(1)); over.append((tr > env.task.cost_limit * 1.01).any(1).float())
        # Same plan with the imposed spread block off: the share of the effect
        # that goes through the learned model's fuel channel.
        block, env.task.block = env.task.block, 0.0
        pl, _ = rollout(env, state, actions, horizon)
        env.task.block = block
        learned.append(pl.flatten(1).mean(1))
    cat = lambda xs: torch.cat(xs)
    return {
        "burned %": 100 * float(cat(burned).mean()),
        "burned % (pop-weighted)": 100 * float(cat(weighted).mean()),
        "reduction vs none %": 100 * float(1 - cat(burned).mean() / cat(none).mean().clamp(min=1e-9)),
        "reduction via fuel channel only %": 100 * float(1 - cat(learned).mean() / cat(none).mean().clamp(min=1e-9)),
        "treated % per day": 100 * float(cat(treated).mean()),
        "over budget": float(cat(over).mean()),
    }


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
    parser.add_argument("--budget", type=float, default=0.03, help="treatable fraction of the patch")
    parser.add_argument("--block", type=float, default=0.9,
                        help="imposed spread suppression in treated cells. --block 0 removes it, "
                             "leaving only the learned fuel response: the ablation that says whether "
                             "the result rests on the action model's imposed physics")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1,
                        help="days planned; >1 feeds the predicted mask back as tomorrow's input")
    parser.add_argument("--plan-steps", type=int, default=50, help="per-instance optimiser steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unet-ckpt", default=None, help="skip surrogate training, load this state_dict")
    parser.add_argument("--out", default="runs/ndws_planning")
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
        unet = make_fire_model("unet", 1 + train["drivers"].shape[1], grid=args.grid, width=args.width).to(device)
        unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device)); unet.eval()
        val_aucpr = float("nan")
    else:
        unet, val_aucpr = train_unet(train, val, args, device)
    print(f"surrogate val AUC-PR {val_aucpr:.4f}", flush=True)

    task = FireTask(cost_limit=args.budget, block=args.block)
    # Plan on training patches, score on held-out patches: the planner must
    # generalise across fires, not memorise the ones it was tuned on.
    plan_env = FireEnv(unet, train, task, grid=args.grid, patches=args.patches,
                       horizon=args.horizon, device=device)
    test_env = FireEnv(unet, val, task, grid=args.grid, patches=args.patches,
                       horizon=args.horizon, device=device)
    torch.save(unet.state_dict(), root / "unet.pt")
    held = torch.arange(test_env.n)
    s, l = test_env.forecast_scores(held)
    forecast_aucpr = average_precision(s, l)

    # --- controls ------------------------------------------------------------ #
    k = args.patches * args.patches
    per_patch_budget = args.budget * k        # sum of intensities that meets the budget exactly

    def none(state, t=0):
        return -torch.ones(state.shape[0], k, device=state.device)

    def random(state, t=0):
        g = torch.Generator().manual_seed(args.seed)
        u = torch.rand(state.shape[0], k, generator=g).to(state.device)
        u = u / u.sum(1, keepdim=True) * per_patch_budget
        return FireEnv.action_for(u)

    def greedy(state, t=0):
        p = test_env.predict(state)                       # forecast once, then act
        risk = F.adaptive_avg_pool2d(p[:, None], args.patches).flatten(1)
        u = torch.zeros_like(risk)
        top = risk.topk(max(1, int(round(per_patch_budget))), dim=1).indices
        u.scatter_(1, top, 1.0)
        # If the budget is not a whole number of patches, scale the last one.
        u = u * min(1.0, per_patch_budget / max(1, int(round(per_patch_budget))))
        return FireEnv.action_for(u)

    plan_cache: dict = {}

    def per_instance(state, t=0):
        # The joint H-day plan is computed once per batch (at t = 0) and replayed.
        if t == 0:
            plan_cache["plan"] = per_instance_plan(test_env, state, args.horizon, steps=args.plan_steps)
        return plan_cache["plan"][:, t]

    rows = []
    for name, act in (("none", none), ("random", random), ("greedy", greedy),
                      ("pspe per-instance", per_instance)):
        rows.append({"policy": name, **score_policy(test_env, act, held, args.horizon), "real samples": args.n_train})
        print(f"[{name}] {rows[-1]}", flush=True)

    # --- PSPE v2 planner, with and without the budget dual ------------------ #
    # Scale-free dual error: the budget is a fraction (0.03) while the reward is
    # in percent, so at raw gains lambda needed ~30x longer than the run to bind
    # (seed 0 first attempt: 16.6% treated at lambda 9.9 and climbing).
    v2 = dict(saturation_coef=1.0, advantage_std_floor=1e-2, dual_ema=0.7, ki=0.5,
              dual_normalize=True)
    for name, extra in (("pspe v2", {}), ("unconstrained", {"kp": 0.0, "ki": 0.0, "kd": 0.0})):
        seed_everything(args.seed)
        policy = GaussianFieldPolicy(plan_env.obs_shape[0], plan_env.action_dim)
        # Start near "no treatment": tanh(-1.0) maps to 1.4% intensity under the
        # squared map, inside the saturation wall. The default init treats a
        # quarter of the patch.
        with torch.no_grad():
            policy.mean_head.bias.fill_(-1.0)
        trainer = HybridPlannerTrainer(
            plan_env, policy,
            cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon, batch=16, seed=args.seed,
                              **{**v2, **extra}),
            eval_env=None, logger=RunLogger(root / name.replace(" ", "_"), use_tensorboard=False),
            device=device, surrogate_train_transitions=args.n_train,
        )
        summary = trainer.train()
        policy.eval()
        act = lambda state, t=0, pol=policy: pol.sample(state, deterministic=True)[0]
        rows.append({"policy": name, **score_policy(test_env, act, held, args.horizon),
                     "real samples": args.n_train, "final lambda": round(summary["final_lambda"], 3)})
        print(f"[{name}] {rows[-1]}", flush=True)

    for r in rows:
        r["forecast AUC-PR"] = round(forecast_aucpr, 4)
        r["budget"] = args.budget
        r["block"] = args.block
        for key, val_ in list(r.items()):
            if isinstance(val_, float):
                r[key] = round(val_, 4)
    table = markdown_table(rows)
    (root / "results.md").write_text(
        table + f"\n\nsurrogate val AUC-PR {val_aucpr:.4f}; budget {args.budget}; block {args.block}\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
