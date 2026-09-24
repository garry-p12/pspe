#!/usr/bin/env python
"""Planning under partial observation: the twin's sync step, on real fire data.

    python eval/run_ndws_partial.py --seed 0 --unet-ckpt runs/ndws_planning/seed_0/unet.pt

A digital twin never sees the state. It sees a cloud-obscured, sensor-gapped
observation and has to infer what is burning before it can decide anything.
Every planning result in this repository so far has been handed the full fire
mask, which is the one thing a twin does not get.

NDWS carries this structure natively: cells the sensor could not label are
marked -1. Measured over 400 training patches, the median patch has none, the
90th percentile has 1.5%, and 22 of 400 exceed 10% — occasionally the whole
patch. So occlusion is real, spatially clustered, and heavy-tailed. This script
extends it to a controlled sweep and asks two questions:

    1. Can a learned estimator recover the fire state from what is visible?
    2. Does the *decision* survive as observation degrades?

Question 2 is the one that matters. A state estimator that halves
reconstruction error but leaves the plan unchanged has bought nothing; one that
keeps the plan good under heavy cloud is the twin's sync step working.

Three ways to handle the gap, all planning through the same frozen surrogate:

    blind       occluded cells treated as "no fire" — what the pipeline does now
    persist     occluded cells filled from the previous day's mask
    perceive    a learned estimator: (visible mask, validity, drivers) -> mask

The estimator is trained self-supervised on real patches: hide a block of cells
whose labels we have, predict them, score against the truth we hid.
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
from eval.run_ndws import average_precision  # noqa: E402
from eval.run_ndws_planning import per_instance_plan, rollout, score_policy, train_unet  # noqa: E402
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.simulate.real import NDWSConfig, driver_stats, load_ndws, normalize_drivers  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.utils import get_device, project_path, seed_everything  # noqa: E402


def cloud_mask(n: int, grid: int, rate: float, gen: torch.Generator,
               scale: int = 8) -> torch.Tensor:
    """Spatially correlated occlusion -> (n, grid, grid) bool, True = hidden.

    Clouds come in patches, not independent pixels. Low-resolution noise is
    upsampled and thresholded so the hidden region is contiguous, which is
    both realistic and much harder than iid dropout: a blob can hide the whole
    fire front, where scattered pixels never do.
    """
    if rate <= 0:
        return torch.zeros(n, grid, grid, dtype=torch.bool)
    low = torch.rand(n, 1, scale, scale, generator=gen)
    field = F.interpolate(low, size=(grid, grid), mode="bicubic", align_corners=False)[:, 0]
    flat = field.reshape(n, -1)
    k = max(1, int(round(rate * grid * grid)))
    thresh = flat.kthvalue(k, dim=1).values[:, None, None]
    return field <= thresh


class Perceive(torch.nn.Module):
    """Small U-Net: what is visible plus the drivers -> the full fire state."""

    def __init__(self, in_channels: int, width: int = 32) -> None:
        super().__init__()
        c = width
        self.e1 = self.block(in_channels, c)
        self.e2 = self.block(c, c * 2)
        self.e3 = self.block(c * 2, c * 4)
        self.d2 = self.block(c * 4 + c * 2, c * 2)
        self.d1 = self.block(c * 2 + c, c)
        self.head = torch.nn.Conv2d(c, 1, 1)

    @staticmethod
    def block(i: int, o: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.GroupNorm(8, o), torch.nn.GELU(),
            torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.GroupNorm(8, o), torch.nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.head(d1)


def train_perceive(data, args, device) -> tuple[Perceive, float]:
    """Self-supervised: hide labelled cells, predict them, score on what we hid."""
    seed_everything(args.seed + 500)
    prev = torch.as_tensor(np.where(data["states"][:, 0] >= 0, data["states"][:, 0], 0.0),
                           dtype=torch.float32)
    drivers = torch.as_tensor(data["drivers"], dtype=torch.float32)
    labelled = torch.as_tensor(data["valid"][:, 0])        # (N,1,H,W), matches hide
    n, _, g, _ = prev.shape
    model = Perceive(2 + drivers.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(args.seed + 500)
    prevalence = float((prev > 0.5).float().mean().clamp(min=1e-6))
    pos_w = torch.tensor(max((1 / prevalence) ** 0.5, 1.0), device=device)

    best = {"ap": -1.0, "state": None}
    for epoch in range(args.perceive_epochs):
        perm = torch.randperm(n, generator=gen)
        model.train()
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            truth = prev[idx].to(device)                        # (B,1,H,W)
            rate = float(torch.empty(1).uniform_(0.05, 0.6, generator=gen))
            hide = cloud_mask(len(idx), g, rate, gen)[:, None].to(device)   # (B,1,H,W)
            visible = truth * (~hide)
            x = torch.cat([visible, (~hide).float(), drivers[idx].to(device)], 1)
            logits = model(x)
            # Score only where we hid a cell whose label we actually have.
            score_here = hide & labelled[idx].to(device)
            loss_map = F.binary_cross_entropy_with_logits(
                logits, (truth > 0.5).float(), pos_weight=pos_w, reduction="none")
            loss = (loss_map * score_here).sum() / score_here.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            idx = torch.arange(min(256, n))
            truth = prev[idx].to(device)
            hide = cloud_mask(len(idx), g, 0.3, torch.Generator().manual_seed(11))[:, None].to(device)
            x = torch.cat([truth * (~hide), (~hide).float(), drivers[idx].to(device)], 1)
            p = torch.sigmoid(model(x))
            sel = hide & labelled[idx].to(device)
            ap = average_precision(p[sel].cpu().numpy(), (truth[sel] > 0.5).float().cpu().numpy())
        if ap > best["ap"]:
            best = {"ap": ap, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        print(f"[perceive] epoch {epoch + 1}/{args.perceive_epochs} hidden-cell AP {ap:.4f}", flush=True)
    model.load_state_dict(best["state"])
    return model.eval(), best["ap"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--perceive-epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--patches", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.03)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--plan-steps", type=int, default=50)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.15, 0.35, 0.6])
    parser.add_argument("--unet-ckpt", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/ndws_partial")
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

    perceive, recon_ap = train_perceive(train, args, device)
    torch.save(perceive.state_dict(), root / "perceive.pt")
    print(f"perceive hidden-cell AP {recon_ap:.4f}", flush=True)

    task = FireTask(cost_limit=args.budget)
    env_kw = dict(grid=args.grid, patches=args.patches, horizon=args.horizon, device=device)
    truth_env = FireEnv(unet, val, task, **env_kw)      # full state: the upper bound
    held = torch.arange(truth_env.n)
    k = args.patches * args.patches
    per_patch = args.budget * k
    drivers_v = torch.as_tensor(val["drivers"], dtype=torch.float32)

    def greedy(env):
        def act(state, t=0):
            p = env.predict(state)
            risk = F.adaptive_avg_pool2d(p[:, None], args.patches).flatten(1)
            u = torch.zeros_like(risk)
            u.scatter_(1, risk.topk(max(1, int(round(per_patch))), dim=1).indices, 1.0)
            return FireEnv.action_for(u * min(1.0, per_patch / max(1, int(round(per_patch)))))
        return act

    def planner(env):
        cache = {}
        def act(state, t=0):
            if t == 0:
                cache["p"] = per_instance_plan(env, state, args.horizon, steps=args.plan_steps)
            return cache["p"][:, t]
        return act

    rows, overlaps = [], []
    for rate in args.rates:
        plan_sets: dict = {}
        gen = torch.Generator().manual_seed(args.seed + 900)
        hide = cloud_mask(truth_env.n, args.grid, rate, gen)          # (N,H,W)

        # Three ways to fill the gap, each producing its own believed state.
        beliefs = {}
        full = truth_env.x.clone()                                    # (N, 12, H, W)
        visible = full[:, 0] * (~hide)
        beliefs["blind"] = visible.clone()
        # persistence: the previous day's mask is the same channel here, so the
        # honest version of "carry the last observation forward" is to keep the
        # visible part and assume unseen cells look like the patch's visible mean.
        mean_vis = (visible.flatten(1).sum(1) / (~hide).flatten(1).sum(1).clamp(min=1))
        beliefs["persist"] = torch.where(hide, mean_vis[:, None, None].expand_as(visible), visible)
        with torch.no_grad():
            est = []
            for b in range(0, truth_env.n, 128):
                sl = slice(b, b + 128)
                x = torch.cat([visible[sl][:, None].to(device), (~hide[sl])[:, None].float().to(device),
                               drivers_v[sl].to(device)], 1)
                est.append(torch.sigmoid(perceive(x))[:, 0].cpu())
            est = torch.cat(est)
        beliefs["perceive"] = torch.where(hide, est, visible)
        # Perceive emits a probability field where the truth is near-binary, so a
        # worse decision could be the surrogate reacting to the REPRESENTATION
        # rather than to the information. Thresholding separates the two: if the
        # hard version recovers the loss, soft imputation was the problem; if it
        # does not, imputing uncertain state genuinely hurts the decision.
        beliefs["perceive-hard"] = torch.where(hide, (est > 0.5).float(), visible)

        # How well does each belief match the state that was actually there?
        truth_mask = full[:, 0]
        scored = hide & torch.as_tensor(val["valid"][:, 0, 0])
        for name, belief in beliefs.items():
            if rate > 0 and scored.any():
                ap = average_precision(belief[scored].numpy(),
                                       (truth_mask[scored] > 0.5).numpy().astype(np.float32))
            else:
                ap = float("nan")
            # Plan on the believed state, but SCORE on the real one: acting on a
            # wrong belief must be paid for in the real world, not the imagined.
            env = FireEnv(unet, val, task, **env_kw)
            env.x = full.clone(); env.x[:, 0] = belief
            for policy_name, mk in (("greedy", greedy), ("PSPE planner", planner)):
                plans = {}
                policy = mk(env)          # one closure, so its plan cache persists
                def act(state, t=0, policy=policy, env=env, plans=plans):
                    if t == 0:            # decide from the BELIEF, not the truth
                        s_belief = env.reset(idx=truth_env._idx)
                        plans["a"] = torch.stack(
                            [policy(s_belief, tt) for tt in range(args.horizon)], 1)
                    return plans["a"][:, t]
                res = score_policy(truth_env, act, held, args.horizon)
                # Does the belief actually reach the plan? If two beliefs produce
                # the same treated cells, the experiment is measuring nothing, and
                # identical downstream numbers would be plumbing, not a finding.
                with torch.no_grad():
                    s_b = env.reset(idx=held[:128])
                    day0 = FireEnv.intensity(policy(s_b, 0))
                plan_sets.setdefault(policy_name, {})[name] = (day0 > 0.02)
                rows.append({"occlusion": rate, "belief": name, "policy": policy_name,
                             "state AP on hidden cells": round(ap, 4) if ap == ap else None,
                             "burn reduction %": round(res["reduction vs none %"], 2),
                             "treated % per day": round(res["treated % per day"], 3)})
                print(f"[rate {rate} | {name} | {policy_name}] "
                      f"AP {ap:.4f} reduction {res['reduction vs none %']:.2f}%", flush=True)
            if rate == 0:
                break            # with nothing hidden every belief is the same state

        # Jaccard between the plans each belief produced, per policy.
        for pol, sets in plan_sets.items():
            keys = list(sets)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a_, b_ = sets[keys[i]], sets[keys[j]]
                    inter = float((a_ & b_).sum()); union = float((a_ | b_).sum())
                    overlaps.append({"occlusion": rate, "policy": pol,
                                     "pair": f"{keys[i]} vs {keys[j]}",
                                     "plan overlap (Jaccard)": round(inter / max(union, 1), 3)})
                    print(f"  [plan overlap {rate} {pol}] {keys[i]} vs {keys[j]}: "
                          f"{inter / max(union, 1):.3f}", flush=True)

    table = markdown_table(rows)
    (root / "results.md").write_text(table + f"\n\nperceive hidden-cell AP {recon_ap:.4f}\n")
    if overlaps:
        table += "\n\n" + markdown_table(overlaps)
    (root / "results.json").write_text(json.dumps(
        {"rows": rows, "overlaps": overlaps, "perceive_ap": recon_ap}, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
