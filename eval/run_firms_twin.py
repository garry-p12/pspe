#!/usr/bin/env python
"""The twin loop, closed on real sequential satellite observations.

    python eval/run_firms_twin.py --device cuda --out runs/firms_twin

Everything else in this repository forecasts one step from a given state. A
digital twin does something different and harder: it carries a state forward
over many days, and every time the satellite looks it has to reconcile what it
believed with what was seen. This is that loop, on the real thing — eight large
US wildfires, 14 to 20 consecutive days each, from NASA FIRMS active-fire
detections, including the days the satellite did not see anything.

Three ways to carry the state, one shared spread model, leave-one-fire-out:

    open loop       forecast from day 0 and never look again. What a pure
                    forecaster does.
    state sync      each day, replace the believed state with what was observed.
                    On days with no overpass, keep the model's own prediction —
                    which is where a twin either holds together or drifts.
    state + model   also take a gradient step on the spread model from the
                    day's observation, so the model itself tracks this fire.

Reported per forecast horizon: how well each mode's predicted fire matches what
the satellite actually saw. The comparison is only meaningful on days that were
observed, so unobserved days advance the loop but are never scored.

What this establishes and what it does not: the loop closes on *observation and
state*, on real data. It says nothing about intervention effects — the record
contains no fire where a firebreak was cut on our instruction, and no dataset
switch fixes that.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws import average_precision  # noqa: E402
from pspe.simulate.real.firms import load_sequences  # noqa: E402
from pspe.utils import get_device, project_path, seed_everything  # noqa: E402


class Spread(torch.nn.Module):
    """Small U-Net: (today's fire, everything burned so far) -> tomorrow's fire."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        c = width
        self.e1, self.e2 = self.block(2, c), self.block(c, c * 2)
        self.d1 = self.block(c * 2 + c, c)
        self.head = torch.nn.Conv2d(c, 1, 1)

    @staticmethod
    def block(i: int, o: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.GroupNorm(4, o), torch.nn.GELU(),
            torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.GroupNorm(4, o), torch.nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        up = F.interpolate(e2, scale_factor=2, mode="nearest")
        return self.head(self.d1(torch.cat([up, e1], 1)))


def pairs(seqs, device):
    """Consecutive observed-day transitions, as (input, target) tensors."""
    xs, ys = [], []
    for s in seqs:
        fire = torch.as_tensor(s.fire)
        burned = torch.cummax(fire, dim=0).values          # everything alight so far
        for t in range(len(s.dates) - 1):
            if not (s.observed[t] and s.observed[t + 1]):
                continue
            xs.append(torch.stack([fire[t], burned[t]]))
            ys.append(fire[t + 1])
    if not xs:
        return None, None
    return torch.stack(xs).to(device), torch.stack(ys)[:, None].to(device)


def fit(train_seqs, device, epochs: int, seed: int, width: int) -> Spread:
    seed_everything(seed)
    model = Spread(width).to(device)
    x, y = pairs(train_seqs, device)
    if x is None:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    prevalence = float((y > 0.1).float().mean().clamp(min=1e-6))
    pw = torch.tensor(max((1 / prevalence) ** 0.5, 1.0), device=device)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for b in range(0, x.shape[0], 8):
            idx = perm[b:b + 8]
            loss = F.binary_cross_entropy_with_logits(model(x[idx]), (y[idx] > 0.1).float(),
                                                      pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def step(model, fire, burned):
    x = torch.stack([fire, burned])[None]
    return torch.sigmoid(model(x))[0, 0]


def run_fire(model, seq, device, mode: str, adapt_lr: float, horizon: int):
    """Carry the state across the sequence; score each forecast against what was seen."""
    fire = torch.as_tensor(seq.fire).to(device)
    T = len(seq.dates)
    local = copy.deepcopy(model) if mode == "state+model" else model
    opt = (torch.optim.SGD(local.parameters(), lr=adapt_lr) if mode == "state+model" else None)

    belief = fire[0].clone()
    burned = fire[0].clone()
    scores = {h: [] for h in range(1, horizon + 1)}

    for t in range(T - 1):
        # Forecast forward `horizon` days from the current belief, without
        # peeking; score each lead time against the day it lands on.
        f, b = belief.clone(), burned.clone()
        for h in range(1, horizon + 1):
            if t + h >= T:
                break
            f = step(local, f, b)
            b = torch.maximum(b, f)
            if seq.observed[t + h]:
                truth = (fire[t + h] > 0).float().flatten().cpu().numpy()
                if truth.sum() > 0:
                    scores[h].append(average_precision(f.flatten().cpu().numpy(), truth))

        # Advance one day. This is where the modes differ.
        nxt = step(local, belief, burned)
        if mode == "open":
            belief = nxt
        else:
            if seq.observed[t + 1]:
                if mode == "state+model" and opt is not None:
                    # One gradient step from today's observation: the model
                    # itself tracks this fire, not just the state.
                    local.train()
                    x = torch.stack([belief, burned])[None]
                    loss = F.binary_cross_entropy_with_logits(
                        local(x), (fire[t + 1] > 0).float()[None, None])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    local.eval()
                belief = fire[t + 1].clone()          # reconcile with what was seen
            else:
                belief = nxt                          # no overpass: trust the model
        burned = torch.maximum(burned, belief)

    return {h: float(np.mean(v)) if v else float("nan") for h, v in scores.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--adapt-lr", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/firms_twin")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences(grid=args.grid)
    print(f"{len(seqs)} fires: " + ", ".join(
        f"{s.name}({int(s.observed.sum())}/{len(s.dates)}d)" for s in seqs), flush=True)

    MODES = ["open", "state sync", "state+model"]
    acc = {m: {h: [] for h in range(1, args.horizon + 1)} for m in MODES}
    per_fire = []

    for seed in args.seeds:
        for held in seqs:                              # leave one fire out
            train = [s for s in seqs if s.name != held.name]
            model = fit(train, device, args.epochs, seed, args.width)
            row = {"seed": seed, "fire": held.name,
                   "observed days": int(held.observed.sum())}
            for m in MODES:
                sc = run_fire(model, held, device, m, args.adapt_lr, args.horizon)
                for h, v in sc.items():
                    if v == v:
                        acc[m][h].append(v)
                    row[f"{m} h{h}"] = round(v, 4) if v == v else None
            per_fire.append(row)
            print(f"[seed {seed}] {held.name}: " + "  ".join(
                f"{m} h1 {row.get(m + ' h1')}" for m in MODES), flush=True)

    # Paired across fires: the fires differ enormously in size and behaviour, so
    # the spread across them says nothing about whether the loop helps. What
    # matters is whether it helps on the SAME fire.
    paired = []
    for h in range(1, args.horizon + 1):
        by_fire = {}
        for r in per_fire:
            for m in MODES:
                v = r.get(f"{m} h{h}")
                if v is not None:
                    by_fire.setdefault(m, {}).setdefault(r["fire"], []).append(v)
        def mean_per_fire(m):
            d = by_fire.get(m, {})
            return {k: float(np.mean(v)) for k, v in d.items()}
        base = mean_per_fire("open")
        for m in ("state sync", "state+model"):
            cur = mean_per_fire(m)
            fires = sorted(set(base) & set(cur))
            if len(fires) < 2:
                continue
            d = np.array([cur[f] - base[f] for f in fires])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            paired.append({"horizon": f"day +{h}", "comparison": f"{m} vs open loop",
                           "mean gain": round(float(d.mean()), 4),
                           "fires improved": f"{int((d > 0).sum())}/{len(d)}",
                           "paired t": round(float(t), 2)})
        cur, alt = mean_per_fire("state+model"), mean_per_fire("state sync")
        fires = sorted(set(cur) & set(alt))
        if len(fires) >= 2:
            d = np.array([cur[f] - alt[f] for f in fires])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            paired.append({"horizon": f"day +{h}", "comparison": "model adaptation vs state sync only",
                           "mean gain": round(float(d.mean()), 4),
                           "fires improved": f"{int((d > 0).sum())}/{len(d)}",
                           "paired t": round(float(t), 2)})

    rows = []
    for m in MODES:
        r = {"mode": m}
        for h in range(1, args.horizon + 1):
            v = acc[m][h]
            r[f"day +{h}"] = f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "—"
        rows.append(r)

    table = (markdown_table(rows) + "\n\nPaired across fires:\n\n" + markdown_table(paired)
             + "\n\nPer fire and seed:\n\n" + markdown_table(per_fire))
    (root / "results.md").write_text(
        table + f"\n\n{len(seqs)} fires, leave-one-fire-out, seeds {args.seeds}. "
        "Scored as average precision of the predicted fire against the observed "
        "detections, on observed days only.\n")
    (root / "results.json").write_text(json.dumps(
        {"summary": rows, "paired": paired, "per_fire": per_fire}, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
