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
from pspe.simulate.real import wsts  # noqa: E402
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


@torch.no_grad()
def forecast_error(model, seqs, device) -> float:
    """RMS one-step error of the spread model, measured on TRAINING fires.

    The ensemble has to be dispersed like the forecast actually is. Set by
    hand at 0.05 the filter had sigma_f^2 = 0.0025 against sigma_o^2 = 0.0225,
    so K came out near 0.1 -- and at sigma_o = 0.5, near 0.01. The analysis
    then ignores the observation and the run degenerates to the open loop,
    which is exactly what the first smoke test showed: 0.003 where state sync
    scored 0.12 to 0.38. That is filter divergence from under-dispersion, and
    reporting it as "the EnKF loses" would be scoring a strawman.

    Measured here rather than tuned, and measured on the fires the model was
    fitted to, so nothing about the held-out fire sets its own error bar.
    """
    # Over the ACTIVE region, not the whole field.
    #
    # Averaged over every cell this returned 0.033 -- smaller than the 0.05
    # picked by hand -- because 97% of the grid is unburnt and the model gets
    # those right for free. That is the model's skill at predicting nothing,
    # and dispersing an ensemble by it leaves the filter as deaf as before.
    # The uncertainty that matters is where the front is, so the error is
    # measured where either the forecast or the truth says something is
    # burning.
    errs = []
    for s in seqs:
        fire = torch.as_tensor(s.fire).to(device)
        for t in range(len(s.dates) - 1):
            if not (s.observed[t] and s.observed[t + 1]):
                continue
            pred = step(model, fire[t], fire[t])
            truth = fire[t + 1]
            active = (truth > 0.05) | (pred > 0.05)
            if active.sum() < 4:
                continue
            errs.append(float(((pred - truth) ** 2)[active].mean()))
    return float(np.sqrt(np.mean(errs))) if errs else 0.3


@torch.no_grad()
def step_ens(model, fire: torch.Tensor, burned: torch.Tensor) -> torch.Tensor:
    """The spread model over a whole ensemble at once. (N,H,W) -> (N,H,W)."""
    x = torch.stack([fire, burned], dim=1)
    return torch.sigmoid(model(x))[:, 0]


def run_fire_enkf(model, seq, device, horizon: int, n_ens: int,
                  obs_sd: float, model_sd: float, seed: int,
                  inflation: float = 1.0):
    """The twin loop with an ensemble analysis instead of a gain-1 nudge.

    Section 5.2a reports state sync against an open loop and calls it a
    control, because replacing the belief with the observation IS assimilation
    with the gain set to one -- no observation error, no forecast spread,
    nothing estimated. Section 9.8 concedes the point. This is the smallest
    method that does not concede it.

    Forecast spread comes from perturbing each member every step, which is the
    model error the deterministic run pretends away. On a day the satellite
    looked, each member is pulled toward a perturbed observation by a per-cell
    gain from the ensemble's own variance against the stated observation
    error:

        K = sigma_f^2 / (sigma_f^2 + sigma_o^2)

    That is the Kalman analysis with a diagonal covariance -- the ensemble
    supplies sigma_f, the instrument supplies sigma_o. Gain-1 nudging is the
    sigma_o -> 0 corner of it, so the comparison is not against a different
    family of method but against the same one with its error model restored.

    `obs_sd` is swept rather than fitted. VIIRS active-fire detection is not
    accompanied by a per-cell error variance, and choosing one to make the
    result come out is the failure this project keeps cataloguing.
    """
    fire = torch.as_tensor(seq.fire).to(device)
    T = len(seq.dates)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def noise(shape, sd):
        return (sd * torch.randn(shape, generator=g)).to(device)

    def perturb(x: torch.Tensor, sd: float) -> torch.Tensor:
        """Ensemble spread, applied ONLY near the front.

        White noise over the whole grid was the third and worst version of this
        mistake. The field is 97% unburnt, so N(0, 0.19) on every cell invents
        fire across the entire domain, and average precision -- which is what
        this is scored by -- is unforgiving about false positives in empty
        space. The EnKF read 0.008 against state sync's 0.203 while its gain
        was a perfectly healthy 0.78: the analysis was fine and the ensemble
        was nonsense.

        What is uncertain is where the front goes next, not whether fire
        appears thirty kilometres away. The perturbation is therefore confined
        to a dilation of the currently active cells, which is the band the
        front can actually reach in a day.
        """
        near = F.max_pool2d(x[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
        band = (near > 0.02).float()
        # In LOGIT space, because the state is a bounded fraction. Adding noise
        # and clamping at zero truncates every negative draw and keeps every
        # positive one, which biases the band upward -- with eight members that
        # does not average out, and average precision charges for each invented
        # cell. A logit perturbation is symmetric and cannot leave (0, 1).
        e = 1e-4
        z = torch.log(x.clamp(e, 1 - e) / (1 - x.clamp(e, 1 - e)))
        return torch.sigmoid(z + band * noise(x.shape, sd * 6.0))

    ens = perturb(fire[0][None].repeat(n_ens, 1, 1), model_sd)
    burned = ens.clone()
    scores = {h: [] for h in range(1, horizon + 1)}

    for t in range(T - 1):
        # Forecast the ensemble forward; the prediction is its mean.
        f, b = ens.clone(), burned.clone()
        for h in range(1, horizon + 1):
            if t + h >= T:
                break
            f = perturb(step_ens(model, f, b), model_sd)
            b = torch.maximum(b, f)
            if seq.observed[t + h]:
                truth = (fire[t + h] > 0).float().flatten().cpu().numpy()
                if truth.sum() > 0:
                    scores[h].append(average_precision(
                        f.mean(0).flatten().cpu().numpy(), truth))

        nxt = perturb(step_ens(model, ens, burned), model_sd)
        if inflation != 1.0:
            # Multiplicative inflation about the ensemble mean, the standard
            # remedy for a filter that is too sure of itself.
            m = nxt.mean(0, keepdim=True)
            nxt = (m + inflation * (nxt - m)).clamp(0.0, 1.0)
        if seq.observed[t + 1]:
            y = fire[t + 1]
            var_f = nxt.var(dim=0, unbiased=False)
            K = var_f / (var_f + obs_sd ** 2 + 1e-9)          # per cell
            # Perturbed observations, so the analysis ensemble keeps the spread
            # the update is entitled to rather than collapsing onto one state.
            y_pert = perturb(y[None].repeat(n_ens, 1, 1), obs_sd)
            ens = (nxt + K[None] * (y_pert - nxt)).clamp(0.0, 1.0)
        else:
            ens = nxt
        burned = torch.maximum(burned, ens)

    return {h: float(np.mean(v)) if v else float("nan") for h, v in scores.items()}


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


def folds(seqs, scheme: str):
    """(name, held-out sequences) pairs for the chosen cross-validation.

    Leave-one-fire-out is right for eight fires and impossible for 607: it
    fits one model per fire per seed. WildfireSpreadTS's own documentation
    recommends cross-validation across years, because the distribution shifts
    between them -- so that is both the affordable protocol and the one the
    benchmark asks for, which is the point of moving to it.
    """
    if scheme == "fire":
        return [(s.name, [s]) for s in seqs]
    # The MODAL year, not the first date's. WildfireSpreadTS groups fires into
    # year directories, but a fire filed under 2018 can have its first frame in
    # late December 2017 -- keying on dates[0] then invents a one-fire "2017"
    # fold, which is not a train/test split of anything.
    by_year: dict[str, list] = {}
    for s in seqs:
        years = [d[:4] for d in s.dates]
        by_year.setdefault(max(set(years), key=years.count), []).append(s)
    return [(y, by_year[y]) for y in sorted(by_year)]


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
    ap.add_argument("--source", default="firms", choices=["firms", "wsts"],
                    help="firms: 8 fires fetched live. wsts: WildfireSpreadTS, 607 fires")
    ap.add_argument("--wsts-root", default="data/wsts/WildfireSpreadTS")
    ap.add_argument("--cv", default=None, choices=["fire", "year"],
                    help="held-out unit; default is fire for firms, year for wsts")
    ap.add_argument("--max-fires", type=int, default=None,
                    help="cap the number of sequences, for smoke tests")
    ap.add_argument("--min-observed", type=int, default=4,
                    help="drop sequences the satellite barely saw")
    ap.add_argument("--n-ens", type=int, default=32,
                    help="ensemble members for the EnKF arm")
    ap.add_argument("--obs-sd", type=float, nargs="+", default=[0.15, 0.3, 0.5],
                    help="observation error SD to sweep; gain-1 nudging is the "
                         "0 corner of this axis")
    ap.add_argument("--model-sd", type=float, default=None,
                    help="per-step perturbation; default is the model's own "
                         "measured one-step RMS error on the training fires")
    ap.add_argument("--inflation", type=float, default=1.0,
                    help="multiplicative covariance inflation")
    args = ap.parse_args()
    if args.cv is None:
        args.cv = "fire" if args.source == "firms" else "year"

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    if args.source == "wsts":
        seqs = wsts.load_sequences(args.wsts_root, grid=args.grid,
                                   max_fires=args.max_fires,
                                   min_observed=args.min_observed)
    else:
        seqs = load_sequences(grid=args.grid)
    if len(seqs) < 2:
        raise SystemExit(f"only {len(seqs)} usable sequences from {args.source}")
    obs = [int(s.observed.sum()) for s in seqs]
    print(f"{len(seqs)} fires from {args.source}; observed days "
          f"min {min(obs)} median {int(np.median(obs))} max {max(obs)}", flush=True)
    if len(seqs) <= 12:
        print("  " + ", ".join(f"{s.name}({int(s.observed.sum())}/{len(s.dates)}d)"
                               for s in seqs), flush=True)

    MODES = ["open", "state sync", "state+model"] + [
        f"enkf s{sd:g}" for sd in args.obs_sd]
    acc = {m: {h: [] for h in range(1, args.horizon + 1)} for m in MODES}
    per_fire = []

    split = folds(seqs, args.cv)
    print(f"{len(split)} {args.cv}-fold(s): " + ", ".join(
        f"{n}({len(h)})" for n, h in split), flush=True)
    for seed in args.seeds:
        for fold_name, held_group in split:
            names = {s.name for s in held_group}
            train = [s for s in seqs if s.name not in names]
            if not train:
                continue
            model = fit(train, device, args.epochs, seed, args.width)
            # Disperse the ensemble like the forecast actually is, measured on
            # the fires this model was fitted to.
            sd = (args.model_sd if args.model_sd is not None
                  else forecast_error(model, train[:24], device))
            for held in held_group:
                row = {"seed": seed, "fold": fold_name, "fire": held.name,
                       "observed days": int(held.observed.sum())}
                for m in MODES:
                    if m.startswith("enkf"):
                        sc = run_fire_enkf(model, held, device, args.horizon,
                                           args.n_ens, float(m.split("s")[-1]),
                                           sd, seed, args.inflation)
                    else:
                        sc = run_fire(model, held, device, m, args.adapt_lr,
                                      args.horizon)
                    for h, v in sc.items():
                        if v == v:
                            acc[m][h].append(v)
                        row[f"{m} h{h}"] = round(v, 4) if v == v else None
                per_fire.append(row)
            print(f"[seed {seed}] fold {fold_name}: {len(held_group)} fires scored, "
                  f"{len(train)} in train, ensemble sd {sd:.3f}", flush=True)

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

        # The comparison this experiment exists for: a real analysis step
        # against the gain-1 nudge that section 5.2a can only call a control.
        for m in [x for x in MODES if x.startswith("enkf")]:
            cur = mean_per_fire(m)
            fires = sorted(set(cur) & set(alt))
            if len(fires) < 2:
                continue
            d = np.array([cur[f] - alt[f] for f in fires])
            tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            paired.append({"horizon": f"day +{h}", "comparison": f"{m} vs state sync",
                           "mean gain": round(float(d.mean()), 4),
                           "fires improved": f"{int((d > 0).sum())}/{len(d)}",
                           "paired t": round(float(tt), 2)})

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
        table + f"\n\n{len(seqs)} fires from {args.source}, held out by {args.cv}, "
        f"seeds {args.seeds}. "
        "Scored as average precision of the predicted fire against the observed "
        "detections, on observed days only.\n")
    (root / "results.json").write_text(json.dumps(
        {"summary": rows, "paired": paired, "per_fire": per_fire}, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
