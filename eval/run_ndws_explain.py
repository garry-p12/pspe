#!/usr/bin/env python
"""Certified briefs for firebreak plans on observed wildfire data.

    python eval/run_ndws_explain.py --seed 0 --unet-ckpt runs/ndws_planning/seed_0/unet.pt \
        --backbone Qwen/Qwen2.5-0.5B-Instruct --out runs/ndws_explain/seed_0

Closes the loop on one real system: the plans are the per-instance firebreak
plans of eval/run_ndws_planning.py (real fire patch in, 8 x 8 break
intensities out, 3% daily budget, three days ahead), and the brief has to
say what the plan does. Same three arms as the synthetic-testbed comparison
(trained-in, no faithfulness term, post-hoc), same certificate.

The plans are computed once for a pool of fires and served to the Explain
trainer through a bank that presents the planner interface it expects: the
brief for a fire names every treated patch (intensity on a 0.05 grid), the
day-1 treated fraction against the budget, and the expected population-
weighted burn. The parser reads the intensities back; F(b) = exp(-KL) between
the plan and what the brief implies.

Calibration and test briefs come from disjoint held-out fires the explainer
never trained on, drawn without replacement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws_planning import per_instance_plan, rollout, train_unet  # noqa: E402
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer  # noqa: E402
from pspe.explain.conformal import certify, minimum_calibration_size  # noqa: E402
from pspe.simulate.real import NDWSConfig, driver_stats, load_ndws, normalize_drivers  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

ARMS = {
    "trained-in":  dict(use_faithfulness=True, posthoc=False),
    "no-faithful": dict(use_faithfulness=False, posthoc=False),
    "post-hoc":    dict(use_faithfulness=False, posthoc=True),
}


def plan_diversity(bank: dict) -> dict[str, float]:
    """How much do the plans differ across fires? If they do not, no brief can
    be state-specific and the permutation gap is zero whatever the model does
    (which is what the dar testbed turned out to be: action std across states
    1.2% of the action magnitude, 8 distinct plans in 64 states)."""
    u = bank["intensity"]
    grid = (u / 0.05).round()
    return {"mean intensity": float(u.mean()),
            "across-fire std / mean": float(u.std(0).mean() / u.abs().mean().clamp(min=1e-9)),
            "distinct plans on the brief grid": len({tuple(r.int().tolist()) for r in grid}),
            "fires": int(u.shape[0])}


@torch.no_grad()
def build_bank(env: FireEnv, idx: torch.Tensor, horizon: int, steps: int, batch: int = 64) -> dict:
    """Per-instance plans for the fires at `idx`: day-1 intensities, cost, reward."""
    states, intens, costs, rewards = [], [], [], []
    for begin in range(0, idx.numel(), batch):
        state = env.reset(idx=idx[begin:begin + batch])
        w = 1 + env.task.population_weight * env.population(state)
        plan = per_instance_plan(env, state, horizon, steps=steps)          # (B, T, K) actions
        p, treated = rollout(env, state, lambda st, t: plan[:, t], horizon)
        states.append(state.cpu())
        intens.append(FireEnv.intensity(plan[:, 0]).cpu())                  # what the brief describes
        costs.append(100 * treated[:, 0].cpu())                             # day-1 treated, percent
        rewards.append(-100 * (p * w).flatten(1).mean(1).cpu())             # pop-weighted burn, percent
    return {"states": torch.cat(states), "intensity": torch.cat(intens),
            "cost": torch.cat(costs), "reward": torch.cat(rewards)}


class PlanBank:
    """Serves precomputed plans through the env interface ExplainTrainer reads.

    `reset` draws fires (with replacement for training, sequentially without
    replacement for calibration/test); `step` returns the stored cost and
    reward of the plan for the current fires, ignoring the action it is
    handed, because the brief is about the stored plan.
    """

    def __init__(self, bank: dict, patches: int, cost_limit_pct: float, device,
                 sequential: bool = False) -> None:
        self.bank, self.device, self.sequential = bank, device, sequential
        self.n = bank["states"].shape[0]
        self._cursor = 0
        self.t = 0
        self.idx: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        centres = (torch.arange(patches).float() + 0.5) / patches
        yy, xx = torch.meshgrid(centres, centres, indexing="ij")
        self.basis = SimpleNamespace(centers=torch.stack([xx.flatten(), yy.flatten()], 1),
                                     n_actuators=patches * patches)
        self.task = SimpleNamespace(cost_limit=cost_limit_pct, equity_enabled=False)
        self.dynamics = "surrogate"

    @property
    def action_dim(self) -> int:
        return self.basis.n_actuators

    @property
    def obs_shape(self):
        return tuple(self.bank["states"].shape[1:])

    def reset(self, batch: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.sequential:
            batch = min(batch, self.n - self._cursor)   # last draw may be short
            if batch <= 0:
                raise RuntimeError(f"plan bank exhausted ({self.n} fires)")
            self.idx = torch.arange(self._cursor, self._cursor + batch)
            self._cursor += batch
        else:
            self.idx = torch.randint(0, self.n, (batch,), generator=generator)
        self.state = self.bank["states"][self.idx].to(self.device)
        self.t = 0
        return self.state

    def step(self, action: torch.Tensor):
        self.t += 1
        return (self.state, self.bank["reward"][self.idx].to(self.device),
                self.bank["cost"][self.idx].to(self.device), True)


class PlanPolicy(nn.Module):
    """The planner as a policy: a point mass on the stored day-1 intensities.

    Pre-tanh mean = atanh(intensity), so the trainer's `tanh(mean)` is the
    intensity the brief quotes; std is a fixed 0.22 (the "likely" bucket), wide
    enough that the 0.05 quantisation grid does not dominate the KL.
    """

    def __init__(self, bank: PlanBank, std: float = 0.22) -> None:
        super().__init__()
        self.bank, self.std = bank, std

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        u = self.bank.bank["intensity"][self.bank.idx].to(state.device).clamp(0, 0.999)
        mean = torch.atanh(u)
        return torch.distributions.Normal(mean, torch.full_like(mean, self.std))

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(state)
        pre = dist.mean if deterministic else dist.rsample()
        return torch.tanh(pre), dist.log_prob(pre).sum(-1)

    def entropy(self, state):
        return self.distribution(state).entropy().sum(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--n-plans", type=int, default=1024, help="training fires to plan and explain")
    parser.add_argument("--patches", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.03)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--plan-steps", type=int, default=50)
    parser.add_argument("--unet-ckpt", default=None)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--gap-every", type=int, default=0,
                        help="log the permutation gap every N iterations, to see whether it "
                             "opens as the NLL falls; 0 disables")
    parser.add_argument("--explain-batch", type=int, default=8)
    parser.add_argument("--contrastive", type=float, default=0.0,
                        help="weight of the differentiable condition-contrastive term")
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        help="subset of arms to run, so each fits a short backfill slot; "
                             "the certificate needs 'trained-in'")
    parser.add_argument("--top-k", type=int, default=12,
                        help="patches a brief names; the parser zeroes the rest, so a "
                             "brief naming every one of 64 patches cannot be matched")
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/ndws_explain")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    for delta in args.deltas:
        if args.n_cal < minimum_calibration_size(delta):
            raise SystemExit(f"delta={delta} needs n_cal >= {minimum_calibration_size(delta)}")
    if -(-(args.n_cal + args.n_test) // 16) * 16 > args.n_eval:
        raise SystemExit("n_cal + n_test must fit in the held-out split")

    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, grid=args.grid))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, grid=args.grid))
    dstats = driver_stats(train)
    train["drivers"] = normalize_drivers(train["drivers"], dstats)
    val["drivers"] = normalize_drivers(val["drivers"], dstats)

    if args.unet_ckpt:
        unet = make_fire_model("unet", 1 + train["drivers"].shape[1], grid=args.grid, width=args.width).to(device)
        unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device)); unet.eval()
    else:
        unet, _ = train_unet(train, val, args, device)
        torch.save(unet.state_dict(), root / "unet.pt")

    task = FireTask(cost_limit=args.budget)
    plan_env = FireEnv(unet, train, task, grid=args.grid, patches=args.patches, horizon=args.horizon, device=device)
    held_env = FireEnv(unet, val, task, grid=args.grid, patches=args.patches, horizon=args.horizon, device=device)

    # Plans: a pool of training fires for the explainer to learn on, and a
    # disjoint block of held-out fires for calibration + test.
    g = torch.Generator().manual_seed(args.seed)
    train_idx = torch.randperm(plan_env.n, generator=g)[: args.n_plans]
    # The trainer draws held-out fires in batches of 16 and builds one brief per
    # requested slot, so the pool is padded to a multiple of 16; the scorer
    # slices back to n_cal + n_test.
    n_held = -(-(args.n_cal + args.n_test) // 16) * 16
    held_idx = torch.randperm(held_env.n, generator=g)[:n_held]
    train_bank = build_bank(plan_env, train_idx, args.horizon, args.plan_steps)
    held_bank = build_bank(held_env, held_idx, args.horizon, args.plan_steps)
    print("plan diversity (train):", json.dumps(plan_diversity(train_bank)), flush=True)
    print(f"plans: {train_bank['states'].shape[0]} train, {held_bank['states'].shape[0]} held-out; "
          f"mean day-1 treated {train_bank['cost'].mean():.2f}% (budget {100 * args.budget:.1f}%), "
          f"patches named per brief {(train_bank['intensity'] >= 0.05).sum(1).float().mean():.1f}", flush=True)

    cost_limit_pct = 100 * args.budget
    rows, trained, trained_scores = [], None, None
    for name, overrides in {k: v for k, v in ARMS.items() if k in args.arms}.items():
        seed_everything(args.seed)
        env = PlanBank(train_bank, args.patches, cost_limit_pct, device)
        trainer = ExplainTrainer(
            env, PlanPolicy(env), ExplainConfig(backbone=args.backbone),
            ExplainTrainConfig(iterations=args.iterations, batch=args.explain_batch,
                               seed=args.seed, log_dir=str(root / name),
                               per_dim_kl=True, brief_top_k=args.top_k,
                               weight_contrastive=args.contrastive, **overrides),
            RunLogger(root / name, use_tensorboard=False), device,
        )
        trainer.gap_every = args.gap_every
        summary = trainer.train()
        # Score on held-out fires, not the training pool.
        trainer.env = PlanBank(held_bank, args.patches, cost_limit_pct, device, sequential=True)
        trainer.policy.bank = trainer.env
        held = trainer.faithfulness_samples(args.n_cal + args.n_test)
        rows.append({
            "arm": name,
            "F(b) held-out": round(float(held.mean()), 4),
            "F(b) train": round(summary["eval/faithfulness"], 4),
            "F(b) reference": round(summary["eval/faithfulness_reference"], 4),
            "KL": round(summary["eval/kl"], 4),
            "stub backbone": summary["backbone_is_stub"],
            "wall (s)": round(summary["wall_clock_s"], 1),
        })
        print(f"[{name}] {rows[-1]}", flush=True)
        if name == "trained-in":
            trained, trained_scores = trainer, held

    # Permutation control: score each brief against a DIFFERENT fire's plan.
    # A generator that learned the brief format and a generic plan, rather than
    # this fire's plan, scores nearly as well shuffled as aligned; the gap is
    # the part of F that is actually fire-specific.
    if trained is not None:
        env = PlanBank(held_bank, args.patches, cost_limit_pct, device, sequential=True)
        trained.env, trained.policy.bank = env, env
        shuffled = trained.faithfulness_samples_permuted(args.n_cal + args.n_test)
        rows.append({"arm": f"trained-in (shuffled plans){' + contrastive' if args.contrastive else ''}",
                     "F(b) held-out": round(float(shuffled.mean()), 4),
                     "F(b) train": None, "F(b) reference": None, "KL": None,
                     "stub backbone": False, "wall (s)": 0.0})
        print(f"[permutation control] F = {shuffled.mean():.4f} "
              f"(aligned {trained_scores.mean():.4f})", flush=True)

    # Certificate for the trained-in explainer: calibration and test are
    # disjoint blocks of the sequential held-out draw above.
    cert_rows = []
    for delta in (args.deltas if trained is not None else []):
        cal, test = trained_scores[: args.n_cal], trained_scores[args.n_cal:]
        cert = certify(cal, test, delta)
        cert_rows.append({"delta": delta, "certified F floor": round(cert.f_min, 4),
                          "empirical coverage": round(cert.empirical_coverage, 4),
                          "nominal": round(1 - delta, 2), "p (one-sided)": round(cert.p_value, 3),
                          "holds": cert.holds, "n_cal": cert.n_cal, "n_test": cert.n_test})
        print(f"[certificate delta={delta}] {cert_rows[-1]}", flush=True)

    # A few briefs for the reader, from the held-out fires.
    if trained is not None:
        env = PlanBank(held_bank, args.patches, cost_limit_pct, device, sequential=True)
        trained.env, trained.policy.bank = env, env
        trained.evaluate(batch=8, dump=root / "briefs_heldout.jsonl")

    table = markdown_table(rows) + ("\n\n" + markdown_table(cert_rows) if cert_rows else "")
    (root / "results.md").write_text(table + f"\n\nbackbone {args.backbone}; budget {cost_limit_pct:.1f}% per day\n")
    (root / "results.json").write_text(json.dumps({"arms": rows, "certificate": cert_rows}, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
