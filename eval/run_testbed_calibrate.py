#!/usr/bin/env python
"""Does this testbed's RETURN separate? Measure before adopting, not after.

    python eval/run_testbed_calibrate.py --testbed swe --targets 0.0 0.2 0.3 0.4

Section 3.4 retired `swe` because its return band was ~7% and method
differences lived inside it: its cost separated, so the constraint bound, but
its joint, disaggregated and unanchored arms all returned the same number.

That diagnosis does not survive a positive control -- see section 3.4a. `rdf`,
which ranks planners fine, scores 7.1% on this same probe, and its trained
planner's span (4.4%) is NARROWER than retired `swe`'s (7.2%). What separates
the two is not span magnitude but (a) whether any random actuation improves
return at all (rdf 23/60, swe 0/60) and (b) whether the reward optimum drives
cost to the limit (rdf 104% of it, swe 62%). Both are reported below.

The diagnosis in section 3.4 was that raising actuator amplitude does not help
-- it scales the useful signal and the exploration noise together -- and that
the fix has to be a different objective. This measures candidates for that.

The mechanism the candidates exploit: `swe` damps the surface toward zero on
its own (`h_damp`), and the shipped objective rewards exactly that. Tracking a
NON-ZERO level instead means doing nothing is wrong by construction and the
control has to sustain the state against the dynamics, which is the regime
where a planner can be told apart from a heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.utils import get_device, seed_everything  # noqa: E402


@torch.no_grad()
def rollout(env, actions: torch.Tensor | None, horizon: int, batch: int,
            ic_seed: int) -> tuple[float, float]:
    """Mean episode return and cost for a fixed action vector (or do-nothing).

    `ic_seed` fixes the initial conditions, so every action vector is scored on
    the SAME batch of episodes. Without that, resetting per rollout draws fresh
    random fields and "best of N" selects lucky initial states as much as good
    actions -- which manufactures a span out of nothing. An earlier version of
    this script did exactly that and reported random beating do-nothing on swe,
    contradicting section 3.4's 400-vector sweep. The contradiction was the bug.
    """
    gen = torch.Generator().manual_seed(ic_seed)
    env.reset(batch=batch, generator=gen)
    total = torch.zeros(batch, device=env.state.device)
    cost_total = torch.zeros(batch, device=env.state.device)
    for _ in range(horizon):
        a = (torch.zeros(batch, env.action_dim, device=env.state.device)
             if actions is None else actions)
        _, r, c, _ = env.step(a)
        total = total + r.detach()
        cost_total = cost_total + c.detach()
    return float(total.mean()), float(cost_total.mean())


def probe(testbed: str, target: float, grid: int, horizon: int,
          n_random: int, device, seed: int, batch: int) -> dict:
    """Do-nothing against the best of N random constant actions, same episodes."""
    seed_everything(seed)
    # make_env forwards overrides to make_task, so the rest of the testbed's
    # calibrated spec (w_track, u_max, budget) is held fixed and only the
    # target moves. That is the controlled version of this question.
    env = make_env(dynamics="truth", testbed=testbed, grid=grid, horizon=horizon,
                   n_actuators=9, device=device, batched=True, target_value=target)

    ic_seed = 10_000 + seed
    do_nothing, dn_cost = rollout(env, None, horizon, batch, ic_seed)
    gen = torch.Generator().manual_seed(seed)
    best, best_cost, n_better = -float("inf"), float("nan"), 0
    for _ in range(n_random):
        a = (2.0 * torch.rand(1, env.action_dim, generator=gen) - 1.0).to(device)
        a = a.expand(batch, -1)
        r, c = rollout(env, a, horizon, batch, ic_seed)
        if r > do_nothing:
            n_better += 1
        if r > best:
            best, best_cost = r, c

    # Span as a fraction of what there is to win. A 7% span cannot rank methods
    # whose differences are a couple of points.
    span = (best - do_nothing) / max(abs(do_nothing), 1e-9)
    return {
        "target": target,
        "do_nothing": round(do_nothing, 4),
        "best_random": round(best, 4),
        "span_abs": round(best - do_nothing, 4),
        "span_pct": round(100 * span, 1),
        "n_beating_nothing": f"{n_better}/{n_random}",
        "cost_do_nothing": round(dn_cost, 4),
        "cost_best": round(best_cost, 4),
    }


# ---------------------------------------------------------------------------
# Cost calibration. Admitting a target is only half of Rule 36; the limit then
# has to be placed so the constraint binds.
#
# The convention is not written down anywhere but it is exact. Every shipped
# TASK_SPECS limit sits at 35% of the way from the do-nothing cost to the
# reward-greedy cost:
#
#   dar  0.229 -> 2.249, limit 0.936   = 35.00%
#   swe  0.108 -> 0.347, limit 0.192   = 35.15%
#   rdf  0.594 -> 8.213, limit 3.26    = 34.99%
#
# So a new family's limit is a measurement, not a taste: trace the fields once,
# sweep the caps in numpy (both cost terms enter only through a relu, so the
# solver does not need re-running per candidate), and read the limit off.
LIMIT_FRACTION = 0.35


@torch.no_grad()
def trace(env, actions, horizon, batch, ic_seed):
    """Per-step tracked field and mean|a| over an episode, plus the return."""
    gen = torch.Generator().manual_seed(ic_seed)
    env.reset(batch=batch, generator=gen)
    fields, acts, ret = [], [], torch.zeros(batch, device=env.state.device)
    for _ in range(horizon):
        a = (torch.zeros(batch, env.action_dim, device=env.state.device)
             if actions is None else actions)
        st, r, _, _ = env.step(a)
        fields.append(st[:, 0].detach().cpu().numpy())
        acts.append(a.abs().mean(dim=-1).detach().cpu().numpy())
        ret = ret + r.detach()
    return np.stack(fields), np.stack(acts), float(ret.mean())


def episode_cost(fields, acts, u_max, budget):
    """Task.cost summed over steps, averaged over the batch, at weights 1.0."""
    exposure = np.maximum(fields - u_max, 0.0).mean(axis=(2, 3))
    overspend = np.maximum(acts - budget, 0.0)
    return float((exposure + overspend).sum(axis=0).mean())


def calibrate_cost(testbed, target, grid, horizon, n_random, device, seed,
                   batch, caps, budgets):
    """Candidate (u_max, budget, cost_limit) triples for a new target."""
    seed_everything(seed)
    env = make_env(dynamics="truth", testbed=testbed, grid=grid, horizon=horizon,
                   n_actuators=9, device=device, batched=True, target_value=target)
    ic = 10_000 + seed
    dn_f, dn_a, dn_ret = trace(env, None, horizon, batch, ic)
    best_ret, best = -float("inf"), None
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_random):
        a = (2.0 * torch.rand(1, env.action_dim, generator=gen) - 1.0).to(device)
        a = a.expand(batch, -1)
        f, aa, r = trace(env, a, horizon, batch, ic)
        if r > best_ret:
            best_ret, best = r, (f, aa)
    bf, ba = best

    print(f"do-nothing return {dn_ret:.4f}   best random {best_ret:.4f}")
    print(f"field: idle max {dn_f.max():.3f} p99 {np.percentile(dn_f, 99):.3f} | "
          f"best max {bf.max():.3f} p99 {np.percentile(bf, 99):.3f}")
    print(f"mean|a|: idle {dn_a.mean():.4f} | best {ba.mean():.4f}\n")

    rows = []
    for u_max in caps:
        for budget in budgets:
            c0 = episode_cost(dn_f, dn_a, u_max, budget)
            c1 = episode_cost(bf, ba, u_max, budget)
            # An exposure-dominated cost is the one worth having: if the budget
            # term carries it, the constraint is about actuation spend and has
            # stopped coupling to the PDE state at all.
            exp_only = episode_cost(bf, ba, u_max, 1e9)
            share = exp_only / c1 if c1 > 1e-12 else 0.0
            rows.append({
                "u_max": round(u_max, 3), "budget": round(budget, 3),
                "idle cost": round(c0, 4), "best cost": round(c1, 4),
                "exposure share": round(share, 3),
                "limit (35%)": round(c0 + LIMIT_FRACTION * (c1 - c0), 4),
                "usable": "**yes**" if (c1 > 10 * max(c0, 1e-6) and c1 > 1e-3
                                        and share > 0.5) else "no",
            })
    print(markdown_table(rows))
    print("\n`usable` wants three things at once: the reward optimum's cost well\n"
          "above idling's, a cost large enough to measure, and more than half of\n"
          "it coming from field exposure rather than actuation overspend.")
    print("\nThe 35% limit is a LOWER BOUND while the upper anchor is best-of-N\n"
          "random: TASK_SPECS anchors on a trained reward-greedy planner, which\n"
          "reaches a higher cost. Re-derive it from a trained unconstrained run\n"
          "before using the family for method comparison.")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="swe")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate-cost", action="store_true",
                    help="sweep u_max/budget and place cost_limit for --targets[0]")
    ap.add_argument("--caps", type=float, nargs="+", default=None)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.2, 0.35, 0.6])
    args = ap.parse_args()

    device = get_device(args.device)
    if args.calibrate_cost:
        t = args.targets[0]
        caps = args.caps or [t + d for d in (0.02, 0.05, 0.10, 0.20, 0.40)]
        calibrate_cost(args.testbed, t, args.grid, args.horizon, args.n_random,
                       device, args.seed, args.batch, caps, args.budgets)
        return 0
    rows = [probe(args.testbed, t, args.grid, args.horizon, args.n_random,
                  device, args.seed, args.batch) for t in args.targets]
    for r in rows:
        print(f"target {r['target']:<5} do-nothing {r['do_nothing']:>9.4f}   "
              f"best random {r['best_random']:>9.4f}   span {r['span_pct']:>7.1f}%"
              f"   beat nothing {r['n_beating_nothing']:>8s}",
              flush=True)

    table = markdown_table(rows)
    print("\n" + table)

    # The admission test, corrected by a positive control. `span_pct >= 25%`
    # was the first criterion here and it REJECTED rdf -- the family the whole
    # section 3.2 safety story runs on -- at 7.1%. A test that fails on a
    # testbed known to rank planners is not a test. What actually separates rdf
    # from retired swe is whether random actuation can find ANY improving
    # direction (rdf 23/60, swe 0/60) and whether chasing reward drives cost to
    # the limit. See section 3.4a.
    print("\nAdmission (rdf reference: 23/60 improving, best-random cost 104% of limit):")
    for r in rows:
        frac = int(r["n_beating_nothing"].split("/")[0]) / n_random
        verdict = ("usable" if frac >= 0.15 else
                   "DEAD — no improving direction, the planner will learn to idle")
        print(f"  target {r['target']:<5} {100*frac:4.0f}% of directions improve"
              f"   -> {verdict}")
    print("\nA target that passes still needs u_max/budget/cost_limit recalibrated:\n"
          "  cost at the reward optimum must land NEAR the limit, not far above or\n"
          "  below it. rdf: 3.385 against 3.26. Retired swe: 0.119 against 0.192.")

    if args.out:
        Path(args.out).write_text(json.dumps({"testbed": args.testbed, "rows": rows},
                                             indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
