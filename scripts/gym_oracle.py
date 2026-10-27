#!/usr/bin/env python
"""Hindsight oracle: the best action sequence per scenario, with full foresight.

Every verdict in `docs/GYM_RESULTS.md` compares a learned policy against a
hand-written one, and neither number says how much room there was. On
`FloodDefend-v0` the whole feasible ladder spans 0.141 of return against a
per-episode spread of 18.1, so "the learner did not beat the incumbent" there
is close to uninformative -- nothing could. This script measures the ceiling
instead of another policy.

The oracle is deliberately NOT a policy. It optimises an open-loop action
sequence per scenario, knowing that scenario's future, which no policy can
do. So it is an upper bound on what any closed-loop policy could achieve, and
the gap between the incumbent and the oracle is the headroom a method is
competing for:

* oracle close to the incumbent -> the heuristic is near-optimal and that is
  the finding; no amount of RL changes it.
* oracle far above the incumbent -> there is something to win, and the
  oracle's own action sequences say what the heuristic is missing.

Constrained CEM, ranked the way the catalog ranks everything: feasible on the
episode budget first, then by return. An open-loop sequence is the right
search space here because these are 1-to-48-step allocation problems, not
long-horizon control.

    scripts/gym_oracle.py --env CMDPToy-v0 --env PDEControl-dar-v0
    scripts/gym_oracle.py --cheap-only --iters 8 --cands 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_gym import FEASIBLE_TOL, INCUMBENT                    # noqa: E402

#: Wall clock per environment step under CEM, MEASURED from the oracle runs
#: themselves rather than from training. The first version of this table used
#: `train_seconds / steps_done` from the 42-run sweep and was wrong by one to
#: two orders of magnitude in both directions, so every estimate this script
#: printed was useless: `HeatRetrofit-v0` was quoted at 84 minutes and took
#: 6.1 hours, `FireDispatch-v0` at 52 minutes and took 9.1.
#:
#: The pattern in the error is the interesting part. On long horizons CEM is
#: CHEAPER per step than training (`FloodCrossing-v0`, 36 steps: 0.02 ms here
#: against 0.85 in training). On three-step environments it is about ten times
#: dearer (`FireDispatch-v0`: 1786 ms against 170). A rollout is one reset plus
#: `horizon` steps, so at horizon 3 the reset dominates, and `rollout` reseeds
#: on every call where training reuses a drawn scenario. Whether that reseed is
#: the whole explanation is UNVERIFIED -- it is the obvious candidate and
#: making it faster is a separate job from measuring headroom.
MS_PER_STEP = {
    "FloodCrossing-v0": 0.02, "CMDPToy-v0": 0.11, "FloodEvacuate-v0": 0.11,
    "PDEControl-dar-v0": 2.34, "PDEControl-rdf-v0": 2.78,
    "PDEControl-swe-v0": 4.41, "HeatRetrofit-v0": 397.61,
    "Firebreak-v0": 1760.02, "FireDispatch-v0": 1785.74,
    # Not yet measured under CEM; these stay at the training figure and are
    # the reason the `slow` tier runs at a reduced budget.
    "FloodOperate-v0": 126.11, "FloodBuyout-v0": 306.31,
    "FloodOperateFine-v0": 530.29, "FloodDefend-v0": 552.24,
    "FloodGreenInfra-v0": 2081.44,
}
#: Environments where a full-budget CEM finishes in minutes rather than hours.
CHEAP = tuple(e for e, ms in MS_PER_STEP.items() if ms <= 25.0)


def rollout(env, actions: np.ndarray, seed: int):
    """One episode driven by a fixed action sequence. Returns (ret, cost, steps).

    Bails out on a non-finite reward or cost. `PDEControl-swe-v0` can be driven
    to numerical divergence by an open-loop sequence -- one eval seed returned
    an infinite episode cost, and the search that produced it ran for nine
    hours against a fifty-minute estimate, because a diverged field is both
    slow to step and attractive to a maximiser that sees `-inf` as merely very
    bad. A diverged rollout is treated as maximally infeasible so the search
    walks away from it, and the fact that it happened is reported rather than
    smoothed over: an environment that can return `inf` is a benchmark defect.
    """
    env.action_space.seed(seed)
    env.reset(seed=seed)
    total_r = total_c = 0.0
    for t in range(len(actions)):
        _obs, rew, term, trunc, info = env.step(actions[t])
        r, c = float(rew), float(info["cost"])
        if not (np.isfinite(r) and np.isfinite(c)):
            return -np.inf, np.inf, t + 1
        total_r += r
        total_c += c
        if term or trunc:
            return total_r, total_c, t + 1
    return total_r, total_c, len(actions)


def closed_loop(env, policy, seed: int):
    """The incumbent, rolled the way `score_gym.roll` rolls it.

    This used to freeze the incumbent's FIRST action and replay it open-loop
    for the whole horizon, which scored `FloodCrossing-v0`'s
    `budgeted_react_action` at -88.9 on a spend of zero -- the `never` policy.
    A reactive rule evaluated without letting it react is not that rule, and
    the oracle's whole purpose is to be compared against the real one.
    """
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    total_r = total_c = 0.0
    for _ in range(512):
        obs, rew, term, trunc, info = env.step(
            np.asarray(policy(obs), dtype=np.float32))
        total_r += float(rew)
        total_c += float(info["cost"])
        if term or trunc:
            break
    return total_r, total_c


def anchor_sequences(env, policy, shape, horizon: int, seed: int):
    """Known-feasible (or near-feasible) starting points for the search.

    The null action and the incumbent's own closed-loop sequence. The second
    is what makes `oracle >= incumbent` structural: whatever else CEM does, it
    begins having already evaluated the policy it is meant to bound.
    """
    null = np.full((horizon, *shape), -1.0)
    zero = np.zeros((horizon, *shape))
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    acts = []
    for _ in range(horizon):
        act = np.asarray(policy(obs), dtype=np.float64)
        acts.append(act)
        obs, _r, term, trunc, _i = env.step(act.astype(np.float32))
        if term or trunc:
            break
    while len(acts) < horizon:
        acts.append(acts[-1] if acts else np.zeros(shape))
    return [null, zero, np.stack(acts)]


def horizon_of(env, shape, seed: int = 0, cap: int = 64) -> int:
    """How many steps this environment actually runs, measured not assumed."""
    env.action_space.seed(seed)
    env.reset(seed=seed)
    for t in range(cap):
        _o, _r, term, trunc, _i = env.step(np.zeros(shape, dtype=np.float32))
        if term or trunc:
            return t + 1
    return cap


def cem(env, shape, horizon: int, seed: int, limit: float,
        cands: int, iters: int, elite_frac: float, rng,
        anchors=()) -> dict:
    """Constrained CEM over an open-loop action sequence.

    Two details are what make this find anything, and the first version of
    this function got both wrong -- it reported "no feasible sequence exists"
    for `FloodEvacuate-v0`, whose `never_action` is feasible at a cost of
    exactly zero.

    **The population is seeded with known-feasible anchors.** Starting from
    `mu = 0` with a wide sigma, every candidate on a 48-step environment
    overspends, and a search that has never seen a feasible point has no
    gradient towards one. The anchors are the null action and the incumbent's
    own sequence, so the oracle is an upper bound on the LADDER by
    construction rather than by luck.

    **With no feasible candidate, rank by VIOLATION, not by return.** Return
    and spend are positively coupled in every one of these environments -- you
    buy outcome with budget -- so ranking an all-infeasible generation by
    return walks the search further out of the feasible set with every
    iteration. That is a constrained-CEM detail, not a tuning preference.
    """
    dim = (horizon, *shape)
    mu = np.zeros(dim, dtype=np.float64)
    sigma = np.full(dim, 0.8)
    n_elite = max(2, int(cands * elite_frac))
    slack = abs(limit) * FEASIBLE_TOL
    best = {"return": -np.inf, "cost": np.inf, "feasible": False}
    anchors = [np.asarray(a, dtype=np.float64).reshape(dim) for a in anchors]

    for it in range(iters):
        draw = np.clip(mu + sigma * rng.standard_normal((cands, *dim)), -1.0, 1.0)
        if it == 0 and anchors:
            # The anchors replace the first candidates rather than extending
            # the population, so the rollout budget stays exactly `cands`.
            for j, anc in enumerate(anchors[:max(1, cands // 4)]):
                draw[j] = anc
        scored = []
        for c in range(cands):
            r, cost, _ = rollout(env, draw[c].astype(np.float32), seed)
            scored.append((cost <= limit + slack, r, cost, c))
        any_feasible = any(t[0] for t in scored)
        if any_feasible:
            scored.sort(key=lambda t: (not t[0], -t[1]))
        else:
            # Least overspend first. `inf` sorts last, which is what a diverged
            # rollout deserves: it is not a near-miss on the budget.
            scored.sort(key=lambda t: (not np.isfinite(t[2]), t[2]))
        top = scored[:n_elite]
        for ok, r, cost, ci in scored:
            if ok and (not best["feasible"] or r > best["return"]):
                best = {"return": r, "cost": cost, "feasible": True,
                        "actions": draw[ci].tolist()}
                break
        elite = draw[[t[3] for t in top]]
        mu = elite.mean(axis=0)
        # A floor on sigma: CEM collapses on these short horizons long before
        # the iteration budget is spent, and a collapsed search stops finding
        # the boundary-hugging sequences that the budget makes optimal.
        sigma = np.maximum(elite.std(axis=0), 0.05)
    return best


def main() -> int:
    import gymnasium as gym
    from pspe.envs import register_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--cheap-only", action="store_true",
                    help="only environments at or under 25 ms/step")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--cands", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--elite-frac", type=float, default=0.125)
    ap.add_argument("--out", default="runs/gym/oracle.json")
    a = ap.parse_args()

    register_all()
    env_ids = a.env or (list(CHEAP) if a.cheap_only else list(MS_PER_STEP))
    seeds = list(range(a.seeds))
    out = {}
    for env_id in env_ids:
        env = gym.make(env_id, split=a.split)
        inner = env.unwrapped
        limit = float(inner.task.cost_limit)
        shape = env.action_space.shape
        h = horizon_of(env, shape)
        est = a.cands * a.iters * h * MS_PER_STEP.get(env_id, 100.0) / 1000.0
        print(f"{env_id:<22} horizon {h:>3}  limit {limit:<12.4g} "
              f"~{est * len(seeds) / 60:.0f} min est", flush=True)
        inc = getattr(inner, INCUMBENT[env_id])
        rows = []
        t0 = time.time()
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            o = cem(env, shape, h, s, limit, a.cands, a.iters, a.elite_frac,
                    rng, anchors=anchor_sequences(env, inc, shape, h, s))
            ir, ic = closed_loop(env, inc, s)
            rows.append({"seed": s, "oracle_return": o["return"],
                         "oracle_cost": o["cost"],
                         "oracle_feasible": bool(o["feasible"]),
                         "incumbent_return": ir, "incumbent_cost": ic})
        env.close()
        feas = [r for r in rows if r["oracle_feasible"]]
        orc = float(np.mean([r["oracle_return"] for r in feas])) if feas else None
        out[env_id] = {"limit": limit, "horizon": h, "rows": rows,
                       "oracle_mean_return": orc,
                       "oracle_feasible_seeds": len(feas),
                       "of": len(rows), "seconds": time.time() - t0,
                       "cands": a.cands, "iters": a.iters}
        om = f"{orc:.4f}" if orc is not None else "none feasible"
        print(f"  -> oracle {om}  feasible {len(feas)}/{len(rows)}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
