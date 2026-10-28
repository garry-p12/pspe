#!/usr/bin/env python
"""How many eval scenarios can actually tell two policies apart.

`of: 12` counts seeds. It does not count information, and on three of these
fourteen environments the difference is most of the number:

* `FloodBuyout-v0`'s incumbent returns **0.0 on eleven of twelve** eval
  scenarios. There is no harm left to prevent on those eleven, so nothing a
  policy does there can show up in the score.
* `FloodEvacuate-v0` has six scenarios with no flood at all.
* `FloodCrossing-v0`'s returns take three distinct values.

A scenario DISCRIMINATES when the span between the worst and the best policy
on it is non-trivial: if `nothing` and the best feasible rung give the same
number, that scenario is a constant and contributes nothing but a smaller
standard error. This measures the span per scenario over the whole ladder, so
the count can be reported beside every result instead of `of: 12` implying
twelve independent observations.

The span is taken over policies that are FEASIBLE on that scenario. A policy
that buys a better outcome by overspending does not establish that the
scenario can be improved within budget, which is the only question a
constrained benchmark asks.

    scripts/gym_discrimination.py                    # every environment
    scripts/gym_discrimination.py --env FloodBuyout-v0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_gym import (FEASIBLE_TOL, INCUMBENT, LADDER_POLICIES,   # noqa: E402
                       _policy, roll)

#: A scenario counts as discriminating when the feasible policies on it span
#: more than this fraction of the whole environment's feasible span. Relative
#: because returns here run from 0.16 to 3.7e3; 1% is low enough to admit a
#: real but small difference and high enough to reject floating-point noise.
SPAN_FRAC = 0.01

#: Two-sided 5% critical t on 11 degrees of freedom is 2.201, so detecting a
#: difference on twelve PAIRED scenarios needs a mean gap of 2.201/sqrt(12) =
#: 0.635 paired standard deviations. A span below that cannot be resolved by
#: the protocol however many rungs are compared.
MIN_SPAN_IN_PAIRED_SD = 2.201 / 12 ** 0.5


def resolvability(span: float, paired_sd: float) -> "tuple[float, str]":
    """Is the span large enough for the paired protocol to see it?

    The denominator has to be the spread of PER-SCENARIO DIFFERENCES, not the
    spread of episode returns. Using the latter is a mistake that changes the
    answer for half this catalog: against unpaired episode noise six of the
    twelve measured environments look unmeasurable, and against the paired
    spread ten of twelve are fine. `FireDispatch-v0` is the clearest case --
    its span of 2.13 sits against an episode sd of 9.10 and a paired sd of
    0.57, and the paired test resolves it at t = -3.9. The blocking factor is
    the scenario, so the scenario's own variance is not noise the comparison
    has to fight.
    """
    if not paired_sd or paired_sd <= 0.0:
        return float("inf"), "exact: per-scenario differences have no spread"
    r = span / paired_sd
    if r < MIN_SPAN_IN_PAIRED_SD:
        return r, "UNRESOLVABLE even paired"
    if r < 2.0:
        return r, "resolvable but tight"
    return r, "usable"


def main() -> int:
    import gymnasium as gym
    from pspe.envs import register_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--paired", default="runs/gym/paired.json",
                    help="where `paired_gym.py` wrote its per-episode "
                         "arrays; the resolvability denominator comes "
                         "from there and is independent of --out")
    ap.add_argument("--out", default="runs/gym/discrimination.json")
    a = ap.parse_args()

    register_all()
    env_ids = a.env or list(LADDER_POLICIES)
    seeds = list(range(a.seeds))
    out = {}
    for env_id in env_ids:
        env = gym.make(env_id, split=a.split)
        inner = env.unwrapped
        limit = float(inner.task.cost_limit)
        slack = abs(limit) * FEASIBLE_TOL
        rungs = dict(LADDER_POLICIES[env_id])
        rungs.setdefault("incumbent", INCUMBENT[env_id])
        # per scenario: the returns of the policies that were feasible on it
        feas_by_seed = {s: [] for s in seeds}
        for _name, spec in rungs.items():
            pol = _policy(env, inner, spec)
            for s in seeds:
                r, c, _lim, _st = roll(env, pol, s)
                if c <= limit + slack:
                    feas_by_seed[s].append(float(r))
        env.close()
        spans = {s: (max(v) - min(v) if len(v) > 1 else 0.0)
                 for s, v in feas_by_seed.items()}
        overall = max(spans.values()) if spans else 0.0
        thresh = overall * SPAN_FRAC
        disc = [s for s in seeds if spans[s] > thresh and spans[s] > 0.0]
        # The paired spread, taken from `paired_gym.py`'s saved arrays if they
        # exist: the smallest per-seed sd of (learned - incumbent) differences,
        # i.e. the best case the protocol faces on this environment.
        paired_sd = None
        pj = Path(a.paired)
        if pj.exists():
            pd = json.loads(pj.read_text()).get(env_id)
            if pd:
                inc = np.asarray(pd["incumbent_returns"], dtype=float)
                sds = []
                for v in pd["seeds"].values():
                    # A checkpoint `paired_gym.py` marked stale carries no
                    # episode arrays, because it was never rolled: its
                    # observation space no longer matches the environment.
                    # There is no paired spread to take from it.
                    if v.get("stale"):
                        continue
                    d = np.asarray(v["returns"], dtype=float) - inc
                    if d.size > 1:
                        sds.append(float(d.std(ddof=1)))
                if sds:
                    paired_sd = min(sds)
        ratio, verdict = (resolvability(overall, paired_sd)
                          if paired_sd is not None else (None, "no paired data"))
        if len(disc) <= len(seeds) // 4:
            verdict = "BROKEN: too few informative scenarios"
        out[env_id] = {
            "discriminating": len(disc), "of": len(seeds),
            "seeds": disc, "span_per_seed": {str(k): v for k, v in spans.items()},
            "largest_span": overall, "threshold": thresh,
            "n_rungs": len(rungs), "paired_sd": paired_sd,
            "span_in_paired_sd": ratio, "verdict": verdict,
        }
        rs = "---" if ratio is None else ("inf" if ratio == float("inf")
                                          else f"{ratio:.2f}")
        print(f"{env_id:<22} discriminating {len(disc):>2}/{len(seeds)}  "
              f"span {overall:>10.4g}  span/paired-sd {rs:>6}  {verdict}",
              flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
