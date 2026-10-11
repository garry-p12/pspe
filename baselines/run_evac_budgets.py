"""Do the learned policies beat the trigger family AT THEIR OWN PRICE?

Two corrections to the first run of this comparison, both of which pushed the
answer the same way.

**The cost statistic.** It reported `eval/cost_mean_over_run`, the mean over
every periodic evaluation during training, while `return` is the final
policy's. Each row therefore paired one policy's benefit with another
policy's price, and the error is largest exactly when a dual is moving the
cost during training -- which is the case here: measured, final cost 5.5-8.5
against an over-run mean of 11.7-14.4, so the policy actually shipped spends
about half what was reported. This reports `episode_cost` and carries the
mean-over-run beside it so the drift stays checkable rather than inferred.

**The comparison.** Every learned arm was scored against the best trigger at
a budget of 32, which the trigger clears with slack while the learned arms
spent a third of it. Returns at different costs do not compare on a
constrained problem. Each arm is now placed against the best trigger costing
no more than that arm itself did, from `run_evac_frontier`.

The hypothesis, recorded before the numbers arrived: a trigger has ONE degree
of freedom, a single threshold applied to every zone at once, so when
disruption is scarce the question stops being *when* to order and becomes
*whom* -- which that family cannot express. Measured, the family has three
distinct behaviours across 0.30-0.99, and returns -3511.4 at every cost from
6.5 to 15.1. If a learned policy is worth anything here it is by landing
inside that gap.

    python3 -m baselines.run_evac_budgets --budgets 12,20,32

Reported with a SIGN TEST over seeds as well as a mean. The per-seed spread
is heavy-tailed -- one run at -824 and another at -4416 in the same
configuration -- so "how many seeds beat the baseline" is the more honest
summary, and a mean plus a standard deviation over four draws of that
distribution is close to meaningless.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time

import torch

from baselines.evac import table as T
from baselines.safe_rl import SafeRLConfig, make_agent
from pspe.envs.evacuate import EvacuateTask, FloodEvacuateEnv

#: The raw objective runs to thousands; the PID dual gains are calibrated
#: against a REWARD gradient, so the environment is scaled to O(1) and the
#: reported return is scaled back for comparison with the baselines.
SCALE = 1500.0


def sign_test(wins: int, n: int) -> float:
    """Two-sided p for `wins` of `n` under a fair coin. Exact, no scipy."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=str, default="12,20,32")
    ap.add_argument("--algos", type=str, default="ppo_lagrangian,cpo")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cpo-seeds", type=int, default=4,
                    help="CPO costs ~3x per run; fewer seeds by default")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--baseline", type=str, default="",
                    help="JSON from run_evac_frontier --out, to score each "
                         "arm against the best trigger it could afford")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    curve = []
    if args.baseline:
        with open(args.baseline) as fh:
            curve = json.load(fh)["curve"]

    tab = T.load()
    print(f"egress table: {tab.members} members x {tab.zones} zones x "
          f"{tab.slots} slots, lead {T.LEAD * T.INTERVAL:.0f} min\n")

    rows = []
    for budget in [float(b) for b in args.budgets.split(",")]:
        for algo in args.algos.split(","):
            n = args.cpo_seeds if algo == "cpo" else args.seeds
            rs, cs, ms = [], [], []
            for sd in range(n):
                t0 = time.time()
                torch.manual_seed(sd)
                env = FloodEvacuateEnv(
                    tab, EvacuateTask(cost_limit=budget, reward_scale=1 / SCALE))
                out = make_agent(algo, env, cfg=SafeRLConfig(
                    iterations=args.iterations, batch=args.batch,
                    horizon=tab.slots, seed=sd)).train()
                rs.append(out.get("return", float("nan")) * SCALE)
                cs.append(out.get("episode_cost", float("nan")))
                ms.append(out.get("eval/cost_mean_over_run", float("nan")))
                print(f"  b{budget:.0f} {algo} seed {sd}: {rs[-1]:10.1f}  "
                      f"final cost {cs[-1]:5.1f}  (over-run {ms[-1]:5.1f})  "
                      f"({time.time() - t0:.0f}s)", flush=True)

            row = {"budget": budget, "algo": algo, "seeds": n,
                   "return": st.fmean(rs), "return_sd": st.pstdev(rs),
                   "cost": st.fmean(cs), "cost_sd": st.pstdev(cs),
                   "cost_over_run": st.fmean(ms),
                   "returns": rs, "costs": cs}

            # Each seed against the best trigger IT could afford, which is
            # the only comparison the constraint permits.
            #
            # Some seeds come in CHEAPER than any trigger in the family --
            # measured, 5.5 and 6.0 against a cheapest trigger of 6.51 -- and
            # dropping those as "no comparison available" would quietly
            # discard exactly the runs that are hardest for the baseline. The
            # family's return is monotone non-increasing in cost, so the
            # cheapest trigger's return is an UPPER bound on what the family
            # could manage at a price it cannot actually reach. Using it is
            # generous to the baseline, which is the direction to err.
            if curve:
                floor = min(curve, key=lambda row_: row_[2])
                per, unreachable = [], 0
                for r_i, c_i in zip(rs, cs):
                    afford = [r for th, r, c in curve if c <= c_i]
                    if afford:
                        per.append(r_i - max(afford))
                    else:
                        unreachable += 1
                        per.append(r_i - floor[1])
                row["vs_affordable_trigger"] = per
                row["cheaper_than_any_trigger"] = unreachable
                row["wins"] = sum(1 for d in per if d > 0)
                row["p_sign"] = sign_test(row["wins"], len(per))
                row["delta_mean"] = st.fmean(per) if per else float("nan")

            rows.append(row)
            msg = (f"b{budget:<4.0f} {algo:<16} {row['return']:10.1f} "
                   f"(sd {row['return_sd']:7.1f})  cost {row['cost']:5.1f}")
            if curve and row.get("wins") is not None:
                msg += (f"  |  beats affordable trigger on {row['wins']}/"
                        f"{len(rs)} seeds, mean {row['delta_mean']:+.1f}, "
                        f"sign p={row['p_sign']:.3f}")
                if row["cheaper_than_any_trigger"]:
                    msg += (f"  ({row['cheaper_than_any_trigger']} seed(s) "
                            f"cheaper than ANY trigger)")
            print(msg + "\n", flush=True)

            if args.out:
                with open(args.out, "w") as fh:
                    json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
