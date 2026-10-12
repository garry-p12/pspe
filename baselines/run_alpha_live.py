"""Does the mixing rule do anything, on a flood twin where it can fire at all?

Every flood number this project has reported was produced with alpha pinned to
zero. `FloodOperateEnv` defaults to `differentiable=False`, and
`HybridPlannerTrainer` responds by setting `alpha_fixed = 0.0`, so the "hybrid
planner" row in `run_operate_comparison` is plain likelihood-ratio with a
critic baseline. The adaptive pathwise/likelihood mixing -- the thing the
method is named for -- has never run on this environment. A negative result
collected that way is not evidence about the method.

The obstacle was real: the gate's orifice derivative `dq/d(dH) ~ dH^(-1/2)`
spans twenty-five orders of magnitude across an episode, so the truth dynamics
cannot carry a usable pathwise gradient at episode length. That is what
`FloodIntervalSurrogate` is for, and it has never been run either.

## The arms, and why they are these three

One surrogate is fitted ONCE and shared, so that a difference between arms is
alpha and not two draws of a network:

* `adaptive`  -- surrogate dynamics, `adaptive_alpha=True`. Alpha is live.
* `alpha0`    -- surrogate dynamics, `alpha_fixed=0.0`. Identical in every
                 other respect. This is the matched control, and it is the
                 only comparison that isolates the rule.
* `truth`     -- true solver dynamics, alpha necessarily 0. What was reported
                 before. Included to separate "training on a surrogate helps"
                 from "mixing the gradients helps", which are different claims
                 that a two-arm study would confound.

Every arm is SCORED on the true solver via `eval_env`, so a planner that only
looks good on its own surrogate scores badly here, which is the point.

## Reporting

Per-seed return, `episode_cost`, `final_alpha` and the measured pathwise bias.
Paired across seeds, because §4.2a is the cautionary tale: at five seeds the
mixing rule looked like a mean effect and at twenty it was a variance effect,
and the only honest comparison was the matched-budget one.

    python -m baselines.run_alpha_live --fit --out runs/alpha_live
    python -m baselines.run_alpha_live --arm adaptive --seed 0 --out runs/alpha_live
    python -m baselines.run_alpha_live --aggregate --out runs/alpha_live
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import torch

from pspe.envs.flood_env import FloodOperateEnv, FloodOperateTask, make_polder_bank
from pspe.envs.flood_surrogate import (
    FloodIntervalSurrogate, collect_transitions, fit_surrogate,
)
from pspe.plan.policy import FieldCritic, GaussianFieldPolicy
from pspe.plan.trainer import HybridPlannerTrainer, PlannerConfig

ARMS = ("adaptive", "alpha0", "truth")

#: The planner settings `eval/run_ndws_planning.py` calls "PSPE v2", which this
#: environment has never used. Every flood number in this project was produced
#: with all five at their defaults, and the trace says that is why.
#:
#: `saturation_coef` is the one that matters here. PlannerConfig documents the
#: failure precisely -- "the reward-greedy sprint drives the pre-tanh policy
#: mean past |mu| ~ 3 within 40 iterations. There tanh'(mu) ~ 0: the pathwise
#: gradient dies, the variance rule then picks the dead branch (alpha -> 0.99)
#: because zero variance looks like precision, and lambda = 38 multiplies
#: nothing" -- and then leaves the guard at 0.0 because dar never needed it.
#: Measured on the polder: pre-tanh past 3 by iteration 30, reaching 1,436;
#: alpha 0.979; lambda 27.7 and still climbing linearly at the last iteration.
#: That is the documented pathology, line for line.
#:
#: `dual_normalize` is the second half. The budget is in pump-hours (limit 2.0)
#: while the reward is a percentage, so the raw dual gains are working across a
#: scale gap -- the same mismatch ndws hit, where lambda needed about thirty
#: times the length of the run to bind.
V2 = dict(saturation_coef=1.0, advantage_std_floor=1e-2, dual_ema=0.7,
          ki=0.5, dual_normalize=True)


def build_bank(args):
    #: One bank for every arm and every seed. The terrain is not the variable.
    g = torch.Generator().manual_seed(0)
    return make_polder_bank(n=64, grid=args.grid, dx=args.dx, generator=g)


def make_env(bank, args, surrogate=None):
    return FloodOperateEnv(
        bank, task=FloodOperateTask(cost_limit=args.budget),
        horizon=args.horizon, control_minutes=30.0, dx=args.dx,
        surrogate=surrogate)


@torch.no_grad()
def score_fixed(env, act, episodes: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    s = env.reset(episodes, generator=g)
    ret = torch.zeros(episodes)
    cost = torch.zeros(episodes)
    while True:
        s, r, c, done = env.step(act(s, env))
        ret += r.detach()
        cost += c.detach()
        if done:
            break
    return {"return": float(ret.mean()), "cost": float(cost.mean())}


def cmd_fit(args) -> int:
    """Fit the shared surrogate on random-control rollouts of the TRUE solver."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = build_bank(args)
    env = make_env(bank, args)

    t0 = time.time()
    data = collect_transitions(env, episodes=args.fit_episodes,
                               generator=torch.Generator().manual_seed(1))
    net, rep = fit_surrogate(data, epochs=args.fit_epochs,
                             generator=torch.Generator().manual_seed(2))
    torch.save(net.state_dict(), out / "surrogate.pt")

    # The incumbent, on the same bank and the same eval seed every arm uses.
    # Without this the arms are compared to a number from another run.
    base = score_fixed(make_env(bank, args),
                       lambda s, e: e.baseline_action(s), args.episodes, 12345)
    report = {
        "transitions": rep.transitions, "val_rmse_m": rep.val_rmse_m,
        "persistence_rmse_m": rep.persistence_rmse_m, "skill": rep.skill,
        "epochs": rep.epochs, "seconds": round(time.time() - t0, 1),
        "incumbent_deadband": base, "budget": args.budget,
    }
    (out / "surrogate_report.json").write_text(json.dumps(report, indent=1))
    print(f"surrogate: {rep.transitions} transitions, val RMSE "
          f"{rep.val_rmse_m:.4f} m vs persistence {rep.persistence_rmse_m:.4f} m, "
          f"skill {rep.skill:.3f}  ({time.time() - t0:.0f}s)")
    print(f"incumbent deadband: {base['return']:.2f} at {base['cost']:.2f} "
          f"pump-hours (budget {args.budget})")
    # A surrogate with no skill cannot carry a gradient worth mixing, and the
    # experiment below would be measuring noise. Say so loudly rather than
    # letting it be inferred from a null result three hours later.
    if rep.skill <= 0.0:
        print("WARNING: surrogate has no skill over persistence. Alpha has "
              "nothing useful to mix and a null result here says nothing "
              "about the rule.")
    return 0


def cmd_arm(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = build_bank(args)

    net = None
    if args.arm in ("adaptive", "alpha0"):
        net = FloodIntervalSurrogate()
        # `weights_only=True`: this file is named on the command line and
        # pickle would execute whatever is in it.
        net.load_state_dict(torch.load(out / "surrogate.pt",
                                       map_location="cpu", weights_only=True))
        net.eval()

    torch.manual_seed(args.seed)
    env = make_env(bank, args, surrogate=net)
    pol = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    cfg = PlannerConfig(iterations=args.iterations, batch=args.batch,
                        horizon=args.horizon)
    if args.config == "v2":
        for k, v in V2.items():
            setattr(cfg, k, v)
    if args.arm == "alpha0":
        # The matched control: same surrogate dynamics, mixing switched off.
        cfg.adaptive_alpha = False
        cfg.alpha_fixed = 0.0

    t0 = time.time()
    trainer = HybridPlannerTrainer(
        env, pol, FieldCritic(env.obs_shape[0]), cfg,
        eval_env=make_env(bank, args))          # scored on the TRUE solver
    res = trainer.train()
    row = {
        "arm": args.arm, "config": args.config, "seed": args.seed,
        "return": res.get("return", float("nan")),
        "episode_cost": res.get("episode_cost", float("nan")),
        "final_alpha": res.get("final_alpha", float("nan")),
        "pathwise_bias_sq": res.get("alpha/final_pathwise_bias_sq", float("nan")),
        # The three that say whether this run means anything.
        #
        # A single-seed run of this comparison returned -32.009300231933594
        # for BOTH alpha=0.000 and alpha=0.979 -- identical to thirteen
        # significant figures, at exactly 6.00 pump-hours, which is 12
        # intervals of pumping at 100% and the largest number the action
        # space can produce. Two policies do not agree to float precision by
        # accident: both had collapsed onto the tanh boundary, where the
        # evaluation trajectory is the same whatever gradient put them there,
        # so alpha was being measured on a policy that could not move.
        #
        # `pre_tanh_mean_abs` is the direct read of that: past about 3 the
        # squashing function is flat and the policy is a corner. `final_lambda`
        # says whether the dual ever engaged or simply never grew.
        "final_lambda": res.get("final_lambda", float("nan")),
        "pre_tanh_mean_abs": res.get("train/pre_tanh_mean_abs", float("nan")),
        "violation_rate": res.get("violation_rate", float("nan")),
        "cost_limit": args.budget,
        "seconds": round(time.time() - t0, 1),
    }
    (out / f"{args.arm}_{args.config}_s{args.seed}.json").write_text(
        json.dumps(row, indent=1))
    print(f"{args.arm}/{args.config} seed {args.seed}: return {row['return']:9.2f}  "
          f"cost {row['episode_cost']:6.2f}  alpha {row['final_alpha']:.3f}  "
          f"lambda {row['final_lambda']:.3f}  pretanh {row['pre_tanh_mean_abs']:.2f}  "
          f"B^2 {row['pathwise_bias_sq']:.3e}  ({row['seconds']:.0f}s)")
    return 0


def sign_p(wins: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def cmd_aggregate(args) -> int:
    out = Path(args.out)
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*_s*.json"))]
    if not rows:
        print("no arm results in " + str(out))
        return 1
    rep_path = out / "surrogate_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    if rep:
        b = rep["incumbent_deadband"]
        print(f"surrogate skill {rep['skill']:.3f}  |  incumbent deadband "
              f"{b['return']:.2f} at {b['cost']:.2f} pump-hours "
              f"(budget {rep['budget']})\n")

    cfgs = sorted({r.get("config", "v1") for r in rows})
    by = {(a, c): {r["seed"]: r for r in rows
                   if r["arm"] == a and r.get("config", "v1") == c}
          for a in ARMS for c in cfgs}
    print(f"{'arm/config':<14} {'n':>3} {'return':>10} {'sd':>8} {'cost':>7} "
          f"{'alpha':>7} {'lambda':>8} {'pretanh':>8} {'satur':>6}")
    for a, c in [(a, c) for c in cfgs for a in ARMS]:
        vals = list(by[(a, c)].values())
        if not vals:
            continue
        rs = [r["return"] for r in vals]
        sat = sum(1 for r in vals if r.get("pre_tanh_mean_abs", 0) > 3.0)
        print(f"{a + '/' + c:<14} {len(rs):>3} {st.fmean(rs):>10.2f} "
              f"{(st.pstdev(rs) if len(rs) > 1 else 0):>8.2f} "
              f"{st.fmean([r['episode_cost'] for r in vals]):>7.2f} "
              f"{st.fmean([r['final_alpha'] for r in vals]):>7.3f} "
              f"{st.fmean([r.get('final_lambda', float('nan')) for r in vals]):>8.3f} "
              f"{st.fmean([r.get('pre_tanh_mean_abs', float('nan')) for r in vals]):>8.2f} "
              f"{sat:>3}/{len(rs)}")

    # The only comparison that isolates the rule: same surrogate, same seed,
    # alpha on versus alpha off.
    for c in cfgs:
        _paired(by, c, args)
    return 0


def _paired(by, c, args) -> None:
    shared = sorted(set(by[("adaptive", c)]) & set(by[("alpha0", c)]))
    if shared:
        d = [by[("adaptive", c)][s]["return"] - by[("alpha0", c)][s]["return"]
             for s in shared]
        n = len(d)
        mean = st.fmean(d)
        sd = st.stdev(d) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 and sd else float("nan")
        wins = sum(1 for x in d if x > 0)
        print(f"\n[{c}] adaptive - alpha0, paired on {n} seeds "
              f"(same surrogate):")
        print(f"  mean {mean:+.2f}  sd {sd:.2f}  SE {se:.2f}  "
              f"t={mean / se if se == se and se else float('nan'):.2f}")
        print(f"  wins {wins}/{n}  sign p={sign_p(wins, n):.3f}")
        if all(abs(by[('adaptive', c)][s]['final_alpha']) < 1e-9 for s in shared):
            print("  NOTE: alpha is 0.000 on every seed, so the two arms are "
                  "the same estimator and this comparison measures nothing "
                  "but seed noise.")
        # The failure this guard exists for, which the alpha check missed.
        same = [s for s in shared
                if abs(by[("adaptive", c)][s]["return"]
                       - by[("alpha0", c)][s]["return"]) < 1e-6]
        if same:
            print(f"  WARNING: {len(same)}/{len(shared)} seeds return an "
                  f"IDENTICAL score for both arms. Two policies do not agree "
                  f"to float precision by chance -- they have collapsed onto "
                  f"the same action-space corner, where the evaluation is the "
                  f"same trajectory whatever gradient produced it. Alpha is "
                  f"then being measured on a policy that cannot move, and the "
                  f"paired comparison above is meaningless for those seeds.")
        hot = [s for s in shared
               if by[("adaptive", c)][s].get("pre_tanh_mean_abs", 0) > 3.0]
        if hot:
            print(f"  WARNING: pre-tanh magnitude above 3 on {len(hot)}/"
                  f"{len(shared)} adaptive seeds: the squashing function is "
                  f"flat there, so the policy is a corner rather than a "
                  f"distribution.")
        over = [s for s in shared
                if by[("adaptive", c)][s]["episode_cost"] > 1.0001 * args.budget]
        if over:
            print(f"  WARNING: {len(over)}/{len(shared)} adaptive seeds end "
                  f"INFEASIBLE (cost > {args.budget}). A constrained method "
                  f"outside its constraint is not a weaker solution, it is "
                  f"not a solution, and its return does not belong in a "
                  f"ranking.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--config", choices=("v1", "v2"), default="v1",
                    help="v1 is the defaults every flood result so far used; "
                         "v2 turns on the saturation wall and the normalised "
                         "dual that PlannerConfig's own notes prescribe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/alpha_live")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--dx", type=float, default=120.0)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--fit-episodes", type=int, default=96)
    ap.add_argument("--fit-epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.fit:
        return cmd_fit(args)
    if args.aggregate:
        return cmd_aggregate(args)
    if args.arm:
        return cmd_arm(args)
    ap.error("one of --fit, --arm, --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
