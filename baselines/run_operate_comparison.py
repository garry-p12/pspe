"""Every controller we have, on the polder, against the same event.

The point of this script is the thing the first run of `FloodOperateEnv` got
wrong by omission: reaching for PPO-Lagrangian because it was nearest to hand
and reporting that as "the RL result". PPO-Lagrangian, CPO, Sauté and
primal-dual NPG are the OPPONENTS in this project. The method is the hybrid
planner in `pspe.plan.trainer` — pathwise and likelihood-ratio gradients mixed
at the variance-minimising coefficient, with a PID dual on the budget. A
comparison that leaves it out is not a comparison of anything.

Three groups, all scored on the same seed and the same budget:

* **fixed controllers** — do nothing, gate only, the solver's own level
  deadband, "pump while the gate is sealed", pump flat out. The third is the
  incumbent and the fourth is the one worth beating.
* **safe-RL baselines** — the four in `baselines.safe_rl`.
* **this project's planner** — `HybridPlannerTrainer`.

One caution about the hybrid planner here, measured and written up in
`pspe/envs/flood_env.py`: the gate's pathwise gradient is an orifice
derivative and diverges as the head closes, spanning twenty-five orders of
magnitude across an episode. `FloodOperateEnv(differentiable=False)` is
therefore the default and the planner runs on its likelihood-ratio branch,
which is the correct behaviour rather than a workaround — but it means alpha
is not exercised, and the pathwise arm needs either a surrogate or a float64
variance estimate before that experiment is meaningful.

    python3 -u baselines/run_operate_comparison.py --iterations 150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.safe_rl import SafeRLConfig, make_agent            # noqa: E402
from pspe.envs.flood_env import (                                 # noqa: E402
    FloodOperateEnv, FloodOperateTask, make_polder_bank,
)
from pspe.envs.flood_surrogate import (                           # noqa: E402
    FloodIntervalSurrogate, collect_transitions, fit_surrogate,
)
from pspe.plan.policy import FieldCritic, GaussianFieldPolicy      # noqa: E402
from pspe.plan.trainer import HybridPlannerTrainer, PlannerConfig  # noqa: E402

EVAL_SEED = 12345


def fixed_controllers(env):
    return {
        "do nothing": lambda s, e: torch.full((s.shape[0], e.action_dim), -1.0),
        "gate only": lambda s, e: torch.cat(
            [torch.ones(s.shape[0], 1),
             torch.full((s.shape[0], e.n_pumps), -1.0)], dim=1),
        "level deadband (incumbent)": lambda s, e: e.baseline_action(s),
        "pump while sealed": lambda s, e: e.sealed_aware_action(s),
        "pump flat out": lambda s, e: torch.ones(s.shape[0], e.action_dim),
    }


@torch.no_grad()
def score(env, act, episodes: int, seed: int = EVAL_SEED) -> dict:
    """Return and cost of one controller, on a fixed seed so rows compare."""
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
    st = env.episode_stats()
    return {"return": float(ret.mean()), "cost": float(cost.mean()),
            "wet_km2": float(st["wet_km2"].mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--dx", type=float, default=120.0)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--surrogate", type=str, default="",
                    help="a FloodIntervalSurrogate state_dict; adds the arm "
                         "where the pathwise branch actually exists")
    ap.add_argument("--fit-surrogate", type=int, default=0,
                    help="episodes of random-control data to fit one first")
    args = ap.parse_args()

    torch.set_num_threads(8)
    g = torch.Generator().manual_seed(0)
    bank = make_polder_bank(n=64, grid=args.grid, dx=args.dx, generator=g)

    def mk():
        return FloodOperateEnv(
            bank, task=FloodOperateTask(cost_limit=args.budget),
            horizon=args.horizon, control_minutes=30.0, dx=args.dx)

    env = mk()
    print(f"polder {bank['protected_ha']:.0f} ha · grid {args.grid} · "
          f"dx {args.dx:.0f} m · budget {args.budget} pump-hours\n")

    rows: list[dict] = []
    # Feasibility is marked, not left to the reader.
    #
    # Ranking by return alone flatters whichever controller ignores the
    # budget hardest: at 2.0 pump-hours "pump flat out" spends 6.0 and tops
    # the table. A constrained problem has one meaningful ranking and it is
    # over the FEASIBLE rows.
    print(f"{'controller':<30} {'return':>9} {'pump-h':>8} {'wet km2':>9}  budget")
    print("-" * 70)

    def mark(cost: float) -> str:
        return "ok " if cost <= args.budget + 1e-6 else "OVER"

    for name, act in fixed_controllers(env).items():
        r = score(env, act, args.episodes)
        r["feasible"] = bool(r["cost"] <= args.budget + 1e-6)
        rows.append({"group": "fixed", "name": name, **r})
        print(f"{name:<30} {r['return']:>9.2f} {r['cost']:>8.2f} "
              f"{r['wet_km2']:>9.3f}  {mark(r['cost'])}")

    cfg = SafeRLConfig(iterations=args.iterations, batch=args.batch,
                       horizon=args.horizon)
    for algo in ("ppo_lagrangian", "cpo", "saute", "primal_dual_npg"):
        t0 = time.time()
        try:
            out = make_agent(algo, mk(), cfg=cfg).train()
        except Exception as exc:                            # noqa: BLE001
            print(f"{algo:<30} FAILED: {str(exc)[:60]}")
            rows.append({"group": "baseline", "name": algo, "error": str(exc)[:200]})
            continue
        # `episode_cost`, not `eval/cost_mean_over_run`. The latter is the
        # mean over EVERY periodic evaluation during training, so pairing it
        # with `return` -- which is the final policy's -- scores one policy's
        # benefit against another policy's price. It matters whenever cost
        # drifts during training, which is exactly what a dual does. The
        # hybrid row below already read `episode_cost`, so until now the
        # opponents and the method were judged by different statistics in
        # the same table.
        r = {"return": out.get("return", float("nan")),
             "cost": out.get("episode_cost", float("nan")),
             "cost_mean_over_run": out.get("eval/cost_mean_over_run",
                                           float("nan")),
             "violation_rate": out.get("violation_rate", float("nan")),
             "seconds": round(time.time() - t0, 1),
             "samples": out.get("samples", 0)}
        r["feasible"] = bool(r["cost"] <= args.budget + 1e-6)
        rows.append({"group": "baseline", "name": algo, **r})
        print(f"{algo:<30} {r['return']:>9.2f} {r['cost']:>8.2f} "
              f"{'':>9}  {mark(r['cost'])}  viol {r['violation_rate']:.3f}"
              f"  ({r['seconds']:.0f}s)")

    # --- this project's planner ------------------------------------------- #
    t0 = time.time()
    penv = mk()
    pol = GaussianFieldPolicy(penv.obs_shape[0], penv.action_dim)
    trainer = HybridPlannerTrainer(
        penv, pol, FieldCritic(penv.obs_shape[0]),
        PlannerConfig(iterations=args.iterations, batch=args.batch,
                      horizon=args.horizon),
        eval_env=mk())
    try:
        out = trainer.train()
        r = {"return": out.get("return", float("nan")),
             "cost": out.get("episode_cost", float("nan")),
             "cost_mean_over_run": out.get("eval/cost_mean_over_run",
                                           float("nan")),
             # `final_alpha`, not `alpha`: the trainer reports the latter
             # only inside its per-iteration record, so reading "alpha" here
             # printed NaN and looked like an estimator fault for two runs.
             "alpha": out.get("final_alpha", float("nan")),
             "seconds": round(time.time() - t0, 1)}
        rows.append({"group": "ours", "name": "hybrid planner", **r})
        r["feasible"] = bool(r["cost"] <= args.budget + 1e-6)
        print(f"{'hybrid planner (ours)':<30} {r['return']:>9.2f} "
              f"{r['cost']:>8.2f} {'':>9}  {mark(r['cost'])}  "
              f"alpha {r['alpha']:.3f}  ({r['seconds']:.0f}s)")
    except Exception as exc:                                # noqa: BLE001
        print(f"{'hybrid planner (ours)':<30} FAILED: {str(exc)[:70]}")
        rows.append({"group": "ours", "name": "hybrid planner",
                     "error": str(exc)[:300]})

    feasible = [r for r in rows if r.get("feasible") and "return" in r]
    if feasible:
        best = max(feasible, key=lambda r: r["return"])
        print(f"\nbest FEASIBLE controller: {best['name']} "
              f"({best['return']:.2f} at {best['cost']:.2f} pump-hours)")

    # --- the planner with a pathwise branch that exists --------------------- #
    #
    # Everything above runs `differentiable=False`, so `alpha` is pinned to 0
    # and the hybrid estimator is likelihood-ratio with a critic baseline. The
    # mixing -- the part the method is named for -- only has something to mix
    # once the dynamics carry a usable gradient, which is what the surrogate is
    # for. Trained on the TRUTH and scored on the TRUTH; only the gradient
    # comes from the model.
    net = None
    if args.surrogate:
        net = FloodIntervalSurrogate()
        # `weights_only=True`: this loads a file named on the command
        # line, and pickle will execute whatever is in it. The default
        # flips in a future torch, and there is no reason to wait.
        net.load_state_dict(torch.load(args.surrogate, map_location="cpu",
                                       weights_only=True))
        net.eval()
    elif args.fit_surrogate:
        data = collect_transitions(mk(), episodes=args.fit_surrogate,
                                   generator=torch.Generator().manual_seed(1))
        net, rep = fit_surrogate(data, epochs=60,
                                 generator=torch.Generator().manual_seed(2))
        print(f"\nsurrogate: {rep.transitions} transitions, val RMSE "
              f"{rep.val_rmse_m:.4f} m vs persistence {rep.persistence_rmse_m:.4f} m, "
              f"skill {rep.skill:.3f}")

    if net is not None:
        t0 = time.time()
        senv = FloodOperateEnv(
            bank, task=FloodOperateTask(cost_limit=args.budget),
            horizon=args.horizon, control_minutes=30.0, dx=args.dx, surrogate=net)
        spol = GaussianFieldPolicy(senv.obs_shape[0], senv.action_dim)
        # eval_env is the TRUE solver. A planner that only looks good on its
        # own surrogate has not solved anything, which is the whole reason
        # `HybridPlannerTrainer` takes the pair.
        strainer = HybridPlannerTrainer(
            senv, spol, FieldCritic(senv.obs_shape[0]),
            PlannerConfig(iterations=args.iterations, batch=args.batch,
                          horizon=args.horizon),
            eval_env=mk())
        try:
            out = strainer.train()
            r = {"return": out.get("return", float("nan")),
                 "cost": out.get("episode_cost", float("nan")),
                 "alpha": out.get("final_alpha", float("nan")),
                 "seconds": round(time.time() - t0, 1)}
            r["feasible"] = bool(r["cost"] <= args.budget + 1e-6)
            rows.append({"group": "ours", "name": "hybrid planner + surrogate", **r})
            print(f"{'hybrid planner + surrogate':<30} {r['return']:>9.2f} "
                  f"{r['cost']:>8.2f} {'':>9}  {mark(r['cost'])}  "
                  f"alpha {r['alpha']:.3f}  ({r['seconds']:.0f}s)")
        except Exception as exc:                            # noqa: BLE001
            print(f"{'hybrid planner + surrogate':<30} FAILED: {str(exc)[:70]}")
            rows.append({"group": "ours", "name": "hybrid planner + surrogate",
                         "error": str(exc)[:300]})

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
