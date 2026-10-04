#!/usr/bin/env python
"""Does closing the surrogate/reality cost gap fix the constraint violation?

    python eval/run_constraint_fix.py --seed 0 --out runs/constraint_fix/seed_0

The failure: the dual controller is driven by cost measured on *surrogate*
rollouts while violation is realised under the *true* dynamics. On `dar` the
planner exceeded a 0.936 limit on 7.3% of evaluations while every model-free
baseline exceeded it on none.

Four arms, chosen so the two mechanisms can be separated rather than confounded:

* **baseline**    - the current planner. The failure, reproduced.
* **probe**       - dual driven by real-environment cost every N iterations,
                    with the surrogate's cost bias-corrected in between.
* **margin**      - no probe; plan against a limit tightened by a fixed margin.
                    Tests whether conservatism alone is enough, without paying
                    any real samples.
* **probe+margin** - both.

The comparison is not only "did violations stop". A planner can satisfy any
constraint by refusing to act, so the table reports return alongside violation,
and the real-sample cost each arm pays: the whole claim against model-free
baselines is sample efficiency, and probe transitions are real interaction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig,
    SimulateTrainer,
    ensure_dataset,
    make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

def arms(probe_every: int, margin_k: float) -> dict[str, dict]:
    return {
        "baseline":     dict(real_cost_every=0,           cost_margin_k=0.0),
        "probe":        dict(real_cost_every=probe_every, cost_margin_k=0.0),
        "margin":       dict(real_cost_every=0,           cost_margin_k=margin_k),
        "probe+margin": dict(real_cost_every=probe_every, cost_margin_k=margin_k),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="surrogate epochs")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-every", type=int, default=20,
                        help="iterations between real-environment cost probes. Must be "
                             "< --iterations or the probe arms never fire and every arm "
                             "silently collapses onto the baseline")
    parser.add_argument("--margin-k", type=float, default=2.0)
    # Saturation guard (see PlannerConfig). Defaults reproduce the original
    # runs; `--sat-coef 1 --adv-floor 1e-2 --dual-ema 0.7` is the rdf fix.
    parser.add_argument("--sat-coef", type=float, default=0.0)
    parser.add_argument("--sat-margin", type=float, default=1.5)
    parser.add_argument("--adv-floor", type=float, default=1e-6)
    parser.add_argument("--dual-ema", type=float, default=0.9)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--dual-normalize", action="store_true")
    parser.add_argument("--lr-policy", type=float, default=3e-4)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--ki", type=float, default=0.05)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--margin-episode-std", action="store_true")
    parser.add_argument("--margin-conformal", action="store_true",
                        help="replace the k-sigma margin with a split-conformal quantile of the "
                             "probe's cost errors, which carries a stated failure rate")
    parser.add_argument("--margin-delta", type=float, default=0.1)
    parser.add_argument("--task-target", type=float, default=None)
    parser.add_argument("--task-umax", type=float, default=None)
    parser.add_argument("--task-budget", type=float, default=None)
    parser.add_argument("--task-cost-limit", type=float, default=None,
                        help="override TASK_SPECS for this run only")
    parser.add_argument("--margin-mode", default="episode",
                        choices=["model_error", "episode", "residual", "adaptive"],
                        help="which distribution the conformal quantile is taken "
                             "over. 'adaptive' is Gibbs-Candes online conformal, "
                             "which §9.3a promoted from diagnostic to arm after "
                             "the lag-1 exchangeability test rejected on rdf "
                             "(p = 0.000/0.002/0.000) while adaptive coverage "
                             "landed on 0.1000 against a requested 0.1.")
    parser.add_argument("--out", default="runs/constraint_fix")
    # The matched-budget comparison. §8.3 gives the model-free four 38,400 real
    # transitions against the planner's 4,160 and still reports a better return
    # for the planner -- but the planner violates a limit they respect, so the
    # document concedes "part of the advantage was bought". Capping them at the
    # planner's own budget, in this same seed loop, is what makes the claim
    # matched instead of assembled from two tables of two different runs.
    parser.add_argument("--null-arm", action="store_true", default=True,
                        help="evaluate the zero policy as an arm. On by default: "
                             "it is free and it is the row that decides whether "
                             "any other row demonstrates control")
    parser.add_argument("--no-null-arm", dest="null_arm", action="store_false")
    parser.add_argument("--baselines-only", action="store_true",
                        help="skip the planner arms; run the null row and the "
                             "model-free baselines at --baseline-budget. For "
                             "re-measuring §8.3 at full budget with varied seeds")
    parser.add_argument("--baselines", action="store_true",
                        help="also run the four model-free safe-RL baselines at "
                             "the planner's real-sample budget, same seed")
    parser.add_argument("--baseline-budget", type=int, default=0,
                        help="real transitions per baseline; 0 matches the planner")
    parser.add_argument("--baseline-batch", type=int, default=16,
                        help="baseline rollout batch; budget = iters x batch x horizon")
    args = parser.parse_args()

    # The probe guard is about the PLANNER arms, which --baselines-only does not
    # run, so it must not reject a baselines-only invocation.
    if args.probe_every >= args.iterations and not args.baselines_only:
        raise SystemExit(
            f"--probe-every {args.probe_every} >= --iterations {args.iterations}: "
            "the probe would never fire and all four arms would be identical."
        )

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    # One surrogate, shared by every arm: the arms must differ in how the dual
    # is driven, not in how good their model of the world happens to be.
    seed_everything(args.seed)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    # The model-free baselines train on the TRUE environment and never touch the
    # surrogate, so --baselines-only skips fitting it. Guarding only the arms
    # loop left this running: wasted GPU time on the full-budget tasks, and it
    # made the path untestable off-GPU because the local FNO backward raises
    # "norm ops are not supported for complex yet" in clip_grad_norm_.
    surrogate, sim_summary = None, {}
    if not args.baselines_only:
        surrogate = make_surrogate("fno", spec.n_channels, grid=args.grid)
        sim = SimulateTrainer(
            surrogate,
            SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                                epochs=args.epochs, grid=args.grid),
            data, RunLogger(root / "surrogate", use_tensorboard=False), device,
        )
        sim_summary = sim.train()
        surrogate = sim.model
        for q in surrogate.parameters():
            q.requires_grad_(False)
        print(f"surrogate rel L2 {sim_summary['rel_l2_final']:.4f}")
    else:
        print("--baselines-only: no surrogate fitted; the baselines train on "
              "the true environment.", flush=True)

    # Real transitions spent fitting the surrogate — the denominator of the
    # sample-efficiency claim, and what the probe arms add to.
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    # Task overrides go to make_task, so a revived testbed spec can be passed on
    # the command line instead of editing TASK_SPECS and invalidating every
    # existing result for that family. §3.4a revived swe at horizon 96 with
    # target 0.12, u_max 0.14, budget 0.60; this is how that spec is exercised.
    task_over = {}
    if args.task_target is not None:     task_over["target_value"] = args.task_target
    if args.task_umax is not None:       task_over["u_max"] = args.task_umax
    if args.task_budget is not None:     task_over["budget"] = args.task_budget
    if args.task_cost_limit is not None: task_over["cost_limit"] = args.task_cost_limit

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      **task_over,
                      n_actuators=9, device=device, batched=True)
    rows = []
    planner_arms = {} if args.baselines_only else arms(args.probe_every, args.margin_k)
    if args.baselines_only:
        print("--baselines-only: skipping the planner arms. Used for the "
              "full-budget correction of §8.3. The null row still runs: it "
              "is free and it is the floor these baselines are measured "
              "against, so every output table stays self-contained.", flush=True)
    for name, overrides in planner_arms.items():
        seed_everything(args.seed)
        train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
        eval_env = make_env(dynamics="truth", **env_kwargs)
        policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
        trainer = HybridPlannerTrainer(
            train_env, policy,
            cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                              seed=args.seed, saturation_coef=args.sat_coef,
                              saturation_margin=args.sat_margin,
                              advantage_std_floor=args.adv_floor,
                              dual_ema=args.dual_ema, lambda_init=args.lambda_init,
                              dual_normalize=args.dual_normalize, lr_policy=args.lr_policy,
                              kp=args.kp, ki=args.ki, kd=args.kd,
                              margin_episode_std=args.margin_episode_std,
                              margin_conformal=args.margin_conformal,
                              margin_mode=args.margin_mode,
                              margin_delta=args.margin_delta,
                              **overrides),
            eval_env=eval_env,
            logger=RunLogger(root / name, use_tensorboard=False),
            device=device,
            surrogate_train_transitions=fit_transitions,
        )
        s = trainer.train()
        rows.append({
            "arm": name,
            "return": round(s["return"], 4),
            "episode_cost": round(s["episode_cost"], 4),
            "cost_limit": s["cost_limit"],
            "violating_evals": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
            "worst_cost": round(s.get("eval/cost_max_over_run", float("nan")), 4),
            "cost_bias": round(s.get("dual/final_cost_bias", 0.0), 4),
            "effective_limit": round(s.get("dual/effective_limit", s["cost_limit"]), 4),
            # A margin that consumes the whole limit leaves the planner nothing
            # to spend, and a violation under that is a property of the TASK,
            # not a failure of coverage. `run_flood_margin.py` has guarded this
            # since section 5.11; this driver did not, so swe's 20-seed result
            # pooled five infeasible seeds with fifteen feasible ones and read
            # as 16.8% against delta = 0.1 when the feasible pool is 12.1% with
            # an interval containing it (defect 39, section 3.4b).
            "margin_exceeds_limit": bool(
                s.get("dual/effective_limit", s["cost_limit"]) <= 1e-9
                and s["cost_limit"] > 1e-9),
            "real_samples": s["samples_real_env"],
            "probe_samples": s.get("samples_real_probe", 0),
            "final_lambda": round(s.get("final_lambda", 0.0), 3),
            "pre_tanh_abs": round(s.get("train/pre_tanh_mean_abs", 0.0), 3),
        })
        print(f"[{name}] return {rows[-1]['return']} "
              f"violating {rows[-1]['violating_evals']} "
              f"real samples {rows[-1]['real_samples']}"
              + ("  INFEASIBLE: the margin consumes the whole limit, so this "
                 "seed's violation rate is not a coverage measurement"
                 if rows[-1]["margin_exceeds_limit"] else ""))

    # ---------------------------------------------------------------------- #
    # Model-free safe RL, AT OUR REAL-SAMPLE BUDGET.
    #
    # Two honest ways to match the comparison: hold performance fixed and count
    # the samples each method needs, or hold the sample budget fixed and compare
    # performance. This is the second, because it is one config change -- cap
    # the baselines at the planner's real-sample count and let them train on the
    # true environment, which is the only one they have.
    #
    # They run in the SAME seed loop, testbed and limit as the planner arms, so
    # the comparison is seed-paired rather than cross-table.
    #
    # PRE-REGISTERED PREDICTION, recorded before the first run and in the style
    # this project used for SafeDreamer: at ~4,200 real transitions the
    # model-free four are nowhere near converged (§8.3 runs them for nine times
    # that), so they should lose to the planner on return AND fail to hold the
    # constraint they satisfy at full budget -- their 0% is bought with data,
    # not with the algorithm. If any of them still holds 0% here, the
    # sample-efficiency claim is weaker than §8.3 implies and that is the
    # result that must be reported.
    # The row this table has never had: DO NOTHING.
    #
    # Free, instant, and it decides whether any of the other rows mean anything.
    # Measured on the published IC set (dar, grid 64, d = 0.936) the zero policy
    # returns -2.5645 at cost 0.2735 -- so Saute's -2.537 is the null policy plus
    # 0.027, CPO's -2.541 plus 0.024, PPO-Lagrangian's -2.558 plus 0.007, each
    # bought with 38,400 real transitions. Averaged over five IC draws instead
    # of the one the published runs shared, do-nothing returns -2.3822 and beats
    # all four. The sign of "does safe RL beat doing nothing here" depends on the
    # draw, which is why this row has to be in the table and why the comparison
    # has to be seed-paired.
    #
    # A method that does not clear the zero policy has not demonstrated control,
    # whatever its violation rate -- and 0% violation is exactly what the zero
    # policy scores (Rule 57).
    if args.null_arm:
        seed_everything(args.seed)
        nenv = make_env(dynamics="truth", **env_kwargs)
        st = nenv.reset(8, torch.Generator().manual_seed(args.seed + 10_000))
        R = torch.zeros(8, device=st.device); C = torch.zeros(8, device=st.device)
        for _ in range(args.horizon):
            st, r, c, _ = nenv.step(torch.zeros(8, nenv.action_dim, device=st.device))
            R += r; C += c
        lim = nenv.task.cost_limit
        rows.append({
            "arm": "do-nothing",
            "return": round(float(R.mean()), 4),
            "episode_cost": round(float(C.mean()), 4),
            "cost_limit": lim,
            "violating_evals": 0.0 if float(C.mean()) <= lim else 1.0,
            "worst_cost": round(float(C.max()), 4),
            "real_samples": 0,
            "null_policy": True,
        })
        print(f"[do-nothing] return {rows[-1]['return']} "
              f"cost {rows[-1]['episode_cost']} (limit {lim}) "
              f"real samples 0  <- every other arm must clear this", flush=True)

    if args.baselines:
        from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent

        planner_real = max((r.get("real_samples", 0) for r in rows), default=0)
        budget = args.baseline_budget or int(planner_real)
        if budget <= 0:
            raise SystemExit(
                "no real-sample budget: --baselines-only needs an explicit "
                "--baseline-budget, since there is no planner arm to match")
        iters = max(1, budget // (args.baseline_batch * args.horizon))
        actual = iters * args.baseline_batch * args.horizon
        print(f"\n=== model-free baselines at a matched budget ===\n"
              f"  planner used {planner_real} real transitions; baselines get "
              f"{iters} x {args.baseline_batch} x {args.horizon} = {actual}",
              flush=True)
        for alg in sorted(ALGORITHMS):
            seed_everything(args.seed)
            benv = make_env(dynamics="truth", **env_kwargs)
            agent = make_agent(
                alg, benv,
                cfg=SafeRLConfig(iterations=iters, batch=args.baseline_batch,
                                 horizon=args.horizon, seed=args.seed,
                                 log_dir=str(root / f"baseline_{alg}")),
                logger=RunLogger(root / f"baseline_{alg}", use_tensorboard=False),
                device=device,
            )
            bs = agent.train()
            rows.append({
                "arm": f"{alg} @matched",
                "return": round(bs["return"], 4),
                "episode_cost": round(bs.get("episode_cost", float("nan")), 4),
                "violating_evals": round(
                    bs.get("eval/violating_eval_fraction", float("nan")), 4),
                "worst_cost": round(bs.get("eval/cost_max_over_run", float("nan")), 4),
                "real_samples": bs.get("samples", actual),
                "model_free": True,
            })
            print(f"[{alg} @matched] return {rows[-1]['return']} "
                  f"violating {rows[-1]['violating_evals']} "
                  f"real samples {rows[-1]['real_samples']}", flush=True)

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
