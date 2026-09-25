#!/usr/bin/env python
"""Which distribution should a conformal margin bound? A head-to-head.

    python eval/run_margin_choice.py --testbed rdf --seed 0 --out runs/margin_choice

The claim under test is not "our margin is better". It is that the right choice
depends on *where the error lives*, and that the field's default answer is the
wrong one in a regime that is easy to fall into and hard to notice.

Two mechanisms push a realised episode cost above the limit:

    bias    the surrogate systematically under-predicts cost, so a dual that
            is satisfied in the model is not satisfied in reality;
    spread  even with a perfect model, a dual that holds the MEAN episode cost
            at the limit is exceeded by individual episodes about half the time.

Four arms, differing only in what the conformal quantile is taken over:

    none          no margin. The failure, reproduced.
    model_error   the published default: matched-pair residuals, same initial
                  condition and same action noise rolled in both environments.
                  Bounds `bias`. The policy's own spread appears in both terms
                  and cancels, so this is blind to `spread` by construction.
    episode       quantile of deviations from the probe mean, plus the measured
                  bias as a separate deterministic term. Covers both, in two
                  pieces, and the bias piece carries no coverage statement.
    residual      one quantile of (realised cost - the quantity the dual
                  controls). Covers both in a single order statistic.

Surrogate fidelity is swept by training budget, so the same code produces the
regime where the default should win (an inaccurate model, bias dominant) and
the regime where it fails (an accurate model, spread dominant). The predicted
ordering is by rho = spread / bias: below 1 the default suffices, above it the
default undercovers. That is the diagnostic, and it is what the paper was
missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

# name -> (apply a margin at all, margin_mode)
ARMS = {
    "none":        (False, "residual"),
    "model_error": (True,  "model_error"),
    "episode":     (True,  "episode"),
    "residual":    (True,  "residual"),
}


def build_surrogate(testbed, grid, epochs, frac, root, seed, device):
    """A surrogate at a chosen fidelity. Returns (model, rel_l2, transitions)."""
    seed_everything(seed)
    data = ensure_dataset(testbed, grid=grid)
    n_all = data["states"].shape[0]
    if frac < 1.0:                       # fewer trajectories -> a worse operator
        # The floor was 8, which silently collapsed every fraction below 0.09
        # onto the same 8 trajectories: two "different" fidelity levels trained
        # on identical data and returned identical rel L2. The count is printed
        # so a repeat is visible rather than inferred.
        keep = max(4, int(n_all * frac))
        data = {k: (v[:keep] if hasattr(v, "shape") and v.shape[0] == n_all else v)
                for k, v in data.items()}
    else:
        keep = n_all
    print(f"  [fidelity] {epochs} epochs on {keep}/{n_all} trajectories", flush=True)
    spec = make_testbed(testbed, grid=grid)
    model = make_surrogate("fno", spec.n_channels, grid=grid)
    sim = SimulateTrainer(
        model,
        SimulateTrainConfig(testbed=testbed, surrogate="fno", epochs=epochs, grid=grid),
        data, RunLogger(root, use_tensorboard=False), device,
    )
    summary = sim.train()
    model = sim.model
    for p in model.parameters():
        p.requires_grad_(False)
    transitions = int(data["states"].shape[0] * data["states"].shape[1])
    return model, float(summary["rel_l2_final"]), transitions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="rdf", choices=["dar", "swe", "rdf"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--probe-episodes", type=int, default=16)
    ap.add_argument("--margin-delta", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=20,
                    help="iterations between evaluations. At the default, 200 "
                         "iterations give 11 evaluations and every violation rate "
                         "is a multiple of 1/11, which cannot resolve a delta of "
                         "0.1. Use 5 for ~41 per run.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    # Fidelity levels as "epochs:fraction-of-trajectories". Fewer of either
    # gives a worse operator and therefore a larger cost bias.
    ap.add_argument("--fidelity", nargs="+", default=["1:0.1", "3:0.3", "20:1.0"])
    # rdf needs the planner fixes; dar does not care.
    ap.add_argument("--sat-coef", type=float, default=1.0)
    ap.add_argument("--adv-floor", type=float, default=1e-2)
    ap.add_argument("--dual-ema", type=float, default=0.7)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--kp", type=float, default=0.5)
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--out", default="runs/margin_choice")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)

    rows = []
    for level in args.fidelity:
        epochs, frac = int(level.split(":")[0]), float(level.split(":")[1])
        tag = f"fid{epochs}x{frac}"
        surrogate, rel_l2, fit_transitions = build_surrogate(
            args.testbed, args.grid, epochs, frac, root / tag / "surrogate",
            args.seed, device)
        print(f"\n=== fidelity {tag}: surrogate rel L2 {rel_l2:.4f} ===", flush=True)

        for name, (use_margin, mode) in ARMS.items():
            seed_everything(args.seed)
            train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
            eval_env = make_env(dynamics="truth", **env_kwargs)
            policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
            trainer = HybridPlannerTrainer(
                train_env, policy,
                cfg=PlannerConfig(
                    iterations=args.iterations, horizon=args.horizon, seed=args.seed,
                    eval_every=args.eval_every,
                    saturation_coef=args.sat_coef, advantage_std_floor=args.adv_floor,
                    dual_ema=args.dual_ema, kp=args.kp, ki=args.ki, kd=args.kd,
                    real_cost_every=args.probe_every,
                    real_cost_episodes=args.probe_episodes,
                    cost_margin_k=1.0 if use_margin else 0.0,
                    margin_conformal=use_margin, margin_mode=mode,
                    margin_delta=args.margin_delta),
                eval_env=eval_env,
                logger=RunLogger(root / tag / name, use_tensorboard=False),
                device=device, surrogate_train_transitions=fit_transitions,
            )
            s = trainer.train()
            bias = abs(float(s.get("dual/final_cost_bias", 0.0)))
            spread = float(trainer.probe_episode_std)
            # The diagnostic is the policy's spread against the model's
            # PER-INSTANCE error, not against its bias. Bias shifts both the
            # coverage requirement and the matched-pair quantile by the same
            # amount and cancels; what decides whether the default covers is
            # whether eps >= sigma. Crossover is at rho = 1 (see
            # eval/run_margin_synthetic.py).
            me = trainer.probe_model_err
            eps = float(np.std(me)) if len(me) > 1 else float("nan")
            rows.append({
                "fidelity": tag,
                "surrogate rel L2": round(rel_l2, 4),
                "arm": name,
                "violating": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
                "worst": round(s.get("eval/cost_max_over_run", float("nan")), 4),
                "return": round(s["return"], 4),
                "limit": s["cost_limit"],
                "d_eff": round(s.get("dual/effective_limit", s["cost_limit"]), 4),
                "|bias|": round(bias, 4),
                "spread sigma": round(spread, 4),
                "model err eps": round(eps, 4),
                "rho = sigma/eps": round(spread / (eps + 1e-9), 2),
                "seed": args.seed,
            })
            print(f"  [{name:12s}] violating {rows[-1]['violating']:.3f}  "
                  f"worst {rows[-1]['worst']:.3f}  return {rows[-1]['return']:.3f}  "
                  f"d_eff {rows[-1]['d_eff']:.3f}", flush=True)

    table = markdown_table(rows)
    (root / "results.md").write_text(
        f"# Which distribution should the margin bound? ({args.testbed}, seed {args.seed})\n\n"
        + table + f"\n\nTarget failure rate delta = {args.margin_delta}. "
        "`rho` is the policy's episode-cost spread over the surrogate's per-instance "
        "cost error: "
        "the model-error recipe is predicted to suffice when rho < 1 and to "
        "undercover when rho > 1.\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
