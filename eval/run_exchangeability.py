#!/usr/bin/env python
"""Are the conformal margin's calibration scores exchangeable across a run?

    python eval/run_exchangeability.py --seeds 0 1 2 --out runs/exchangeability

Split conformal guarantees its rate under exchangeability of the calibration
scores. Section 3.2's margin calibrates on residuals collected *during training*,
and a learning policy is a distribution shift, so the assumption is not free.
Every coverage result we have (`run_margin_coverage.py`) draws its scores i.i.d.
by construction, which tests the estimator and not the assumption.

This script collects the real sequence and asks three things of it:

  1. Does the score distribution drift with the iteration it came from?
     Spearman rho of residual, and of |residual|, against iteration.
  2. Do early and late scores come from the same distribution? Two-sample KS
     between the first and last third.
  3. Is there serial dependence beyond the drift? Lag-1 autocorrelation against
     a permutation null, which is the exchangeability null stated directly:
     under exchangeability every ordering is equally likely, so the observed
     statistic should sit inside the permutation distribution.

Then the question a reviewer actually cares about, which the tests above only
motivate: **does the stated rate still hold?** Calibrating on an early prefix
and scoring the later scores is exactly how the margin is used in a run, so
the realised exceedance rate there is the honest coverage number. It is
reported against delta, and against the same quantity computed on a shuffled
sequence -- the difference between them is the cost of non-exchangeability,
with the estimator held fixed.

Where the fixed quantile loses the rate, adaptive conformal inference
[Gibbs & Candes, 2021] is the standard repair: carry delta_t and update it by
gamma * (delta - err_t) after each observation. It is measured here on the same
sequence, so the comparison isolates the update rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.plan.margins import conformal_quantile  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig, SimulateTrainer, ensure_dataset, make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


def collect_scores(args, device) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one conformal arm and return its calibration scores, in order."""
    seed_everything(args.seed)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    surrogate = make_surrogate("fno", spec.n_channels, grid=args.grid)
    root = project_path(args.out) / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    sim = SimulateTrainer(
        surrogate,
        SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                            epochs=args.epochs, grid=args.grid),
        data, RunLogger(root / "surrogate", use_tensorboard=False), device,
    )
    sim_summary = sim.train()
    surrogate = sim.model
    for p in surrogate.parameters():
        p.requires_grad_(False)

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    seed_everything(args.seed)
    train_env = make_env(dynamics="surrogate", surrogate=surrogate, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    trainer = HybridPlannerTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                          seed=args.seed, real_cost_every=args.probe_every,
                          cost_margin_k=args.margin_k, margin_conformal=True,
                          margin_mode="residual", margin_delta=args.delta),
        eval_env=eval_env,
        logger=RunLogger(root / "conformal", use_tensorboard=False),
        device=device,
    )
    summary = trainer.train()
    scores = np.asarray(trainer.probe_residual, dtype=float)
    iters = np.asarray(trainer.probe_iteration, dtype=float)
    meta = {"seed": args.seed, "n_scores": int(scores.size),
            "n_probes": int(len(set(iters.tolist()))),
            "surrogate_rel_l2": round(float(sim_summary["rel_l2_final"]), 4),
            "return": round(float(summary["return"]), 4)}
    np.savez_compressed(root / "scores.npz", scores=scores, iters=iters)
    return scores, iters, meta


def exchangeability_tests(scores: np.ndarray, iters: np.ndarray,
                          n_perm: int, rng: np.random.Generator) -> dict:
    """Drift, two-sample and serial-dependence, against the exchangeability null."""
    out: dict[str, float] = {}

    # 1. Drift with training progress, on the score and on its magnitude. The
    #    magnitude matters more: the quantile is an upper order statistic, so a
    #    widening spread breaks it even with the mean pinned.
    for label, y in (("score", scores), ("|score|", np.abs(scores))):
        rho, p = stats.spearmanr(iters, y)
        out[f"spearman_{label}"] = round(float(rho), 4)
        out[f"spearman_{label}_p"] = round(float(p), 5)

    # 2. First third against last third.
    k = max(1, len(scores) // 3)
    ks, p = stats.ks_2samp(scores[:k], scores[-k:])
    out["ks_first_vs_last_third"] = round(float(ks), 4)
    out["ks_p"] = round(float(p), 5)
    out["sd_first_third"] = round(float(scores[:k].std()), 4)
    out["sd_last_third"] = round(float(scores[-k:].std()), 4)

    # 3. Lag-1 autocorrelation against the permutation null. Under
    #    exchangeability every ordering is equally likely, so this IS the null.
    def lag1(x: np.ndarray) -> float:
        a, b = x[:-1] - x.mean(), x[1:] - x.mean()
        denom = float((x - x.mean()) @ (x - x.mean()))
        return float(a @ b / denom) if denom > 0 else 0.0

    obs = lag1(scores)
    null = np.array([lag1(rng.permutation(scores)) for _ in range(n_perm)])
    out["lag1_autocorr"] = round(obs, 4)
    out["lag1_perm_p"] = round(float((np.abs(null) >= abs(obs)).mean()), 5)
    return out


def resolution_note(n_calib: int, n_test: int, delta: float) -> str:
    """Can this split resolve the claim at all? (Defect 18.)

    Two ways it cannot. The realised rate is a multiple of 1/n_test, so a test
    set smaller than about 2/delta cannot distinguish delta from twice delta.
    And the quantile's rank is min(n, ceil((n+1)(1-delta))): once that clamps to
    n, the "quantile" is the sample maximum and the margin is maximally
    conservative by construction rather than by calibration.
    """
    notes = []
    if n_test < 2 / delta:
        notes.append(f"n_test={n_test} < 2/delta={2/delta:.0f}: rate resolution "
                     f"{1/max(n_test,1):.3f} against delta {delta}")
    if math.ceil((n_calib + 1) * (1 - delta)) >= n_calib:
        notes.append(f"n_calib={n_calib}: quantile rank clamps to the maximum")
    return "; ".join(notes)


def prefix_coverage(scores: np.ndarray, delta: float, frac: float) -> float | None:
    """Exceedance rate on the tail when the quantile is fitted on the prefix.

    This is how the margin is actually used: calibrate on what has been seen,
    apply to what comes next. Under exchangeability it should sit near delta.
    """
    k = int(round(frac * len(scores)))
    if k < 2 or k >= len(scores):
        return None
    q = conformal_quantile(scores[:k].tolist(), delta)
    if q is None:
        return None
    return float((scores[k:] > q).mean())


def adaptive_coverage(scores: np.ndarray, delta: float, frac: float,
                      gamma: float) -> tuple[float, float]:
    """Gibbs & Candes online update on the same sequence.

    delta_t moves against realised error, so a widening score distribution
    tightens the quantile instead of silently undercovering. Returns the
    realised exceedance rate and the mean delta_t it used.
    """
    k = int(round(frac * len(scores)))
    d_t, errs, deltas = delta, [], []
    for i in range(k, len(scores)):
        d_use = float(np.clip(d_t, 1e-3, 0.5))
        q = conformal_quantile(scores[:i].tolist(), d_use)
        err = 1.0 if (q is None or scores[i] > q) else 0.0
        errs.append(err)
        deltas.append(d_use)
        d_t = d_t + gamma * (delta - err)
    return float(np.mean(errs)), float(np.mean(deltas))


def aggregate(root: Path) -> int:
    """Merge per-seed results written by a job array into one table.

    One seed per node is the convention here and it is the robust one: a seed
    that overruns its wall clock costs that seed, not the experiment.
    """
    seeds = []
    for f in sorted(root.glob("seed_*/results.json")):
        seeds += json.loads(f.read_text())["seeds"]
    if not seeds:
        print(f"no per-seed results under {root}", flush=True)
        return 1
    rows = [{"seed": s["seed"], "n": s["n_scores"],
             "calib/test": f"{s['n_calib']}/{s['n_test']}",
             "spearman |score|": s["spearman_|score|"],
             "p": s["spearman_|score|_p"],
             "sd early": s["sd_first_third"], "sd late": s["sd_last_third"],
             "KS p": s["ks_p"], "lag1 p": s["lag1_perm_p"],
             "cover fixed": round(s["coverage_fixed"], 4),
             "cover shuffled": round(s["coverage_shuffled"], 4),
             "cover adaptive": round(s["coverage_adaptive"], 4)}
            for s in seeds]
    means = {k: round(float(np.mean([r[k] for r in rows])), 4)
             for k in ("cover fixed", "cover shuffled", "cover adaptive")}
    table = markdown_table(rows)
    (root / "results.md").write_text(table + f"\n\n{means}\n")
    (root / "results.json").write_text(json.dumps(
        {"seeds": seeds, **{f"mean_{k}": v for k, v in means.items()}}, indent=2, default=float))
    under = [s["seed"] for s in seeds if s.get("underpowered")]
    print("\n" + table)
    print(f"\n{means}")
    if under:
        print(f"UNDERPOWERED seeds: {under}", flush=True)
    print(f"written to {root / 'results.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--margin-k", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="fraction of the sequence used to fit the quantile")
    ap.add_argument("--gamma", type=float, default=0.02, help="adaptive step size")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/exchangeability")
    ap.add_argument("--aggregate", action="store_true",
                    help="merge seed_*/results.json under --out, run nothing")
    args = ap.parse_args()

    if args.aggregate:
        return aggregate(project_path(args.out))

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows, payload = [], []
    for seed in args.seeds:
        args.seed = seed
        scores, iters, meta = collect_scores(args, device)
        if scores.size < 20:
            print(f"[seed {seed}] only {scores.size} scores; skipping tests", flush=True)
            continue
        tests = exchangeability_tests(scores, iters, args.n_perm, rng)
        fixed = prefix_coverage(scores, args.delta, args.calib_frac)
        shuffled = prefix_coverage(rng.permutation(scores), args.delta, args.calib_frac)
        adaptive, mean_delta = adaptive_coverage(scores, args.delta, args.calib_frac, args.gamma)
        k = int(round(args.calib_frac * len(scores)))
        warn = resolution_note(k, len(scores) - k, args.delta)
        if warn:
            print(f"[seed {seed}] UNDERPOWERED -- {warn}", flush=True)
        rows.append({
            "seed": seed,
            "n": meta["n_scores"],
            "calib/test": f"{k}/{len(scores) - k}",
            "spearman |score|": tests["spearman_|score|"],
            "p": tests["spearman_|score|_p"],
            "sd early": tests["sd_first_third"],
            "sd late": tests["sd_last_third"],
            "KS p": tests["ks_p"],
            "lag1 p": tests["lag1_perm_p"],
            "cover fixed": round(fixed, 4) if fixed is not None else None,
            "cover shuffled": round(shuffled, 4) if shuffled is not None else None,
            "cover adaptive": round(adaptive, 4),
        })
        payload.append({**meta, **tests, "underpowered": warn or None,
                        "n_calib": k, "n_test": len(scores) - k,
                        "coverage_fixed": fixed,
                        "coverage_shuffled": shuffled,
                        "coverage_adaptive": adaptive,
                        "adaptive_mean_delta": round(mean_delta, 4)})
        print(f"[seed {seed}] {rows[-1]}", flush=True)

    if not rows:
        print("no seed produced enough calibration scores", flush=True)
        return 1

    def mean_of(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "delta": args.delta,
        "calib_frac": args.calib_frac,
        "gamma": args.gamma,
        "mean_coverage_fixed": mean_of("cover fixed"),
        "mean_coverage_shuffled": mean_of("cover shuffled"),
        "mean_coverage_adaptive": mean_of("cover adaptive"),
        "seeds": payload,
        "config": vars(args),
    }
    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n" + table)
    print(f"\ndelta {args.delta}   fixed {summary['mean_coverage_fixed']}   "
          f"shuffled {summary['mean_coverage_shuffled']}   "
          f"adaptive {summary['mean_coverage_adaptive']}")
    print(f"written to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
