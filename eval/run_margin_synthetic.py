#!/usr/bin/env python
"""Coverage of each conformal recipe, in a setting where everything is visible.

    python eval/run_margin_synthetic.py --out runs/margin_synthetic

The PDE and wildfire experiments show the recipes failing or holding on real
planners, where many things vary at once. This isolates the claim. Bias and
spread are dialled independently, nothing is learned, and the only quantity
measured is the one a conformal paper should report: does the realised failure
rate exceed the stated delta?

The setting is the minimal CMDP the argument is about. A policy has true mean
episode cost mu and episode-to-episode spread sigma. The model is wrong by a
systematic bias b and by per-instance noise eps:

    truth        c_i   = mu + n_i,            n_i ~ N(0, sigma^2)
    model, mean  ghat  = mu - b               <- what the dual holds at d_eff
    model, each  g_i   = c_i - b + e_i,       e_i ~ N(0, eps^2)

`g_i` is deliberately generous to the default recipe: it tracks each episode's
own realisation, so the matched pair c_i - g_i contains only the bias and a
little noise. That is the point. A dual that drives ghat to d_eff leaves the
realised cost at mu = d_eff + b, and an episode exceeds the true limit d
whenever n_i > d - d_eff - b.

What the algebra says, and it is not what we first guessed. With d_eff = d - q
the realised mean is d_eff + b, so an episode violates when n_i > q - b, and
coverage at level delta needs

    q  >=  b + z_delta * sigma,        z_delta = Phi^{-1}(1 - delta).

The matched-pair residual is c_i - g_i = b - e_i, so its quantile is about
b + z_delta * eps. The default therefore covers exactly when

    eps  >=  sigma,

that is, when the model's PER-INSTANCE error is at least as large as the
policy's episode-to-episode spread. The relevant ratio is rho = sigma / eps,
not spread over bias: bias shifts both the requirement and the quantile by the
same amount and drops out. Sweeping rho traces the crossover.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.plan import margins  # noqa: E402
from pspe.utils import project_path  # noqa: E402

ARMS = ("none", "model_error", "episode", "residual")


def trial(rng, mu, d, bias, sigma, eps, n_cal, n_test, delta):
    """One calibration/evaluation split. Returns realised violation rate per arm."""
    c = mu + rng.normal(0, sigma, n_cal)          # real episode costs (the probe)
    g = c - bias + rng.normal(0, eps, n_cal)      # per-instance model costs
    ghat = mu - bias                              # the quantity the dual controls

    m = {
        "none": 0.0,
        "model_error": margins.margin("model_error", delta, model_err=c - g),
        "episode": margins.margin("episode", delta,
                                  deviations=c - c.mean(), bias=bias),
        "residual": margins.margin("residual", delta, residuals=c - ghat),
    }
    out = {}
    for arm, q in m.items():
        d_eff = d - q
        # The dual holds the MODEL's mean at d_eff, so the realised mean sits a
        # bias above it. This is the silent-violation mechanism, in one line.
        realised_mean = d_eff + bias
        fresh = realised_mean + rng.normal(0, sigma, n_test)
        out[arm] = float((fresh > d).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--bias", type=float, default=0.10)
    ap.add_argument("--sigma", type=float, default=0.10,
                    help="policy episode-cost spread, held fixed while eps varies")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--rho", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0],
                    help="sigma / eps: policy spread over per-instance model error")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/margin_synthetic")
    args = ap.parse_args()

    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for rho in args.rho:
        # rho = sigma / eps. Spread is held fixed and the model's per-instance
        # accuracy is varied, which is what improving a surrogate actually does.
        sigma = args.sigma
        eps = sigma / rho
        acc = {a: [] for a in ARMS}
        for _ in range(args.trials):
            r = trial(rng, args.mu, args.limit, args.bias, sigma, eps,
                      args.n_cal, args.n_test, args.delta)
            for a in ARMS:
                acc[a].append(r[a])
        row = {"rho = spread/model err": rho, "spread sigma": round(sigma, 4),
               "per-instance model err eps": round(eps, 4), "bias": args.bias}
        for a in ARMS:
            v = np.array(acc[a])
            row[a] = round(float(v.mean()), 4)
            # Fraction of calibration draws whose realised rate exceeds delta.
            # Split conformal is marginal over the draw, so some excess is
            # expected; a recipe that is structurally wrong sits near 1.
            row[f"{a} P(rate>delta)"] = round(float((v > args.delta).mean()), 3)
        rows.append(row)
        print(f"rho {rho:5.2f}  " + "  ".join(
            f"{a} {row[a]:.3f}" for a in ARMS), flush=True)

    table = markdown_table(rows)
    (root / "results.md").write_text(
        f"# Coverage by recipe, controlled setting (delta = {args.delta})\n\n"
        f"{args.trials} calibration draws per row, n_cal = {args.n_cal}, "
        f"n_test = {args.n_test}, bias = {args.bias}, policy spread "
        f"= {args.sigma}. Entries are realised violation rates; a correct recipe "
        f"sits at or below {args.delta}.\n\n" + table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
