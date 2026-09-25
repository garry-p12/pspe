#!/usr/bin/env python
"""Does each recipe deliver the rate it states, across levels?

    python eval/run_margin_coverage.py --out runs/margin_coverage

The standard question of a conformal method, which the other experiments do not
answer: they fix delta = 0.1 and ask whether it holds. Here delta is swept, and
a recipe is calibrated if its realised violation rate tracks the diagonal.

Same controlled setting as `run_margin_synthetic.py`: a policy with true mean
cost mu and spread sigma, a model wrong by a systematic bias b and per-instance
noise eps, a dual that holds the model's estimate at d_eff. Two regimes are
swept because the claim is conditional:

    rho = sigma/eps = 0.25   the model's per-instance error dominates; the
                             matched-pair recipe should be calibrated here
    rho = 4.0                the policy's spread dominates; it should not be

A recipe above the diagonal is undercovering. Below it is conservative, which
costs return rather than safety.
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

ARMS = ("model_error", "episode", "residual")


def realised(rng, delta, mu, d, bias, sigma, eps, n_cal, n_test):
    c = mu + rng.normal(0, sigma, n_cal)
    g = c - bias + rng.normal(0, eps, n_cal)
    ghat = mu - bias
    q = {
        "model_error": margins.margin("model_error", delta, model_err=c - g),
        "episode": margins.margin("episode", delta, deviations=c - c.mean(), bias=bias),
        "residual": margins.margin("residual", delta, residuals=c - ghat),
    }
    out = {}
    for arm, m in q.items():
        fresh = (d - m + bias) + rng.normal(0, sigma, n_test)
        out[arm] = float((fresh > d).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.25, 4.0])
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--bias", type=float, default=0.10)
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/margin_coverage")
    args = ap.parse_args()

    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for rho in args.rhos:
        eps = args.sigma / rho
        for delta in args.deltas:
            acc = {a: [] for a in ARMS}
            for _ in range(args.trials):
                r = realised(rng, delta, 1.0, 1.0, args.bias, args.sigma, eps,
                             args.n_cal, args.n_test)
                for a in ARMS:
                    acc[a].append(r[a])
            row = {"rho": rho, "delta": delta}
            for a in ARMS:
                v = np.array(acc[a])
                row[a] = round(float(v.mean()), 4)
                row[f"{a} sd"] = round(float(v.std(ddof=1)), 4)
            rows.append(row)
            print(f"rho {rho:>5} delta {delta:>5}  " +
                  "  ".join(f"{a} {row[a]:.3f}" for a in ARMS), flush=True)

    (root / "results.json").write_text(json.dumps(rows, indent=2))
    (root / "results.md").write_text(
        f"# Realised violation rate against the stated level\n\n"
        f"{args.trials} calibration draws per point, n_cal {args.n_cal}, "
        f"n_test {args.n_test}. A calibrated recipe tracks the diagonal.\n\n"
        + markdown_table(rows) + "\n")
    print("\n" + markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
