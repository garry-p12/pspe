"""Fit a fast interaction surrogate to solver runs, with a conformal margin.

One implementation, used by the offline script that builds the district's
scenario library and by the service that plans an arbitrary location on demand.
They must agree: rule 13 in Part 7 exists because two numbers produced by
different code are not comparable, and a planner whose guarantee means one thing
in Richmond and another in Iowa is worse than one with no guarantee at all.

The model, for a plan with per-site normalised contributions g_i:

    lead = max_i g_i
    effect = lead + S * tanh((sum_i g_i - lead) / S)

Single-measure effects are exact by construction -- they come straight from a
solver run -- so everything the surrogate guesses about lives in the interaction
between measures, and the conformal quantile of that residual is the margin.

The lead term is not decoration. Saturating the WHOLE sum makes the model
discontinuous between one measure and two, so adding a measure that helps can
lower the prediction (defect 28). Only what is stacked on top of the largest
single contribution saturates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import margins


def effect(alpha: Sequence[float], heights: Sequence[float], max_h: float,
           S: float) -> float:
    """Predicted effect of a height vector. Exact for a single measure."""
    parts = [alpha[i] * (h / max_h) for i, h in enumerate(heights) if h > 0]
    if not parts:
        return 0.0
    lead = max(parts)
    return lead + S * math.tanh((sum(parts) - lead) / S)


def fit(alpha: Sequence[float], plans: Sequence[Sequence[float]],
        measured: Sequence[float], max_h: float, delta: float = 0.1) -> dict:
    """Fit S on the multi-measure runs and take a conformal margin of what is left.

    `alpha` is per-site effect at full height, exact from the single-measure
    runs. `plans` and `measured` cover every run; single-measure and do-nothing
    rows are skipped for calibration because they carry no interaction error.
    """
    rows = []
    for plan, m in zip(plans, measured):
        parts = [alpha[i] * (h / max_h) for i, h in enumerate(plan) if h > 0]
        if len(parts) < 2:
            continue
        lead = max(parts)
        rows.append({"lead": lead, "rest": sum(parts) - lead, "measured": m})

    # S large means "no saturation needed", which is the right answer when the
    # measures are small enough not to overlap. Report when the search lands on
    # its own boundary rather than presenting an edge value as a fitted one.
    best_S, best_err = 100.0, float("inf")
    grid = list(range(2, 400, 2)) + [500, 1000, 5000, 20000]
    for S in grid:
        err = math.sqrt(sum((r["lead"] + S * math.tanh(r["rest"] / S) - r["measured"]) ** 2
                            for r in rows) / max(len(rows), 1))
        if err < best_err:
            best_S, best_err = float(S), err
    saturating = best_S < grid[-1]

    resid = [abs(r["measured"] - (r["lead"] + best_S * math.tanh(r["rest"] / best_S)))
             for r in rows]
    n = len(rows)
    need = max(2, int(round(1.0 / delta)) - 1)
    attainable = n >= need
    band = margins.margin("residual", delta, residuals=resid) if attainable else 0.0

    return {
        "saturation_S": best_S,
        "saturating": saturating,
        "fit_rmse": best_err,
        "band": band,
        "delta": delta,
        "band_attainable": attainable,
        "n_calibration_runs": n,
        "band_note": (f"split-conformal {100 * (1 - delta):.0f}% quantile over "
                      f"{n} held-out combinations" if attainable else
                      f"NOT ATTAINABLE: {n} calibration runs, {need} needed for "
                      f"delta = {delta}; no margin is applied"),
    }


def enumerate_plans(k: int, max_h: float, seed: int = 0) -> list[list[float]]:
    """Do-nothing, each site alone, then a calibration set of combinations.

    Heights are varied because the calibration sample has to be exchangeable
    with what a planner actually proposes, and a planner spending a budget picks
    partial heights far more often than full ones.
    """
    import random

    rng = random.Random(seed)
    plans = [[0.0] * k]
    for i in range(k):
        h = [0.0] * k
        h[i] = max_h
        plans.append(h)

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    triples = [(i, j, l) for i in range(k) for j in range(i + 1, k)
               for l in range(j + 1, k)]
    combos = [(c, [max_h] * len(c)) for c in pairs]
    combos += [(c, [max_h] * len(c)) for c in rng.sample(triples, min(6, len(triples)))]
    combos += [(c, [rng.choice([0.5, 1.0, 1.5, 2.0, 2.5]) for _ in c])
               for c in rng.sample(pairs, min(6, len(pairs)))]
    if k >= 2:
        combos.append((tuple(range(k)), [max_h] * k))

    seen = {tuple(p) for p in plans}
    for combo, hs in combos:
        h = [0.0] * k
        for i, hi in zip(combo, hs):
            h[i] = hi
        if tuple(h) in seen:
            continue
        seen.add(tuple(h))
        plans.append(h)
    return plans
