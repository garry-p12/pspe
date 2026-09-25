"""Conformal margins, and the question of what they should bound.

Three recipes appear in the literature and in this repository. They differ only
in which sample the quantile is taken over, and that difference decides whether
the stated failure rate holds.

Write the realised cost of evaluation i as c_i, and let the planner's model
predict cost g_i for the same instance. A margin tightens the limit to
d_eff = d - q, so that driving the *controlled* quantity to d_eff leaves the
realised cost below d with probability at least 1 - delta.

    model_error   q over |c_i - g_i|, matched pairs.
                  Correct when the controlled quantity is per instance, since
                  then {c > d} is implied by {c - g > q} given g <= d_eff.
                  In a CMDP the dual controls an EXPECTATION, so g_i is not
                  what is held at the limit, and the per-instance scatter that
                  actually breaches d cancels between c_i and g_i.

    episode       q over (c_i - mean c), plus max(0, bias) as a separate term.
                  Covers spread and bias, but in two pieces, only one of which
                  carries a coverage statement.

    residual      q over (c_i - ghat), where ghat is the quantity the dual
                  controls. One order statistic covers bias and spread
                  together, and the coverage argument needs one assumption.

`residual` reduces to `model_error` when the controlled quantity is per
instance (ghat = g_i), which is why the default is right in the pointwise
control settings it was developed for and wrong here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MODES = ("model_error", "episode", "residual")


def conformal_quantile(sample: Sequence[float], delta: float) -> float | None:
    """Upper (1-delta) order statistic, or None if the level is unattainable.

    ceil((n+1)(1-delta)) <= n requires n >= 1/delta - 1. Below that no finite
    order statistic carries the guarantee and returning a number would be
    theatre, so the caller is told to apply no margin instead.
    """
    n = len(sample)
    if n < max(2, int(round(1.0 / delta)) - 1):
        return None
    rank = min(n, math.ceil((n + 1) * (1.0 - delta)))
    return sorted(sample)[rank - 1]


def margin(mode: str, delta: float, *, model_err: Sequence[float] = (),
           deviations: Sequence[float] = (), residuals: Sequence[float] = (),
           bias: float = 0.0) -> float:
    """The margin for `mode`, or 0.0 when the level is not yet attainable.

    Margins are clipped at zero: a negative quantile would *loosen* the limit,
    which no recipe intends.
    """
    if mode not in MODES:
        raise ValueError(f"margin mode {mode!r} not in {MODES}")
    if mode == "model_error":
        q = conformal_quantile([abs(e) for e in model_err], delta)
        return max(0.0, q) if q is not None else 0.0
    if mode == "residual":
        q = conformal_quantile(list(residuals), delta)
        return max(0.0, q) if q is not None else 0.0
    q = conformal_quantile(list(deviations), delta)
    return 0.0 if q is None else max(0.0, q) + max(0.0, bias)
