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

MODES = ("model_error", "episode", "residual", "adaptive")


def adaptive_quantile(sequence: Sequence[float], delta: float,
                      gamma: float = 0.02) -> tuple[float | None, float]:
    """Gibbs & Candes adaptive conformal, replayed over an ORDERED sequence.

    Split conformal assumes the calibration scores are exchangeable. Section 9.3a
    measured that assumption on rdf and it FAILS: the lag-1 permutation test
    rejects on all three seeds (p = 0.000, 0.002, 0.000), because consecutive
    probes see consecutive policies. The realised coverage there was 0.1104
    against a requested 0.1000, and this recipe landed on 0.1000 exactly -- which
    is why it is promoted here from a diagnostic in run_exchangeability.py to a
    margin anyone can request.

    The level moves against realised error,

        delta_{t+1} = delta_t + gamma * (delta - err_t)

    so a widening score distribution tightens the quantile instead of silently
    undercovering. Unlike the other modes this one needs the scores IN ORDER;
    passing a shuffled sequence silently turns it back into split conformal.

    Returns (margin, mean level used). The margin is None when no prefix is long
    enough to carry the level, matching `conformal_quantile`.
    """
    seq = list(sequence)
    n = len(seq)
    if n < 4:
        return None, delta
    start = max(2, n // 2)
    d_t, used, q = delta, [], None
    for i in range(start, n):
        d_use = min(max(d_t, 1e-3), 0.5)
        used.append(d_use)
        q = conformal_quantile(seq[:i], d_use)
        err = 1.0 if (q is None or seq[i] > q) else 0.0
        d_t = d_t + gamma * (delta - err)
    # The margin to deploy is the quantile at the level the run converged to.
    d_final = min(max(d_t, 1e-3), 0.5)
    q = conformal_quantile(seq, d_final)
    return q, (sum(used) / len(used) if used else delta)


def min_calibration(delta: float) -> int:
    """Smallest n for which some order statistic carries level `delta`.

    Solved by searching the rank condition `ceil((n+1)(1-delta)) <= n` rather
    than by rearranging it to `n >= 1/delta - 1`, because both sides of this
    project got the rearrangement wrong in opposite directions and a parity
    check between them is what found it:

      * here it was `round(1/delta) - 1`, which under-counts whenever `1/delta`
        is not an integer -- at delta = 0.30, 0.12, 0.08 it admits an `n` whose
        rank exceeds `n`, where the old `min(n, rank)` clamp then returned the
        SAMPLE MAXIMUM. That is exactly the theatre this function exists to
        refuse, and it refused it everywhere except the values where it
        mattered.
      * the portal's mirror used `ceil(1/delta - 1)` on a delta obtained as
        `1 - confidence/100`, which in binary floating point makes delta =
        0.09999999999999998 and the expression overshoot by one -- so it
        declined delta = 0.1 at n = 9, where 9 is sufficient.

    Searching the condition is exact, needs no algebra, and cannot drift: it is
    the definition. delta = 0.1 gives 9, 0.05 gives 19, 0.3 gives 3.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    n = 2
    while n < 10_000:
        if math.ceil((n + 1) * (1.0 - delta)) <= n:
            return n
        n += 1
    raise ValueError(f"no attainable calibration size for delta = {delta}")


def conformal_quantile(sample: Sequence[float], delta: float) -> float | None:
    """Upper (1-delta) order statistic, or None if the level is unattainable.

    ceil((n+1)(1-delta)) <= n requires n >= 1/delta - 1. Below that no finite
    order statistic carries the guarantee and returning a number would be
    theatre, so the caller is told to apply no margin instead.
    """
    n = len(sample)
    if n < min_calibration(delta):
        return None
    rank = math.ceil((n + 1) * (1.0 - delta))
    # The guard above guarantees rank <= n, so no clamp is needed. The clamp
    # that used to be here was load-bearing precisely when the guard was wrong,
    # and silently returned the sample maximum -- see `min_calibration`.
    return sorted(sample)[rank - 1]


def conditional_quantile(
    strata: "dict[object, Sequence[float]]", delta: float,
) -> "dict[object, float | None]":
    """Split-conformal quantile within each stratum -- Mondrian conformal.

    Pooling residuals across strata gives MARGINAL coverage: the limit holds
    over the whole population and may be violated systematically inside any
    subgroup. When the subgroup is chosen by the planner, that is the wrong
    guarantee, because the planner can walk into the subgroup where the pooled
    band is too narrow and the pooled number still looks valid.

    The portal's levee task is exactly that case. Surrogate residuals grow with
    the number of active sites -- mean |measured - predicted| of 4.74 over 21
    two-site runs, 8.57 over 8 three-site runs, 11.50 on the single six-site run
    -- so a pooled band is too wide for a two-site plan and too narrow for a
    six-site one, and a search maximising predicted effect prefers more sites.
    It walks into the stratum its own margin does not cover.

    Validity: the taxonomy must be a function of the INPUT only, never of the
    outcome. `n_active` is a property of the height vector, so conditioning on
    it is legitimate and each stratum keeps the usual exchangeability
    requirement among its own members.

    The price is sample size. Each stratum must independently satisfy the rank
    condition, so a stratum below `min_calibration(delta)` gets `None` and the
    caller must refuse for plans landing there rather than borrow a neighbour's
    band. That refusal is the honest output: it says the evidence covers
    two-site plans at this level and does not cover six-site ones.
    """
    return {k: conformal_quantile(list(v), delta) for k, v in strata.items()}


def margin(mode: str, delta: float, *, model_err: Sequence[float] = (),
           deviations: Sequence[float] = (), residuals: Sequence[float] = (),
           bias: float = 0.0, gamma: float = 0.02) -> float:
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
    if mode == "adaptive":
        # Ordered residuals: the sequence is the point (see adaptive_quantile).
        q, _ = adaptive_quantile(list(residuals), delta, gamma)
        return max(0.0, q) if q is not None else 0.0
    q = conformal_quantile(list(deviations), delta)
    return 0.0 if q is None else max(0.0, q) + max(0.0, bias)
