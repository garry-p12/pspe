"""The conditional (Mondrian) band, and why pooling is the wrong guarantee here.

A pooled split-conformal band gives MARGINAL coverage: it holds across the
population of plans and may be violated systematically inside any subgroup.
When the planner chooses the subgroup, that is the wrong guarantee -- and on the
portal's levee task the planner chooses it adversarially, because predicted
effect rises with the number of active sites and so does the surrogate's error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pspe.plan import margins
from pspe.plan.surrogate import saturating

SURROGATE = Path("portal/public/data/surrogate.json")


def _strata(fit: dict) -> dict[int, list[float]]:
    S = fit["saturation_S"]
    out: dict[int, list[float]] = {}
    for p in fit["calibration"]:
        out.setdefault(p["n_active"], []).append(
            abs(p["measured"] - saturating(p["lead"], p["rest"], S)))
    return out


def test_conditional_quantile_refuses_a_thin_stratum():
    """Each stratum satisfies the rank condition on its own, or gets None."""
    need = margins.min_calibration(0.1)
    q = margins.conditional_quantile(
        {"fat": [float(i) for i in range(need + 5)],
         "exact": [float(i) for i in range(need)],
         "thin": [float(i) for i in range(need - 1)],
         "single": [1.0]},
        0.1,
    )
    assert q["fat"] is not None
    assert q["exact"] is not None, "a stratum at exactly the threshold must pass"
    assert q["thin"] is None, "a stratum one run short must refuse, not borrow"
    assert q["single"] is None


def test_conditional_quantile_matches_per_stratum_conformal():
    """It is split conformal within each cell, nothing more."""
    strata = {"a": [3.0, 1.0, 2.0] * 4, "b": [9.0, 7.0] * 6}
    q = margins.conditional_quantile(strata, 0.1)
    for k, v in strata.items():
        assert q[k] == margins.conformal_quantile(v, 0.1)


@pytest.mark.skipif(not SURROGATE.exists(), reason="portal surrogate.json absent")
def test_pooling_is_wrong_in_both_directions_on_the_real_fit():
    """The measured reason the portal's displayed band had to change.

    Pooled 13.37 against a two-site band of 10.01: the pooled figure understates
    what the tool can promise for the plan it recommends, and covers strata for
    which it has too few runs to promise anything.
    """
    fit = json.loads(SURROGATE.read_text())["fits"]["1.0"]
    strata = _strata(fit)
    pooled = margins.conformal_quantile(
        [r for v in strata.values() for r in v], 0.1)
    cond = margins.conditional_quantile(strata, 0.1)

    assert cond[2] is not None, "21 two-site runs must support 90%"
    assert cond[2] < pooled, (
        f"two-site band {cond[2]:.3f} should be tighter than pooled {pooled:.3f}"
    )
    assert cond[3] is None, "8 three-site runs cannot support 90%"
    assert cond[6] is None, "1 six-site run cannot support anything"


@pytest.mark.skipif(not SURROGATE.exists(), reason="portal surrogate.json absent")
def test_surrogate_error_grows_with_active_sites():
    """The premise. If this ever stops holding, conditioning buys nothing."""
    strata = _strata(json.loads(SURROGATE.read_text())["fits"]["1.0"])
    means = {k: sum(v) / len(v) for k, v in strata.items()}
    ordered = [means[k] for k in sorted(means)]
    assert ordered == sorted(ordered), (
        f"mean residual is not increasing in n_active: {means}"
    )


def test_single_measure_prediction_is_exact_only_at_full_height():
    """Why the one-site stratum has no calibration points and still refuses.

    `alpha[i]` is read off a solver run at FULL height, so a one-site plan at
    full height is a measurement and owes no margin. A partial height scales
    that measurement linearly, which no run checks -- so it cannot claim the
    zero band, and there are no one-site runs to bound it either.
    """
    alpha, max_h, S = [5.0, 0.0, 0.0], 3.0, 6.0
    from pspe.plan.surrogate import effect

    assert effect(alpha, [max_h, 0.0, 0.0], max_h, S) == pytest.approx(5.0)
    # Partial height is a linear interpolation, not a measurement.
    assert effect(alpha, [1.5, 0.0, 0.0], max_h, S) == pytest.approx(2.5)
