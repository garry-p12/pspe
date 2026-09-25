"""What a conformal margin should bound, as executable claims.

The paper's contribution is a choice between three recipes, so the properties
that distinguish them belong in tests rather than only in prose. Each test
below states one claim from the paper and fails if the code stops making it.
"""

import math

import numpy as np
import pytest

from pspe.plan import margins


def test_quantile_refuses_a_level_it_cannot_attain():
    """ceil((n+1)(1-delta)) <= n requires n >= 1/delta - 1.

    Below that no finite order statistic carries the guarantee, and returning a
    number anyway would be theatre.
    """
    assert margins.conformal_quantile([1.0, 2.0, 3.0], 0.1) is None
    assert margins.conformal_quantile(list(range(20)), 0.1) is not None


def test_quantile_is_the_right_order_statistic():
    sample = list(range(100))                      # 0..99
    q = margins.conformal_quantile(sample, 0.1)
    assert q == sample[math.ceil(101 * 0.9) - 1]   # 91st smallest, zero-indexed 90


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        margins.margin("quantile_of_vibes", 0.1, residuals=[1.0] * 50)


def test_margins_never_loosen_the_limit():
    """A negative quantile would raise d_eff above d. No recipe intends that."""
    over_predicting = [-0.5] * 60                  # realised costs below what the model said
    for mode in margins.MODES:
        m = margins.margin(mode, 0.1, model_err=over_predicting,
                           deviations=over_predicting, residuals=over_predicting,
                           bias=-0.3)
        assert m >= 0.0, mode


def test_no_margin_when_the_model_over_predicts_cost():
    """If reality is cheaper than the model says, the dual is already
    conservative and the signed recipes ask for nothing.

    `model_error` is excluded: it takes the absolute residual, because the
    barrier constraints it was built for can be breached from either side. That
    is why it charges a margin here even though none is needed, and it is a
    second, milder way the recipe mismatches an episode-cost budget.
    """
    over_predicting = [-0.5] * 60
    for mode in ("episode", "residual"):
        assert margins.margin(mode, 0.1, deviations=over_predicting,
                              residuals=over_predicting, bias=-0.3) == 0.0, mode
    assert margins.margin("model_error", 0.1, model_err=over_predicting) == 0.5


def test_matched_pairs_cancel_policy_spread():
    """The paper's central claim, as an inequality.

    With an accurate per-instance model and a policy that scatters widely, the
    matched-pair quantile is far below what coverage needs, while the residual
    quantile is what coverage needs.
    """
    rng = np.random.default_rng(0)
    ghat, sigma, eps = 1.0, 0.5, 0.01
    c = ghat + rng.normal(0, sigma, 400)           # realised episode costs
    g = c - rng.normal(0, eps, 400)                # per-instance model: accurate

    needed = float(np.quantile(c - ghat, 0.9))
    mo = margins.margin("model_error", 0.1, model_err=c - g)
    re = margins.margin("residual", 0.1, residuals=c - ghat)

    assert mo < 0.1 * needed, "matched pairs should not see the policy's spread"
    assert re == pytest.approx(needed, rel=0.05)


def test_residual_reduces_to_matched_pairs_under_per_instance_control():
    """When the controller holds each instance, the two recipes coincide.

    This is why the default is right in the pointwise-control setting it was
    developed for, and the reason the paper frames them as one family.
    """
    rng = np.random.default_rng(1)
    g = rng.normal(1.0, 0.5, 400)                  # per-instance controlled quantity
    c = g + rng.normal(0.05, 0.02, 400)            # realised, close to its own prediction

    matched = margins.margin("model_error", 0.1, model_err=c - g)
    per_instance_residual = margins.margin("residual", 0.1, residuals=c - g)
    assert matched == pytest.approx(per_instance_residual, rel=1e-6)


def test_crossover_sits_at_spread_equals_per_instance_error():
    """The diagnostic: the default covers iff eps >= sigma, crossover at rho = 1.

    Coverage needs bias + z*sigma; matched pairs supply about bias + z*eps; the
    bias cancels. Checked by realised violation rate either side of rho = 1.
    """
    rng = np.random.default_rng(2)
    bias, sigma, d, delta = 0.1, 0.1, 1.0, 0.1

    def rate(eps, n_cal=400, n_test=20_000):
        c = (d + bias) - bias + rng.normal(0, sigma, n_cal)   # mu = d, probe draws
        g = c - bias + rng.normal(0, eps, n_cal)
        q = margins.margin("model_error", delta, model_err=c - g)
        realised_mean = (d - q) + bias
        return float((realised_mean + rng.normal(0, sigma, n_test) > d).mean())

    assert rate(eps=sigma * 4) < delta, "should cover comfortably when rho << 1"
    assert rate(eps=sigma / 4) > delta, "should undercover when rho >> 1"


def test_episode_mode_needs_its_bias_term_where_residual_does_not():
    """The residual form absorbs bias into one order statistic.

    Dropping the bias argument from `episode` loses coverage; `residual` has no
    such argument to drop, which is the simplification the paper claims.
    """
    rng = np.random.default_rng(3)
    ghat, bias, sigma = 1.0, 0.3, 0.05
    c = ghat + bias + rng.normal(0, sigma, 300)

    with_bias = margins.margin("episode", 0.1, deviations=c - c.mean(), bias=bias)
    without = margins.margin("episode", 0.1, deviations=c - c.mean(), bias=0.0)
    residual = margins.margin("residual", 0.1, residuals=c - ghat)

    # Dropping the bias loses most of the margin; the residual form recovers it
    # from one sample, landing where the two-piece construction lands.
    assert without < 0.5 * residual
    assert residual == pytest.approx(with_bias, rel=0.05)
