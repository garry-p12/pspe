"""Prop. 3: the split-conformal faithfulness certificate.

The guarantee is distribution-free, so it is tested against distributions the
model has never seen — including adversarially skewed ones — rather than
against briefs. What matters is that coverage holds whenever exchangeability
does, and that the certificate says so when it cannot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pspe.explain.conformal import (
    certify,
    conformal_quantile,
    minimum_calibration_size,
    nonconformity,
)


def test_quantile_index_uses_n_plus_one() -> None:
    """The finite-sample correction is the whole difference from a naive quantile."""
    scores = np.arange(1, 11) / 10.0          # 0.1 ... 1.0, n = 10
    # ceil(11 * 0.9) = 10 -> the 10th smallest = 1.0, not the naive 90th pct.
    assert conformal_quantile(scores, delta=0.1) == 1.0
    # ceil(11 * 0.5) = 6 -> 0.6
    assert conformal_quantile(scores, delta=0.5) == 0.6


def test_too_little_calibration_data_gives_a_vacuous_not_a_wrong_bound() -> None:
    assert conformal_quantile(np.array([0.2, 0.3]), delta=0.1) == math.inf
    cert = certify(np.array([0.9, 0.8]), np.array([0.5, 0.1]), delta=0.1)
    assert cert.f_min == 0.0, "with n_cal < 9 the only honest floor is 0"
    assert cert.empirical_coverage == 1.0, "a vacuous bound is trivially covered"


def test_minimum_calibration_size_matches_the_quantile_rule() -> None:
    for delta in (0.1, 0.05, 0.2):
        n = minimum_calibration_size(delta)
        assert conformal_quantile(np.linspace(0.1, 0.9, n), delta) < math.inf
        assert conformal_quantile(np.linspace(0.1, 0.9, n - 1), delta) == math.inf


@pytest.mark.parametrize("dist", ["beta_left", "beta_right", "bimodal", "uniform"])
def test_coverage_holds_under_exchangeability_for_any_distribution(dist: str) -> None:
    """Marginal coverage >= 1 - delta, averaged over many calibration draws."""
    rng = np.random.default_rng(0)
    delta, n_cal, n_test, trials = 0.1, 50, 200, 300

    def draw(n: int) -> np.ndarray:
        if dist == "beta_left":
            return rng.beta(2, 8, n)
        if dist == "beta_right":
            return rng.beta(8, 2, n)
        if dist == "bimodal":
            return np.where(rng.random(n) < 0.5, rng.beta(2, 10, n), rng.beta(10, 2, n))
        return rng.random(n)

    coverages = [certify(draw(n_cal), draw(n_test), delta).empirical_coverage
                 for _ in range(trials)]
    mean_cov = float(np.mean(coverages))
    # Split conformal guarantees E[coverage] >= 1 - delta, and (with the +1
    # correction) also <= 1 - delta + 1/(n_cal + 1). Both sides checked.
    assert mean_cov >= 1 - delta - 0.01, f"{dist}: coverage {mean_cov:.3f} below 0.9"
    assert mean_cov <= 1 - delta + 1 / (n_cal + 1) + 0.02, f"{dist}: over-conservative {mean_cov:.3f}"


def test_certificate_detects_exchangeability_failure() -> None:
    """Calibrate on faithful briefs, test on unfaithful ones: coverage must collapse."""
    rng = np.random.default_rng(1)
    cal = rng.beta(8, 2, 100)          # mostly F ~ 0.8
    test = rng.beta(2, 8, 200)         # mostly F ~ 0.2 — a different policy
    cert = certify(cal, test, delta=0.1)
    assert cert.empirical_coverage < 0.5
    assert not cert.holds, "a certificate on shifted data must report that it fails"


def test_faithfulness_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        nonconformity(np.array([0.5, 1.2]))


def test_holds_is_a_binomial_test_not_a_std_threshold() -> None:
    """A correct certificate on a small test set is noisy; do not reject it on 1 sigma."""
    rng = np.random.default_rng(3)
    cal = rng.random(200)
    # Force a realised coverage of 0.87 on n_test = 120 at delta = 0.1: within
    # binomial noise of 0.9 (p ~ 0.18), so it must hold.
    cert = certify(cal, rng.random(120), delta=0.1)
    # Reconstruct with a chosen coverage rather than rely on the draw.
    from pspe.explain.conformal import Certificate
    ok = Certificate(0.1, 200, cert.s_hat, cert.f_min, 120, 0.87)
    bad = Certificate(0.1, 200, cert.s_hat, cert.f_min, 120, 0.75)
    assert ok.holds and ok.p_value > 0.05
    assert not bad.holds and bad.p_value < 0.05
