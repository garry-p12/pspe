"""Split-conformal certificate for explanation faithfulness (paper, Prop. 3).

`F(b) = exp(-KL(pi || pi_b))` is a training-time signal. This turns it into a
deployment-time statement: with probability at least 1 - delta over a fresh
(state, brief) pair, F(b) >= 1 - s_hat, where s_hat is a quantile of the
nonconformity score s = 1 - F on a held-out calibration set.

The guarantee needs exactly one assumption — the calibration pairs and the test
pair are exchangeable draws from the same distribution — and nothing about the
policy, the language model, or the parser. That is the whole point: it is a
bound a reviewer can check without trusting the model.

Two things this module refuses to do:

* it never fits or tunes anything on the calibration set beyond taking one
  order statistic, because any adaptation there breaks exchangeability and the
  certificate with it;
* it never reports a bound without also reporting the empirical coverage on a
  disjoint test set, because the bound is only as good as its assumption and
  the coverage check is the only thing that can catch that assumption failing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Certificate:
    delta: float                 # miscoverage level
    n_cal: int
    s_hat: float                 # nonconformity quantile
    f_min: float                 # certified faithfulness floor, 1 - s_hat
    # Empirical check on a disjoint test set. Should be >= 1 - delta up to
    # finite-sample noise; if it is not, exchangeability failed.
    n_test: int
    empirical_coverage: float

    @property
    def p_value(self) -> float:
        """One-sided binomial test: P[coverage this low or lower | true rate = 1 - delta].

        A single test set is one draw from Binomial(n_test, 1 - delta); a
        one-std tolerance rejected a correct certificate on a 120-brief test
        set at coverage 0.825 (p = 0.011 is unusual but the *mean* over
        re-splits was 0.900 exactly). The p-value states the evidence instead
        of a threshold that fails 16% of correct certificates by construction.
        """
        if self.n_test == 0:
            return float("nan")
        from scipy.stats import binom
        covered = int(round(self.empirical_coverage * self.n_test))
        return float(binom.cdf(covered, self.n_test, 1.0 - self.delta))

    @property
    def holds(self) -> bool:
        """Not significantly below nominal coverage (one-sided, alpha = 0.05)."""
        return self.n_test > 0 and self.p_value >= 0.05


def nonconformity(faithfulness: np.ndarray) -> np.ndarray:
    """s = 1 - F, in [0, 1). Larger means a less faithful brief."""
    f = np.asarray(faithfulness, dtype=np.float64)
    if f.size and (f.min() < 0.0 or f.max() > 1.0):
        raise ValueError(f"faithfulness must lie in (0, 1]; got range [{f.min()}, {f.max()}]")
    return 1.0 - f


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """The ceil((n + 1)(1 - delta))-th smallest score, per split conformal.

    The +1 is what makes the finite-sample guarantee exact rather than
    asymptotic: it accounts for the test point's own rank among n + 1
    exchangeable values. With n small the index can exceed n, in which case
    the quantile is +inf — the certificate then says nothing, which is the
    correct answer when there is not enough calibration data to say anything.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    k = math.ceil((n + 1) * (1.0 - delta))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


def certify(
    cal_faithfulness: np.ndarray,
    test_faithfulness: np.ndarray,
    delta: float = 0.1,
) -> Certificate:
    """Calibrate on one set, check coverage on a disjoint one.

    Returns the certified floor `f_min` and the fraction of test briefs whose
    faithfulness actually met it. The two numbers must be read together.
    """
    cal = nonconformity(cal_faithfulness)
    test = nonconformity(test_faithfulness)
    s_hat = conformal_quantile(cal, delta)
    f_min = max(0.0, 1.0 - s_hat) if math.isfinite(s_hat) else 0.0
    coverage = float(np.mean(test <= s_hat)) if test.size else float("nan")
    return Certificate(
        delta=delta,
        n_cal=int(cal.size),
        s_hat=s_hat,
        f_min=f_min,
        n_test=int(test.size),
        empirical_coverage=coverage,
    )


def minimum_calibration_size(delta: float) -> int:
    """Smallest n_cal for which the certificate is non-vacuous.

    Need ceil((n + 1)(1 - delta)) <= n, i.e. n >= (1 - delta) / delta. At
    delta = 0.1 that is 9; at delta = 0.05 it is 19. Below this the quantile is
    +inf and f_min is 0 — the honest, useless bound.
    """
    return math.ceil((1.0 - delta) / delta)
