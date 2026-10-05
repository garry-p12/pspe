"""rho = sigma_ep / epsilon on a real planner, and R(rho)'s prediction for it.

Proposition 3's closed form is validated in `runs/margin_synthetic/` to 0.003,
where sigma and epsilon are dialled by hand. These tests pin the two pieces
needed to test it against a learned planner instead: that the quantities the
trainer accumulates are the ones the synthetic harness defines, and that the
closed form we compute from them is the published one.
"""

from __future__ import annotations

import importlib.util
import random
import statistics as st
from pathlib import Path

import pytest

from pspe.plan.trainer import _rho_summary

spec = importlib.util.spec_from_file_location(
    "cfx", Path(__file__).resolve().parents[1] / "eval" / "run_constraint_fix.py")
cfx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfx)


@pytest.mark.parametrize("rho,published", [
    (0.10, 0.000), (0.50, 0.005), (0.80, 0.055),
    (1.00, 0.100), (1.25, 0.153), (2.00, 0.261), (10.0, 0.449),
])
def test_closed_form_reproduces_the_published_table(rho, published):
    assert cfx._r_of_rho(rho, 0.1) == pytest.approx(published, abs=5e-4)


def test_r_of_one_is_exactly_delta():
    """R(1) = 1 - Phi(z_delta) = delta by construction. The crossing point."""
    for d in (0.01, 0.05, 0.1, 0.2):
        assert cfx._r_of_rho(1.0, d) == pytest.approx(d, abs=1e-9)


def test_r_is_increasing_in_rho():
    """Realised violation is strictly increasing in the spread ratio."""
    vals = [cfx._r_of_rho(r, 0.1) for r in (0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 20.0)]
    assert vals == sorted(vals)
    assert vals[-1] < 0.5, "R tends to 1/2 from below for symmetric error"


def test_r_undefined_without_a_measured_rho():
    """No probe means rho is not measured; 0.0 would read as 'no spread'."""
    import math
    assert math.isnan(cfx._r_of_rho(None, 0.1))
    assert math.isnan(cfx._r_of_rho(0.0, 0.1))


def test_summary_recovers_sigma_epsilon_and_rho():
    """The accumulated probe samples must give back the generating parameters."""
    rng = random.Random(0)
    SIG, EPS, BIAS = 0.40, 0.25, 0.10
    dev = [rng.gauss(0, SIG) for _ in range(4000)]
    err = [rng.gauss(BIAS, EPS) for _ in range(4000)]
    o = _rho_summary(dev, err)
    assert o["probe/sigma_ep"] == pytest.approx(SIG, rel=0.05)
    assert o["probe/epsilon"] == pytest.approx(EPS, rel=0.05)
    assert o["probe/rho"] == pytest.approx(SIG / EPS, rel=0.05)
    # Bias is reported separately: epsilon is the per-instance spread only, and
    # folding the bias into it would inflate rho's denominator.
    assert o["probe/bias_model_err"] == pytest.approx(BIAS, abs=0.02)


def test_summary_omits_rho_when_nothing_was_probed():
    """The unprobed arms must not report a rho of zero."""
    assert _rho_summary([], []) == {}
    assert "probe/rho" not in _rho_summary([0.1, 0.2, 0.3], [])
    # epsilon identically zero leaves rho undefined rather than infinite
    assert "probe/rho" not in _rho_summary([0.1, 0.2], [0.5, 0.5])


def test_the_model_error_arm_exists_and_selects_that_recipe():
    """R(rho) is a prediction about the EXCLUDED recipe, so it needs an arm."""
    a = cfx.arms(10, 2.0)
    assert "probe+model-error" in a
    ov = a["probe+model-error"]
    assert ov["margin_mode"] == "model_error"
    assert ov["margin_conformal"] is True
    assert ov["real_cost_every"] == 10, "it must probe, or rho is unmeasured"


def test_every_arm_override_is_a_real_config_field():
    import dataclasses as dc

    from pspe.plan.trainer import PlannerConfig
    fields = {f.name for f in dc.fields(PlannerConfig)}
    for name, ov in cfx.arms(10, 2.0).items():
        assert not (set(ov) - fields), f"{name} has unknown keys"


# --------------------------------------------------------------------------- #
# The trap that cost 18 seeds: cost_margin_k is an ENABLE GATE as well as a
# multiplier. `_effective_limit` returns the untightened limit when it is <= 0,
# before the conformal branch is reached -- so an arm with margin_conformal=True
# and cost_margin_k=0.0 produces a margin of exactly zero and becomes a copy of
# the unmargined arm, with every reported column plausible.
# --------------------------------------------------------------------------- #

def test_conformal_arms_request_a_positive_margin_k():
    """Any arm asking for a conformal margin must pass the enable gate."""
    for name, ov in cfx.arms(10, 2.0).items():
        if ov.get("margin_conformal"):
            assert ov.get("cost_margin_k", 0.0) > 0, (
                f"arm {name!r} sets margin_conformal=True with cost_margin_k="
                f"{ov.get('cost_margin_k')}; _effective_limit short-circuits on "
                "cost_margin_k <= 0 and the arm will silently apply NO margin"
            )


def test_effective_limit_short_circuits_on_zero_margin_k():
    """Pin the gate itself, so the trap is documented by a test and not a comment."""
    import inspect

    from pspe.plan.trainer import HybridPlannerTrainer
    src = inspect.getsource(HybridPlannerTrainer._effective_limit)
    assert "cost_margin_k <= 0" in src, (
        "the enable-gate behaviour this test documents has moved; re-check every "
        "arm that sets margin_conformal"
    )


def test_model_error_arm_is_distinguishable_from_probe_only():
    """The two arms must differ in configuration, not just in name.

    They came back numerically identical on 18/18 seeds because the only
    difference was a flag that had been short-circuited.
    """
    a = cfx.arms(10, 2.0)
    probe, me = a["probe"], a["probe+model-error"]
    assert me != probe
    assert me["cost_margin_k"] > 0 and probe["cost_margin_k"] == 0.0, (
        "probe-only must have no margin and the model-error arm must have one"
    )
