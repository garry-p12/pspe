"""The learned departure-timing policy, and the leakage it must not have.

`filtered` departs at the first slot where a margin-feasible route exists. It is
caught by water 14 times in 96 where the oracle is caught none, so the rule is
myopic and there is headroom in the timing alone. These tests pin the policy
family, the hindsight bound, and -- the part that matters -- that the member
being scored never enters training.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "evac", Path(__file__).resolve().parents[1] / "eval" / "run_evac_policy.py")
evac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evac)


def slot(t_frac, surv_frac, feasible=True, arrives=False, t_now=None):
    return {"t_frac": t_frac, "surv_frac": surv_frac, "spread": 0.0,
            "margin_m": 0.0, "feasible": feasible, "arrives": arrives,
            "minutes": 1.0 if arrives else None,
            "t_now": t_now if t_now is not None else t_frac * 100, "route": object()}


def test_tau_zero_reproduces_the_filtered_rule():
    """The policy family must CONTAIN the baseline, or the search can lose to it."""
    tr = [slot(0.0, 1.0), slot(0.5, 0.5), slot(1.0, 0.2)]
    got = evac.apply_policy(tr, theta=0.0, tau=0.0)
    assert got is tr[0], "tau=0 must depart at the first feasible slot"


def test_waits_for_the_belief_to_narrow():
    tr = [slot(0.0, 1.0), slot(0.3, 0.8), slot(0.6, 0.3), slot(0.9, 0.1)]
    got = evac.apply_policy(tr, theta=0.4, tau=1.0)
    assert got is tr[2], "should wait until the survivor fraction reaches theta"


def test_forced_departure_stops_it_waiting_forever():
    """Without the tau clause a narrow-belief threshold can never trigger."""
    tr = [slot(0.0, 1.0), slot(0.5, 0.9), slot(1.0, 0.9)]
    assert evac.apply_policy(tr, theta=0.1, tau=2.0) is None
    assert evac.apply_policy(tr, theta=0.1, tau=0.5) is tr[1]


def test_infeasible_slots_are_never_departed_on():
    tr = [slot(0.0, 0.1, feasible=False), slot(0.5, 0.1, feasible=True)]
    assert evac.apply_policy(tr, theta=1.0, tau=0.0) is tr[1]


def test_policy_can_return_none_when_nothing_is_feasible():
    tr = [slot(0.0, 1.0, feasible=False), slot(1.0, 1.0, feasible=False)]
    assert evac.apply_policy(tr, theta=1.0, tau=0.0) is None


def test_hindsight_bound_takes_the_earliest_arriving_slot():
    tr = [slot(0.0, 1.0, arrives=False, t_now=0),
          slot(0.4, 0.6, arrives=True, t_now=40),
          slot(0.8, 0.2, arrives=True, t_now=80)]
    got = evac.best_timing(tr)
    assert got is tr[1], "bound must be the earliest arriving slot, not the last"


def test_hindsight_bound_is_none_when_no_slot_arrives():
    tr = [slot(0.0, 1.0, arrives=False), slot(1.0, 0.2, arrives=False)]
    assert evac.best_timing(tr) is None


def test_hindsight_bound_dominates_every_causal_policy():
    """The bound is an upper envelope: no (theta, tau) may beat it on a trace."""
    tr = [slot(0.0, 1.0, arrives=False, t_now=0),
          slot(0.25, 0.7, arrives=False, t_now=25),
          slot(0.5, 0.4, arrives=True, t_now=50),
          slot(0.75, 0.2, arrives=False, t_now=75)]
    bound = evac.best_timing(tr)
    assert bound is not None
    for th in evac.THETA:
        for ta in evac.TAU:
            got = evac.apply_policy(tr, th, ta)
            if got is not None and got["arrives"]:
                assert got["t_now"] >= bound["t_now"], (
                    f"policy ({th},{ta}) arrived earlier than the bound"
                )


def test_the_policy_family_includes_a_rule_that_beats_filtered_when_waiting_helps():
    """If waiting never helped, a learned arm would be pointless. Construct a
    trace where the myopic rule is caught and a patient one is not."""
    tr = [slot(0.0, 1.0, feasible=True, arrives=False, t_now=0),
          slot(0.5, 0.3, feasible=True, arrives=True, t_now=50)]
    myopic = evac.apply_policy(tr, theta=0.0, tau=0.0)
    assert myopic is tr[0] and not myopic["arrives"]
    patient = evac.apply_policy(tr, theta=0.4, tau=1.0)
    assert patient is tr[1] and patient["arrives"]


def test_arrives_is_a_label_and_not_among_the_features():
    """A policy reading `arrives` would be the oracle wearing a disguise."""
    import inspect

    src = inspect.getsource(evac.apply_policy)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "arrives" not in code, "apply_policy must not read the outcome label"
    assert "minutes" not in code


# --------------------------------------------------------------------------- #
# The nesting. These are the tests that matter: a leak here would show up as a
# learned arm that beats `filtered` for a reason that is not timing.
# --------------------------------------------------------------------------- #

def rec(member, trace):
    return {"member": member, "from": 0, "to": 1, "trace": trace}


def test_nested_folds_exclude_the_evaluated_member():
    traces = [rec(m, [slot(0.0, 1.0)]) for m in range(4)]
    folds = evac.nested_folds(traces)
    assert set(folds) == {0, 1, 2, 3}
    for k, train in folds.items():
        assert all(r["member"] != k for r in train), (
            f"fold {k} trained on its own evaluation member"
        )
        assert len(train) == 3


def test_nested_folds_keeps_every_other_member():
    traces = [rec(m, [slot(0.0, 1.0)]) for m in (0, 0, 1, 2)]
    folds = evac.nested_folds(traces)
    assert len(folds[0]) == 2, "both non-zero members must remain"
    assert len(folds[1]) == 3, "member 0 appears twice and both must remain"


def test_fit_reduces_to_filtered_when_waiting_never_helps():
    """If departing immediately always works, the fit must not invent a wait."""
    tr = [slot(0.0, 1.0, arrives=True, t_now=0), slot(0.5, 0.4, arrives=True, t_now=50)]
    th, ta = evac.fit_policy([rec(0, tr), rec(1, tr)])
    chosen = evac.apply_policy(tr, th, ta)
    assert chosen["t_now"] == 0, f"fit chose to wait ({th},{ta}) with no reason to"


def test_fit_learns_to_wait_when_the_myopic_rule_is_caught():
    tr = [slot(0.0, 1.0, arrives=False, t_now=0),
          slot(0.5, 0.3, arrives=True, t_now=50)]
    th, ta = evac.fit_policy([rec(m, tr) for m in range(4)])
    chosen = evac.apply_policy(tr, th, ta)
    assert chosen is not None and chosen["arrives"], (
        f"fit ({th},{ta}) kept the losing myopic departure"
    )


def test_fit_never_scores_below_the_filtered_rule_in_sample():
    """The family contains tau=0, so the search can only match or beat it."""
    import random

    rng = random.Random(0)
    for _ in range(25):
        traces = []
        for m in range(5):
            tr = [slot(i / 4, 1.0 - i / 5, feasible=rng.random() > 0.2,
                       arrives=rng.random() > 0.5, t_now=i * 25) for i in range(4)]
            traces.append(rec(m, tr))
        th, ta = evac.fit_policy(traces)
        assert evac.policy_score(traces, th, ta) >= \
            evac.policy_score(traces, 0.0, 0.0) - 1e-12
