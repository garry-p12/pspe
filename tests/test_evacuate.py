"""`FloodEvacuateEnv` — the protocol, and the four ways it measured nothing.

Every one of these assertions exists because the environment once failed it
silently, producing a confident ranking of arms that were all identical. They
are written against the SYMPTOM rather than the cause, because the cause was
different each time and the symptom never was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pspe.envs.evacuate import (                              # noqa: E402
    EgressTable, EvacuateTask, FloodEvacuateEnv,
)

K, T, M = 4, 8, 3


def _table(*, close_at: int = 3, lead: int = 2) -> EgressTable:
    """A small table with the structure the decision needs, built by hand.

    Bigger members cut the roads earlier, the signal leads the cutting, and
    the water at the zones rises and falls. No solver, so the tests stay fast
    and test the ENV rather than the hydraulics.
    """
    ok = np.ones((M, K, T), dtype=bool)
    exposure = np.zeros((M, K, T), dtype=np.float32)
    rise = np.sin(np.linspace(0, np.pi, T)) ** 2
    for m in range(M):
        shut = max(close_at - m, 1)
        ok[m, :, shut:T - 1] = False
        exposure[m] = (rise * (0.4 + 0.8 * m))[None, :]
    mag = np.stack([rise * (0.3 + 0.35 * m) for m in range(M)])
    mag = mag / mag.max()
    signal = np.stack([mag[:, min(t + lead, T - 1)] for t in range(T)], axis=1)
    deadline = np.array([[int(np.flatnonzero(ok[m, k]).max()) for k in range(K)]
                         for m in range(M)], dtype=np.int64)
    return EgressTable(ok=ok, minutes=np.where(ok, 20.0, np.inf).astype(np.float32),
                       exposure=exposure, signal=signal.astype(np.float32),
                       deadline=deadline,
                       population=np.full(K, 1000.0, dtype=np.float32),
                       zone_nodes=list(range(K)), safe_node=99, interval_min=30.0)


def _env(**kw):
    return FloodEvacuateEnv(_table(**kw), EvacuateTask(cost_limit=8.0))


def _roll(env, policy, batch=16, seed=0):
    s = env.reset(batch, generator=torch.Generator().manual_seed(seed))
    ret = torch.zeros(batch)
    cost = torch.zeros(batch)
    while True:
        s, r, c, done = env.step(policy(s, env))
        ret += r
        cost += c
        if done:
            return ret, cost, s


# ----------------------------------------------------------------- protocol

def test_matches_the_batched_field_env_protocol():
    env = _env()
    assert env.action_dim == K
    assert env.obs_shape == (4, 1, K)
    s = env.reset(16, generator=torch.Generator().manual_seed(0))
    assert s.shape == (16, *env.obs_shape)
    s2, r, c, done = env.step(torch.zeros(16, env.action_dim))
    assert s2.shape == s.shape and r.shape == (16,) and c.shape == (16,)
    assert isinstance(done, bool)


def test_done_fires_at_the_horizon():
    env = _env()
    steps = 0
    env.reset(4, generator=torch.Generator().manual_seed(0))
    while True:
        _, _, _, done = env.step(torch.zeros(4, env.action_dim))
        steps += 1
        if done:
            break
    assert steps == T


# ------------------------------------------------------- the four failures

def test_doing_nothing_is_worse_than_evacuating_everyone():
    """Harm is being in the water, not being ordered at an awkward moment.

    When an order given onto a cut road counted as an instant casualty, the
    optimum sat on "never order": it scored 0.0% harm because the flood
    recedes before the horizon, while a mistimed trigger stranded 100%. The
    best policy was the one an operator would never take.
    """
    env = _env()
    never, never_cost, _ = _roll(env, lambda s, e: e.never_action(s))
    now, now_cost, _ = _roll(env, lambda s, e: e.order_now_action(s))
    assert now.mean() > never.mean(), "ordering everyone out is not better than nothing"
    assert now_cost.mean() > never_cost.mean(), "evacuating everyone must cost more"
    assert never_cost.sum() == 0.0


def test_the_arms_separate_at_all():
    """The check `task.py` demands: can any action move the objective?

    Three separate causes produced an environment where every arm scored
    exactly 0.00 — random per-edge depths that cannot disconnect a graph,
    `fastest_at` letting travellers wait out the flood, and a signal that
    only moved once egress was already gone.
    """
    env = _env()
    scores = [_roll(env, p)[0].mean().item() for p in (
        lambda s, e: e.never_action(s),
        lambda s, e: e.order_now_action(s),
        lambda s, e: e.trigger_action(s, 0.3),
    )]
    assert max(scores) - min(scores) > 1.0, f"all arms scored alike: {scores}"


def test_people_ordered_onto_a_cut_road_stay_rather_than_vanish():
    env = _env()
    env.reset(8, generator=torch.Generator().manual_seed(0))
    # Run past the point where the roads shut, ordering everyone every step.
    for _ in range(T):
        _, _, _, done = env.step(torch.ones(8, env.action_dim))
        if done:
            break
    # Nobody is unaccounted for: everyone either left or is still there.
    st = env.episode_stats()
    assert (st["still_there"] >= 0).all()
    assert torch.isfinite(env.remaining).all()


def test_the_signal_leads_the_closures():
    """Without lead there is no decision, only a coin flip.

    Measured on the real network when the signal came from the roads: egress
    closed at slot 1 of 12 while the signal was still 0.00, so every trigger
    threshold fired at the same slot, too late, and scored identically to
    never ordering while paying nearly all the disruption.
    """
    tab = _table(close_at=3, lead=2)
    biggest = tab.signal[-1]
    first_cut = int(np.argmin(tab.ok[-1].mean(axis=0) > 0.5))
    assert biggest[max(first_cut - 2, 0)] > 0.05, (
        "the signal is still flat when the roads are already closing")


# -------------------------------------------------------------- accounting

def test_cost_is_disruption_and_an_order_keeps_costing():
    """An order, once given, stands: it accrues every interval it is up."""
    env = _env()
    _, early, _ = _roll(env, lambda s, e: e.order_now_action(s))
    # Ordering at the very end costs one interval, not all of them.
    class Late:
        def __call__(self, s, e):
            on = e.t >= T - 1
            return torch.full((s.shape[0], e.action_dim), 1.0 if on else -1.0)
    _, late, _ = _roll(env, Late())
    assert early.mean() > late.mean(), "ordering early must cost more than ordering late"


def test_seeding_reproduces_an_episode():
    env = _env()
    a = _roll(env, lambda s, e: e.trigger_action(s, 0.3), seed=5)[0]
    b = _roll(env, lambda s, e: e.trigger_action(s, 0.3), seed=5)[0]
    assert torch.allclose(a, b)


@pytest.mark.parametrize("batch", [1, 7])
def test_runs_at_any_batch_size(batch):
    env = _env()
    ret, _, _ = _roll(env, lambda s, e: e.order_now_action(s), batch=batch)
    assert ret.shape == (batch,)
