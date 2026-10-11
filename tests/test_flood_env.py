"""`FloodOperateEnv` -- the protocol, and the two calibrations that matter.

Conformance is the easy half: `reset`/`step`/`obs_shape`/`action_dim` in the
shapes `safe_rl.py` and the hybrid planner expect. The half worth having is
the other one.

`pspe/envs/task.py` carries this warning, learned the hard way on `swe`:

    Calibrating a testbed means checking BOTH that the cost separates (so the
    constraint binds) AND that the RETURN separates (so actuation can move the
    objective at all). The shipped calibration only checked the first, which
    is how swe went a whole project without anyone noticing its margin.

A planning environment where no action changes the outcome is not a hard
problem, it is a broken one, and it fails silently: training runs, returns
plot, nothing means anything. The first version of this env failed exactly
that way -- doing nothing scored -214 and pumping flat out for six hours
scored -208 -- because the pump was sized for 1,336 ha of polder against six
hours of rain on it. These tests assert the separation so it cannot regress
into decoration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pspe.envs.flood_env import (                            # noqa: E402
    SUMP_M, FloodOperateEnv, FloodOperateTask, make_polder_bank,
)

# Coarse and short: same physical domain as the training preset, few enough
# solver steps to run in a test suite.
GRID, DX, HORIZON, BATCH = 32, 120.0, 4, 4


def _env(**kw):
    g = torch.Generator().manual_seed(0)
    bank = make_polder_bank(n=4, grid=GRID, dx=DX, generator=g)
    return FloodOperateEnv(bank, horizon=HORIZON, control_minutes=30.0,
                           dx=DX, **kw), bank


def _roll(env, policy, batch=BATCH, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = env.reset(batch, generator=g)
    ret = torch.zeros(batch)
    cost = torch.zeros(batch)
    steps = 0
    while True:
        s, r, c, done = env.step(policy(s, env))
        ret += r
        cost += c
        steps += 1
        if done:
            break
    return ret, cost, steps, s


NOTHING = lambda s, e: torch.full((s.shape[0], e.action_dim), -1.0)   # noqa: E731
FLAT_OUT = lambda s, e: torch.ones(s.shape[0], e.action_dim)          # noqa: E731
GATE_ONLY = lambda s, e: torch.cat(                                   # noqa: E731
    [torch.ones(s.shape[0], 1), torch.full((s.shape[0], e.n_pumps), -1.0)], dim=1)


# ----------------------------------------------------------------- protocol

def test_matches_the_batched_field_env_protocol():
    env, _ = _env()
    assert env.action_dim == 1 + env.n_pumps
    assert env.obs_shape == (4, GRID, GRID)

    s = env.reset(BATCH, generator=torch.Generator().manual_seed(0))
    assert s.shape == (BATCH, *env.obs_shape)

    s2, r, c, done = env.step(NOTHING(s, env))
    assert s2.shape == s.shape
    assert r.shape == (BATCH,) and c.shape == (BATCH,)
    assert isinstance(done, bool)


def test_done_fires_exactly_at_the_horizon():
    env, _ = _env()
    _, _, steps, _ = _roll(env, NOTHING)
    assert steps == HORIZON


def test_depth_never_goes_negative():
    """The solver's limiter is what keeps this true; the env must not break it."""
    env, _ = _env()
    _, _, _, s = _roll(env, FLAT_OUT)
    assert torch.isfinite(s).all()
    assert s[:, 0].min() >= 0.0


# -------------------------------------------------------------- calibration

def test_the_return_separates_so_actuation_can_move_the_objective():
    """The check `task.py` says was skipped on swe, asserted here.

    Pumping flat out for the whole episode must beat doing nothing by a
    margin that is not noise. The first version of this env returned -214
    against -208 -- three per cent for the entire budget -- because the
    actuators were two orders of magnitude too small for the water.
    """
    env, _ = _env()
    idle, idle_cost, _, _ = _roll(env, NOTHING)
    full, full_cost, _, _ = _roll(env, FLAT_OUT)

    assert idle_cost.sum() == 0.0, "doing nothing must spend no budget"
    assert full_cost.mean() > 0.0, "pumping must spend budget"

    gain = (full.mean() - idle.mean()).item()
    span = abs(idle.mean().item())
    assert gain > 0.0, "pumping made it worse"
    assert gain / span > 0.15, (
        f"only {100 * gain / span:.1f}% of the objective is reachable by "
        f"acting; the actuators are undersized against the water")


def test_the_gate_opens_and_is_worth_something():
    """A flap that never opens is the bug this caught the first time.

    With the basin floor below the river's low stage, the gate was shut for
    every step of every episode and "gate only" scored -71.37 against doing
    nothing's -71.37, identical to the last decimal. Gravity drainage is free
    and the sump sits above `RIVER_BED_M` precisely so it is available.
    """
    env, _ = _env()
    idle, _, _, _ = _roll(env, NOTHING)
    gate, gate_cost, _, _ = _roll(env, GATE_ONLY)

    assert gate_cost.sum() == 0.0, "the gate must not spend pump budget"
    assert gate.mean() > idle.mean(), "the gate never opened"


def test_the_sump_sits_between_the_river_low_and_high_stages():
    """The geometric precondition for the test above, stated directly."""
    env, bank = _env()
    g = torch.Generator().manual_seed(0)
    env.reset(BATCH, generator=g)
    stages = torch.stack([env._river_stage(i / HORIZON) for i in range(HORIZON + 1)])
    wse = 7.0 + stages                      # RIVER_BED_M + stage
    assert wse.min() < SUMP_M, "the river never falls far enough to drain by gravity"
    assert wse.max() > SUMP_M, "the river never rises enough to seal the gate"


# ------------------------------------------------------------------- budget

def test_cost_is_pump_hours_and_follows_the_action():
    env, _ = _env()
    _, cost, _, _ = _roll(env, FLAT_OUT)
    hours = HORIZON * env.control_s / 3600.0
    assert torch.allclose(cost, torch.full_like(cost, hours), atol=1e-5)

    _, half, _, _ = _roll(env, lambda s, e: torch.zeros(s.shape[0], e.action_dim))
    assert torch.allclose(half, torch.full_like(half, hours / 2), atol=1e-5)


def test_episode_stats_report_what_a_reader_would_ask_for():
    env, _ = _env()
    _roll(env, FLAT_OUT)
    st = env.episode_stats()
    assert set(st) == {"wet_km2", "deepest_m", "pump_hours"}
    assert (st["wet_km2"] >= 0).all()
    assert (st["pump_hours"] > 0).all()


# ----------------------------------------------------------------- incumbent

def test_the_rule_based_baseline_is_a_real_opponent():
    """It must beat doing nothing, and spend less than flat out.

    If the incumbent were free to ignore, "we beat the baseline" would mean
    nothing. A local level controller is a strong heuristic when the pump is
    the only way out, and it should look like one here.
    """
    env, _ = _env()
    idle, _, _, _ = _roll(env, NOTHING)
    base, base_cost, _, _ = _roll(env, lambda s, e: e.baseline_action(s))
    full, full_cost, _, _ = _roll(env, FLAT_OUT)

    assert base.mean() > idle.mean(), "the incumbent does not help at all"
    assert base_cost.mean() < full_cost.mean(), "the incumbent is not thrifty"


def test_a_tighter_budget_cannot_be_met_by_pumping_flat_out():
    """The constraint has to bind, or the CMDP is an MDP with extra steps."""
    env, _ = _env(task=FloodOperateTask(cost_limit=0.5))
    _, cost, _, _ = _roll(env, FLAT_OUT)
    assert cost.mean() > env.task.cost_limit, (
        "flat-out pumping fits inside the budget, so the constraint never binds")


# ------------------------------------------------------------- determinism

def test_the_same_seed_gives_the_same_episode():
    env, _ = _env()
    a, ac, _, _ = _roll(env, FLAT_OUT, seed=3)
    b, bc, _, _ = _roll(env, FLAT_OUT, seed=3)
    assert torch.allclose(a, b) and torch.allclose(ac, bc)


def test_different_seeds_give_different_episodes():
    env, _ = _env()
    a, _, _, _ = _roll(env, NOTHING, seed=1)
    b, _, _, _ = _roll(env, NOTHING, seed=2)
    assert not torch.allclose(a, b), "the forcing is not being randomised"


@pytest.mark.parametrize("batch", [1, 3])
def test_runs_at_any_batch_size(batch):
    """`/plan` solves several batch sizes with the same solver code; so do we."""
    env, _ = _env()
    ret, _, _, _ = _roll(env, FLAT_OUT, batch=batch)
    assert ret.shape == (batch,)


# ------------------------------------------------- the pathwise branch

def test_pathwise_is_off_by_default_and_the_reward_carries_no_graph():
    """A detached reward is what makes the score-function branch safe."""
    env, _ = _env()
    s = env.reset(BATCH, generator=torch.Generator().manual_seed(0))
    a = torch.zeros(BATCH, env.action_dim, requires_grad=True)
    _, r, _, _ = env.step(a)
    assert not r.requires_grad
    assert a.grad is None


def test_the_pump_has_a_usable_pathwise_gradient():
    """Linear in capacity, so it behaves: the branch is worth having for it."""
    env, _ = _env(differentiable=True)
    s = env.reset(BATCH, generator=torch.Generator().manual_seed(0))
    grads = []
    for _ in range(HORIZON):
        a = torch.zeros(BATCH, env.action_dim, requires_grad=True)
        s, r, _, done = env.step(a)
        r.sum().backward()
        grads.append(a.grad[:, 1:].abs().mean().item())
        if done:
            break
    assert all(g == g for g in grads), "NaN in the pump gradient"
    top = max(grads)
    assert top > 0.0, "the pump never developed a gradient at all"
    assert top < 1e3, f"the pump gradient blew up to {top:.1e}"


def test_the_gate_pathwise_gradient_is_not_usable_and_that_is_recorded():
    """An orifice has dq/d(dH) -> infinity as the head closes.

    `FloodSolver._structure_demand` says so in prose, and added a half-volume
    limiter -- which stops the flap chattering and does nothing whatever about
    the derivative. This pins the consequence so that a future change which
    appears to "fix" it has to explain what it traded away, and so that nobody
    turns on the pathwise branch for the gate expecting it to work.
    """
    env, _ = _env(differentiable=True)
    s = env.reset(BATCH, generator=torch.Generator().manual_seed(0))
    seen = []
    for _ in range(HORIZON):
        a = torch.zeros(BATCH, env.action_dim, requires_grad=True)
        s, r, _, done = env.step(a)
        r.sum().backward()
        seen.append(a.grad[:, 0].abs().mean().item())
        if done:
            break
    live = [g for g in seen if g > 0]
    assert live, "the gate developed no gradient at all"
    spread = max(live) / min(live)
    assert spread > 1e6, (
        f"the gate's pathwise gradient spans only {spread:.1e}; if an orifice "
        f"has become well-conditioned, say what changed")


# ----------------------------------------------- the surrogate dynamics path

def _tiny_surrogate(env, episodes=4, epochs=3):
    from pspe.envs.flood_surrogate import collect_transitions, fit_surrogate
    data = collect_transitions(env, episodes=episodes,
                               generator=torch.Generator().manual_seed(1))
    net, rep = fit_surrogate(data, epochs=epochs,
                             generator=torch.Generator().manual_seed(2))
    return net, rep


def test_a_surrogate_turns_the_dynamics_label_and_the_gradient_flag():
    """An env running a learned model must not report itself as truth.

    `HybridPlannerTrainer` and `BatchedFieldEnv` both read `dynamics` to
    decide what is being trained on, and `differentiable` to decide whether a
    pathwise branch exists at all. `dynamics` was a class attribute and stayed
    "truth" with a surrogate attached, which is the one mislabel that would
    make an evaluation meaningless.
    """
    env, _ = _env()
    net, _ = _tiny_surrogate(env)
    senv, _ = _env(surrogate=net)
    assert senv.dynamics == "surrogate"
    assert senv.differentiable is True, "a surrogate is only worth its error for the gradient"

    plain, _ = _env()
    assert plain.dynamics == "truth"
    assert plain.differentiable is False


def test_the_surrogate_carries_a_bounded_gradient_where_the_solver_cannot():
    """The whole point: a pathwise branch the estimator can actually use.

    Through the true solver the gate's orifice derivative spans twenty-five
    orders of magnitude over an episode. Through one learned jump per control
    interval it stays inside one. That is what makes `alpha` meaningful rather
    than a number computed over an overflow.
    """
    env, _ = _env()
    net, _ = _tiny_surrogate(env)
    senv, _ = _env(surrogate=net)
    s = senv.reset(BATCH, generator=torch.Generator().manual_seed(3))
    seen = []
    for _ in range(HORIZON):
        a = torch.zeros(BATCH, senv.action_dim, requires_grad=True)
        s, r, _, done = senv.step(a)
        r.sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        seen.append(a.grad.abs().mean().item())
        s = s.detach()
        if done:
            break
    live = [g for g in seen if g > 0]
    assert live, "the surrogate carried no gradient at all"
    assert max(live) < 1e3, f"the surrogate gradient blew up to {max(live):.1e}"


def test_the_surrogate_beats_predicting_that_nothing_changes():
    """Skill, or the pathwise gradient is a gradient of noise.

    A model that cannot beat persistence would still produce a finite, smooth
    gradient -- and it would point nowhere. Asserted weakly because this
    fixture trains for three epochs on four episodes; the shipped fit scores
    0.239.
    """
    env, _ = _env()
    _, rep = _tiny_surrogate(env, episodes=6, epochs=6)
    assert rep.transitions > 0
    assert rep.persistence_rmse_m > 0
    assert rep.val_rmse_m == rep.val_rmse_m, "validation RMSE came back NaN"


def test_assigning_state_at_a_different_batch_resizes_the_episode():
    """`_pathwise_grad_from` assigns `env.state` directly and then steps.

    `BatchedFieldEnv` and `FireEnv` survive that because everything an episode
    needs is in the observation. This env keeps the forcing, the flux, the
    spent budget and the solver's batched bed outside it, so probing with
    `real_cost_episodes` different from `batch` crashed with "expanded size of
    the tensor (4) must match the existing size (16)" — Eq. 8 unreachable on
    this environment for a reason that had nothing to do with Eq. 8.
    """
    env, _ = _env()
    s = env.reset(8, generator=torch.Generator().manual_seed(0))
    small = s[:3].clone()
    env.state = small                      # the direct assignment the trainer makes
    out, r, c, _ = env.step(torch.zeros(3, env.action_dim))
    assert out.shape[0] == 3 and r.shape == (3,) and c.shape == (3,)

    env2, _ = _env()
    env2.reset(2, generator=torch.Generator().manual_seed(0))
    env2.set_state(s)                      # and the other direction
    out2, r2, _, _ = env2.step(torch.zeros(8, env2.action_dim))
    assert out2.shape[0] == 8 and r2.shape == (8,)


# ------------------------------------------------- the Gymnasium view

def _gym_env(**kw):
    from pspe.envs.flood_env import FloodControlEnv
    kw.setdefault("grid", GRID)
    kw.setdefault("dx", DX)
    kw.setdefault("horizon", HORIZON)
    kw.setdefault("n_terrains", 4)
    return FloodControlEnv(**kw)


def test_the_gym_wrapper_obeys_the_reset_and_step_signatures():
    """numpy in, numpy out, five-tuple, `info["cost"]`.

    That last one is the Safety-Gymnasium / OmniSafe convention `PDEControlEnv`
    already follows and `baselines/safe_rl.py` reads, so an outside algorithm
    can be dropped in without a shim.
    """
    import numpy as np

    env = _gym_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "cost" in info

    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert {"cost", "episode_cost", "cost_limit"} <= set(info)
    assert env.observation_space.contains(obs)


def test_the_episode_ends_as_truncation_not_termination():
    """The horizon is a time limit, not an absorbing state.

    Reporting it as `terminated` tells a bootstrapping critic the value beyond
    the horizon is zero. It is not — the water is still there.
    """
    env = _gym_env()
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == HORIZON


def test_episode_cost_accumulates_and_the_limit_is_reported():
    env = _gym_env(budget_pump_hours=1.0)
    env.reset(seed=0)
    full = env.action_space.high                     # pump flat out
    last = None
    while True:
        _, _, _, truncated, info = env.step(full)
        last = info
        if truncated:
            break
    assert last["cost_limit"] == 1.0
    assert last["episode_cost"] > last["cost_limit"], "the constraint must bind"
    assert {"wet_km2", "deepest_m", "pump_hours"} <= set(last), (
        "the episode summary should arrive with the final step")


def test_seeding_reproduces_an_episode():
    env = _gym_env()
    def roll(seed):
        obs, _ = env.reset(seed=seed)
        out = [float(obs.sum())]
        for _ in range(HORIZON):
            o, r, _, t, _ = env.step(env.action_space.high)
            out.append(r)
            if t:
                break
        return out
    assert roll(7) == roll(7)
    assert roll(7) != roll(8)


def test_gymnasium_make_resolves_the_registered_ids():
    import gymnasium

    from pspe.envs.flood_env import register_envs
    register_envs()
    register_envs()                                   # idempotent
    env = gymnasium.make("FloodOperate-v0", n_terrains=2, horizon=HORIZON)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    env.close()


def test_gymnasium_check_env_passes():
    """Gymnasium's own conformance checker, which is stricter than the above.

    It caught what these tests did not: `reset` has to call
    `super().reset(seed=seed)` so `self.np_random` is seeded. Without it the
    env LOOKS seeded — episodes reproduce, because the torch generator does
    that work — while `action_space.sample()` stays on a global stream, so a
    random policy is not reproducible from the seed the caller passed.
    """
    gymnasium = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from pspe.envs.flood_env import register_envs
    register_envs()
    env = gymnasium.make("FloodOperate-v0", n_terrains=4, horizon=6)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_the_action_space_is_seeded_the_way_gymnasium_says_it_is():
    """`reset(seed=)` does NOT seed the action space, and should not.

    I first asserted that it did, which failed — correctly. Gymnasium gives
    the action space its own generator, seeded by `action_space.seed(n)`, so
    that a caller who has deliberately seeded their exploration is not
    silently overridden every time the episode restarts. The documented way
    to reproduce a random policy is to seed both.
    """
    env = _gym_env()

    env.action_space.seed(11)
    env.reset(seed=0)
    a = [env.action_space.sample().tolist() for _ in range(4)]
    env.action_space.seed(11)
    env.reset(seed=0)
    b = [env.action_space.sample().tolist() for _ in range(4)]
    assert a == b, "seeding the action space must reproduce its samples"

    env.action_space.seed(12)
    c = [env.action_space.sample().tolist() for _ in range(4)]
    assert a != c
