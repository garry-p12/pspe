"""The four `gymnasium.make` ids, and the claims the catalog makes about them.

`tests/test_flood_env.py` already covers `FloodControlEnv` thoroughly. What is
new here is that there are now four environments rather than two, that the
other two conform to the same protocol, and -- the part worth a test -- that
`pspe/envs/catalog.py` cannot keep describing them after they change.

The expensive assertions are skipped rather than failed when the data they
need is absent: the egress table is eight minutes of routing and the fire
model needs a checkpoint, so a clean checkout should report "skipped", not
"broken".
"""

from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from pspe.envs import BY_ID, SPECS, catalog, register_all      # noqa: E402

#: Which incumbent each environment exposes, and whether it reads the
#: observation. Kept beside `scripts/score_gym.py`'s copy deliberately: if the
#: two disagree, one of them is describing an environment that no longer
#: exists.
INCUMBENTS = {
    "FloodOperate-v0": "baseline_action",
    "FloodOperateFine-v0": "baseline_action",
    "FloodEvacuate-v0": "trigger_action",
    "Firebreak-v0": "uniform_action",
}

#: Cheap to construct. The other two want data on disk.
CHEAP = ("FloodOperate-v0", "FloodOperateFine-v0")


def _make(env_id: str, **kw):
    register_all()
    try:
        return gymnasium.make(env_id, **kw)
    except Exception as exc:                                   # noqa: BLE001
        if env_id in CHEAP:
            raise
        pytest.skip(f"{env_id} needs data not present: {type(exc).__name__}")


def test_all_four_ids_register():
    ids = register_all()
    assert ids == [s.env_id for s in SPECS]
    assert set(ids) == set(INCUMBENTS)
    assert register_all() == ids, "registration must be idempotent"


def test_the_catalog_describes_every_registered_id_and_nothing_else():
    assert set(BY_ID) == set(INCUMBENTS)
    for spec in SPECS:
        # A spec with an empty caveat is a spec that has stopped being honest
        # about something; every one of these environments has a limit worth
        # stating, and three of them are a learned or synthetic stand-in for
        # something unmeasured.
        assert spec.caveat.strip(), f"{spec.env_id} states no caveat"
        assert spec.incumbent.strip(), f"{spec.env_id} names no incumbent"
        assert spec.status.strip(), f"{spec.env_id} reports no status"
        assert spec.dynamics in {"physics", "learned"}
        assert spec.scores, f"{spec.env_id} records no measured scores"
        for arm in ("random", "incumbent"):
            s = spec.scores[arm]
            assert s["of"] > 0 and 0 <= s["feasible"] <= s["of"]


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_check_env_passes(env_id):
    env = _make(env_id)
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_the_cost_convention_is_the_one_safe_rl_reads(env_id):
    """`info["cost"]` per step, `cost_limit` for the whole episode.

    The split matters: `safe_rl.py` winds its multiplier on
    `episode_cost - cost_limit`, a sum over the episode, so an incumbent that
    spends `cost_limit` on every step is infeasible by a factor of the horizon.
    That is exactly the bug `Firebreak-v0`'s `uniform_action` shipped with.
    """
    env = _make(env_id)
    inner = env.unwrapped
    obs, info = env.reset(seed=0)
    assert info["cost"] == 0.0
    assert obs.shape == env.observation_space.shape

    total = 0.0
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32))
        total += info["cost"]
        steps += 1
        assert info["cost"] >= 0.0, "cost is a spend, never a credit"
        assert np.isfinite(reward)
        assert info["cost_limit"] == pytest.approx(inner.task.cost_limit)
        assert info["episode_cost"] == pytest.approx(total)
        # The horizon is a time limit, not an absorbing state: a bootstrapping
        # critic must not be told the value beyond it is zero.
        assert terminated is False
        if truncated or steps > 256:
            break
    assert truncated, "the episode must end on its horizon"
    env.close()


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_the_incumbent_exists_and_produces_a_legal_action(env_id):
    env = _make(env_id)
    inner = env.unwrapped
    name = INCUMBENTS[env_id]
    fn = getattr(inner, name)
    obs, _ = env.reset(seed=0)
    action = fn() if name == "uniform_action" else fn(obs)
    assert env.action_space.contains(np.asarray(action, dtype=np.float32)), \
        f"{env_id}.{name} produced an action outside the action space"
    env.close()


def test_the_uniform_firebreak_stays_inside_its_episode_budget():
    """The incumbent has to be feasible or it is not a baseline.

    `uniform_action` first shipped treating `cost_limit` of the patch on every
    step, which spends the horizon times the budget -- 0.09 against 0.03. A
    policy allowed to overspend by 3x is not something a learned policy has to
    beat.
    """
    env = _make("Firebreak-v0")
    inner = env.unwrapped
    env.reset(seed=0)
    total = 0.0
    while True:
        _, _, _, truncated, info = env.step(inner.uniform_action())
        total += info["cost"]
        if truncated:
            break
    assert total <= inner.task.cost_limit + 1e-6, \
        f"uniform_action spent {total:.4f} of a {inner.task.cost_limit} budget"
    env.close()


def test_resetting_with_a_seed_does_not_seed_the_action_space():
    """The fact two inline comments in this package used to get wrong.

    Gymnasium gives the action space its own generator so that restarting an
    episode does not silently reset a caller's exploration stream. Reproducing
    a random policy takes both seeds, which is why `scripts/score_gym.py` calls
    both and why its numbers moved by up to 8% between runs before it did.
    """
    env = _make("FloodOperate-v0", n_terrains=4, horizon=4)
    env.reset(seed=3)
    a = env.action_space.sample()
    env.reset(seed=3)
    b = env.action_space.sample()
    assert not np.array_equal(a, b), \
        "reset(seed=) must not reach the action space's generator"

    env.action_space.seed(3)
    env.reset(seed=3)
    c = env.action_space.sample()
    env.action_space.seed(3)
    env.reset(seed=3)
    d = env.action_space.sample()
    assert np.array_equal(c, d), "seeding both must reproduce the arm"
    env.close()


def test_probing_the_catalog_reports_shapes_or_a_reason_but_never_raises():
    rows = catalog(probe=True)
    assert len(rows) == len(SPECS)
    for row in rows:
        assert "available" in row
        if row["available"]:
            assert row["observation_space"].startswith("Box(")
            assert row["action_space"].startswith("Box(")
        else:
            assert row["reason"], f"{row['id']} unavailable with no reason given"
