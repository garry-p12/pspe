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

#: The strongest FEASIBLE policy known on each environment. `score_gym.py` and
#: `service/gym_routes.py` hold the same mapping and a test below asserts all
#: three agree, because the thing most likely to rot here is one of them naming
#: a baseline the others have moved past.
INCUMBENTS = {
    "FloodOperate-v0": "budget_edge_action",
    "FloodOperateFine-v0": "budget_edge_action",
    "FloodEvacuate-v0": "never_action",
    "Firebreak-v0": "greedy_action",
    "FloodDefend-v0": "uniform_action",
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


#: Words that mean something only to somebody who works on this repository.
#: The portal has readers who do not, and the `plain_*` fields exist so the
#: page can be honest without being unreadable -- a check that stops them
#: silently reverting to the register of the paper, which is how the first
#: version of that page got written.
JARGON = ("incumbent", "feasible", "infeasible", "episode", "policy",
          "return", "gymnasium", "cost_limit", "pump-hours", "surrogate",
          "baseline", "budget-edge", "lagrangian", "u-net", "ndws")


def test_every_environment_is_also_described_without_jargon():
    for spec in SPECS:
        for field in ("plain_title", "plain_decision", "plain_score",
                      "plain_budget", "plain_status", "plain_caveat"):
            text = getattr(spec, field)
            assert text.strip(), f"{spec.env_id} has no {field}"
            lowered = text.lower()
            for word in JARGON:
                assert word not in lowered, (
                    f"{spec.env_id}.{field} uses {word!r}, which the portal's "
                    f"readers have no reason to know")
        for rung in spec.ladder:
            assert rung.get("plain", "").strip(), (
                f"{spec.env_id} ladder rung {rung['key']!r} has no plain name")
            assert "`" not in rung["plain"], (
                f"{spec.env_id} ladder rung {rung['key']!r} names a method")


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_the_catalog_budget_matches_the_environment(env_id):
    """The portal turns a spend into "all of it" or "34x over" from this.

    Compared at a tenth of a per cent, the same scale `score_gym.py` uses:
    these budgets span 0.03 of a patch to $261,755, and the one that is
    derived from the ground rather than chosen carries the terrain's own
    arithmetic in its last digits.
    """
    env = _make(env_id)
    limit = env.unwrapped.task.cost_limit
    assert BY_ID[env_id].cost_limit == pytest.approx(limit, rel=1e-3)
    env.close()


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
    # Every baseline takes an observation whether or not it reads one, so a
    # harness can call them all the same way.
    action = fn(obs)
    assert env.action_space.contains(np.asarray(action, dtype=np.float32)), \
        f"{env_id}.{name} produced an action outside the action space"
    env.close()


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_the_incumbent_is_feasible_in_every_episode(env_id):
    """The point of naming an incumbent: it has to be a policy you may run.

    Three of these environments previously named a baseline that a table of
    returns ranked BELOW a random policy, which reads as "random beats the
    heuristic" and was an artefact of a weak baseline. The replacement rule is
    that the bar is the strongest policy that stays inside the budget, so the
    bar being feasible is the property worth a test.

    `FloodEvacuate-v0`'s entry is doing nothing, because on that environment
    nothing known is both feasible and better than doing nothing.
    """
    env = _make(env_id)
    inner = env.unwrapped
    fn = getattr(inner, INCUMBENTS[env_id])
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            obs, _r, _te, truncated, info = env.step(fn(obs))
            total += info["cost"]
            if truncated:
                break
        assert total <= inner.task.cost_limit, (
            f"{env_id}.{INCUMBENTS[env_id]} spent {total:.4f} of a "
            f"{inner.task.cost_limit} budget on seed {seed}")
    env.close()


def test_the_three_incumbent_tables_agree():
    """This file, the scorer and the service must name the same policies."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from service.gym_routes import INCUMBENT as served          # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "_score_gym", root / "scripts" / "score_gym.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert served == INCUMBENTS, "service/gym_routes.py disagrees"
    assert mod.INCUMBENT == INCUMBENTS, "scripts/score_gym.py disagrees"
    # Every ladder rung must have a policy behind it, or it is a number nobody
    # can reproduce.
    for env_id, rungs in mod.LADDER_POLICIES.items():
        recorded = {r["key"] for r in BY_ID[env_id].ladder}
        assert recorded - {"random"} == set(rungs), (
            f"{env_id}: catalog ladder {sorted(recorded)} vs scorer "
            f"{sorted(rungs)}")


def test_the_evacuation_trigger_collapses_to_two_policies():
    """The trigger has one degree of freedom and expresses two behaviours.

    At 0.70 and above it scores exactly what never ordering scores, because
    the signal never crosses it. At 0.55 it scores exactly what ordering
    everyone immediately scores. That is why the frontier of this family is a
    step function and why its best feasible point was doing nothing.

    The seed is chosen rather than fixed, because a third of episodes here
    score 0.0 for every policy -- only 3 of the 6 ensemble members ever put a
    zone over 0.10 m -- and on one of those the collapse is unobservable
    instead of absent. The test asserts both: that a seed with signal exists,
    and that a seed without signal exists.
    """
    env = _make("FloodEvacuate-v0")
    inner = env.unwrapped
    always = lambda _o: np.ones(env.action_space.shape, dtype=np.float32)  # noqa: E731

    def total(policy, seed):
        obs, _ = env.reset(seed=seed)
        ret = 0.0
        while True:
            obs, reward, _te, truncated, _i = env.step(policy(obs))
            ret += reward
            if truncated:
                return ret

    live = dead = None
    for seed in range(12):
        if total(inner.never_action, seed) != total(always, seed):
            live = seed if live is None else live
        else:
            dead = seed if dead is None else dead
    assert live is not None, "no seed in 0..11 separates ordering from not"
    assert dead is not None, (
        "every seed carried reward signal -- the ensemble used to have 3 dry "
        "members of 6, so either the table changed or this test is stale")

    never, order_all = total(inner.never_action, live), total(always, live)
    assert total(lambda o: inner.trigger_action(o, 0.90), live) == pytest.approx(never)
    assert total(lambda o: inner.trigger_action(o, 0.70), live) == pytest.approx(never)
    assert total(lambda o: inner.trigger_action(o, 0.55), live) == pytest.approx(order_all)
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
