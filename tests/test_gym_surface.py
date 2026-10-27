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
    'CMDPToy-v0': 'optimal_action_vector',
    'PDEControl-dar-v0': 'proportional_action',
    'PDEControl-swe-v0': 'proportional_action',
    'PDEControl-rdf-v0': 'proportional_action',
    'FireDispatch-v0': 'biggest_fire_action',
    'FloodBuyout-v0': 'lowest_first_action',
    'FloodCrossing-v0': 'budgeted_react_action',
    'FloodGreenInfra-v0': 'most_paved_action',
    'HeatRetrofit-v0': 'exposure_first_action',
}

#: Cheap to construct. The other two want data on disk.
CHEAP = ("FloodOperate-v0", "FloodOperateFine-v0")


def _make(env_id: str, **kw):
    register_all()
    try:
        return gymnasium.make(env_id, **kw)
    except ImportError:
        # NOT a skip. A missing dependency is a declaration bug, not absent
        # data, and skipping it is how six of the fourteen environments came to
        # be unconstructable while this suite reported 109 passes: without
        # rasterio the four terrain environments raise here, without fastapi so
        # do the two that reach their fire model through `service.fire`, and
        # every assertion about them quietly vanished. Both are now in
        # `requirements.txt` and in `scripts/check_env.py`'s REQUIRED list.
        raise
    except Exception as exc:                                   # noqa: BLE001
        if env_id in CHEAP:
            raise
        pytest.skip(f"{env_id} needs data not present: {type(exc).__name__}")


def test_all_four_ids_register():
    ids = register_all()
    assert ids == [s.env_id for s in SPECS]
    assert set(ids) == set(INCUMBENTS)
    assert register_all() == ids, "registration must be idempotent"


def test_train_and_eval_draw_from_disjoint_scenarios():
    """The property that makes a number worth reporting.

    Without this, a policy is scored on the scenarios it was fitted to and
    nothing anywhere notices. `FloodBuyout-v0` is the standing example: its
    headline claim held over the whole pool and vanished on the held-out
    windows, and the difference was invisible until the split existed.
    """
    from pspe.envs.splits import SPLITS, split_indices       # noqa: PLC0415

    for n in (6, 12, 33, 35, 64, 1689):
        train = set(split_indices(n, "train").tolist())
        held = set(split_indices(n, "eval").tolist())
        assert not (train & held), f"splits overlap at n={n}"
        assert train | held == set(range(n)), f"splits lose scenarios at n={n}"
        assert train and held, f"a split is empty at n={n}"
        # Stable across calls, or two runs reporting "eval" report different
        # things.
        assert split_indices(n, "eval").tolist() == sorted(held)
    assert set(SPLITS) == {"train", "eval", "all"}


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_every_environment_accepts_a_split(env_id):
    """Including the two with no pool, which take it and say why it is moot."""
    for split in ("train", "eval"):
        env = _make(env_id, split=split)
        assert getattr(env.unwrapped, "split", None) == split or \
            getattr(getattr(env.unwrapped, "core", None), "split", None) == split
        env.close()


@pytest.mark.parametrize("env_id", list(INCUMBENTS))
def test_the_budget_does_not_depend_on_the_split(env_id):
    """Train and eval have to be the SAME problem.

    Three budgets here are derived from the ground -- mean bank length, a
    share of at-risk homes, the median time a crossing is wet -- and deriving
    them after the split gave train and eval different limits. A policy fitted
    under one constraint would then have been reported under another, and the
    catalog would have recorded whichever split it happened to measure.
    """
    limits = []
    for split in ("train", "eval", "all"):
        env = _make(env_id, split=split)
        limits.append(float(env.unwrapped.task.cost_limit))
        env.close()
    assert max(limits) - min(limits) <= abs(limits[0]) * 1e-6, (
        f"{env_id} has a different budget per split: {limits}")


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
        # Three categories, not two. "observed" is a real third answer -- the
        # field is measured data replayed, nothing is predicted -- and the
        # two-way version of this assertion shipped a portal label calling
        # measured temperature "guessed by a model".
        assert spec.dynamics in {"physics", "learned", "observed"}
        assert spec.as_dict()["plain"]["dynamics"] in {
            "really simulated", "guessed by a model", "really measured"}
        assert spec.scores, f"{spec.env_id} records no measured scores"
        assert spec.split == "eval", (
            f"{spec.env_id} records numbers from the {spec.split!r} split; the "
            f"catalog reports held-out numbers")
        assert spec.pool.strip(), f"{spec.env_id} does not say what its pool is"
        for arm in ("random", "incumbent"):
            s = spec.scores[arm]
            assert s["of"] > 0 and 0 <= s["feasible"] <= s["of"]


#: Words that mean something only to somebody who works on this repository.
#: The portal has readers who do not, and the `plain_*` fields exist so the
#: page can be honest without being unreadable -- a check that stops them
#: silently reverting to the register of the paper, which is how the first
#: version of that page got written.
#: Environments whose episode ends in an absorbing state rather than on a
#: clock. Everything else here stops because time ran out.
TERMINATES = {"CMDPToy-v0"}

#: Environments on which NO policy is feasible in every episode, so "the
#: incumbent stays inside the budget" is not a property anything can have.
#:
#: `PDEControl-swe-v0`'s cost carries a state-driven exposure term: the limit
#: can be exceeded whatever the actuators do, and leaving all nine dials alone
#: still busts it in 1 episode of 12. That is a property of the testbed, not a
#: defect in its baseline, and the test below asserts it rather than asserting
#: around it -- if swe is ever recalibrated so that something IS feasible
#: throughout, this set is what has to change.
NOT_ALWAYS_FEASIBLE = {"PDEControl-swe-v0"}

#: The tolerance `scripts/score_gym.py` uses. An episode cost is a sum of
#: float32 steps against one limit, and `CMDPToy-v0`'s optimum sits ON the
#: boundary by definition, so a strict `<=` fails on arithmetic rather than
#: on overspending.
FEASIBLE_TOL = 1.0e-6

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
        # Terminated or truncated is not a style choice, and it is not the
        # same answer for every environment here.
        #
        # A flood or a fire episode ends because the clock ran out: there IS a
        # value beyond the horizon and a bootstrapping critic must not be told
        # it is zero, so those truncate. `CMDPToy-v0` is a one-step bandit --
        # its episode is over because the problem is over, which is an
        # absorbing terminal state, and Gymnasium's own checker rejects an
        # environment that truncates after one step. Asserting one rule for
        # all of them was wrong, and it was wrong in the direction that would
        # have mislabelled the only environment whose answer is known.
        if env_id in TERMINATES:
            assert truncated is False
        else:
            assert terminated is False
        if terminated or truncated or steps > 256:
            break
    assert terminated or truncated, "the episode must end"
    assert (terminated if env_id in TERMINATES else truncated)
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
    nothing known is both feasible and better than doing nothing. And
    `PDEControl-swe-v0` is exempt, because on that one nothing is feasible
    throughout -- see `NOT_ALWAYS_FEASIBLE`.
    """
    env = _make(env_id)
    inner = env.unwrapped
    fn = getattr(inner, INCUMBENTS[env_id])
    limit = inner.task.cost_limit
    slack = abs(limit) * FEASIBLE_TOL
    over = []
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            obs, _r, terminated, truncated, info = env.step(fn(obs))
            total += info["cost"]
            if terminated or truncated:
                break
        if total > limit + slack:
            over.append((seed, total))
    env.close()

    if env_id in NOT_ALWAYS_FEASIBLE:
        # Assert the exemption is still earned. If this environment becomes
        # feasible throughout, the catalog is claiming something false and
        # this set is stale.
        recorded = BY_ID[env_id].scores["incumbent"]
        assert recorded["feasible"] < recorded["of"], (
            f"{env_id} is in NOT_ALWAYS_FEASIBLE but its catalog entry now "
            f"records {recorded['feasible']}/{recorded['of']} -- drop it "
            f"from the set")
        return
    assert not over, (
        f"{env_id}.{INCUMBENTS[env_id]} spent " +
        "; ".join(f"{t:.4f} on seed {s}" for s, t in over) +
        f" of a {limit} budget")


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


@pytest.mark.parametrize("split", ["all", "train", "eval"])
def test_the_evacuation_trigger_never_does_anything_of_its_own(split):
    """Every threshold coincides with never-ordering or with ordering everyone.

    The trigger has one degree of freedom on one global signal, and on this
    ensemble it never expresses a third behaviour: whatever threshold it is
    given, its score equals one of the two degenerate arms exactly. That is
    why the family's "frontier" is a step function and why its best feasible
    point was doing nothing.

    Checked on every split, because WHICH it coincides with depends on which
    members are in the pool -- on the full six it tracks ordering everyone,
    and on either half it tracks never ordering. A one-dimensional threshold
    over six members has no stable behaviour to report, which is this
    environment's real defect and is worse than the thin support alone.
    """
    env = _make("FloodEvacuate-v0", split=split)
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

    saw_signal = False
    for seed in range(6):
        never = total(inner.never_action, seed)
        order_all = total(always, seed)
        if never != pytest.approx(order_all):
            saw_signal = True
        for threshold in (0.35, 0.55, 0.70, 0.90):
            got = total(lambda o, t=threshold: inner.trigger_action(o, t), seed)
            assert got == pytest.approx(never) or got == pytest.approx(order_all), (
                f"the trigger at {threshold} scored {got:.2f} on seed {seed}, "
                f"which is neither never ({never:.2f}) nor ordering everyone "
                f"({order_all:.2f}) -- it has found a third behaviour, and "
                f"this environment's write-up says it cannot")
    assert saw_signal, (
        f"no seed on the {split!r} split separates ordering from not, so this "
        f"test asserted nothing")
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
