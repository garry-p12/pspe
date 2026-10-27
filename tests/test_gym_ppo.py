"""The learned arm on the catalog gym: the pieces, and the one oracle.

`pspe/plan/gym_ppo.py` exists so the ladder in `pspe/envs/catalog.py` can carry
a row that was learned rather than written down. Four of the tests below are
cheap unit checks on the machinery that is easy to get silently wrong -- a
streaming normaliser, a GAE recursion, a scale-free dual, a checkpoint -- and
the fifth is the one that matters: `CMDPToy-v0` has a closed-form optimum, so
on that environment "the learner works" is a comparison against the right
answer rather than against the best thing anyone has tried.

The fifth is marked `slow` (about thirty seconds). Everything above it runs in
under a second, which is the point: a convergence test that is the only test
tells you something broke but not which part.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")

from pspe.plan.gym_ppo import (                               # noqa: E402
    GymPPOConfig, GymPPOLagrangian, ObsNorm, ReturnScale, _gae, load_policy,
)


# --------------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(3, 8, 8), (4,)])
def test_obs_norm_streams_to_the_same_answer_as_one_pass(shape):
    """A parallel-variance update is where an off-by-one count hides.

    Three unequal batches, compared against the mean and variance of their
    concatenation. For an image observation the statistics are per channel, so
    the reference pools over N, H and W -- which is the behaviour the physics
    wants (five quantities, not 32000 of them) and the behaviour that is easy
    to lose by reducing over the wrong axis.
    """
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, (n, *shape)).astype(np.float32)
               for n in (5, 17, 2)]
    norm = ObsNorm(shape)
    for b in batches:
        norm.update(b)
    allx = np.concatenate(batches)
    if len(shape) == 3:
        ref = np.moveaxis(allx, 1, 0).reshape(shape[0], -1)
    else:
        ref = allx.T
    assert np.allclose(norm.mean, ref.mean(axis=1), rtol=1e-5, atol=1e-5)
    assert np.allclose(norm.var, ref.var(axis=1), rtol=1e-4, atol=1e-4)


def test_obs_norm_clips_and_survives_a_dead_channel():
    """`Firebreak-v0` reaches 5 on a channel whose mean is near zero, and
    several environments have a channel that never varies. A constant channel
    must not produce a division by zero, and an outlier must not saturate the
    first tanh layer."""
    norm = ObsNorm((2, 4, 4))
    x = np.zeros((6, 2, 4, 4), dtype=np.float32)
    x[:, 0] = 1.0                      # dead channel
    x[:, 1] = np.arange(6)[:, None, None]
    norm.update(x)
    out = norm(np.array([[[1e6]], [[1e6]]], dtype=np.float32)
               * np.ones((2, 4, 4), dtype=np.float32))
    assert np.isfinite(out).all()
    assert out.max() <= 10.0 and out.min() >= -10.0


def test_return_scale_tracks_spread_not_level():
    """Only the scale, never the mean. Subtracting a mean from the reward
    changes the optimal policy when episodes differ in length, and
    `FloodCrossing-v0` episodes differ in length."""
    s = ReturnScale()
    s.update([100.0, 102.0, 98.0, 100.0])
    assert s.scale == pytest.approx(np.std([100.0, 102.0, 98.0, 100.0]), rel=1e-6)


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
def test_gae_with_lambda_one_is_reward_to_go_minus_baseline():
    """At gamma = lam = 1 the advantage collapses to (sum of future rewards +
    bootstrap) - V(s_t), which is the identity a sign error in the recursion
    breaks."""
    rew = np.array([1.0, -2.0, 3.0])
    val = np.array([0.5, 0.25, -1.0])
    last = 7.0
    adv, ret = _gae(rew, val, last, gamma=1.0, lam=1.0)
    want = np.array([rew[t:].sum() + last - val[t] for t in range(3)])
    assert np.allclose(adv, want)
    assert np.allclose(ret, adv + val)


def test_gae_bootstraps_on_truncation_and_not_on_termination():
    """Thirteen of the fourteen environments truncate at a horizon. A learner
    told the value beyond a time limit is zero is solving a shorter problem
    than the one being scored, so the two cases must differ."""
    rew = np.array([0.0, 0.0])
    val = np.array([0.0, 0.0])
    trunc, _ = _gae(rew, val, 5.0, gamma=1.0, lam=1.0)
    term, _ = _gae(rew, val, 0.0, gamma=1.0, lam=1.0)
    assert trunc[-1] == pytest.approx(5.0)
    assert term[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The dual
# --------------------------------------------------------------------------- #
def test_dual_is_scale_free_across_the_catalogs_cost_limits():
    """Cost limits in this gym span 0.03 (burned-area fraction) to 1.48e6
    (dollars). The same *relative* overspend has to produce the same
    multiplier, or one setting of the gains cannot serve both -- which is the
    reason `PIDLagrangian.normalize` exists."""
    from pspe.plan.lagrangian import PIDLagrangian

    def final(limit: float) -> float:
        d = PIDLagrangian(cost_limit=limit, kp=1.0, ki=0.05, kd=0.5,
                          ema=0.5, normalize=True)
        lam = 0.0
        for _ in range(20):
            lam = d.update(1.5 * limit)      # 50% over, every iteration
        return lam

    small, large = final(0.03), final(1.48e6)
    assert small > 0.4, "a 50% overspend must move the multiplier"
    assert large == pytest.approx(small, rel=1e-6)


# --------------------------------------------------------------------------- #
# The checkpoint
# --------------------------------------------------------------------------- #
def test_checkpoint_reproduces_the_agents_own_action():
    """`load_policy` is what `scripts/score_gym.py` scores. If it disagreed
    with the agent that produced it by even a hair, every learned row in the
    ladder would be measuring a different policy from the one that was
    trained."""
    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("CMDPToy-v0", split="train")
    agent = GymPPOLagrangian(env, GymPPOConfig(total_steps=1, seed=3))
    obs, _ = env.reset(seed=0)
    agent.obs_norm.update(np.stack([np.asarray(obs, dtype=np.float32)] * 4))
    mine = agent.deterministic(obs)
    theirs = load_policy(agent.state(), env.action_space.shape)(obs)
    assert np.array_equal(mine, theirs)
    assert theirs.shape == env.action_space.shape
    assert theirs.dtype == np.float32
    env.close()


def test_dual_target_sits_below_the_limit_feasibility_is_judged_against():
    """The margin is deliberate: a dual drives the policy TO the boundary, and
    a policy on the boundary is over it in about half of episodes, so the
    deployed deterministic policy would fail the per-episode feasibility test
    the catalog applies."""
    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("CMDPToy-v0", split="train")
    agent = GymPPOLagrangian(env, GymPPOConfig(total_steps=1))
    assert agent.dual_target < agent.cost_limit
    assert agent.dual_target == pytest.approx(agent.cost_limit * 0.95)
    env.close()


# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_learner_recovers_the_closed_form_optimum_on_the_toy():
    """`CMDPToy-v0`'s optimum is min(peak, limit) = 0.400 at multiplier 0.800.

    The learner is trained against 0.95 * 0.400 by design, so it should land
    *just under* the optimum and be feasible on every episode -- not reach it.
    A run that beats -0.16 has broken the constraint; a run far below it has
    broken the learner. Both failure modes are caught by one assertion pair,
    which is what an environment with a known answer buys.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from score_gym import score                               # noqa: PLC0415

    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("CMDPToy-v0", split="train")
    agent = GymPPOLagrangian(env, GymPPOConfig(total_steps=20_000, seed=0))
    agent.fit()
    env.close()

    eval_env = gymnasium.make("CMDPToy-v0", split="eval")
    limit = float(eval_env.unwrapped.task.cost_limit)
    s = score(eval_env, load_policy(agent.state(), eval_env.action_space.shape),
              list(range(12)), limit)
    eval_env.close()

    assert s["feasible"] == s["of"], (
        f"learned policy broke the budget: cost {s['cost']:.4f} > {limit}")
    assert s["return"] <= -0.16 + 1e-6, (
        "beat the closed-form optimum, which is impossible inside the budget")
    assert s["return"] > -0.30, f"did not get near the optimum: {s['return']:.3f}"
    # The multiplier is the other half of the closed form. Trained against a
    # 5%-tighter budget, so slightly above 0.800 is right and far from it is not.
    assert 0.5 < agent.dual.multiplier < 1.6, agent.dual.multiplier


# --------------------------------------------------------------------------- #
# The spread schedule
# --------------------------------------------------------------------------- #
def test_sigma_anneal_closes_the_sampler_deterministic_gap():
    """The dual is fed the SAMPLER's cost; the deterministic mean is what gets
    scored. In the first 42-run sweep sigma never shrank -- 0.47 to 0.61 on
    thirteen of fourteen environments, near-uniform on a [-1, 1] box -- so the
    dual regulated a policy nobody deployed. The cap has to actually fall, and
    it has to be a cap rather than an assignment so a policy that tightens on
    its own is left alone."""
    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("CMDPToy-v0", split="train")
    cfg = GymPPOConfig(total_steps=1000, log_std_init=-0.5,
                       log_std_final=-2.3, log_std_anneal_frac=0.7)
    agent = GymPPOLagrangian(env, cfg)

    agent.steps_done = 0
    agent._anneal_log_std()
    assert float(agent.pi.log_std_cap) == pytest.approx(-0.5)

    agent.steps_done = int(0.35 * cfg.total_steps)        # half way to the floor
    agent._anneal_log_std()
    assert float(agent.pi.log_std_cap) == pytest.approx(-1.4)

    agent.steps_done = cfg.total_steps                    # past the anneal point
    agent._anneal_log_std()
    assert float(agent.pi.log_std_cap) == pytest.approx(-2.3)

    # A cap, not an assignment: a tighter learned spread survives it.
    with torch.no_grad():
        agent.pi.log_std.fill_(-4.0)
    assert float(agent.pi.effective_log_std().detach().mean()) == pytest.approx(-4.0)
    env.close()


def test_checkpoint_carries_the_annealed_spread():
    """`log_std_cap` is a buffer so it rides in the state dict. If it did not,
    `load_policy` would rebuild the policy at the INITIAL spread -- harmless
    for the deterministic action, but it would silently change any sampling a
    later caller did from a checkpoint."""
    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("CMDPToy-v0", split="train")
    agent = GymPPOLagrangian(env, GymPPOConfig(total_steps=1000))
    agent.steps_done = agent.cfg.total_steps
    agent._anneal_log_std()
    assert "log_std_cap" in agent.state()["policy"]
    assert float(agent.state()["policy"]["log_std_cap"]) == pytest.approx(-2.3)
    env.close()


# --------------------------------------------------------------------------- #
# A diverged solve is not a good policy
# --------------------------------------------------------------------------- #
def test_pde_env_never_returns_a_non_finite_reward_or_cost():
    """`PDEControl-swe-v0` can be driven to numerical divergence by an
    open-loop actuation sequence -- one eval scenario did exactly that under
    CEM and returned an infinite episode cost. `inf` poisons every mean
    computed downstream and, worse, sorts a diverged episode into a winning
    position on return. The env must report such a step as terminated and
    unambiguously infeasible instead."""
    from pspe.envs import register_all

    register_all()
    env = gymnasium.make("PDEControl-swe-v0", split="eval")
    limit = float(env.unwrapped.task.cost_limit)
    rng = np.random.default_rng(0)
    for seed in range(6):
        env.reset(seed=seed)
        for _ in range(64):
            # Corners of the box, which is where the instability lives.
            act = rng.choice([-1.0, 1.0], size=env.action_space.shape)
            obs, rew, term, trunc, info = env.step(act.astype(np.float32))
            assert np.isfinite(rew), f"reward {rew} on seed {seed}"
            assert np.isfinite(info["cost"]), f"cost {info['cost']}"
            assert np.isfinite(np.asarray(obs)).all(), "observation"
            if info.get("diverged"):
                assert term, "a diverged solve must end the episode"
                assert info["cost"] > limit, (
                    "a diverged solve must be infeasible, not merely costly")
            if term or trunc:
                break
    env.close()
