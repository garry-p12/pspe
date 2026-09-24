"""Saturation guard, advantage floor and dual warm start (rdf planner fix)."""

import math

import torch

from pspe.plan.lagrangian import PIDLagrangian
from pspe.plan.policy import GaussianFieldPolicy
from pspe.plan.trainer import HybridPlannerTrainer, PlannerConfig
from pspe.envs import make_env


def test_lambda_init_survives_first_update_below_limit():
    dual = PIDLagrangian(cost_limit=1.0, kp=0.5, ki=0.05, kd=0.1, lambda_init=2.0)
    assert dual.multiplier == 2.0
    lam = dual.update(0.5)  # under the limit: the P term is negative
    assert lam > 1.0, "warm start was wiped by one under-limit step"


def test_policy_records_pre_tanh_mean():
    policy = GaussianFieldPolicy(2, 4)
    state = torch.randn(3, 2, 16, 16)
    policy.sample(state)
    assert policy.last_pre_tanh_mean.shape == (3, 4)


def _trainer(**cfg):
    env = make_env(testbed="dar", grid=16, horizon=3, n_actuators=4, dynamics="truth",
                   device=torch.device("cpu"), batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    return HybridPlannerTrainer(env, policy, cfg=PlannerConfig(iterations=1, horizon=3, batch=4,
                                                               seed=0, **cfg),
                                device=torch.device("cpu"))


def test_saturation_penalty_appears_in_batch_and_loss():
    tr = _trainer(saturation_coef=1.0, saturation_margin=0.0)
    batch = tr.collect(tr.env, 4)
    assert batch.saturation is not None and batch.saturation.shape == (4, 3)
    pw, lr, diag = tr._per_sample_losses(batch, 0.0)
    assert diag["train/saturation"] > 0
    assert torch.isfinite(pw).all() and torch.isfinite(lr).all()


def test_saturation_off_by_default_matches_legacy():
    tr = _trainer()
    batch = tr.collect(tr.env, 4)
    assert batch.saturation is None
    _, _, diag = tr._per_sample_losses(batch, 0.0)
    assert diag["train/saturation"] == 0.0


def test_advantage_floor_bounds_lr_gradient():
    """Identical returns must not blow up the likelihood branch."""
    tr = _trainer(advantage_std_floor=1e-2)
    batch = tr.collect(tr.env, 4)
    batch.rewards = torch.ones_like(batch.rewards)   # every rollout identical
    batch.costs = torch.zeros_like(batch.costs)
    _, lr, _ = tr._per_sample_losses(batch, 0.0)
    g = torch.autograd.grad(lr.mean(), list(tr.policy.parameters()), allow_unused=True)
    norm = math.sqrt(sum(float((x ** 2).sum()) for x in g if x is not None))
    assert norm < 1e3


def test_normalized_dual_is_scale_free():
    small = PIDLagrangian(cost_limit=1.0, kp=0.5, ki=0.05, kd=0.1, ema=0.0, normalize=True)
    big = PIDLagrangian(cost_limit=10.0, kp=0.5, ki=0.05, kd=0.1, ema=0.0, normalize=True)
    assert abs(small.update(1.5) - big.update(15.0)) < 1e-9
    raw = PIDLagrangian(cost_limit=10.0, kp=0.5, ki=0.05, kd=0.1, ema=0.0)
    assert raw.update(15.0) > big.update(15.0) * 5


def test_conformal_margin_is_the_right_quantile_and_waits_for_data():
    """The margin must be the ceil((n+1)(1-delta)) order statistic, and 0 until attainable."""
    import math
    import torch
    from pspe.envs import make_env
    from pspe.plan.policy import GaussianFieldPolicy
    from pspe.plan.trainer import HybridPlannerTrainer, PlannerConfig

    env = make_env(testbed="dar", grid=16, horizon=2, n_actuators=4, dynamics="truth",
                   device=torch.device("cpu"), batched=True)
    tr = HybridPlannerTrainer(env, GaussianFieldPolicy(env.obs_shape[0], env.action_dim),
                              cfg=PlannerConfig(iterations=1, horizon=2, batch=2, seed=0,
                                                margin_conformal=True, margin_delta=0.1),
                              device=torch.device("cpu"))
    tr.probe_episode_dev = [0.1, 0.2]                 # too few for delta = 0.1
    assert tr.conformal_margin() == 0.0
    tr.probe_episode_dev = [0.05 * i for i in range(1, 11)]   # n = 10
    rank = math.ceil((10 + 1) * 0.9)                          # = 10
    expected = sorted(tr.probe_episode_dev)[rank - 1] + max(0.0, tr.cost_bias)
    assert abs(tr.conformal_margin() - expected) < 1e-9
    # It must cover the policy's own spread, which is what actually violates.
    tr.probe_episode_dev = [0.0] * 9 + [0.9]
    assert tr.conformal_margin() > 0.5
