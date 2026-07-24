"""Phase 2: envs, hybrid gradient estimator, PID-Lagrangian, planner, baselines."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from baselines.safe_rl import ALGORITHMS, SafeRLConfig, make_agent
from pspe.envs import PDEControlEnv, make_env
from pspe.envs.actuators import GaussianActuatorBasis
from pspe.plan import (
    GaussianFieldPolicy,
    HybridGradientEstimator,
    HybridPlannerTrainer,
    PIDLagrangian,
    PlannerConfig,
)
from pspe.utils import RunLogger

GRID = 16


# --------------------------------------------------------------------------- #
# Actuators and envs
# --------------------------------------------------------------------------- #
def test_actuator_field_is_differentiable_and_localised() -> None:
    basis = GaussianActuatorBasis(4, GRID, sigma=0.08)
    amps = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    field = basis.field(amps)
    assert field.shape == (1, 1, GRID, GRID)
    field.sum().backward()
    assert amps.grad is not None

    # A single actuator's mass should sit near its own centre.
    peak = int(torch.argmax(field.detach().flatten()))
    py, px = divmod(peak, GRID)
    assert abs(px / GRID - float(basis.centers[0, 0])) < 0.2
    assert abs(py / GRID - float(basis.centers[0, 1])) < 0.2


def test_gymnasium_env_follows_the_api() -> None:
    env = PDEControlEnv(testbed="dar", grid=GRID, horizon=3)
    assert isinstance(env, gym.Env)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "cost" in info

    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        assert info["cost"] >= 0.0
    assert steps == 3


def test_batched_env_reward_is_differentiable_in_the_action() -> None:
    env = make_env("dar", grid=GRID, horizon=2, batched=True)
    env.reset(2)
    action = torch.zeros(2, env.action_dim, requires_grad=True)
    _, reward, _, _ = env.step(action)
    reward.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_cost_rises_when_the_budget_is_exceeded() -> None:
    env = make_env("dar", grid=GRID, horizon=2, batched=True)
    state = env.reset(1)
    small = env.task.cost(state, torch.zeros(1, env.action_dim))
    large = env.task.cost(state, torch.ones(1, env.action_dim))
    assert float(large) > float(small)


# --------------------------------------------------------------------------- #
# Hybrid gradient estimator
# --------------------------------------------------------------------------- #
def test_alpha_stays_in_the_unit_interval_and_grads_are_set() -> None:
    layer = torch.nn.Linear(4, 1)
    estimator = HybridGradientEstimator(layer.parameters(), estimate_every=1, n_folds=2)
    x = torch.randn(8, 4)

    for _ in range(3):
        pathwise = (layer(x).squeeze(-1) ** 2)
        likelihood = (layer(x).squeeze(-1) * torch.randn(8))
        stats = estimator.step(pathwise, likelihood)
        assert 0.0 <= stats.alpha <= 1.0
    assert layer.weight.grad is not None


def test_fixed_alpha_ablation_pins_the_coefficient() -> None:
    layer = torch.nn.Linear(4, 1)
    estimator = HybridGradientEstimator(layer.parameters(), adaptive=False, alpha_init=0.25)
    x = torch.randn(8, 4)
    for _ in range(3):
        stats = estimator.step((layer(x) ** 2).squeeze(-1), (layer(x)).squeeze(-1))
    assert stats.alpha == pytest.approx(0.25)


def test_mixture_matches_a_hand_computed_combination() -> None:
    """With alpha fixed, .grad must equal alpha*g_pw + (1-alpha)*g_lr exactly."""
    layer = torch.nn.Linear(3, 1, bias=False)
    x = torch.randn(6, 3)
    estimator = HybridGradientEstimator(layer.parameters(), adaptive=False, alpha_init=0.3,
                                        grad_clip=None)

    pathwise = (layer(x) ** 2).squeeze(-1)
    likelihood = (3.0 * layer(x)).squeeze(-1)
    g_pw = torch.autograd.grad(pathwise.mean(), layer.weight, retain_graph=True)[0]
    g_lr = torch.autograd.grad(likelihood.mean(), layer.weight, retain_graph=True)[0]

    estimator.step(pathwise, likelihood)
    expected = 0.3 * g_pw + 0.7 * g_lr
    assert torch.allclose(layer.weight.grad, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# PID-Lagrangian
# --------------------------------------------------------------------------- #
def test_multiplier_rises_under_violation_and_decays_when_safe() -> None:
    dual = PIDLagrangian(cost_limit=1.0, kp=0.5, ki=0.1, kd=0.1, ema=0.0)
    for _ in range(10):
        dual.update(5.0)
    high = dual.multiplier
    assert high > 0.0

    for _ in range(50):
        dual.update(0.0)
    assert dual.multiplier < high


def test_multiplier_is_zero_while_the_constraint_is_slack() -> None:
    dual = PIDLagrangian(cost_limit=2.0, ema=0.0)
    for _ in range(5):
        dual.update(0.1)
    assert dual.multiplier == 0.0


# --------------------------------------------------------------------------- #
# Planner and baselines
# --------------------------------------------------------------------------- #
def test_planner_runs_and_reports_the_acceptance_metrics(tmp_path) -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    eval_env = make_env("dar", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    trainer = HybridPlannerTrainer(
        env, policy,
        cfg=PlannerConfig(iterations=3, batch=4, horizon=3, eval_every=100, eval_episodes=2),
        eval_env=eval_env,
        logger=RunLogger(tmp_path / "plan"),
    )
    summary = trainer.train()
    for key in ("return", "episode_cost", "violation_rate", "samples", "final_alpha"):
        assert key in summary
    assert 0.0 <= summary["violation_rate"] <= 1.0
    assert summary["samples"] == 3 * 4 * 3


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_safe_rl_baselines_run(name: str, tmp_path) -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    agent = make_agent(
        name, env,
        cfg=SafeRLConfig(iterations=2, batch=4, horizon=3, eval_episodes=2,
                         train_iters=2, cg_iters=3),
        logger=RunLogger(tmp_path / name),
    )
    summary = agent.train()
    assert summary["algorithm"] == name
    assert summary["samples"] == 2 * 4 * 3
    for key in ("return", "episode_cost", "violation_rate"):
        assert key in summary


def test_saute_augments_the_observation_with_the_budget() -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    agent = make_agent("saute", env, cfg=SafeRLConfig(iterations=1, batch=2, horizon=2))
    state = env.reset(2)
    observed = agent.observe(state, torch.full((2,), env.task.cost_limit))
    assert observed.shape[1] == state.shape[1] + 1
    assert torch.allclose(observed[:, -1], torch.ones_like(observed[:, -1]))


# --------------------------------------------------------------------------- #
# Equity constraint (g_2)
# --------------------------------------------------------------------------- #
def test_equity_cost_penalises_concentrated_harm() -> None:
    from pspe.envs.task import TaskSpec

    spec = TaskSpec(equity_enabled=True, n_subregions=2)
    uniform = torch.full((2, 1, 32, 32), 0.5)
    concentrated = torch.zeros(2, 1, 32, 32)
    concentrated[:, 0, :16, :16] = 2.0
    action = torch.zeros(2, 9)

    assert float(spec.equity_cost(uniform, action).mean()) < 1e-4
    assert float(spec.equity_cost(concentrated, action).mean()) > 0.1
    assert set(spec.costs(concentrated, action)) == {"safety", "equity"}


def test_equity_cost_is_differentiable() -> None:
    from pspe.envs.task import TaskSpec

    spec = TaskSpec(equity_enabled=True)
    state = torch.rand(2, 1, 16, 16, requires_grad=True)
    spec.equity_cost(state, torch.zeros(2, 9)).sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_planner_enforces_the_equity_constraint_with_a_second_dual(tmp_path) -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    env.task.equity_enabled = True
    eval_env = make_env("dar", grid=GRID, horizon=3, batched=True)
    eval_env.task.equity_enabled = True

    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    trainer = HybridPlannerTrainer(
        env, policy,
        cfg=PlannerConfig(iterations=3, batch=4, horizon=3, eval_every=100, eval_episodes=2),
        eval_env=eval_env, logger=RunLogger(tmp_path / "eq"),
    )
    summary = trainer.train()
    assert trainer.equity_dual is not None
    assert "equity_cost" in summary and "equity_violation_rate" in summary
    assert 0.0 <= summary["equity_violation_rate"] <= 1.0


def test_equity_off_by_default_keeps_single_constraint() -> None:
    env = make_env("dar", grid=GRID, horizon=2, batched=True)
    assert env.task.equity_enabled is False
    assert set(env.task.costs(env.reset(1), torch.zeros(1, env.action_dim))) == {"safety"}
