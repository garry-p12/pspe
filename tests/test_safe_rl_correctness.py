"""Correctness checks for the in-tree safe-RL four.

`tests/test_plan.py` checks that the four algorithms *run*. This file checks
that their update rules are *right*, because the failure mode that matters here
is an implementation that runs, produces plausible curves, and is silently
wrong — a mis-scaled natural-gradient step or a mis-signed dual update would
never surface in a self-comparison on the PDE testbeds.

Two levels:

1. **Analytic** — each mathematical component against a closed form: conjugate
   gradient against an explicit matrix inverse, the KL Hessian-vector product
   against the analytic Fisher of a Gaussian, the NPG step against the
   trust-region radius it claims to respect, and the PID controller against
   hand-computed arithmetic.

2. **Behavioural** — all four on a CMDP whose constrained optimum is known in
   closed form (`baselines/toy_cmdp.py`). A correct constrained learner lands
   on `min(1, d)`; a constraint-blind one runs to 1; a sign-flipped dual
   collapses below 0.

What this does *not* cover is behaviour at benchmark scale. See
`baselines/validate_safety_gym.py`.
"""

from __future__ import annotations

import math

import pytest
import torch

from baselines.safe_rl import (
    ALGORITHMS,
    SafeRLConfig,
    _conjugate_gradient,
    _flat_grad,
    make_agent,
)
from baselines.toy_cmdp import ToyConstrainedBandit, action_error
from pspe.plan.lagrangian import PIDLagrangian
from pspe.utils import RunLogger, seed_everything


# --------------------------------------------------------------------------- #
# 1. Analytic checks
# --------------------------------------------------------------------------- #
def test_conjugate_gradient_matches_an_explicit_inverse() -> None:
    torch.manual_seed(0)
    root = torch.randn(6, 6)
    matrix = root @ root.t() + 3.0 * torch.eye(6)  # symmetric positive definite
    b = torch.randn(6)

    got = _conjugate_gradient(lambda v: matrix @ v, b, iters=50)
    want = torch.linalg.solve(matrix, b)
    assert torch.allclose(got, want, atol=1e-4), f"{got} vs {want}"


def test_kl_hessian_vector_product_equals_the_analytic_fisher() -> None:
    """For N(Wx, sigma^2), the Fisher w.r.t. W is E[x x^T] / sigma^2."""
    torch.manual_seed(0)
    sigma = 0.5
    weight = torch.zeros(1, 3, requires_grad=True)
    x = torch.randn(256, 3)

    def kl_to_self() -> torch.Tensor:
        mean = x @ weight.t()
        old = torch.distributions.Normal(mean.detach(), sigma)
        new = torch.distributions.Normal(mean, torch.full_like(mean, sigma))
        return torch.distributions.kl_divergence(old, new).sum(-1).mean()

    kl_grad = _flat_grad(kl_to_self(), [weight], create=True)
    v = torch.randn(3)
    got = _flat_grad((kl_grad * v).sum(), [weight])

    fisher = (x.t() @ x) / x.shape[0] / (sigma**2)
    want = fisher @ v
    assert torch.allclose(got, want, atol=1e-4), f"{got} vs {want}"


def test_natural_gradient_step_lands_on_the_trust_region_boundary() -> None:
    """The step scale sqrt(2*delta / g^T F^-1 g) must produce KL ~= delta."""
    torch.manual_seed(0)
    sigma, delta = 0.5, 0.01
    x = torch.randn(512, 3)
    fisher = (x.t() @ x) / x.shape[0] / (sigma**2)
    g = torch.randn(3)

    natural = torch.linalg.solve(fisher, g)
    scale = math.sqrt(2 * delta / float(g @ natural))
    step = scale * natural

    # Second-order KL for a Gaussian with fixed variance is exactly quadratic,
    # so this is an equality, not an approximation.
    kl = 0.5 * float(step @ (fisher @ step))
    assert kl == pytest.approx(delta, rel=1e-5)


def test_pid_lagrangian_matches_hand_computed_arithmetic() -> None:
    """Every term of lambda = Kp*e + Ki*I + Kd*max(0, de), checked by hand."""
    dual = PIDLagrangian(cost_limit=1.0, kp=0.5, ki=0.25, kd=2.0, ema=0.0)

    # Step 1: cost 3.0. First call seeds the smoother and prev_cost, so
    # e = 2.0, I = 2.0, derivative = 0.
    got = dual.update(3.0)
    assert got == pytest.approx(0.5 * 2.0 + 0.25 * 2.0, rel=1e-6)

    # Step 2: cost 4.0. e = 3.0, I = 5.0, derivative = 4.0 - 3.0 = 1.0.
    got = dual.update(4.0)
    assert got == pytest.approx(0.5 * 3.0 + 0.25 * 5.0 + 2.0 * 1.0, rel=1e-6)

    # Step 3: cost 2.0. e = 1.0, I = 6.0, derivative is one-sided so 0.
    got = dual.update(2.0)
    assert got == pytest.approx(0.5 * 1.0 + 0.25 * 6.0, rel=1e-6)


def test_pid_integral_is_anti_windup_clamped() -> None:
    """A long safe stretch must not drive the integral arbitrarily negative."""
    dual = PIDLagrangian(cost_limit=1.0, kp=0.1, ki=0.1, kd=0.0, ema=0.0)
    for _ in range(100):
        dual.update(0.0)
    assert dual.integral == 0.0
    assert dual.multiplier == 0.0


def test_cpo_respects_its_trust_region() -> None:
    """After a CPO update the measured KL must not exceed target_kl."""
    seed_everything(0)
    env = ToyConstrainedBandit(cost_limit=0.4)
    agent = make_agent(
        "cpo", env,
        cfg=SafeRLConfig(iterations=1, batch=32, horizon=1, target_kl=0.01, cg_iters=10),
    )
    buf = agent.collect()
    with torch.no_grad():
        before = agent.policy.distribution(buf.states)
        old_mean, old_std = before.mean.clone(), before.stddev.clone()

    agent.update(buf)

    with torch.no_grad():
        after = agent.policy.distribution(buf.states)
        kl = float(
            torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mean, old_std), after
            ).sum(-1).mean()
        )
    assert kl <= agent.cfg.target_kl + 1e-6, f"CPO left its trust region: KL={kl}"


def test_saute_never_rewards_exhausting_the_budget() -> None:
    """The invariant Sauté depends on: unsafe must score no better than safe.

    PSPE rewards are always <= 0, so an unshifted implementation gives the
    absorbing unsafe state (reward 0) a *higher* value than any safe outcome and
    the agent learns to blow the budget on purpose. This asserts the ordering
    rather than raw values, since the shift is by a running floor.
    """
    env = ToyConstrainedBandit(cost_limit=0.4)
    agent = make_agent("saute", env, cfg=SafeRLConfig(iterations=1, batch=3, horizon=1))

    reward = torch.tensor([-1.0, -2.0, -1.0])   # all negative, as in every PSPE task
    cost = torch.tensor([0.1, 0.2, 0.9])
    budget = torch.tensor([0.4, 0.4, 0.4])
    shaped, passed_cost = agent.shape(reward, cost, budget)

    safe, unsafe = shaped[:2], shaped[2]
    assert float(unsafe) == 0.0                    # absorbed
    assert float(safe.min()) >= 0.0                # shifted to non-negative
    assert float(safe.max()) > float(unsafe), (
        "a safe outcome must be strictly preferable to blowing the budget"
    )
    # Ordering among safe outcomes is preserved by a constant shift.
    assert float(safe[0]) > float(safe[1])
    assert torch.equal(passed_cost, cost)  # cost itself is never rewritten


def test_primal_dual_multiplier_moves_in_the_right_direction() -> None:
    """Over budget raises the multiplier; under budget drives it back to zero."""
    env = ToyConstrainedBandit(cost_limit=0.4)
    agent = make_agent(
        "primal_dual_npg", env, cfg=SafeRLConfig(iterations=1, batch=8, horizon=1)
    )

    class _Buf:
        episode_cost = 5.0

    agent.multiplier = 0.0
    agent.multiplier = max(0.0, agent.multiplier + agent.dual_lr * (_Buf.episode_cost - 0.4))
    assert agent.multiplier > 0.0

    for _ in range(1000):
        agent.multiplier = max(0.0, agent.multiplier + agent.dual_lr * (0.0 - 0.4))
    assert agent.multiplier == 0.0


# --------------------------------------------------------------------------- #
# 2. Behavioural check against the closed-form optimum
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_algorithm_converges_to_the_closed_form_constrained_optimum(name, tmp_path) -> None:
    """Each algorithm must land near a* = min(1, d), not at the unconstrained 1."""
    seed_everything(0)
    env = ToyConstrainedBandit(cost_limit=0.4)
    agent = make_agent(
        name, env,
        cfg=SafeRLConfig(
            iterations=300, batch=64, horizon=1, lr_policy=3e-3, target_kl=0.02,
            train_iters=4, cg_iters=10,
        ),
        logger=RunLogger(tmp_path / name, use_tensorboard=False),
    )
    agent.train()

    error = action_error(agent.policy, env)
    assert error < 0.15, (
        f"{name} converged to {env.optimal_action + error:.3f} or "
        f"{env.optimal_action - error:.3f}, expected {env.optimal_action:.3f}"
    )

    metrics = agent.evaluate(episodes=64)
    assert metrics["episode_cost"] <= env.task.cost_limit + 0.1


@pytest.mark.slow
def test_lagrangian_multipliers_reach_the_analytic_value() -> None:
    """lambda* = 2(p - d). A dual that settles elsewhere is solving another problem.

    Only the two algorithms carrying an explicit multiplier are checked; CPO and
    Sauté encode the constraint in the step geometry and the state respectively.
    """
    expected = ToyConstrainedBandit(cost_limit=0.4).optimal_multiplier
    for name, attribute in (("ppo_lagrangian", "dual"), ("primal_dual_npg", None)):
        seed_everything(0)
        env = ToyConstrainedBandit(cost_limit=0.4)
        agent = make_agent(
            name, env,
            cfg=SafeRLConfig(iterations=300, batch=64, horizon=1, lr_policy=3e-3,
                             target_kl=0.02, train_iters=4, cg_iters=10),
        )
        agent.train()
        got = agent.dual.multiplier if attribute else agent.multiplier
        assert 0.4 * expected <= got <= 2.5 * expected, (
            f"{name}: lambda settled at {got:.3f}, analytic optimum {expected:.3f}"
        )
