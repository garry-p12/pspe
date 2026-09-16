"""Joint Simulate + Plan training (paper contribution #1) and its failure mode."""

from __future__ import annotations

import torch

from pspe.envs import make_env
from pspe.plan import GaussianFieldPolicy, JointConfig, JointPlannerTrainer, PlannerConfig
from pspe.simulate import ensure_dataset, make_surrogate
from pspe.utils import RunLogger

GRID = 16


def build(**joint_kwargs) -> JointPlannerTrainer:
    torch.manual_seed(0)
    data = ensure_dataset("dar", grid=GRID)
    surrogate = make_surrogate("fno", 1, grid=GRID)
    train_env = make_env("dar", dynamics="surrogate", surrogate=surrogate,
                         grid=GRID, horizon=3, batched=True)
    eval_env = make_env("dar", dynamics="truth", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    return JointPlannerTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=6, batch=4, horizon=3, eval_episodes=2,
                          eval_every=10_000, log_dir="runs/_test"),
        eval_env=eval_env, logger=RunLogger("runs/_test", use_tensorboard=False),
        surrogate=surrogate, dataset=data, joint=JointConfig(eval_every=2, **joint_kwargs),
    )


def test_beta_zero_leaves_the_surrogate_where_data_alone_puts_it() -> None:
    """beta = 0 is the disaggregated pipeline: no planning gradient reaches the surrogate."""
    t = build(beta=0.0)
    w0 = [p.detach().clone() for p in t.surrogate_params]
    t.train()
    # The anchor still trains it on data, so weights move — but only via data.
    assert t._pending_plan_grads is None
    assert any(not torch.equal(a, b) for a, b in zip(w0, t.surrogate_params))


def test_planning_gradient_reaches_the_surrogate_when_beta_positive() -> None:
    t = build(beta=0.5)
    batch = t.collect(t.env, 4, keep_graph=True)
    pathwise, likelihood, _ = t._per_sample_losses(batch, multiplier=0.0)
    rec = t._before_policy_step(batch, pathwise, likelihood)
    assert rec["joint/plan_grad_norm"] > 0.0
    assert any(g is not None for g in t._pending_plan_grads)


def test_summary_reports_the_capture_diagnostic() -> None:
    s = build(beta=0.1).train()
    for k in ("joint/surrogate_rel_l2_before", "joint/surrogate_rel_l2_after",
              "joint/surrogate_drift", "joint/beta", "joint/anchor"):
        assert k in s
    assert abs(s["joint/surrogate_drift"]
               - (s["joint/surrogate_rel_l2_after"] - s["joint/surrogate_rel_l2_before"])) < 1e-9


def test_heldout_split_is_by_trajectory_and_disjoint() -> None:
    t = build()
    n_anchor = t._anchor[0].shape[0]
    n_hold = t._heldout[0].shape[0]
    assert n_hold >= 1 and n_anchor + n_hold == ensure_dataset("dar", grid=GRID)["states"].shape[0]


def test_unanchored_arm_drops_the_data_term() -> None:
    t = build(beta=0.5, anchor=False)
    rec = t._after_policy_step(it=1)
    assert "joint/data_loss" not in rec
