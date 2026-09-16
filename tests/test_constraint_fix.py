"""The surrogate/reality cost gap: probes, bias correction, and accounting.

The failure being fixed: the dual is driven by cost measured on surrogate
rollouts while violation is realised under the true dynamics, so a surrogate
that under-predicts cost leaves the policy safe in-model and unsafe in reality.
"""

from __future__ import annotations

import torch

from pspe.envs import make_env
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig
from pspe.simulate import make_surrogate
from pspe.utils import RunLogger

GRID = 16


def build(**cfg_kwargs) -> HybridPlannerTrainer:
    torch.manual_seed(0)
    surrogate = make_surrogate("fno", 1, grid=GRID)
    train_env = make_env("dar", dynamics="surrogate", surrogate=surrogate,
                         grid=GRID, horizon=3, batched=True)
    eval_env = make_env("dar", dynamics="truth", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    cfg = PlannerConfig(iterations=6, batch=4, horizon=3, eval_episodes=2,
                        eval_every=10_000, log_dir="runs/_test", **cfg_kwargs)
    return HybridPlannerTrainer(
        train_env, policy, cfg=cfg, eval_env=eval_env,
        logger=RunLogger("runs/_test", use_tensorboard=False),
        surrogate_train_transitions=3072,
    )


def test_probe_measures_true_dynamics_not_the_surrogate() -> None:
    trainer = build()
    probe = trainer.probe_real_cost(episodes=4)
    assert probe == probe, "probe returned NaN with an eval env present"
    assert trainer.real_probe_transitions == 4 * 3


def test_probe_transitions_are_counted_as_real_samples() -> None:
    """The sample-efficiency claim is made on this number, so it must include probes."""
    trainer = build(real_cost_every=2, real_cost_episodes=2)
    summary = trainer.train()

    expected_probes = (6 // 2) * 2 * 3  # iterations/every * episodes * horizon
    assert summary["samples_real_probe"] == expected_probes
    assert summary["samples_real_env"] == 3072 + expected_probes


def test_no_probe_means_no_real_samples_and_no_bias() -> None:
    trainer = build(real_cost_every=0)
    summary = trainer.train()
    assert summary["samples_real_probe"] == 0
    assert summary["samples_real_env"] == 3072
    assert summary["dual/final_cost_bias"] == 0.0


def test_bias_correction_moves_the_dual_input_toward_real_cost() -> None:
    """Between probes the dual must see surrogate cost *plus* the learned bias."""
    trainer = build(real_cost_every=4, real_cost_episodes=2)
    trainer.cost_bias = 0.5

    corrected, record = trainer._dual_input(surrogate_cost=0.2, it=3)  # not a probe step
    assert corrected == 0.7, "bias not applied between probes"
    assert record["dual/cost_bias"] == 0.5

    on_probe, record = trainer._dual_input(surrogate_cost=0.2, it=4)  # probe step
    assert "dual/real_cost" in record
    assert on_probe == record["dual/real_cost"], "probe step must use measured cost"


def test_margin_tightens_the_limit_only_when_asked() -> None:
    nominal = build().env.task.cost_limit

    plain = build(cost_margin_k=0.0)
    plain.cost_bias_var = 0.04
    assert plain._effective_limit() == nominal

    tightened = build(cost_margin_k=2.0)
    tightened.cost_bias_var = 0.04          # sigma = 0.2
    assert abs(tightened._effective_limit() - (nominal - 0.4)) < 1e-9
    assert tightened._effective_limit() < nominal


def test_probe_does_not_perturb_training() -> None:
    """Same lesson as periodic evaluation: the instrument must not move the result.

    A probe that consumed the training RNG stream would make the probe schedule
    part of the trained policy, and the comparison between probe-on and
    probe-off arms would then confound two changes at once.
    """
    def final_return(**kw) -> float:
        trainer = build(**kw)
        # Probe repeatedly, but discard the result: only the RNG side effects
        # (or their absence) can show up in the trained policy.
        for _ in range(3):
            trainer.probe_real_cost(episodes=2)
        return trainer.train()["return"]

    assert final_return() == build().train()["return"]
