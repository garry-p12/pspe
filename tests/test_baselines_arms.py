"""Section 7.2 baseline arms: the controls the comparison claims rest on."""

from __future__ import annotations

import torch

from pspe.envs import make_env
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer
from pspe.perceive.encoder import ConvVisionBackbone, PerceiveConfig, PerceiveModule
from pspe.pipeline import planning_transfer_gap
from pspe.plan import GaussianFieldPolicy
from pspe.simulate import make_surrogate
from pspe.utils import RunLogger

GRID = 16


def test_cnn_arm_trains_every_parameter() -> None:
    """The baseline is only a baseline if nothing is frozen."""
    model = PerceiveModule(PerceiveConfig(backbone="cnn", image_size=GRID, out_grid=GRID))
    frozen = [n for n, p in model.backbone.named_parameters() if not p.requires_grad]
    assert not frozen, f"CNN baseline has frozen parameters: {frozen[:3]}"
    assert isinstance(model.backbone, ConvVisionBackbone)


def test_probe_arm_freezes_the_backbone_and_adds_no_adapters() -> None:
    model = PerceiveModule(
        PerceiveConfig(backbone="tiny", use_lora=False, image_size=GRID, out_grid=GRID)
    )
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any("lora" in n for n, _ in model.backbone.named_parameters())
    # The decoder must still train, or the arm measures nothing.
    assert any(p.requires_grad for p in model.decoder.parameters())


def test_lora_arm_trains_adapters_only() -> None:
    model = PerceiveModule(PerceiveConfig(backbone="tiny", image_size=GRID, out_grid=GRID))
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n in trainable)


def test_the_three_perception_arms_share_one_output_contract() -> None:
    images = torch.randn(2, 3, GRID, GRID)
    shapes = set()
    for cfg in (
        PerceiveConfig(backbone="tiny", image_size=GRID, out_grid=GRID),
        PerceiveConfig(backbone="tiny", use_lora=False, image_size=GRID, out_grid=GRID),
        PerceiveConfig(backbone="cnn", image_size=GRID, out_grid=GRID),
    ):
        out = PerceiveModule(cfg)(images)
        shapes.add(tuple((out[0] if isinstance(out, tuple) else out).shape))
    assert len(shapes) == 1, f"arms disagree on output shape: {shapes}"


def test_posthoc_arm_never_updates_the_generator() -> None:
    """Post-hoc means the explainer never saw this policy during training."""
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    trainer = ExplainTrainer(
        env, policy, ExplainConfig(),
        ExplainTrainConfig(iterations=5, batch=2, posthoc=True, log_dir="runs/_test"),
        RunLogger("runs/_test", use_tensorboard=False),
    )
    before = [p.detach().clone() for p in trainer.model.parameters() if p.requires_grad]
    summary = trainer.train()
    after = [p.detach() for p in trainer.model.parameters() if p.requires_grad]

    assert summary["train_iterations"] == 0.0
    assert summary["posthoc"] is True
    for b, a in zip(before, after):
        assert torch.equal(b, a), "post-hoc arm modified the generator"


def test_planning_transfer_gap_reports_both_policies_on_the_target() -> None:
    surrogate = make_surrogate("fno", 0, grid=GRID, padded=True)
    out = planning_transfer_gap(surrogate, "dar", "rdf", grid=GRID,
                                iterations=3, horizon=3, episodes=2)
    assert out["protocol"] == "planning transfer dar->rdf"
    # The gap is native minus transferred, so it must reconstruct exactly.
    assert out["planning_transfer_gap"] == out["return_native"] - out["return_transferred"]
    for key in ("cost_transferred", "cost_native", "violating_evals_transferred"):
        assert key in out


def test_periodic_evaluation_does_not_change_baseline_training() -> None:
    """Instrumentation must observe, not intervene.

    Adding periodic evaluation moved every baseline's return on `dar` (-2.51 ->
    -2.30) because `evaluate` reset the training env and consumed the training
    RNG stream. With a separate eval generator and a restored env state, a run
    with frequent evaluation must match one with none.
    """
    from baselines.safe_rl import SafeRLConfig, make_agent

    def run(eval_every: int) -> float:
        torch.manual_seed(0)
        env = make_env("dar", grid=GRID, horizon=3, batched=True)
        agent = make_agent(
            "ppo_lagrangian", env,
            cfg=SafeRLConfig(iterations=6, batch=4, horizon=3, eval_episodes=2,
                             eval_every=eval_every, log_dir="runs/_test"),
        )
        return agent.train()["return"]

    assert run(eval_every=2) == run(eval_every=0), "evaluation perturbed training"


def test_periodic_evaluation_does_not_change_planner_training() -> None:
    """Same isolation requirement on the planner side."""
    from pspe.plan import HybridPlannerTrainer, PlannerConfig

    def run(eval_every: int) -> float:
        torch.manual_seed(0)
        env = make_env("dar", grid=GRID, horizon=3, batched=True)
        eval_env = make_env("dar", grid=GRID, horizon=3, batched=True)
        policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
        trainer = HybridPlannerTrainer(
            env, policy,
            cfg=PlannerConfig(iterations=6, batch=4, horizon=3, eval_episodes=2,
                              eval_every=eval_every or 10_000, log_dir="runs/_test"),
            eval_env=eval_env,
            logger=RunLogger("runs/_test", use_tensorboard=False),
        )
        return trainer.train()["return"]

    assert run(eval_every=2) == run(eval_every=0), "evaluation perturbed training"
