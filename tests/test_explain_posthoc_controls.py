"""The post-hoc controls, and why `posthoc=True` alone is not one.

`ExplainTrainConfig(posthoc=True)` sets iterations to zero, so it scores a
randomly initialised prefix, brief embedding and output head. Its briefs do not
parse, which makes its faithfulness the parser's fallback
`exp(-KL(pi || N(0, 0.4)))` -- a function of how close the policy sits to zero
action, not of anything about explanation. "Trained-in beats post-hoc" measured
against it says only that a supervised generator beats an untrained one
(audit item 12.1, defect 48).

These tests pin the two controls that replace it.
"""

from __future__ import annotations

import pytest
import torch

from pspe.envs import make_env
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer
from pspe.plan import GaussianFieldPolicy
from pspe.utils import RunLogger

DEV = torch.device("cpu")


def _env():
    return make_env(testbed="dar", grid=16, horizon=4, n_actuators=9,
                    device=DEV, batched=True, dynamics="truth")


def _policy(env, jitter: float = 0.0):
    p = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    if jitter:
        with torch.no_grad():
            for q in p.parameters():
                q.add_(torch.randn_like(q) * jitter)
    return p


def _trainer(env, policy, tmp_path, *, source=None, **cfg):
    return ExplainTrainer(
        env, policy, ExplainConfig(max_len=160),
        ExplainTrainConfig(iterations=cfg.pop("iterations", 1), batch=3,
                           log_dir=str(tmp_path), **cfg),
        RunLogger(str(tmp_path), use_tensorboard=False), DEV,
        source_policy=source,
    )


def test_transfer_control_refuses_without_a_source_policy(tmp_path):
    """Without a different policy to learn from it is ordinary training."""
    env = _env()
    with pytest.raises(ValueError, match="source_policy"):
        _trainer(env, _policy(env), tmp_path, posthoc_transfer=True)


def test_transfer_control_trains_on_the_other_policy(tmp_path):
    """The briefs it learns from must be the SOURCE policy's, not the target's."""
    env = _env()
    target, source = _policy(env), _policy(env, jitter=0.8)
    t = _trainer(env, target, tmp_path, source=source, posthoc_transfer=True)

    t.generator = torch.Generator().manual_seed(7)
    _, from_source, _ = t.sample_batch(3, policy=t.source_policy)
    t.generator = torch.Generator().manual_seed(7)
    _, from_target, _ = t.sample_batch(3, policy=t.policy)

    assert from_source != from_target, (
        "same briefs from both policies: the control collapses to ordinary training"
    )


def test_ordinary_training_defaults_its_source_to_the_policy(tmp_path):
    env = _env()
    policy = _policy(env)
    t = _trainer(env, policy, tmp_path)
    assert t.source_policy is t.policy


def test_transfer_control_actually_trains(tmp_path):
    """Unlike `posthoc`, the transfer arm is a TRAINED generator."""
    env = _env()
    t = _trainer(env, _policy(env), tmp_path, source=_policy(env, jitter=0.8),
                 posthoc_transfer=True, iterations=2)
    before = [p.detach().clone() for p in t.model.parameters() if p.requires_grad]
    t.train()
    after = [p for p in t.model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), (
        "no trainable parameter moved: the transfer arm did not train"
    )


def test_untrained_posthoc_does_not_train(tmp_path):
    """The original arm, pinned so the distinction stays visible."""
    env = _env()
    t = _trainer(env, _policy(env), tmp_path, posthoc=True, iterations=5)
    before = [p.detach().clone() for p in t.model.parameters() if p.requires_grad]
    t.train()
    after = [p for p in t.model.parameters() if p.requires_grad]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_prompted_control_refuses_the_stub_backbone(tmp_path):
    """Prompting a randomly initialised stub would score noise as a baseline."""
    env = _env()
    t = _trainer(env, _policy(env), tmp_path, posthoc_prompted=True)
    with pytest.raises(RuntimeError, match="real backbone"):
        t.evaluate(2)


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (dict(), "trained_in"),
        (dict(posthoc=True), "posthoc_untrained"),
        (dict(posthoc_prompted=True), "posthoc_prompted"),
    ],
)
def test_evaluate_labels_its_arm(tmp_path, cfg, expected):
    """A summary.json without the arm name can be read as the wrong comparison."""
    env = _env()
    t = _trainer(env, _policy(env), tmp_path, **cfg)
    if expected == "posthoc_prompted":
        pytest.skip("covered by the stub-backbone refusal; evaluate() cannot run")
    assert t.evaluate(3)["eval/arm"] == expected


def test_reference_briefs_exceed_the_old_default_length(tmp_path):
    """Defect 47: max_len 96 cut briefs before the cost/reward/confidence footer.

    Seeded explicitly. A policy left near zero by whatever ran before emits the
    short "no actuator exceeds the deadband" brief, which made this test pass
    alone and fail in a suite -- the brief length is a property of the policy's
    action magnitudes, so the policy has to be pinned for the length to mean
    anything.
    """
    env = _env()
    torch.manual_seed(0)
    policy = _policy(env)
    with torch.no_grad():                 # amplitudes well clear of the deadband
        for q in policy.parameters():
            q.mul_(0.0).add_(torch.randn_like(q) * 0.5)
    t = _trainer(env, policy, tmp_path)
    t.generator = torch.Generator().manual_seed(0)
    _, briefs, _ = t.sample_batch(4)
    longest = max(len(b.split()) for b in briefs)
    assert longest > 96, (
        f"longest reference brief is {longest} words; this test exists to show "
        "the old 96-token cap truncated it"
    )
    assert any("confidence" in b for b in briefs)
