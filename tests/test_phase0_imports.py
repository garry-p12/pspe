"""Phase 0 acceptance check: all four modules import and run a forward pass."""

from __future__ import annotations

import importlib

import pytest
import torch

MODULES = [
    "pspe.simulate",
    "pspe.plan",
    "pspe.perceive",
    "pspe.explain",
    "pspe.envs",
    "pspe.pipeline",
    "baselines.safe_rl",
    "eval.metrics",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_device_selection_never_raises() -> None:
    from pspe.utils import get_device

    for prefer in ("auto", "cuda", "mps", "cpu"):
        assert isinstance(get_device(prefer), torch.device)


def test_end_to_end_forward_pass_cpu() -> None:
    """One step through Perceive -> Simulate -> Plan -> Explain."""
    from pspe.envs import make_env
    from pspe.explain import ExplainConfig, ExplainModule, WordTokenizer, build_vocabulary
    from pspe.explain.trainer import condition_dim
    from pspe.perceive import PerceiveConfig, PerceiveModule
    from pspe.plan import GaussianFieldPolicy
    from pspe.simulate import make_surrogate

    grid = 16
    env = make_env("dar", grid=grid, horizon=2, batched=True)
    state = env.reset(2)

    perceive = PerceiveModule(PerceiveConfig(image_size=grid, out_grid=grid))
    field, embed = perceive(torch.rand(2, 3, grid, grid))
    assert field.shape == (2, 1, grid, grid)
    assert embed.shape[0] == 2

    surrogate = make_surrogate("fno", 1, grid=grid, modes=4, width=8, n_layers=2)
    assert surrogate(state).shape == state.shape

    policy = GaussianFieldPolicy(1, env.action_dim)
    action, log_prob = policy.sample(state)
    _, reward, cost, _ = env.step(action)
    assert reward.shape == (2,) and cost.shape == (2,)
    assert torch.isfinite(log_prob).all()

    tokenizer = WordTokenizer(build_vocabulary(env.action_dim, env.basis.centers))
    explain = ExplainModule(
        tokenizer, ExplainConfig(max_len=16), cond_features=condition_dim(env.action_dim)
    )
    briefs, logp = explain.generate(
        torch.zeros(2, condition_dim(env.action_dim)), max_new_tokens=6
    )
    assert len(briefs) == 2 and torch.isfinite(logp).all()
