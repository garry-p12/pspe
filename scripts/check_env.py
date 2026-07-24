#!/usr/bin/env python
"""Phase 0 smoke test: report the environment and exercise all four modules.

Prints one line per dependency (required vs optional) and per module, then runs
a one-step forward pass through Perceive -> Simulate -> Plan -> Explain on CPU.
"""

from __future__ import annotations

import importlib
import sys

import torch

from pspe.utils import get_device, peak_memory_mb

REQUIRED = ["torch", "numpy", "scipy", "gymnasium", "hydra", "tensorboard", "matplotlib"]
OPTIONAL = {
    "transformers": 'Phase 3/4 open-weight backbones      -> pip install -e ".[llm]"',
    "peft": 'LoRA on HuggingFace backbones        -> pip install -e ".[llm]"',
    "accelerate": 'mixed precision / device placement   -> pip install -e ".[llm]"',
    "bitsandbytes": "4-bit quantisation (CUDA only; unavailable on macOS/arm64)",
    "neuralop": 'reference FNO                        -> pip install -e ".[extras]"',
    "pde": 'py-pde reference solvers             -> pip install -e ".[extras]"',
    "omnisafe": "external safe-RL baselines (repo ships its own in baselines/)",
}


def _probe(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "?")


def main() -> int:
    print(f"python           {sys.version.split()[0]}")
    device = get_device()
    print(f"torch device     {device}")
    print(f"  cuda available {torch.cuda.is_available()}")
    print(f"  mps available  {torch.backends.mps.is_available()}")

    print("\nrequired:")
    missing = []
    for name in REQUIRED:
        version = _probe(name)
        print(f"  {name:<16} {version or 'MISSING'}")
        if version is None:
            missing.append(name)

    print("\noptional:")
    for name, note in OPTIONAL.items():
        version = _probe(name)
        status = version or "absent"
        print(f"  {name:<16} {status:<10} {note if version is None else ''}")

    print("\nmodule smoke test (CPU):")
    from pspe.envs import make_env
    from pspe.explain import ExplainConfig, ExplainModule, WordTokenizer, build_vocabulary
    from pspe.perceive import PerceiveConfig, PerceiveModule
    from pspe.plan import GaussianFieldPolicy
    from pspe.simulate import make_surrogate, make_testbed

    testbed = make_testbed("dar", grid=32)
    state = testbed.initial_condition(2)
    print(f"  simulate: solver step   {tuple(testbed.step(state).shape)}")

    surrogate = make_surrogate("fno", testbed.n_channels, grid=32, modes=6, width=16, n_layers=2)
    print(f"  simulate: surrogate     {tuple(surrogate(state).shape)}")

    perceive = PerceiveModule(PerceiveConfig(image_size=32, out_grid=32))
    field, embed = perceive(torch.rand(2, 3, 32, 32))
    print(f"  perceive: field {tuple(field.shape)} embed {tuple(embed.shape)}")

    env = make_env("dar", grid=32, horizon=4, batched=True)
    env.reset(2)
    policy = GaussianFieldPolicy(testbed.n_channels, env.action_dim)
    action, _ = policy.sample(env.state)
    _, reward, cost, _ = env.step(action)
    print(f"  plan:     reward {float(reward.mean()):+.4f}  cost {float(cost.mean()):.4f}")

    tokenizer = WordTokenizer(build_vocabulary(env.action_dim, env.basis.centers))
    explain = ExplainModule(tokenizer, ExplainConfig(max_len=24), cond_features=2 * env.action_dim + 5)
    briefs, _ = explain.generate(torch.zeros(1, 2 * env.action_dim + 5), max_new_tokens=8)
    print(f"  explain:  sample brief  {briefs[0][:60]!r}")

    print(f"\npeak memory      {peak_memory_mb():.0f} MB")
    if missing:
        print(f"\nMISSING REQUIRED: {missing} -> make setup")
        return 1
    print("\nall four modules import and run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
