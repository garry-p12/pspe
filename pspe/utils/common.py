"""Device selection, seeding, and compute-budget instrumentation.

The proposal reports wall-clock time and peak memory per run; `timer` and
`peak_memory_mb` are the primitives every training script uses for that.
"""

from __future__ import annotations

import contextlib
import os
import random
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Resolve a repo-relative path.

    Hydra may run a job from an `outputs/` directory; anchoring to the repo root
    keeps `runs/`, `data/` and checkpoint paths pointing at one place regardless.
    """
    path = Path(*parts)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_device(prefer: str = "auto") -> torch.device:
    """Local GPU if available, else Apple MPS, else CPU.

    `prefer` may be "auto", "cuda", "mps" or "cpu". Unavailable requests fall
    back to CPU with no error so the same config runs on a free-tier CPU box.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts.

    Phase 3/4 acceptance checks assert the backbone stays frozen, i.e. that
    trainable is a small fraction of total.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def peak_memory_mb(device: torch.device | None = None) -> float:
    """Peak resident memory in MB (GPU peak on CUDA, RSS elsewhere)."""
    if device is not None and device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return usage / 1e6 if usage > 1e7 else usage / 1e3


@dataclass
class TimerResult:
    seconds: float = field(default=0.0)


@contextlib.contextmanager
def timer() -> Iterator[TimerResult]:
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


def constraint_summary(
    eval_costs: Sequence[float], limit: float, prefix: str = "eval"
) -> dict[str, float]:
    """Run-level constraint statistics from every evaluation of a training run.

    A single final evaluation cannot characterise constraint satisfaction here.
    Measured on `dar`: the planner's episode cost spikes to 5-8x the limit and
    decays again within one eval interval, so whether the *last* snapshot lands
    on an excursion is luck. Two runs with identical excursion behaviour were
    reported as "violation 0.0" and "violation 1.0" purely on that timing.

    So the honest run-level numbers are: how many evaluations violated, and how
    bad the worst one was. Both are reported alongside the final snapshot rather
    than replacing it, since the final policy is what a deployment would ship.
    """
    costs = [float(c) for c in eval_costs]
    if not costs:
        return {}
    violating = [c for c in costs if c > limit]
    return {
        f"{prefix}/evals": float(len(costs)),
        f"{prefix}/cost_mean_over_run": sum(costs) / len(costs),
        f"{prefix}/cost_max_over_run": max(costs),
        f"{prefix}/violating_eval_fraction": len(violating) / len(costs),
    }
