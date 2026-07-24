"""Explanation briefs: the structured text the Explain module emits.

A brief is short, templated and *parseable*: the faithfulness objective needs a
deterministic map from text back to an action distribution, so the surface form
is constrained even though the model still has to learn to produce it.

Amplitudes are quantised to a 0.05 grid. That is not cosmetic - it bounds the
vocabulary of a word-level model and makes the parse exact, so a faithfulness
score of 1.0 means the brief really does describe the planner's action, not a
rounding of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor

AMP_STEP = 0.05
CONFIDENCE_LEVELS = {"confident": 0.10, "likely": 0.20, "uncertain": 0.40}


def quantise(value: float, step: float = AMP_STEP) -> float:
    return round(round(value / step) * step, 2)


def fmt(value: float) -> str:
    """Signed fixed-width number, e.g. '+0.35'. Stable tokens for the vocabulary."""
    return f"{value:+.2f}"


@dataclass
class BriefContext:
    """Everything a brief may mention, gathered from the planner and the task."""

    step: int
    amplitudes: Tensor              # (K,)
    centers: Tensor                 # (K, 2)
    predicted_cost: float
    cost_limit: float
    predicted_reward: float
    log_std: Tensor | None = None   # (K,), the policy's own uncertainty
    top_k: int = 3


def confidence_from_std(log_std: Tensor | None) -> str:
    if log_std is None:
        return "likely"
    std = float(log_std.exp().mean())
    if std <= 0.15:
        return "confident"
    return "likely" if std <= 0.30 else "uncertain"


def render_brief(ctx: BriefContext) -> str:
    """Reference brief for the planner's action - the supervised target."""
    amps = ctx.amplitudes.detach().flatten()
    order = torch.argsort(amps.abs(), descending=True)[: ctx.top_k]

    parts = [f"step {ctx.step} ."]
    for idx in order.tolist():
        amp = quantise(float(amps[idx]))
        if abs(amp) < AMP_STEP:
            continue
        x, y = float(ctx.centers[idx, 0]), float(ctx.centers[idx, 1])
        direction = "increase" if amp > 0 else "reduce"
        parts.append(
            f"actuator {idx} at {x:.2f} {y:.2f} set to {fmt(amp)} to {direction} the field ."
        )
    if len(parts) == 1:
        parts.append("no actuator exceeds the deadband , hold all settings .")

    status = "within" if ctx.predicted_cost <= ctx.cost_limit else "above"
    parts.append(
        f"predicted cost {ctx.predicted_cost:.1f} is {status} the limit {ctx.cost_limit:.1f} ."
    )
    parts.append(f"expected reward {ctx.predicted_reward:.1f} .")
    parts.append(f"confidence {confidence_from_std(ctx.log_std)} .")
    return " ".join(parts)


def build_vocabulary(n_actuators: int, centers: Tensor) -> list[str]:
    """Closed vocabulary covering every token any brief can contain."""
    words = [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "step", ".", ",", "actuator", "at", "set", "to", "the", "field",
        "increase", "reduce", "no", "exceeds", "deadband", "hold", "all",
        "settings", "predicted", "cost", "is", "within", "above", "limit",
        "expected", "reward", "confidence",
    ]
    words += [str(i) for i in range(max(n_actuators, 32))]
    words += [f"{c:.2f}" for c in torch.unique(centers).tolist()]
    # Amplitudes on the quantisation grid, plus cost/reward at one decimal.
    words += [fmt(quantise(v / 100)) for v in range(-100, 105, int(AMP_STEP * 100))]
    words += [f"{v / 10:.1f}" for v in range(-400, 401)]
    words += list(CONFIDENCE_LEVELS)
    seen: dict[str, None] = {}
    for word in words:
        seen.setdefault(word, None)
    return list(seen)
