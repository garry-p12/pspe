"""Phase 5: the end-to-end PSPE loop.

    image  --Perceive-->  field estimate
           --Simulate-->  differentiable rollout
           --Plan-->      constrained intervention
           --Explain-->   brief, scored for faithfulness against the plan

`PSPEPipeline.rollout` runs one closed loop and returns, per step, the perceived
field error, the action, the reward and cost, the brief, and the brief's
faithfulness - i.e. every metric in the proposal's Section 7 measured on the
*same* trajectory rather than four separately-run modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .envs.pde_env import BatchedFieldEnv
from .explain.brief import BriefContext, render_brief
from .explain.faithfulness import faithfulness_score
from .explain.model import ExplainModule
from .explain.parser import FrozenBriefParser
from .explain.trainer import build_condition
from .perceive.dataset import render_field
from .perceive.encoder import PerceiveModule
from .plan.policy import GaussianFieldPolicy
from .simulate.losses import relative_l2

Tensor = torch.Tensor


@dataclass
class StepRecord:
    step: int
    perception_rel_l2: float
    reward: float
    cost: float
    cumulative_cost: float
    faithfulness: float
    brief: str
    action: list[float]


@dataclass
class RolloutReport:
    records: list[StepRecord] = field(default_factory=list)

    def summary(self, cost_limit: float) -> dict[str, Any]:
        if not self.records:
            return {}
        n = len(self.records)
        total_cost = self.records[-1].cumulative_cost
        return {
            "e2e/return": sum(r.reward for r in self.records),
            "e2e/episode_cost": total_cost,
            "e2e/cost_limit": cost_limit,
            "e2e/violated": float(total_cost > cost_limit),
            "e2e/perception_rel_l2": sum(r.perception_rel_l2 for r in self.records) / n,
            "e2e/faithfulness": sum(r.faithfulness for r in self.records) / n,
            "e2e/steps": n,
        }


class PSPEPipeline:
    """Wires the four trained modules into one rollout loop."""

    def __init__(
        self,
        env: BatchedFieldEnv,
        policy: GaussianFieldPolicy,
        perceive: PerceiveModule | None = None,
        explain: ExplainModule | None = None,
        parser: FrozenBriefParser | None = None,
        use_perception: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.env = env
        self.policy = policy.to(self.device).eval()
        self.perceive = perceive.to(self.device).eval() if perceive is not None else None
        self.explain = explain.to(self.device).eval() if explain is not None else None
        self.parser = parser or FrozenBriefParser(env.action_dim)
        self.use_perception = use_perception and perceive is not None

    # -- perception --------------------------------------------------------- #
    @torch.no_grad()
    def perceive_state(self, true_state: Tensor) -> tuple[Tensor, float]:
        """Render the true field to imagery, then estimate it back.

        Returns (estimated_state, relative L2 of the estimate). With perception
        off, the planner is handed the true state - the proposal's
        perfect-perception upper bound.
        """
        if not self.use_perception or self.perceive is None:
            return true_state, 0.0

        images = torch.stack([render_field(s.cpu()) for s in true_state]).to(self.device)
        estimate, _ = self.perceive(images)
        # The perception head reconstructs the tracked channel; the rest of the
        # state is passed through, since imagery does not observe it.
        merged = true_state.clone()
        channels = min(estimate.shape[1], merged.shape[1])
        merged[:, :channels] = estimate[:, :channels]
        return merged, float(relative_l2(estimate[:, :channels], true_state[:, :channels]))

    # -- explanation -------------------------------------------------------- #
    @torch.no_grad()
    def explain_action(
        self,
        state: Tensor,
        dist: torch.distributions.Normal,
        action: Tensor,
        cost: Tensor,
        reward: Tensor,
    ) -> tuple[list[str], Tensor]:
        centers = self.env.basis.centers.cpu()
        reference = [
            render_brief(
                BriefContext(
                    step=self.env.t,
                    amplitudes=action[i].cpu(),
                    centers=centers,
                    predicted_cost=float(cost[i]),
                    cost_limit=self.env.task.cost_limit,
                    predicted_reward=float(reward[i]),
                    log_std=dist.stddev[i].log().cpu(),
                )
            )
            for i in range(action.shape[0])
        ]
        if self.explain is None:
            briefs = reference
        else:
            condition = build_condition(
                state, dist.mean, dist.stddev.log(), cost, self.env.task.cost_limit, reward
            )
            briefs, _ = self.explain.generate(condition, greedy=True)

        parsed, _ = self.parser.batch_distribution(briefs, self.device)
        score, _ = faithfulness_score(
            torch.distributions.Normal(action, dist.stddev), parsed
        )
        return briefs, score

    # -- loop --------------------------------------------------------------- #
    @torch.no_grad()
    def rollout(self, batch: int = 4, generator: torch.Generator | None = None) -> RolloutReport:
        state = self.env.reset(batch, generator)
        report = RolloutReport()
        cumulative = torch.zeros(batch, device=self.device)

        for step in range(self.env.horizon):
            observed, perception_error = self.perceive_state(state)
            dist = self.policy.distribution(observed)
            action = torch.tanh(dist.mean)

            state, reward, cost, done = self.env.step(action)
            cumulative = cumulative + cost

            briefs, score = self.explain_action(observed, dist, action, cost, reward)
            report.records.append(
                StepRecord(
                    step=step,
                    perception_rel_l2=perception_error,
                    reward=float(reward.mean()),
                    cost=float(cost.mean()),
                    cumulative_cost=float(cumulative.mean()),
                    faithfulness=float(score.mean()),
                    brief=briefs[0],
                    action=[round(a, 3) for a in action[0].cpu().tolist()],
                )
            )
            if done:
                break
        return report


# Within-family parameter shifts: a different physical regime of the same PDE
# (faster advection, weaker diffusion, sharper fronts). Unlike a cross-family
# swap these keep the channel count, so one surrogate can be scored on both.
TRANSFER_PROTOCOLS: dict[str, dict[str, float]] = {
    "dar": {"D": 0.003, "vx": -0.5, "vy": 0.45, "r": 1.0},
    "swe": {"g": 1.6, "H": 0.6, "drag": 0.25},
    "rdf": {"Du": 0.010, "Dv": 0.001, "eps": 0.15, "b": 0.5},
}


def transfer_gap(
    surrogate: nn.Module,
    source_testbed: str,
    target_testbed: str | None = None,
    target_params: dict[str, float] | None = None,
    grid: int = 64,
    steps: int = 8,
    batch: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, float | str]:
    """Transfer gap of a surrogate trained on `source_testbed`.

    Two protocols, both reporting `target_rel_l2 - source_rel_l2`:

    * cross-family  - pass `target_testbed`. Testbeds of differing arity require
      a channel-padded surrogate (`make_surrogate(..., padded=True)`); with a
      bespoke single-family model the mismatch is reported rather than fudged.
    * parameter shift - pass `target_params` (or neither, to use
      `TRANSFER_PROTOCOLS[source_testbed]`): the same PDE family in a different
      physical regime.

    Both regimes are scored on the *same* initial conditions, so the gap
    reflects the dynamics rather than a resampled evaluation set. Cross-family
    initial conditions come from the *target* testbed, since a dar initial
    condition is not a valid shallow-water state.
    """
    from .simulate.multifamily import supports_cross_family
    from .simulate.rollout import surrogate_fidelity
    from .simulate.solvers import make_testbed

    device = torch.device(device)
    source = make_testbed(source_testbed, grid=grid, device=device)
    cross_family = target_testbed is not None and target_testbed != source_testbed

    if cross_family:
        target = make_testbed(target_testbed, grid=grid, device=device)  # type: ignore[arg-type]
        protocol = f"cross-family {source_testbed}->{target_testbed}"
        if source.n_channels != target.n_channels and not supports_cross_family(surrogate):
            return {
                "transfer_gap": float("nan"),
                "protocol": protocol,
                "note": f"{source_testbed} has {source.n_channels} channel(s), "
                        f"{target_testbed} has {target.n_channels}, and this surrogate "
                        "is single-family. Train with make_surrogate(..., padded=True) "
                        "to score cross-family transfer.",
            }
    else:
        params = target_params or TRANSFER_PROTOCOLS[source_testbed]
        target = make_testbed(source_testbed, grid=grid, device=device, **params)
        protocol = f"parameter shift {source_testbed} {params}"

    controls = torch.zeros(batch, steps, 1, grid, grid, device=device)
    results = {}
    for name, testbed in (("source", source), ("target", target)):
        # Same initial conditions across a parameter shift (the state space is
        # shared); each family's own initial conditions across a family swap.
        seed_source = testbed if cross_family else source
        u0 = seed_source.initial_condition(batch, torch.Generator().manual_seed(0)).to(device)
        results[name] = float(
            surrogate_fidelity(surrogate, testbed, u0, controls, steps)["rel_l2_final"]
        )
    return {
        "source_rel_l2": results["source"],
        "target_rel_l2": results["target"],
        "transfer_gap": results["target"] - results["source"],
        "protocol": protocol,
    }
