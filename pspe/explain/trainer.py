"""Phase 4 trainer: faithfulness-regularised brief generation.

Training data is generated on the fly by rolling the planner out in the
environment: each visited state gives a policy action distribution, a predicted
cost and reward, and hence a reference brief. The model learns to reproduce
those briefs (supervised term) and is rewarded for briefs whose *parse* matches
the policy distribution (faithfulness term).

`F(b_t)` is logged every step, which is the Phase 4 acceptance check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..envs.pde_env import BatchedFieldEnv
from ..plan.policy import GaussianFieldPolicy
from ..utils.common import count_parameters, peak_memory_mb, timer
from ..utils.logging import RunLogger
from .brief import BriefContext, build_vocabulary, render_brief
from .faithfulness import FaithfulnessObjective, faithfulness_score
from .model import ExplainConfig, ExplainModule
from .parser import FrozenBriefParser
from .tokenizer import WordTokenizer

Tensor = torch.Tensor


@dataclass
class ExplainTrainConfig:
    iterations: int = 200
    batch: int = 16
    lr: float = 1e-3
    weight_supervised: float = 1.0
    weight_faithful: float = 1.0
    use_faithfulness: bool = True   # ablation switch: faithfulness loss on/off
    eval_every: int = 25
    seed: int = 0
    log_dir: str = "runs/explain"


def build_condition(
    state: Tensor,
    mean: Tensor,
    log_std: Tensor,
    cost: Tensor,
    cost_limit: float,
    reward: Tensor,
) -> Tensor:
    """Planner summary -> conditioning vector (B, 2K + 5)."""
    tracked = state[:, 0]
    field_stats = torch.stack(
        [
            tracked.mean(dim=(-2, -1)),
            tracked.amax(dim=(-2, -1)),
            cost,
            torch.full_like(cost, cost_limit),
            reward,
        ],
        dim=-1,
    )
    return torch.cat([mean, log_std, field_stats], dim=-1)


def condition_dim(n_actuators: int) -> int:
    return 2 * n_actuators + 5


class ExplainTrainer:
    def __init__(
        self,
        env: BatchedFieldEnv,
        policy: GaussianFieldPolicy,
        model_cfg: ExplainConfig | None = None,
        train_cfg: ExplainTrainConfig | None = None,
        logger: RunLogger | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = train_cfg or ExplainTrainConfig()
        self.device = torch.device(device)
        self.env = env
        self.policy = policy.to(self.device).eval()
        for p in self.policy.parameters():  # the planner is fixed in Phase 4
            p.requires_grad_(False)

        n_actuators = env.action_dim
        self.tokenizer = WordTokenizer(build_vocabulary(n_actuators, env.basis.centers.cpu()))
        self.parser = FrozenBriefParser(n_actuators)
        self.model = ExplainModule(
            self.tokenizer, model_cfg, cond_features=condition_dim(n_actuators)
        ).to(self.device)
        self.objective = FaithfulnessObjective(
            self.cfg.weight_supervised, self.cfg.weight_faithful
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.logger = logger or RunLogger(self.cfg.log_dir)
        self.generator = torch.Generator().manual_seed(self.cfg.seed)

    # -- data ---------------------------------------------------------------- #
    @torch.no_grad()
    def sample_batch(self, batch: int) -> tuple[Tensor, list[str], torch.distributions.Normal]:
        """Roll the planner one step and build (condition, reference brief, pi)."""
        state = self.env.reset(batch, self.generator)
        # Advance a random number of steps so briefs cover the whole episode,
        # not only the initial condition.
        for _ in range(int(torch.randint(0, 4, (1,), generator=self.generator))):
            action, _ = self.policy.sample(state, deterministic=False)
            state, _, _, _ = self.env.step(action)
            state = state.detach()
            self.env.state = state

        dist = self.policy.distribution(state)
        mean, log_std = dist.mean, dist.stddev.log()
        action = torch.tanh(mean)
        next_state, reward, cost, _ = self.env.step(action)

        condition = build_condition(
            state, mean, log_std, cost, self.env.task.cost_limit, reward
        )
        centers = self.env.basis.centers.cpu()
        briefs = [
            render_brief(
                BriefContext(
                    step=self.env.t,
                    amplitudes=action[i].cpu(),
                    centers=centers,
                    predicted_cost=float(cost[i]),
                    cost_limit=self.env.task.cost_limit,
                    predicted_reward=float(reward[i]),
                    log_std=log_std[i].cpu(),
                )
            )
            for i in range(batch)
        ]
        # The parser reads amplitudes, so the reference policy distribution must
        # be expressed in the same (pre-tanh mean, std) coordinates the brief
        # quotes - i.e. the action the brief actually names.
        reference = torch.distributions.Normal(action, dist.stddev)
        return condition, briefs, reference

    # -- training ------------------------------------------------------------ #
    def train(self) -> dict[str, float]:
        trainable, total = count_parameters(self.model)
        self.logger.log(
            0,
            **{
                "params/trainable": float(trainable),
                "params/total": float(total),
                "params/trainable_fraction": trainable / max(total, 1),
            },
        )
        with timer() as clock:
            for it in range(1, self.cfg.iterations + 1):
                condition, briefs, reference = self.sample_batch(self.cfg.batch)
                token_ids, mask = self.tokenizer.batch_encode(
                    briefs, max_len=self.model.cfg.max_len
                )
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device).float()

                supervised_nll = self.model(condition, token_ids, mask)
                # Memory-efficient: sample without grad, score with one forward.
                # `generate` (grad through the whole loop) OOMs a 1.5B backbone.
                sampled, sample_logprob = self.model.sample_and_score(condition)
                parsed, extras = self.parser.batch_distribution(sampled, self.device)
                score, kl = faithfulness_score(reference, parsed)

                loss, components = self.objective(
                    supervised_nll, sample_logprob, score, self.cfg.use_faithfulness
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 5.0
                )
                self.optimizer.step()

                self.logger.log(
                    it,
                    **components,
                    **{
                        "metric/kl": float(kl.mean()),
                        "metric/mentioned_actuators": sum(
                            e["mentioned_actuators"] for e in extras
                        ) / len(extras),
                        "metric/unk_rate": sum(
                            self.tokenizer.unknown_rate(s) for s in sampled
                        ) / len(sampled),
                    },
                )

        metrics = self.evaluate()
        summary = {
            **metrics,
            "params/trainable": float(trainable),
            "params/total": float(total),
            # Watermark: a randomly-initialised 4-layer stand-in is not a
            # Qwen2.5/Phi-3.5 faithfulness result.
            "backbone": self.model.cfg.backbone,
            "backbone_is_stub": self.model.is_stub_backbone,
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
        }
        self.logger.log_summary(**summary)
        return summary

    # -- evaluation ---------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(self, batch: int = 16, dump: str | Path | None = None) -> dict[str, float]:
        """Faithfulness of greedily generated briefs, plus a brief dump for raters."""
        condition, references, reference_dist = self.sample_batch(batch)
        generated, _ = self.model.generate(condition, greedy=True)
        parsed, extras = self.parser.batch_distribution(generated, self.device)
        score, kl = faithfulness_score(reference_dist, parsed)

        # The reference brief is the faithfulness ceiling: it is the exact
        # rendering of the planner's action, so its score bounds what any
        # generated brief can reach given the quantisation.
        ref_parsed, _ = self.parser.batch_distribution(references, self.device)
        ref_score, _ = faithfulness_score(reference_dist, ref_parsed)

        path = Path(dump) if dump else Path(self.cfg.log_dir) / "briefs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for i in range(batch):
                handle.write(
                    json.dumps(
                        {
                            "id": i,
                            "generated": generated[i],
                            "reference": references[i],
                            "faithfulness": float(score[i]),
                            "kl": float(kl[i]),
                            "mentioned_actuators": extras[i]["mentioned_actuators"],
                        }
                    )
                    + "\n"
                )

        return {
            "eval/faithfulness": float(score.mean()),
            "eval/faithfulness_reference": float(ref_score.mean()),
            "eval/kl": float(kl.mean()),
            "eval/briefs_path": str(path),  # kept as a string in summary.json
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "config": self.model.cfg.__dict__}, path
        )
