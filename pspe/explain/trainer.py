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
from .faithfulness import (
    FaithfulnessObjective,
    contrastive_faithfulness,
    faithfulness_score,
)
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
    # REINFORCE rollout settings. Measured: at the generation temperature (0.8)
    # an under-trained generator's samples never parse — zero actuators
    # mentioned in every sample — so every score equals the parser fallback and
    # the faithfulness gradient is exactly zero. Sampling close to the mode keeps
    # samples parseable; the self-critical baseline (greedy decode's score)
    # makes the advantage informative even when parse rates are low.
    reinforce_temperature: float = 0.3
    self_critical: bool = True
    # Report F from per-dimension KL. Required above ~16 action dimensions,
    # where summed KL sends exp(-KL) to 0 for every generator (measured on the
    # 64-patch firebreak plans: every arm exactly 0, reference brief 0.985).
    # Off by default so the dar/rdf numbers stay comparable with the report.
    per_dim_kl: bool = False
    # Contrastive faithfulness: a brief must score better on its own state than
    # on the other states in the batch. Added after the permutation control
    # found aligned and shuffled faithfulness equal to four decimals on every
    # arm (docs/results/EXPLAIN_PERMUTATION.md): the supervised term alone is
    # minimised by a constant brief. Differentiable through the reference
    # brief's teacher-forced logits, not through sampled text.
    weight_contrastive: float = 0.0
    contrastive_temperature: float = 1.0
    # Pairs scored per step are sub_batch^2, so this caps the cost: the term
    # needs B forward passes per brief, which at batch 8 is 8x the supervised
    # path. 4 keeps it at 2x for a log-4 discrimination task, still far above
    # chance if the prefix carries anything at all.
    contrastive_sub_batch: int = 4
    # Patches a brief names. The parser zeroes every actuator the brief omits,
    # so on a 64-patch plan a brief must either name all of them (long, and the
    # generator degenerates) or the score must accept a shortlist. Default 64
    # keeps the dar/rdf behaviour (9 actuators, all named).
    brief_top_k: int = 64
    # Post-hoc control (Section 7.2): the policy is fixed and the explainer is
    # never trained against it — briefs come out of the untouched generator
    # after the fact, TalkToAgent-style. This is the comparison the whole
    # trained-in argument rests on, and `use_faithfulness=False` is *not* it:
    # that arm still trains the generator on the brief templates, so it has
    # already seen the policy's behaviour through supervised NLL.
    posthoc: bool = False
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
            self.cfg.weight_supervised, self.cfg.weight_faithful,
            weight_contrastive=self.cfg.weight_contrastive,
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.logger = logger or RunLogger(self.cfg.log_dir)
        self.generator = torch.Generator().manual_seed(self.cfg.seed)

    def _condition_contrastive(self, condition: Tensor, token_ids: Tensor,
                               mask: Tensor) -> Tensor:
        """-log softmax over conditions of brief i's own condition -> (B,).

        NLL of every (brief, condition) pair in the batch; the diagonal must
        win. A model that ignores the prefix scores every pair alike and pays
        log B on every brief, so the only way down is to use the condition.
        """
        b = min(condition.shape[0], self.cfg.contrastive_sub_batch)
        condition, token_ids, mask = condition[:b], token_ids[:b], mask[:b]
        cond = condition[None].expand(b, b, condition.shape[-1]).reshape(b * b, -1)
        toks = token_ids[:, None].expand(b, b, token_ids.shape[1]).reshape(b * b, -1)
        msk = mask[:, None].expand(b, b, mask.shape[1]).reshape(b * b, -1)
        nll = self.model(cond, toks, msk).view(b, b)        # (brief i, condition j)
        return torch.nn.functional.cross_entropy(
            -nll / self.cfg.contrastive_temperature,
            torch.arange(b, device=nll.device), reduction="none")
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
                    top_k=self.cfg.brief_top_k,
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
        iterations = 0 if self.cfg.posthoc else self.cfg.iterations
        with timer() as clock:
            for it in range(1, iterations + 1):
                condition, briefs, reference = self.sample_batch(self.cfg.batch)
                token_ids, mask = self.tokenizer.batch_encode(
                    briefs, max_len=self.model.cfg.max_len
                )
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device).float()

                supervised_nll = self.model(condition, token_ids, mask)
                # Memory-efficient: sample without grad, score with one forward.
                # `generate` (grad through the whole loop) OOMs a 1.5B backbone.
                sampled, sample_logprob = self.model.sample_and_score(
                    condition, temperature=self.cfg.reinforce_temperature
                )
                parsed, extras = self.parser.batch_distribution(sampled, self.device)
                score, kl = faithfulness_score(reference, parsed, self.cfg.per_dim_kl)

                critic_score = None
                if self.cfg.use_faithfulness and self.cfg.self_critical:
                    with torch.no_grad():
                        greedy_texts, _ = self.model.generate(condition, greedy=True)
                    greedy_parsed, _ = self.parser.batch_distribution(greedy_texts, self.device)
                    critic_score, _ = faithfulness_score(reference, greedy_parsed, self.cfg.per_dim_kl)

                contrastive = None
                if self.cfg.weight_contrastive > 0:
                    # Differentiable, and deliberately NOT on sampled text: a
                    # REINFORCE version rides the same dead path as the
                    # faithfulness term (unparseable samples -> constant
                    # advantage -> zero gradient), which is what made that term
                    # null. Here the reference brief must be cheaper under its
                    # own condition than under the other conditions in the
                    # batch, so the prefix has to carry state information for
                    # the loss itself to go down.
                    contrastive = self._condition_contrastive(condition, token_ids, mask)
                loss, components = self.objective(
                    supervised_nll, sample_logprob, score, self.cfg.use_faithfulness,
                    critic_score=critic_score, contrastive=contrastive,
                )
                # Parse rate is the starvation diagnostic: near zero means the
                # faithfulness term has nothing to learn from, whatever it reports.
                components["metric/parse_rate"] = sum(
                    1.0 for e in extras if e["mentioned_actuators"] > 0
                ) / len(extras)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 5.0
                )
                self.optimizer.step()

                if getattr(self, "gap_every", 0) and it % self.gap_every == 0:
                    # Aligned minus shuffled faithfulness on fresh states: the
                    # only number that says whether briefs are state-specific.
                    al = self.faithfulness_samples(32)
                    sh = self.faithfulness_samples_permuted(32)
                    self.logger.log(it, **{"metric/gap": float(al.mean() - sh.mean()),
                                           "metric/f_aligned": float(al.mean())})
                    print(f"  it {it}: nll {float(supervised_nll.mean()):.3f} "
                          f"F {al.mean():.4f} gap {al.mean() - sh.mean():+.4f}", flush=True)

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
            "posthoc": self.cfg.posthoc,
            "train_iterations": float(iterations),
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
        }
        self.logger.log_summary(**summary)
        return summary

    # -- evaluation ---------------------------------------------------------- #
    @torch.no_grad()
    def faithfulness_samples(self, n: int, batch: int = 16) -> "np.ndarray":
        """Per-brief F on `n` fresh states — the input to the conformal certificate.

        Fresh draws every call, so a calibration set and a test set built from
        two calls are disjoint and, under a fixed policy, exchangeable.
        """
        import numpy as np
        scores = []
        while sum(len(s) for s in scores) < n:
            condition, _, reference_dist = self.sample_batch(min(batch, n))
            generated, _ = self.model.generate(condition, greedy=True)
            parsed, _ = self.parser.batch_distribution(generated, self.device)
            score, _ = faithfulness_score(reference_dist, parsed, self.cfg.per_dim_kl)
            scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores)[:n]

    @torch.no_grad()
    def faithfulness_samples_permuted(self, n: int, batch: int = 16) -> "np.ndarray":
        """F with each brief scored against another state's action distribution.

        The control for a generator that learned the brief format and a
        generic action rather than the state's own: such a generator scores
        nearly as well shuffled as aligned, and the aligned-minus-shuffled gap
        is what is actually state-specific. Same draws as
        `faithfulness_samples`, only the pairing differs.
        """
        import numpy as np
        scores = []
        while sum(len(s) for s in scores) < n:
            condition, _, reference_dist = self.sample_batch(min(batch, n))
            generated, _ = self.model.generate(condition, greedy=True)
            parsed, _ = self.parser.batch_distribution(generated, self.device)
            roll = torch.roll(reference_dist.mean, 1, dims=0)
            shifted = torch.distributions.Normal(roll, torch.roll(reference_dist.stddev, 1, dims=0))
            score, _ = faithfulness_score(shifted, parsed, self.cfg.per_dim_kl)
            scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores)[:n]

    @torch.no_grad()
    def evaluate(self, batch: int = 16, dump: str | Path | None = None) -> dict[str, float]:
        """Faithfulness of greedily generated briefs, plus a brief dump for raters."""
        condition, references, reference_dist = self.sample_batch(batch)
        generated, _ = self.model.generate(condition, greedy=True)
        parsed, extras = self.parser.batch_distribution(generated, self.device)
        score, kl = faithfulness_score(reference_dist, parsed, self.cfg.per_dim_kl)

        # The reference brief is the faithfulness ceiling: it is the exact
        # rendering of the planner's action, so its score bounds what any
        # generated brief can reach given the quantisation.
        ref_parsed, _ = self.parser.batch_distribution(references, self.device)
        ref_score, _ = faithfulness_score(reference_dist, ref_parsed, self.cfg.per_dim_kl)

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
