"""Faithfulness score and loss.

Faithfulness is agreement between the planner's action distribution and the
distribution a reader would infer from the brief:

    F(b_t) = exp( -KL( pi(.|s_t) || parse(b_t) ) )

**Deliberate divergence from the proposal**, which writes
`F(b_t) = 1 - KL(pi || parse(b_t))`. The linear form is unbounded below: any KL
above 1 makes F negative, and since KL is unbounded a badly wrong brief scores
arbitrarily negative. That breaks it both as a reported metric (no fixed scale,
so scores are not comparable across runs or testbeds) and as a REINFORCE
reward (one catastrophic sample dominates the batch's advantage). The
exponential form is bounded in (0, 1], is monotone in KL exactly as the linear
form is, and agrees with it to first order at small KL - which is the regime a
converged model sits in. The proposal text must be updated to match; see
`docs/proposal_deltas.md`.

F is computed *during* training - it is the learning signal, not a post-hoc
audit metric.

Because the parse of a sampled brief is non-differentiable text, the training
signal has two parts:

* a supervised term - cross-entropy toward the reference brief rendered from
  the planner's actual action (dense, low-variance);
* a REINFORCE term on sampled briefs weighted by (F - baseline), which is what
  actually optimises faithfulness of what the model *generates* rather than
  what it is force-fed.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def kl_normal(p: torch.distributions.Normal, q: torch.distributions.Normal,
              per_dim: bool = False) -> Tensor:
    """KL(p || q), summed over action dimensions -> (B,).

    `per_dim` divides by the action dimension. Summed KL makes F = exp(-KL)
    incomparable across action spaces and unusable above a few dimensions: on
    the 64-patch firebreak plans every arm scored exactly 0 (KL 15 to 22)
    while the reference brief scored 0.985, so the metric had no resolution
    left to rank generators by. Per-dimension KL is the same quantity in
    nats per actuator and keeps F on one scale whatever the plan's size."""
    out = torch.distributions.kl_divergence(p, q).sum(-1)
    return out / p.mean.shape[-1] if per_dim else out


def faithfulness_score(
    policy_dist: torch.distributions.Normal,
    parsed_dist: torch.distributions.Normal,
    per_dim: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return (F, KL) with F = exp(-KL), both shape (B,).

    `per_dim` reports KL in nats per action dimension; see `kl_normal`. The
    dar/rdf numbers in the report are the summed form (9 actuators); anything
    with a large action space should use the per-dimension form, and must say
    which it used, because the two are not comparable."""
    kl = kl_normal(policy_dist, parsed_dist, per_dim=per_dim).clamp(min=0.0)
    return torch.exp(-kl), kl


def contrastive_faithfulness(policy_dist: torch.distributions.Normal,
                             parsed_dist: torch.distributions.Normal,
                             per_dim: bool = False, temperature: float = 1.0) -> Tensor:
    """-log softmax over states of the brief's own state, per brief -> (B,).

    The permutation control showed briefs carrying no state-specific
    information: aligned faithfulness equalled shuffled to four decimals on
    every arm and both testbeds. This is that control turned into a loss. For
    brief i, score it against every state j in the batch and require the
    diagonal to win:

        L_i = -log [ exp(-KL_ii / T) / sum_j exp(-KL_ij / T) ]

    A constant brief scores every state alike and pays the full log B; only a
    brief that matches its own state better than its neighbours' escapes.
    """
    b, k = parsed_dist.mean.shape
    pol = torch.distributions.Normal(policy_dist.mean[:, None].expand(b, b, k),
                                     policy_dist.stddev[:, None].expand(b, b, k))
    par = torch.distributions.Normal(parsed_dist.mean[None].expand(b, b, k),
                                     parsed_dist.stddev[None].expand(b, b, k))
    kl = torch.distributions.kl_divergence(pol, par).sum(-1)      # (state j, brief i)
    if per_dim:
        kl = kl / k
    logits = (-kl / temperature).transpose(0, 1)                  # (brief i, state j)
    return torch.nn.functional.cross_entropy(
        logits, torch.arange(b, device=logits.device), reduction="none")


class FaithfulnessObjective:
    """Combines the supervised, REINFORCE and contrastive terms."""

    def __init__(self, weight_supervised: float = 1.0, weight_faithful: float = 1.0,
                 baseline_ema: float = 0.9, weight_contrastive: float = 0.0) -> None:
        self.weight_supervised = weight_supervised
        self.weight_faithful = weight_faithful
        self.weight_contrastive = weight_contrastive
        self.baseline_ema = baseline_ema
        self.baseline = 0.0
        self._initialised = False

    def __call__(
        self,
        supervised_nll: Tensor,     # (B,) NLL of the reference brief
        sample_logprob: Tensor,     # (B,) log-prob of the sampled brief
        score: Tensor,              # (B,) F of the sampled brief, detached
        use_faithfulness: bool = True,
        critic_score: Tensor | None = None,   # (B,) F of the greedy brief, if SCST
        contrastive: Tensor | None = None,    # (B,) per-brief contrastive loss, differentiable
    ) -> tuple[Tensor, dict[str, float]]:
        score = score.detach()
        mean_score = float(score.mean())
        if not self._initialised:
            self.baseline = mean_score
            self._initialised = True
        else:
            self.baseline = (
                self.baseline_ema * self.baseline + (1 - self.baseline_ema) * mean_score
            )

        loss = self.weight_supervised * supervised_nll.mean()
        reinforce = torch.zeros((), device=supervised_nll.device)
        if use_faithfulness:
            # Self-critical baseline when available: the greedy brief's own
            # score. A batch-EMA baseline gave exactly zero gradient on both the
            # stub and Qwen2.5 — every stochastic sample was unparseable, so all
            # scores equalled the parser fallback and the advantage vanished.
            # Against the greedy decode, a sample that parses *worse* than the
            # mode is penalised and one that parses better is rewarded, which is
            # signal even when most samples fail.
            baseline = critic_score.detach() if critic_score is not None else self.baseline
            advantage = score - baseline
            reinforce = -(advantage * sample_logprob).mean()
            loss = loss + self.weight_faithful * reinforce

        contrastive_value = 0.0
        if contrastive is not None and self.weight_contrastive > 0:
            loss = loss + self.weight_contrastive * contrastive.mean()
            contrastive_value = float(contrastive.mean().detach())

        return loss, {
            "loss/contrastive": contrastive_value,
            "loss/total": float(loss.detach()),
            "loss/supervised_nll": float(supervised_nll.mean().detach()),
            "loss/reinforce": float(reinforce.detach()),
            "metric/faithfulness": mean_score,
            "metric/faithfulness_baseline": self.baseline,
            "metric/advantage_std": float((score - (critic_score.detach() if critic_score is not None else self.baseline)).std()) if score.numel() > 1 else 0.0,
        }
