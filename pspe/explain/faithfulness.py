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


def kl_normal(p: torch.distributions.Normal, q: torch.distributions.Normal) -> Tensor:
    """KL(p || q), summed over action dimensions -> (B,)."""
    return torch.distributions.kl_divergence(p, q).sum(-1)


def faithfulness_score(
    policy_dist: torch.distributions.Normal,
    parsed_dist: torch.distributions.Normal,
) -> tuple[Tensor, Tensor]:
    """Return (F, KL) with F = exp(-KL), both shape (B,)."""
    kl = kl_normal(policy_dist, parsed_dist).clamp(min=0.0)
    return torch.exp(-kl), kl


class FaithfulnessObjective:
    """Combines the supervised and REINFORCE terms with a moving baseline."""

    def __init__(self, weight_supervised: float = 1.0, weight_faithful: float = 1.0,
                 baseline_ema: float = 0.9) -> None:
        self.weight_supervised = weight_supervised
        self.weight_faithful = weight_faithful
        self.baseline_ema = baseline_ema
        self.baseline = 0.0
        self._initialised = False

    def __call__(
        self,
        supervised_nll: Tensor,     # (B,) NLL of the reference brief
        sample_logprob: Tensor,     # (B,) log-prob of the sampled brief
        score: Tensor,              # (B,) F of the sampled brief, detached
        use_faithfulness: bool = True,
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
            advantage = score - self.baseline
            reinforce = -(advantage * sample_logprob).mean()
            loss = loss + self.weight_faithful * reinforce

        return loss, {
            "loss/total": float(loss.detach()),
            "loss/supervised_nll": float(supervised_nll.mean().detach()),
            "loss/reinforce": float(reinforce.detach()),
            "metric/faithfulness": mean_score,
            "metric/faithfulness_baseline": self.baseline,
        }
