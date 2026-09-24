# The permutation control: Explain briefs are not state-specific

Vista, 2026-09-22, jobs 1015981 (NDWS fire plans) and 1016097 / 1016098 (dar).
Raw output in `runs/explain_perm_ti/`, `runs/explain_perm_ctrl/`,
`runs/ndws_explain_perm_*`. Qwen2.5-0.5B-Instruct, 200 iterations, 3 seeds,
128 greedy-decoded briefs per arm.

## The test

F(b) = exp(-KL(pi(.|s) || parse(b))) is supposed to measure whether the brief
for state s describes the action taken at s. The control scores each brief
against a **different** state's action distribution (roll by one within the
batch). If briefs are state-specific, aligned F should exceed shuffled F. If
the generator emits the same brief regardless of state, aligned and shuffled
use the identical multiset of (policy, brief) pairs and must be exactly equal.

## Result

dar, the testbed the published Explain numbers come from:

| seed | arm | F aligned | F shuffled | gap |
|---|---|---|---|---|
| 0 | trained-in | 0.2809 | 0.2805 | +0.0004 |
| 0 | no-faithful | 0.2809 | 0.2805 | +0.0004 |
| 0 | post-hoc | 0.0034 | 0.0034 | 0.0000 |
| 1 | trained-in | 0.5570 | 0.5570 | 0.0000 |
| 1 | post-hoc | 0.2448 | 0.2447 | +0.0001 |
| 2 | trained-in | 0.5657 | 0.5659 | -0.0002 |
| 2 | post-hoc | 0.2978 | 0.2983 | -0.0005 |

NDWS firebreak plans (64-dimensional, real fires):

| seed | F aligned | F shuffled |
|---|---|---|
| 0 | 0.7439 | 0.7436 |
| 1 | 0.7673 | 0.7674 |
| 2 | 0.8630 | 0.8639 |

Zero state-specific information, on both testbeds, for every arm including
the post-hoc control. The generated briefs are constant (up to the odd token)
across states; the sample briefs in
`runs/ndws_explain_ti/seed_0/briefs_heldout.jsonl` show the same two patches
with the same amplitudes for different fires.

## What this does to the claims

**Withdrawn.** "Trained-in explanations beat a post-hoc control" (dar:
0.468 vs 0.182, t = +21; NDWS: 0.814 vs 0.741, t = +3.61) is a true
measurement of the wrong thing. Both generators emit a near-constant brief;
the trained one's constant brief sits closer to the average policy action.
That is format and calibration learning, not faithfulness. No claim about
explaining a particular decision survives.

**Unaffected.** The conformal certificate machinery: coverage holds at 90%
and 95% on 3 of 3 seeds on real data, and the floor it certifies is honest
about the generator it is given. Certifying a constant brief is not a defect
of the certificate.

**Reinforced.** The faithfulness term adds nothing (t = -0.01 here, null in
every previous run). Now there is a mechanism for why: the term rewards
briefs whose parse matches the policy, but the generator never learned to
vary its output with the state, so there is nothing for the term to shape.

## Why it happened

The condition vector (2K + 5 numbers) is projected to a short prefix in front
of a frozen backbone whose weights never see the state. With 200 iterations
of supervised NLL on templated briefs, the fastest loss reduction is to
memorise the template and the marginal action, and the prefix gets ignored.
The faithfulness REINFORCE term is two orders of magnitude weaker than the
supervised term at its default weight, so it cannot pull the model off that
solution. Nothing in the training loop ever penalises a brief for being the
same as the last one.

## Three attempts to fix it, all negative

Vista, 2026-09-22. All on NDWS, the only testbed where the question is
answerable (see the section below on dar).

**1. More conditioning capacity.** Prefix 4 -> 16 vectors, cond_dim 64 -> 256,
deeper projection with LayerNorm, plus 10% condition dropout so the supervised
term cannot be minimised by the template alone. No effect: gap still zero.

**2. A differentiable contrastive objective** (`_condition_contrastive`). The
reference brief must be cheaper under its own condition than under the other
conditions in the batch, so the prefix has to carry state information for the
loss to fall. Deliberately not the REINFORCE form, which rides the same dead
path that made the faithfulness term null. Verified by unit test to reach the
prefix parameters. In training the loss sat at log B, the value for a model
that ignores its condition, and in one seed rose.

**3. 2.5x the training budget** (500 iterations, one seed, with and without
the contrastive term), logging the gap every 100 iterations:

| iteration | NLL/token, plain | gap | NLL/token, contrastive | gap |
|---|---|---|---|---|
| 100 | 0.662 | +0.0015 | 0.669 | +0.0012 |
| 200 | 0.544 | +0.0027 | 0.542 | -0.0004 |
| 300 | 0.757 | -0.0097 | 0.533 | -0.0019 |
| 400 | 1.000 | +0.0000 | 0.515 | +0.0041 |
| 500 | 0.524 | -0.0229 | 0.674 | -0.0209 |

The NLL plateaus near 0.5 nats per token and the gap never leaves zero.

**Not a plumbing bug.** On the Qwen path the prefix receives gradient
(norm 1.1e5 against 4.6e3 for the LoRA adapters), the NLL responds to the
condition (1.51 when it is zeroed), and the prefix embeddings differ across
rows (std 0.048). The model *can* use the condition; the optimisation does
not find a solution that does.

**What the plateau means.** The brief is mostly template tokens, which are
free, with the entropy concentrated on exactly the state-specific tokens (the
actuator index and the amplitude). At 0.5 nats per token averaged, those
number tokens are still uncertain, so greedy decoding takes the modal actuator
and the modal amplitude every time. The model has learned the marginal
distribution over plans, not the conditional.

## dar could never have answered this question

The trained dar policy is effectively state-independent: across 64 states the
action std is 1.2% of the action magnitude, and on the brief's 0.05
quantisation grid those 64 states give **8 distinct action vectors**. Its
reference briefs are therefore nearly identical, its supervised NLL reaches
0.05 nats per token, and a constant brief is close to correct. On dar the
permutation gap is zero no matter what the generator does.

So every Explain number measured on dar -- the 0.468 vs 0.182 comparison, the
faithfulness weight sweep, the conformal certificate -- was computed on a task
with nothing to explain. That is a testbed defect, not a model result, and it
is why the failure went unnoticed through three previous rounds of scrutiny.
NDWS, where 1,024 fires give 1,013 distinct plans (across-fire intensity std
3.06x the mean), is the only testbed in this repository on which explanation
faithfulness can be measured at all.

## What would have to change

1. **Cross-attention conditioning.** Not tried. Widening the prefix was the
   cheap version and it failed; letting the decoder attend to a per-state
   encoding at every layer is the version that changes the information path
   rather than its width.
2. **A structured output head.** The state-specific content is a short list of
   (patch, amplitude) pairs. Predicting those with a dedicated head, and
   generating prose around them, removes the requirement that a frozen LM
   learn to emit exact numbers from a continuous prefix. This is probably the
   right design and it is a different module, not a tuning change.
3. **Report the gap, not F.** Aligned-minus-shuffled is the quantity that
   means "state-specific", and it should be the reported metric from now on.
   F alone can be high for a constant brief, as every run here shows.
4. **Retire dar as an Explain testbed** and calibrate any replacement for
   action diversity before using it (`plan_diversity` in
   `eval/run_ndws_explain.py` is the check: distinct actions on the brief
   grid, and across-state std relative to the action magnitude).

## Process note

This check cost 40 minutes of GPU and invalidated a headline claim that had
survived three previous rounds of scrutiny, including a real-backbone
replication and a conformal certificate. Every faithfulness-style metric that
compares a generated artefact against a target should ship with a permutation
control; without one, "the metric went up" and "the model learned the task"
are indistinguishable.
