# Planning under partial observation: the twin's sync step, on real fire data

`eval/run_ndws_partial.py`, Vista jobs 1021237-9, 2026-09-24. Three occlusion
rates x three seeds, 800 held-out NDWS fires each, planning through the same
frozen next-day U-Net. Raw output in `runs/ndws_partial_r*/`.

## The question

Every other planning result in this project is handed the full fire mask, which
is the one thing a twin never gets. A real twin sees a cloud-obscured,
sensor-gapped observation and must decide anyway. NDWS carries this natively:
cells the sensor could not label are marked -1, and over 400 training patches
the median has none, the 90th percentile 1.5%, and 22 of 400 exceed 10%.

Occlusion here is spatially correlated (low-resolution noise upsampled and
thresholded) rather than scattered pixels, because a blob over the fire front is
the case that actually happens and the one that actually hurts.

Four ways to fill the gap, all planning through the same surrogate, **planning
on the belief and scoring on the truth** — acting on a wrong belief has to be
paid for in the real world, not the imagined one:

| belief | what it assumes about hidden cells |
|---|---|
| blind | nothing is burning there (what the pipeline did before) |
| persist | they look like the patch's visible mean |
| perceive | a learned estimator's probability, trained self-supervised |
| perceive-hard | the same estimator, thresholded to a binary mask |

## Results

Burn reduction (%), PSPE planner, mean ± std over 3 seeds:

| occlusion | blind | persist | perceive | perceive-hard |
|---|---|---|---|---|
| 0% | 41.11 ± 0.75 | — | — | — |
| 15% | **39.72 ± 0.77** | 39.72 ± 0.66 | 38.20 ± 1.15 | 39.44 ± 0.66 |
| 35% | **37.44 ± 0.19** | 37.41 ± 0.29 | 33.30 ± 1.98 | 36.63 ± 0.93 |
| 60% | **34.52 ± 0.56** | 34.44 ± 0.56 | 27.34 ± 2.44 | 33.04 ± 1.10 |

Reconstruction quality on the hidden cells (average precision):

| occlusion | blind | persist | perceive | perceive-hard |
|---|---|---|---|---|
| 15% | 0.006 | 0.041 | **0.594** | 0.284 |
| 35% | 0.006 | 0.037 | **0.489** | 0.224 |
| 60% | 0.005 | 0.032 | **0.346** | 0.147 |

Paired against `blind`, PSPE planner, across all 9 (rate, seed) configurations:

| belief | mean change | better in | paired t |
|---|---|---|---|
| persist | −0.04 pts | 5/9 | −1.04 (n.s.) |
| **perceive** | **−4.28 pts** | **0/9** | **−4.33** |
| perceive-hard | −0.86 pts | 1/9 | −2.79 |

## Two findings

### 1. Planning degrades gracefully with observation loss

Losing **60% of the observation costs 6.6 points of burn reduction out of 41**
— 16% relative — and the planner still beats the full-state greedy heuristic
(22.2%) by a wide margin. Variance across seeds stays tight (±0.19 to ±0.75).
This is the practically useful number: the planning advantage survives the
observation quality a real deployment actually has.

### 2. Reconstructing the hidden state makes the decision worse

The learned estimator recovers hidden cells ~100x better than assuming they are
empty (AP 0.594 against 0.006), and it genuinely changes the plan — Jaccard
overlap between the blind and perceive plans is 0.49, so half the treated cells
differ. And yet the outcome is worse in **0 of 9** configurations, by 4.3 points
on average.

The `perceive-hard` control separates two explanations:

* **Representation (most of it).** Thresholding the estimator's probabilities to
  a binary mask recovers 3.4 of the 4.3 points. The surrogate was trained on a
  near-binary mask and reacts badly to a smeared probability field, so most of
  the damage was the *format* of the belief, not its content.
* **A real residual cost (the rest).** Even hard-thresholded imputation is
  slightly worse than blind (−0.86 pts, 1/9 better, t = −2.79). Under a hard
  crew budget, spending on cells that merely *might* be burning takes crews away
  from cells confirmed to be burning. Imputing uncertain state has a genuine
  price when resources are scarce.

Persistence is indistinguishable from blind (−0.04, n.s.) — it adds information
by the reconstruction metric (AP 0.041 vs 0.006) and none by the decision one.

## The caveat that bounds this

Fire is a **rare-event field**: prevalence is 1.3%. "Assume nothing is burning
where I cannot see" is therefore close to the base rate, which makes `blind` a
strong prior almost for free. On a dense field — flood depth, surface
temperature, pollutant concentration — the same experiment could easily come out
the other way, because there the uninformed prior is badly wrong everywhere.

So the finding is not "twins should never impute". It is: **on a sparse field,
under a hard budget, act on what you observe; the value of reconstruction has to
be demonstrated on the decision, not on the reconstruction metric.**

## For the twin claim

This is the sync step measured on real data with a decision-relevant metric,
and it answers the question posed when the experiment was designed — whether a
state estimator that improves reconstruction improves the decision. Here it does
not, and the measurement that shows it (paired across rates and seeds, with a
representation control) is the contribution.
