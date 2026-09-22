# PSPE experiment slides (corrected)

Slide-by-slide content for the experiments section of the deck. Every number
traces to a run under `runs/` or a section of `docs/technical_report.md`; the
source is named on each slide. Numbers marked *stub* were measured with the
stand-in backbone and are superseded by the real-backbone slide that follows.

Phase 1 and 2a results (joint training, Eq. 8 rule, Lipschitz bound,
faithfulness weight sweep, conformal certificate on Qwen) were fetched from
Vista on 2026-09-21 and are on slides 13 and 14; the rdf planner fix and baselines are slide 12. Full tables in
`runs/PHASE1_ANALYSIS.md`.

---

## Slide 1. Experimental protocol

- 5 seeds by default, mean ± sample std, paired t-test (df = 4; |t| > 2.78 is p < 0.05)
- Three measurement bugs found and fixed before any number was trusted:
  - constraint read from one final evaluation snapshot (identical runs scored 0% or 100% on timing)
  - evaluation perturbed training through the global RNG (baseline return shifted from -2.51 to -2.30)
  - optional-dependency tests skipped silently (a "tested" loader failed on first real use)
- Full sweeps: about 9.5 minutes on 5 GH200 nodes versus about 9 hours on one CPU
- 156 tests, including regression tests for each of the bugs above

Source: `docs/technical_report.md` section 6, `tests/`.

---

## Slide 2. Surrogate accuracy on a public benchmark

PDEBench 2D shallow water, 512 real trajectories, relative L2.

| Model | 1-step | Rollout | Params |
|---|---|---|---|
| FNO | 0.0018 | 0.0098 | 1.19M |
| DeepONet | 0.0044 | 0.0137 | 0.09M |
| GNOT | 0.0056 | 0.0543 | 0.31M |
| Published FNO (PDEBench, nRMSE) | -- | 0.0044 | -- |

- Scored on the benchmark's own trajectories, not our solver
- Metrics differ (relative L2 versus nRMSE): an anchor, not an exact match
- Same ranking on synthetic data, 5 seeds: FNO 0.057, DeepONet 0.256, GNOT 0.268

Source: `runs/simulate/`, report section 7.1.

---

## Slide 3. Planning against safe RL

dar testbed, 5 seeds, full budget. Cost limit 0.936.

| Method | Return | Cost | Violating evals | Worst cost | Real samples |
|---|---|---|---|---|---|
| PSPE (adaptive alpha, as built) | -2.448 ± 0.029 | 0.412 ± 0.37 | 7.3% | 1.109 | 3,072 |
| PSPE (fixed alpha) | -2.738 ± 0.365 | 2.752 ± 3.6 | 12.7% | 2.758 | 3,072 |
| PPO-Lagrangian | -2.558 ± 0.039 | 0.266 ± 0.009 | 0% | 0.297 | 38,400 |
| CPO | -2.541 ± 0.035 | 0.263 ± 0.011 | 0% | 0.277 | 38,400 |
| Saute RL | -2.537 ± 0.024 | 0.264 ± 0.008 | 0% | 0.275 | 38,400 |
| Primal-dual NPG | -2.544 ± 0.026 | 0.263 ± 0.008 | 0% | 0.275 | 38,400 |

- Beats every baseline on return: t = +3.84 (PPO-Lag), +4.87 (CPO), +17.95 (Saute), +4.52 (PD-NPG)
- 12.5x fewer real transitions as built (3,072 versus 38,400); 9.2x after the safety fix on the next slide (4,160)
- The return edge is partly bought by crossing the limit: see next slide

Source: `runs/seeds_v3/`, report section 7.2.

---

## Slide 4. Safety failure: diagnosed and fixed

**Before.** 7.3% of evaluations violate the constraint, worst case 1.109 against a 0.936 limit. Every model-free baseline: 0%.

**Why.** The dual controller is updated from cost measured on surrogate rollouts; violation is realised on the true dynamics. When the surrogate under-predicts cost, the policy is safe in-model and unsafe in reality. A property of model-based planning, not a coding error.

**Fix.** Probe: roll the current policy in the true environment every 20 iterations, feed that cost to the dual, bias-correct the surrogate cost in between. Margin: plan against limit minus 2 sigma, sigma being the measured spread of surrogate cost error.

dar, 5 seeds, paper budget. Returns differ from slide 3 because this run postdates the RNG fix (baseline moved from -2.45 to -2.30 and its own violation rate from 7.3% to 1.8%):

| Arm | Return | Violating evals | Worst cost | Real samples |
|---|---|---|---|---|
| baseline | -2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 (max 1.73) | 3,200 |
| probe only | -2.307 ± 0.16 | 1.8% | 0.56 ± 0.48 | 4,160 |
| margin only | -2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 | 3,200 |
| probe + margin | -2.301 ± 0.17 | 0% on all 5 seeds | 0.29 ± 0.15 | 4,160 |

- Violations eliminated in this run at no measurable return cost, for 960 extra real transitions
- Neither half works alone. Margin only is identical to baseline: with no probe there is no measured error to build a margin from
- Across all 15 seed-runs that use probe + margin (this run plus the alpha-rule and joint runs), 13 are clean and two had a single evaluation over the limit (0.98 and 1.71). Honest rate: about 1% of evaluations, down from 7.3%. A 7x reduction, not a guarantee

Source: `runs/constraint_fix/results_seeds.md`, report section 7.2.

---

## Slide 5. Adaptive gradient weighting

| Arm | Return advantage | Std | Violating evals |
|---|---|---|---|
| Adaptive alpha | +0.290 | 0.029 | 7.3% |
| Fixed alpha | baseline | 0.365 | 12.7% |

- The +0.290 mean difference is not significant (t = +1.66)
- Fixed alpha: one seed diverged (cost 8.1, lambda 19.1)
- Defensible claim: adaptive alpha prevents dual divergence. It does not reliably raise mean return. The signal is in the variance.
- Violation numbers here are pre-fix; with probe + margin the rate is about 1% of evaluations (slide 4)

Eq. 8 rule with the measured pathwise bias term: see slide 12.

Source: `runs/seeds_v3/`, report section 7.2.

---

## Slide 6. Ablations

5 seeds, full budget, stub backbones.

| Component | On | Off | t | Verdict |
|---|---|---|---|---|
| Physics-informed loss (rel L2) | 0.0567 ± 0.026 | 0.0496 ± 0.044 | +0.52 | no effect on final accuracy |
| Frozen perception (rel L2) | 0.0681 ± 0.013 | 0.0908 ± 0.017 | -3.97 | frozen wins |
| Faithfulness term (F(b)) | 0.643 ± 0.15 | 0.598 ± 0.006 | +0.67 | no effect |

- Physics loss buys convergence speed only: at a quick budget, final rel L2 0.117 with it versus 0.273 without
- Frozen perception confirmed on real SigLIP: frozen probe beats a CNN, t = -4.87 (slide 8)
- Faithfulness term confirmed null on real Qwen: trained-in and no-term identical to four decimals (slide 9)

Source: `runs/ablations/`, `runs/seeds_full/`, report section 7.3.

---

## Slide 7. Cross-family transfer

One surrogate per PDE family, rolled out on every family, 5 seeds.

| Transfer | Fidelity gap | Measurable? |
|---|---|---|
| rdf to dar | 0.358 ± 0.033 | yes, about 11 sigma from zero |
| swe to rdf | 0.311 ± 0.13 | yes |
| dar to rdf | 1.81 ± 1.1 | marginal |
| swe to dar | 0.087 ± 0.14 | no, crosses zero |
| rdf to swe | 10.5 ± 8.2 | no, std about equal to mean |
| dar to swe | 27.4 ± 26 | no, std about equal to mean |

- Direction survives: the two parabolic families transfer, transfer into the wave family collapses
- Magnitude does not: quote the direction, never the number
- Forecast error and decision loss decouple. dar to swe: fidelity gap 27, planning gap 0.007 ± 0.014 (nothing). dar to rdf: fidelity gap 1.8, planning gap 0.58 ± 0.22 (about 8% of return). The fidelity number does not predict the decision-relevant one.

Source: `runs/transfer_seeds/`, `runs/transfer_planning/`, report sections 7.4 and 7.7.

---

## Slide 8. Perception on a real backbone

SigLIP base (google/siglip-base-patch16-224), 3 seeds.

| Arm | Field rel L2 | Retrieval acc | Trainable params |
|---|---|---|---|
| PSPE (frozen + LoRA + contrastive) | 0.0596 ± 0.027 | 0.754 ± 0.026 | 1.64M |
| Probe (frozen + decoder only) | 0.0345 ± 0.011 | 0.135 ± 0.009 | 1.64M |
| CNN from scratch | 0.0564 ± 0.015 | 0.130 ± 0.016 | 0.56M |

- Frozen probe reconstructs best and beats the CNN (t = -4.87); real SigLIP features are strong enough that adapters do not help reconstruction
- The win is alignment: retrieval 0.754 versus 0.130, t = +37
- PSPE trades reconstruction accuracy for text-image alignment. Same shape as the stub result (stub: PSPE 0.0524, probe 0.0476, CNN 0.0422, retrieval 0.510 versus 0.137)

Source: `runs/perception_real/results_seeds.md`, report section 7.7.

---

## Slide 9. Explanation quality on a real language model

Qwen2.5-0.5B with LoRA, 3 seeds, faithfulness F(b) against a trained policy.

| Arm | F(b) | versus post-hoc |
|---|---|---|
| Trained-in (with faithfulness term) | 0.468 ± 0.16 | t = +21.3 |
| No faithfulness term | 0.468 ± 0.16 | same |
| Post-hoc control (never trained on this policy) | 0.182 ± 0.16 | -- |

- Joint training beats a real post-hoc control. This claim holds and widens on the real LM (stub: 0.168 versus 0.101, t = +5.53)
- The faithfulness term itself adds nothing: identical to four decimals. A zero-gradient bug (samples never parsed at the generation temperature) was found and fixed; the result did not change. At its default weight the term is two orders of magnitude below the supervised term
- What buys faithfulness is training the generator on the policy's briefs at all. A weight sweep is queued, not run
- The `no faithfulness term` arm is not the post-hoc baseline: it has still seen the policy through the supervised term

Source: `runs/explain_real_v2/results_seeds.md`, report section 7.5.

---

## Slide 10. Resolution generalization

One FNO trained at 64 by 64, evaluated at three grids, 5 seeds, relative L2.

| Grid | Cells | Rel L2 |
|---|---|---|
| 64 by 64 | 4,096 | 0.0298 ± 0.014 |
| 96 by 96 | 9,216 | 0.0298 ± 0.014 |
| 128 by 128 | 16,384 | 0.0300 ± 0.014 |

- Flat to three decimals: discretization invariance, not a memorised grid
- The first sweep returned NaN above 64 by 64. Cause was the truth solver, not the surrogate: fixed time step exceeded the explicit-scheme stability limit at finer grids. Sub-steps now scale with (grid / 64) squared

Replaces the earlier single-seed 32 / 48 / 64 numbers (0.089 / 0.089 / 0.088).

Source: `runs/resolution_seeds/resolution_seeds.json`, report section 7.6.

---

## Slide 11. Real wildfire data: Next Day Wildfire Spread

About 18k real 64 km patches of US fires. Day t fire mask plus eleven drivers to the day t+1 mask. Fire prevalence 1.3%. AUC-PR on the validation split, 3 seeds.

| Model | AUC-PR | Versus published 0.284 |
|---|---|---|
| U-Net | 0.277 ± 0.002 | 97.5% |
| Hybrid (U-Net + gated FNO) | 0.235 ± 0.005 | 83% |
| FNO alone (1 seed) | 0.200 | 70% |
| All-zeros floor | 0.011 | -- |

- First dynamics result scored against observed rather than simulated truth
- U-Net beats the hybrid on every seed (t = +12.5). The learned gate on the spectral path settles at 0.12 on every seed: offered the Fourier operator, the model declines it
- The operator's scope limit, measured: a spectral model truncates the high frequencies that are a fire front. The same FNO reaches 0.0098 on smooth shallow water
- Remaining gap to published 0.284 and to the 0.34 state of the art is architectural
- Three pipeline defects fixed on the way (driver normalisation, a class weight of 77, a residual across the input stack) account for most of the jump from 0.035 to 0.277

Figure: `docs/figures/pspe_app_wildfire.png`.

Source: `runs/ndws_seeds/results_seeds.md`, report section 7.8.

---

## Slide 12. The planner on a second family: rdf

Source: `runs/RDF_PLANNER_FIX.md`. dar cost limit 0.94; rdf cost limit 3.26 (reward-greedy cost 8.2).

**Three defects on rdf, each read from the training trace**

| step | symptom | fix | violating evals |
|---|---|---|---|
| as shipped | cost pinned at 8.0, lambda 38 useless, pathwise gradient norm 0.036 | | 85% |
| 1. action saturation | policy mean past the tanh slope; variance rule picks the dead branch | soft wall on the pre-tanh mean, floor on advantage std | 44% |
| 2. dual oscillation | lambda collapses to 0 on every dip, policy sprints, overshoots | integral gain 0.05 to 0.5 (lowering gain made it worse: 60%) | 14.5% |
| 3. tail | mean held at the limit violates on half of evaluations | margin covers the policy's own episode spread | 5.5%, worst 3.23 |

Same configuration on dar: return unchanged, 0% violations, worst case 3x smaller.

**Against safe RL on rdf** (5 seeds, 38,400 real transitions each)

| method | return | violating evals | worst | real samples |
|---|---|---|---|---|
| PPO-Lagrangian, stock | -7.66 ± 0.31 | 1.8% | 2.15 | 38,400 |
| PPO-Lagrangian with the same integral-gain fix | -7.69 ± 0.31 | 0% | 2.02 | 38,400 |
| CPO, Saute, primal-dual NPG | -7.97 to -8.08 | 0% | 0.63 | 38,400 |
| PSPE v2, probe + margin | -7.49 ± 0.62 | 5.5% | 3.23 | 7,360 |

- Paired t versus PPO-Lagrangian: +0.82. Parity on return; PPO-Lag has the better tail
- CPO, Saute and NPG are safe by inaction: cost sits at the do-nothing level
- The dual fix transfers: the same gain change took PPO-Lagrangian from 1.8% to 0%
- Planning claim across families: return edge on dar (t 3.8 to 18), parity on rdf, 5x to 9x fewer real samples on both

---

## Slide 13. Joint training and the theory checks

dar, 5 seeds, paper budget, probe + margin on. Source: `runs/PHASE1_ANALYSIS.md`.

**Joint versus disaggregated (the paper's central comparison)**

| Arm | Return | Surrogate held-out rel L2, before to after | Drift |
|---|---|---|---|
| Disaggregated (frozen surrogate) | -2.302 ± 0.17 | 0.0048 to 0.0048 | 0 |
| Joint, anchored (beta 0.1) | -2.296 ± 0.18 | 0.0048 to 0.0005 | -0.004 |
| Joint, unanchored | -2.369 ± 0.17 | 0.0048 to 0.975 | +0.970 |

- Joint versus disaggregated on return: t = +0.66, no difference
- Anchored joint surrogate is not captured; its held-out error falls 9x on every seed
- Unanchored joint surrogate stops predicting the data and return falls (t = -5.84): the failure mode, run on purpose
- Claim that holds: joint training is safe when anchored and destructive when not. Claim that does not: joint beats disaggregated
- Replicates on swe and rdf: return unchanged (t = +1.00 on rdf, identical on swe), anchored surrogate improves 5x to 9x, unanchored surrogate destroyed on all three testbeds

**Mixing rule** (fixed 0.5 / variance / Eq. 8 with measured bias)

| Arm | Return | Violating evals | Worst cost | Final alpha | Real samples |
|---|---|---|---|---|---|
| Fixed 0.5 | -2.415 ± 0.30 | 3.6% | 1.13 | 0.5 | 4,160 |
| Variance rule | -2.302 ± 0.17 | 1.8% | 0.38 | 0.992 | 4,160 |
| Eq. 8 | -2.299 ± 0.15 | 0% on 5 seeds | 0.34 | 0.992 | 6,080 |

- Measured pathwise bias is about 0.001, so Eq. 8 and the variance rule choose the same alpha; return t = +0.22
- Eq. 8's zero-violation result comes from its 1,920 extra truth rollouts, not from the formula

**Lipschitz assumption and Proposition 1**

| Arm | L_G | L_G below 1 | Measured return bias | Prop 1 bound, infinite horizon | Prop 1 bound, horizon 12 |
|---|---|---|---|---|---|
| Default FNO | 0.985 ± 0.013 | 4 of 5 | 0.10 | 422 | 9.8 |
| Spectral normalised | 0.964 ± 0.003 | 5 of 5 | 0.11 | 748 | 17.3 |

- Assumption 1 holds after training (initial L_G was 1.61)
- Infinite-horizon bound is vacuous (prefactor 2,450 at gamma 0.98). Finite-horizon bound holds on every seed and is 100x loose. Report the finite-horizon form and say so

---

## Slide 14. Explain on Qwen: weight sweep and certificate

3 seeds, Qwen2.5-0.5B. Source: `runs/faith_weights/`, `runs/conformal_real/`.

**Faithfulness weight sweep**

| Arm | F(b) | KL |
|---|---|---|
| Trained-in, w = 1 | 0.468 ± 0.16 | 0.81 |
| No faithfulness term | 0.468 ± 0.16 | 0.81 |
| w = 10 | 0.366 ± 0.29 | 1.59 |
| w = 100 | 0.218 ± 0.19 | 9.48 |
| w = 1000 | 0.399 ± 0.14 | 0.96 |
| Post-hoc | 0.182 ± 0.16 | 2.77 |

- Null at the default weight, harmful above it: at w = 100 the briefs are no better than post-hoc
- Decision: drop the term from the method, keep trained-in supervision

**Conformal certificate** (n_cal 200, n_test 400)

| Seed | Mean F | Certified floor at 90% | Coverage at 90% | Holds at 80 / 90 / 95% |
|---|---|---|---|---|
| 0 | 0.28 | 0.28 | 0.878 | yes / yes / no |
| 1 | 0.56 | 0.56 | 0.893 | yes / yes / yes |
| 2 | 0.57 | 0.56 | 0.895 | yes / yes / yes |

- Certificate holds at 90% on all three seeds; one seed fails at 95% (0.928 observed)
- The floor tracks the generator: a weak generator gets a low certified floor, not a false one

---

## Slide 15. What survives, what does not

**Survives**
- Beats all four safe-RL baselines on return on dar (t = +3.84 to +17.95); parity with PPO-Lagrangian on rdf
- Constraint violations cut about 7x by probe + margin (13 of 15 seed-runs clean), no return cost
- Joint training is safe when anchored: surrogate held-out error falls 9x, decision unchanged
- Assumption 1 (Lipschitz constant below 1) holds after training
- Conformal faithfulness certificate holds at 90% on real Qwen, 3 of 3 seeds
- Real-sample efficiency: 9x on dar, 5x on rdf
- FNO above DeepONet above GNOT, on benchmark and synthetic data
- Surrogate within reach of the published PDEBench number
- Trained-in explanation beats post-hoc, on stub and on real Qwen (t = +21)
- Resolution invariance from 64 to 128 squared
- Direction of cross-family transfer
- U-Net on real wildfire data at 97.5% of published

**Does not**
- Constraint satisfaction as originally built (7.3% versus 0%)
- Physics loss improving final accuracy (speed only)
- Adaptive weighting raising mean return (variance only)
- The faithfulness loss term (null on stub and on Qwen)
- LoRA adapters beating a frozen probe on reconstruction
- Transfer magnitude into the wave family
- The Fourier operator on sharp fire fronts (gate settles at 0.12)
- Joint training beating the disaggregated pipeline on return (t = +0.66)
- Eq. 8 improving on the variance-only rule (measured bias too small to matter)
- Proposition 1's infinite-horizon bound as a usable number (4,000x loose)
- Constraint satisfaction as a guarantee (a 7x reduction, two excursions in 15 runs)
- swe as a planning testbed: the policy never leaves its initial behaviour, every arm returns -0.1733
- A return edge on rdf: after the planner fix, PSPE and PPO-Lagrangian tie at matched safety (t = +0.82)

**Not yet measured**
- Flood testbed on real data, planning on real wildfire data, human study on real briefs
- Safe-RL baselines on swe (blocked on the swe task redesign)
