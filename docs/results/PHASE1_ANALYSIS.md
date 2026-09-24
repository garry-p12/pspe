# Vista Phase 1 and 2a results: analysis

Fetched 2026-09-21 from `$WORK/pspe/runs` on Vista (jobs 999843, 999856,
999858, 999899; all COMPLETED 2026-09-16). Raw aggregated tables are in
`runs/VISTA_PHASE1_RAW.md`; per-seed files under `runs/{joint, alpha_rule,
lipschitz, faith_weights, conformal_real}/`. dar testbed, grid 64, surrogate
epochs 20, planner iterations 200 (the paper-scale budget; the "quick budgets"
label in the per-run headers is the aggregator's default string, not the
setting). Probe + margin on in every planning arm. Paired t uses df = 4
(5 seeds) or df = 2 (3 seeds).

## 1. Joint versus disaggregated (Phase 2a, the paper's central experiment)

`runs/joint/`, 5 seeds. One pretrained FNO, deep-copied per arm.

| arm | return | violating evals | worst cost | held-out surrogate rel L2 before | after | drift | real samples |
|---|---|---|---|---|---|---|---|
| disaggregated | -2.302 ± 0.17 | 1.8% (1 seed of 5) | 0.38 ± 0.33 | 0.0048 ± 0.0042 | 0.0048 | 0 | 4,160 |
| joint (beta 0.1, anchored) | -2.296 ± 0.18 | 1.8% (1 seed of 5) | 0.56 ± 0.65 | 0.0048 ± 0.0042 | 0.0005 ± 0.0002 | -0.0043 | 4,160 |
| joint, unanchored | -2.369 ± 0.17 | 0% | 0.38 ± 0.23 | 0.0048 ± 0.0042 | 0.975 ± 0.034 | +0.970 | 4,160 |

Paired t on return: joint versus disaggregated **+0.66** (delta +0.006, not
significant); unanchored versus disaggregated **-5.84** (delta -0.067).

Per seed return, joint versus disaggregated: -2.492 / -2.484, -2.090 / -2.086,
-2.313 / -2.313, -2.148 / -2.191, -2.438 / -2.438.

What this says:

- **Joint training does not change the decision.** Same return to two decimals on four of five seeds.
- **The anchored joint surrogate gets better on held-out data, not worse.** Held-out one-step error drops from 0.0048 to 0.0005 on every seed (9x). The planning gradient, scaled to one tenth of the data gradient, acts as extra regularised training on trajectories the policy visits. No capture.
- **The unanchored arm is the failure mode, on purpose.** Held-out error goes from 0.005 to 0.975 (the surrogate stops predicting the data at all) and return falls by 0.07 (t = -5.84). Its 0% violation rate is not safety: a surrogate that predicts nothing predicts no cost either, and the probe is what kept the real cost in range.
- Worst-cost caveat: the joint arm's seed 2 had one evaluation at 1.71 against the 0.936 limit; the disaggregated arm's seed 3 had one at 0.977. See section 5.

Verdict for the paper: contribution 1 as stated ("joint beats disaggregated")
does not hold on this testbed at this budget. The defensible claim is that
joint training is safe when anchored (surrogate improves, decision unchanged)
and destructive when not.

## 2. Mixing rule: fixed, variance, Eq. 8 (Phase 1c)

`runs/alpha_rule/`, 5 seeds.

| arm | return | violating evals | worst cost | final alpha | measured B^2 | real samples |
|---|---|---|---|---|---|---|
| fixed (alpha 0.5) | -2.415 ± 0.30 | 3.6% | 1.13 ± 1.5 | 0.5 | -- | 4,160 |
| variance rule (as shipped) | -2.302 ± 0.17 | 1.8% (1 seed) | 0.38 ± 0.33 | 0.992 | -- | 4,160 |
| Eq. 8 (with measured bias) | -2.299 ± 0.15 | **0% (all 5 seeds)** | 0.34 ± 0.22 | 0.992 | 0.0013 ± 0.0015 | 6,080 |

Paired t on return: Eq. 8 versus variance **+0.22** (nothing); fixed versus
variance -1.51 (not significant, same tail-only story as before).

- The measured pathwise bias is tiny (B^2 about 0.001), so Eq. 8 and the variance rule pick the same alpha (0.992). No return difference.
- Eq. 8 did remove the one excursion the variance rule had (seed 3: worst 0.73 versus 0.98), at 1,920 extra real transitions for the bias measurement. That is a tail effect from the extra truth rollouts, not from the formula.
- Fixed alpha 0.5 is worse on every axis, as before.

## 3. Lipschitz assumption and Proposition 1 (Phase 1a)

`runs/lipschitz/`, 5 seeds. L_G by power iteration on the trained surrogate;
eps = one-step error against the truth solver; return bias = |return under
surrogate minus return under truth| for the same policy.

| arm | rel L2 | L_G | L_G <= 1 | eps max | return bias (measured) | Prop 1 bound, infinite horizon | Prop 1 bound, horizon H | bias <= H bound |
|---|---|---|---|---|---|---|---|---|
| default FNO | 0.069 ± 0.062 | 0.985 ± 0.013 | 4 of 5 | 0.17 ± 0.12 | 0.10 ± 0.09 | 422 ± 290 | 9.8 ± 6.6 | 5 of 5 |
| spectral-normalised | 0.079 ± 0.026 | 0.964 ± 0.003 | 5 of 5 | 0.31 ± 0.08 | 0.11 ± 0.04 | 748 ± 200 | 17.3 ± 4.6 | 5 of 5 |

- Assumption 1 (L_G <= 1) holds on 4 of 5 default seeds after training (init 1.61) and on all 5 with spectral normalisation, which costs 0.01 rel L2.
- Proposition 1's infinite-horizon bound is vacuous: 422 against a measured bias of 0.10 (4,000x loose), because the 1/(1-gamma) prefactor at gamma 0.98 is 2,450. The finite-horizon form (H = 12) holds on every seed but is still 100x loose. Report the finite-horizon form and say it is loose.

## 4. Explain on Qwen: faithfulness weight sweep and conformal certificate (Phase 1b, 1d)

### Weight sweep, `runs/faith_weights/`, 3 seeds

| arm | F(b) | KL |
|---|---|---|
| trained-in, w = 1 (default) | 0.468 ± 0.16 | 0.81 |
| no faithfulness term | 0.468 ± 0.16 | 0.81 |
| trained-in, w = 10 | 0.366 ± 0.29 | 1.59 |
| trained-in, w = 100 | 0.218 ± 0.19 | 9.48 |
| trained-in, w = 1000 | 0.399 ± 0.14 | 0.96 |
| post-hoc | 0.182 ± 0.16 | 2.77 |

The term is null at w = 1 and harmful above it: F(b) falls and KL rises as the
weight grows, and at w = 100 the briefs are no better than post-hoc. The
REINFORCE signal, even with the parse-rate fix and self-critical baseline,
pushes the generator away from the parser-readable format the metric depends
on. Drop the term from the method; keep trained-in supervision.

### Conformal certificate, `runs/conformal_real/`, 3 seeds, n_cal 200, n_test 400

| seed | mean F | delta 0.2: floor / coverage / holds | delta 0.1: floor / coverage / holds | delta 0.05: floor / coverage / holds |
|---|---|---|---|---|
| 0 | 0.281 | 0.278 / 0.798 / yes | 0.277 / 0.878 / yes (p 0.08) | 0.276 / 0.928 / **no** (p 0.03) |
| 1 | 0.557 | 0.557 / 0.793 / yes | 0.556 / 0.893 / yes | 0.556 / 0.940 / yes |
| 2 | 0.567 | 0.562 / 0.773 / yes | 0.559 / 0.895 / yes | 0.558 / 0.935 / yes |

- The certificate holds at delta = 0.1 (90% coverage) on all three seeds and at delta = 0.2 on all three. At delta = 0.05 one seed fails the binomial test (0.928 observed against 0.95).
- The certified floor is only as good as the generator: seed 0 certifies F >= 0.28, seeds 1 and 2 certify F >= 0.56. The certificate is honest about a weak generator rather than hiding it.

## 5. Constraint fix, revisited across all probe + margin runs

Every planning arm above ran with the probe + margin fix. That gives 15
seed-runs of the "disaggregated with fix" configuration across
`constraint_fix`, `alpha_rule` (variance arm) and `joint` (disaggregated arm):

| run | seeds with any violating evaluation | worst cost seen |
|---|---|---|
| constraint_fix, probe + margin | 0 of 5 | 0.29 |
| alpha_rule, variance arm | 1 of 5 (seed 3, 9% of evals) | 0.98 |
| joint, disaggregated arm | 1 of 5 (seed 3, same run) | 0.98 |
| joint, joint arm | 1 of 5 (seed 2) | 1.71 |

So "0% on all 5 seeds" was one draw. Across 15 runs the fix holds on 13, the
excursions are one evaluation each, and the pre-fix full-budget rate of 7.3%
of evaluations becomes about 1% of evaluations. Eq. 8's extra truth rollouts
took the rate to 0 of 5 in its run. The honest statement: probe + margin cuts
violations by about 7x and removes the large excursions on most seeds; it is
not a guarantee. A guarantee needs a margin derived from a conformal quantile
of the measured cost error rather than 2 sigma, which is the obvious next
change.

## 6. What changes in the paper

| claim | before Phase 1 | after |
|---|---|---|
| joint > disaggregated | untested | **no return difference** (t 0.66); anchored joint improves surrogate 9x, unanchored destroys it |
| Eq. 8 alpha rule | untested | same alpha as variance rule (B^2 tiny); tail benefit only, from extra rollouts |
| Assumption 1, L_G <= 1 | untested | holds after training (4 of 5; 5 of 5 with spectral norm) |
| Prop 1 bound | untested | vacuous at infinite horizon; finite-horizon form holds, 100x loose |
| faithfulness term | null at w = 1 | null at w = 1, harmful above |
| conformal certificate | verified in expectation | holds at 90% on real Qwen, 3 of 3 seeds |
| constraint fix | 0 of 5 | 13 of 15 seed-runs clean; about 1% of evaluations violating versus 7.3% |

## 7. Queued next (Vista jobs 1013191 to 1013194, submitted 2026-09-21)

Constraint fix and joint training on the swe and rdf testbeds, 5 seeds each.
Fetcher writes `runs/VISTA_PHASE3_RESULTS.md` when they finish.
