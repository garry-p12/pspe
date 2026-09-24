# Phase 3: constraint fix and joint training on swe and rdf

Vista, 2026-09-21. Same budgets as the dar runs (surrogate epochs 20, planner
iterations 200, probe every 20, margin k = 2). The aggregated tables carry the
aggregator's default "dar" label in their header; the runs are the testbed
named in the directory. Paired t uses df = 4.

Status: all four complete (GPU; jobs 1013192, 1013193, 1013316, 1013315).

## 1. Joint versus disaggregated on swe (`runs/joint_swe/`)

| arm | return | violating evals | held-out surrogate rel L2 before | after | drift | real samples |
|---|---|---|---|---|---|---|
| disaggregated | -0.1733 ± 0.010 | 0% | 0.0246 ± 0.003 | 0.0246 | 0 | 7,360 |
| joint, anchored | -0.1733 ± 0.010 | 0% | 0.0246 ± 0.003 | 0.0027 ± 0.0006 | -0.022 | 7,360 |
| joint, unanchored | -0.1733 ± 0.010 | 0% | 0.0246 ± 0.003 | 1.14 ± 0.27 | +1.12 | 7,360 |

- Return is identical to four decimals across all three arms, including the
  one whose surrogate stopped predicting the data (rel L2 1.14). On shallow
  water at this budget the decision does not depend on the surrogate at all.
- This matches the planning-transfer finding on dar to swe (planning gap
  0.007, nothing): interventions barely move the swe return. swe is not a
  discriminating testbed for the planner; it discriminates surrogates only.
- The anchored-joint surrogate improvement replicates: held-out error falls
  9x, as on dar.

## 1b. Constraint fix on swe (`runs/constraint_fix_swe/`)

| arm | return | episode cost | limit | effective limit | violating evals | real samples |
|---|---|---|---|---|---|---|
| baseline | -0.1733 ± 0.010 | 0.118 ± 0.017 | 0.192 | 0.192 | 0% | 6,400 |
| probe only | -0.1733 ± 0.010 | 0.118 ± 0.017 | 0.192 | 0.192 | 0% | 7,360 |
| margin only | -0.1733 ± 0.010 | 0.118 ± 0.017 | 0.192 | 0.192 | 0% | 6,400 |
| probe + margin | -0.1733 ± 0.010 | 0.118 ± 0.017 | 0.192 | 0.060 ± 0.04 | 0% | 7,360 |

- Identical per seed across arms, and identical to the joint runs above.
- Cost 0.118 is the calibrated do-nothing cost (0.108). The limit is never
  approached, so the dual never activates. probe + margin pulled the
  effective limit to 0.06, below the realised cost, and the policy still did
  not move.
- Conclusion: on swe the policy never leaves its initial near-zero behaviour.
  At the calibrated actuation scale (u_max 0.04) the return gradient through
  the dynamics is effectively zero. swe is a surrogate benchmark, not a
  planning benchmark, until the task is redesigned (more actuation authority,
  or an objective the actuation can move).

## 2. Constraint fix on rdf (`runs/constraint_fix_rdf/`)

| arm | return | episode cost | limit | violating evals | worst cost | real samples |
|---|---|---|---|---|---|---|
| baseline | -6.78 ± 0.69 | 8.00 ± 0.42 | 3.26 | 85% | 8.14 | 6,400 |
| probe only | -6.78 ± 0.69 | 8.00 ± 0.42 | 3.26 | 85% | 8.14 | 7,360 |
| margin only | -6.76 ± 0.69 | 8.00 ± 0.43 | 3.26 | 85% | 8.14 | 6,400 |
| probe + margin | -6.77 ± 0.69 | 7.97 ± 0.49 | 3.26 | 85% | 8.14 | 7,360 |

The fix changes nothing because the failure is upstream of it. Training
trace, seed 0 baseline arm:

| iteration | train cost | lambda | train return |
|---|---|---|---|
| 0 | 1.29 | 0.0 | -9.21 |
| 5 | 1.02 | 0.0 | -7.76 |
| 50 | 7.83 | 1.6 | -6.45 |
| 100 | 8.19 | 13.5 | -6.48 |
| 199 | 8.19 | 36.6 | -6.75 |

- Cost sits near 1 (limit 3.26) with lambda at zero, then the reward-greedy
  updates between iterations 20 and 50 drive it to the reward-greedy cost
  (8.2 from calibration) and it never comes back, although lambda climbs to
  37.
- The surrogate is not the cause: measured cost bias -0.08, rel L2 0.011.
- The signature is action saturation: the policy mean sits at u_max, the
  pathwise gradient through the squashed action is near zero, and a large
  lambda multiplies a near-zero gradient. The dual's PID gains, tuned on dar,
  warm up too slowly for how fast the greedy policy sprints on rdf.
- Safe-RL baselines were never run on rdf at 5 seeds, so there is no
  comparison row yet.

What to change (planner, not surrogate): a lambda warm start or a cap on the
early policy step size; and a likelihood-ratio cost gradient (alpha toward 0
for the cost term) once actions saturate, since the pathwise path is what
vanishes. Both are small edits to `pspe/plan/trainer.py`.

## 2b. Joint versus disaggregated on rdf (`runs/joint_rdf/`)

| arm | return | violating evals | held-out surrogate rel L2 before | after | drift | real samples |
|---|---|---|---|---|---|---|
| disaggregated | -6.76 ± 0.69 | 85% | 0.0024 ± 0.0013 | 0.0024 | 0 | 7,360 |
| joint, anchored | -6.70 ± 0.70 | 87% | 0.0024 ± 0.0013 | 0.0005 ± 0.0001 | -0.002 | 7,360 |
| joint, unanchored | -7.46 ± 0.45 | 31% | 0.0024 ± 0.0013 | 0.83 ± 0.34 | +0.83 | 7,360 |

Paired t on return: joint versus disaggregated **+1.00** (identical on 4 of 5
seeds); unanchored versus disaggregated **-2.41**.

- Same three findings as on dar and swe: the decision does not change; the
  anchored surrogate improves 5x on held-out data; the unanchored surrogate is
  destroyed.
- The unanchored arm's lower violation rate (31% against 85%) is the policy
  going passive on a surrogate that no longer predicts cost, at a return cost
  of 0.7. Not safety.
- The rdf dual failure from section 2 (cost pinned at the reward-greedy value)
  is present in every arm here too.

## 2c. Joint training across all three testbeds

| testbed | joint vs disaggregated return, t | anchored surrogate rel L2 before to after | unanchored after |
|---|---|---|---|
| dar | +0.66 | 0.0048 to 0.0005 (9x) | 0.975 |
| swe | identical | 0.0246 to 0.0027 (9x) | 1.14 |
| rdf | +1.00 | 0.0024 to 0.0005 (5x) | 0.83 |

Robust across families: joint training with a data anchor never changes the
decision and always improves the surrogate on held-out data; without the
anchor it always destroys the surrogate. "Joint beats disaggregated" is not
supported on any testbed; "joint is safe when anchored, destructive when not"
is supported on all three.

## 3. Reading across the three testbeds

| testbed | planner sensitive to surrogate? | constraint active? | probe + margin |
|---|---|---|---|
| dar | yes (return edge over safe RL, t 3.8 to 18) | yes | 7x fewer violations, not zero |
| swe | no (identical return with a destroyed surrogate; policy never leaves init) | no (cost at do-nothing level) | untestable here |
| rdf | no (joint t 1.00; unanchored only hurts) | yes, and unmet by every arm | irrelevant until the dual works |

Update (later the same day, `docs/results/RDF_PLANNER_FIX.md`): the rdf dual was
fixed in three steps (saturation wall, integral gain, episode-spread margin)
and the safe-RL baselines were run on rdf. Result: PSPE v2 at 5.5% violations
and -7.49 return against PPO-Lagrangian at 0% and -7.69 (t = +0.82, parity),
with 5.2x fewer real samples. The return edge is dar-only; sample efficiency
and the safety mechanism hold on both. swe still needs a task redesign.
