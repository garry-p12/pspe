# Experiment 1: the planner on rdf, diagnosed and fixed

Vista, 2026-09-21. rdf testbed, grid 64, surrogate epochs 20, planner
iterations 200, probe every 20, margin k = 2, 5 seeds. Every run below is
`eval/run_constraint_fix.py` with different planner flags; per-seed files under
`runs/constraint_fix_<variant>_rdf/`. Cost limit 3.26 (calibrated: do-nothing
0.59, reward-greedy 8.2).

## The chain of failures

Three defects, each visible in the training trace once the previous one was
removed. Each fix is an opt-in flag in `PlannerConfig`; defaults reproduce
every earlier run.

### 1. Action saturation (`saturation_coef`, `advantage_std_floor`)

Original run, seed 0, baseline arm:

| iteration | cost | lambda | alpha | grad norm pathwise | grad norm LR | var pathwise |
|---|---|---|---|---|---|---|
| 20 | 1.05 | 0 | 0.70 | 0.55 | 2.9 | 1e-2 |
| 40 | 5.87 | 0 | 0.81 | 1.9 | 196 | 0.7 |
| 100 | 8.08 | 14.5 | 0.95 | 0.036 | 5,160 | 0.0 |
| 198 | 8.21 | 38.5 | 0.99 | 0.036 | 12,700 | 0.0 |

The reward-greedy sprint drives the pre-tanh policy mean past |mu| = 3. There
tanh has no slope: the pathwise gradient dies (norm 0.036, variance exactly
zero). The variance-optimal mixing rule then picks the dead branch (alpha to
0.99) because zero variance looks like precision. At the same time the
likelihood-ratio branch explodes (norm 1e4): every rollout returns the same
value, and the advantage normaliser divides by a std of ~0. The mixed gradient
is noise clipped to 10, and lambda = 38 multiplies nothing.

Fix: a soft wall `c * relu(|mu| - 1.5)^2` on both gradient branches
(`saturation_coef 1.0`), and a floor of 1e-2 on the advantage std. dar never
saturates (|mu| 0.2 to 0.3), so neither touches it.

### 2. Dual oscillation (`ki`)

With the wall in place, seed 0, baseline arm:

| iteration | cost | lambda |
|---|---|---|
| 40 | 4.54 | 0.59 |
| 60 | 1.66 | 0.00 |
| 100 | 2.85 | 0.00 |
| 120 | 3.12 | 0.02 |
| 160 | 3.50 | 0.51 |
| 180 | 2.58 | 0.00 |

Every time cost dips under the limit lambda collapses to zero, the policy
sprints greedy again, overshoots. 44% of evaluations land on a peak. Lowering
the gain (scale-free error, `dual_normalize`) made it worse (60%): the
controller was not too strong, it was too weak at holding. Raising the
integral gain from 0.05 to 0.5 lets lambda settle at its equilibrium.

### 3. Tail violations (`margin_episode_std`)

A dual that holds the mean cost at the limit violates on about half of the
evaluations by construction. The existing margin (`limit - 2 sigma`) uses the
spread of the surrogate's cost error, which does not see the policy's own
episode-to-episode spread. `margin_episode_std` adds the std of real episode
costs measured at each probe into sigma.

## Results

| variant | arm | return | cost | violating evals | worst cost | per-seed violating % |
|---|---|---|---|---|---|---|
| original | any | -6.78 | 8.00 | 85% | 8.14 | 91 82 91 82 82 |
| wall + floor | baseline | -7.04 | 3.35 | 44% | 4.52 | 36 55 45 45 36 |
| wall + floor | probe + margin | -7.12 | 2.46 | 33% | 4.25 | 55 18 27 45 18 |
| + dual normalise | baseline | -6.99 | 3.86 | 60% | 4.85 | worse |
| + lr 1e-4 (normalised) | baseline | -6.90 | 4.67 | 42% | 5.22 | |
| + ki 0.5 | baseline | -7.18 | 1.60 | 14.5% | 4.00 | 18 9 18 0 27 |
| + ki 0.5, lr 1e-4 | baseline | -7.07 | 2.80 | 12.7% | 3.93 | 9 9 9 9 27 |
| **+ ki 0.5, episode-std margin** | **probe + margin** | **-7.49** | **0.55** | **5.5%** | **3.23** | 18 0 9 0 0 |

Pass criterion was under 10% with return not collapsing to do-nothing (-9.2).
Met by the last row: 5.5% violating, mean worst case 3.23 against a 3.26
limit, return -7.49 (reward-greedy -6.5, do-nothing -9.2). The constraint now
binds and costs about 0.45 return, which is what a constrained planner should
show.

## dar regression check

Wall + floor + dual EMA 0.7 on dar (`runs/constraint_fix_sat/`): return
-2.27 (was -2.30), 0% violating on every arm (was 1.8% baseline), worst cost
0.23 to 0.39 (was 0.29 to 0.75). No harm, slightly better tails. The wall
never activates on dar.

Full stack (ki 0.5 + episode-std margin) on dar, `runs/constraint_fix_ki10_epstd/`:

| dar arm | return | violating evals | worst cost | effective limit | final lambda |
|---|---|---|---|---|---|
| original baseline | -2.302 | 1.8% | 0.75 | 0.936 | 0 |
| original probe + margin | -2.301 | 0% | 0.29 | 0.758 | 0 |
| v2 baseline | -2.304 | 0% | 0.23 | 0.936 | 0.10 |
| v2 probe + margin | -2.364 | 0% | 0.23 | 0.627 | 2.04 |

No regression: baseline return identical, violations gone, worst case 3x
smaller. probe + margin gives up 0.06 return because the episode-spread margin
tightens the effective limit to 0.63; still far above the safe-RL baselines
(-2.51 to -2.53). One configuration now works on both testbeds.

## Final planner configuration ("v2")

    saturation_coef 1.0, saturation_margin 1.5, advantage_std_floor 1e-2,
    dual_ema 0.7, ki 0.5 (kp 0.5, kd 0.1 unchanged), margin_episode_std on,
    cost_margin_k 2.0, real_cost_every 20

Runner flags: `--sat-coef 1 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5
--margin-episode-std`. To do: expose as `--preset v2` in every planning runner
and use it for the rdf safe-RL baselines (experiment 2).

## Experiment 2: safe-RL baselines on rdf

`runs/baselines_rdf/` (stock dual gains) and `runs/baselines_rdf_ki0.5/`
(PPO-Lagrangian given the same integral-gain fix), 5 seeds, 38,400 real
transitions each.

| method | return | cost | violating evals | worst cost | real samples |
|---|---|---|---|---|---|
| PPO-Lagrangian, stock | -7.66 ± 0.31 | 2.04 | 1.8% | 2.15 | 38,400 |
| PPO-Lagrangian, ki 0.5 | -7.69 ± 0.31 | 1.77 | 0% | 2.02 | 38,400 |
| CPO | -8.05 ± 0.09 | 0.59 | 0% | 0.63 | 38,400 |
| Saute RL | -8.08 ± 0.06 | 0.58 | 0% | 0.63 | 38,400 |
| Primal-dual NPG | -7.97 ± 0.12 | 0.58 | 0% | 0.63 | 38,400 |
| PSPE v2, probe + margin | -7.49 ± 0.62 | 0.55 | 5.5% | 3.23 | 7,360 |
| PSPE v2, no margin | -7.18 ± 0.55 | 1.60 | 14.5% | 4.00 | 7,360 |

Paired t on return, PSPE v2 probe + margin against: PPO-Lag stock +0.74,
PPO-Lag ki 0.5 +0.82, CPO +2.13, Saute +1.97, NPG +2.00. None reaches 2.78.
PSPE v2 without the margin beats every baseline (t +2.1 to +3.8) but at 14.5%
violations against PPO-Lag's 0%, which is a different operating point.

Reading:

- No return edge on rdf at matched safety. PSPE and PPO-Lagrangian are
  indistinguishable; PPO-Lag has the better tail (0% against 5.5%).
- Sample efficiency holds: 5.2x fewer real transitions at the same return.
- CPO, Saute and NPG are safe by inaction: cost at the calibrated do-nothing
  level (0.59), return -8.
- The dual diagnosis transfers. The same ki change that fixed our planner took
  PPO-Lagrangian from 1.8% to 0% violations. The failure and the fix belong to
  the PID controller, not to PSPE.
- PSPE's seed spread on rdf (-6.6 to -8.2) is twice the baselines'; that is
  what keeps the t small and is worth a look (surrogate quality varies by
  seed: rel L2 0.0008 to 0.012 in the joint run).

Planning claim across testbeds after experiments 1 and 2:

| | dar | rdf | swe |
|---|---|---|---|
| return vs best baseline | +0.09 to +0.11, t 3.8 to 18 | +0.2, t 0.8 (parity) | policy never moves |
| violations, probe + margin | 0% | 5.5% (PPO-Lag 0%) | n/a |
| real samples vs model-free | 9x fewer | 5x fewer | n/a |
