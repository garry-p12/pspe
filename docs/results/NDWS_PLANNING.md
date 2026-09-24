# Experiment 4: constrained firebreak planning on observed wildfire data

Vista job 1014505, 2026-09-22 (10 min, 3 seeds packed). `eval/run_ndws_planning.py
--horizon 3`, results under `runs/ndws_planning_vista/seed_{0,1,2}/`. Two local
(MPS) seeds under `runs/ndws_planning/` replicate the direction.

## Setup

- Data: Next Day Wildfire Spread. 8,000 training patches, 1,500 held-out
  patches (the eval split), 64 x 64 km, day-t fire mask plus eleven drivers.
- Surrogate: the U-Net from `eval/run_ndws.py`, trained per seed and frozen.
  Forecast AUC-PR on the held-out patches: 0.264, 0.280, 0.276 (published
  0.284). The only number here checked against what the fire actually did.
- Action: an 8 x 8 grid of firebreak intensities per day. Action model, stated
  in `pspe/envs/fire_env.py`: a break lowers the NDVI channel by 2 standard
  deviations times intensity (the learned model decides what that does), and
  blocks spread into treated cells by 0.9 times intensity (imposed physics).
- Objective: final-day expected burned fraction after three days, weighted
  1 + 4 x normalised population density. Constraint: at most 3% of the patch
  treated per day (crew budget).
- Horizon 3, autoregressive: the predicted mask becomes tomorrow's input.
- Policies: none; random within budget; greedy (forecast today, treat the
  riskiest cells within budget, repeat each day: the operational heuristic);
  PSPE per-instance (optimise all three days' intensities jointly through the
  surrogate rollout, projected gradient, exact budget); PSPE v2 amortised
  policy (one CNN policy trained across fires with the budget dual);
  unconstrained (the same policy without the dual, as a no-budget reference).

## Results, 3 seeds, mean ± std

| policy | burn reduction vs none | pop-weighted burn % | treated per day | patches over budget | reduction via fuel channel only |
|---|---|---|---|---|---|
| none | 0 | 14.7 ± 0.2 | 0 | 0 | 0 |
| random | 4.5 ± 0.3 | 14.0 ± 0.1 | 3.0% | 0 | 0.6 |
| greedy | 24.1 ± 3.8 | 11.1 ± 0.6 | 3.0% | 0 | 5.7 ± 2.9 |
| **PSPE per-instance** | **36.5 ± 1.0** | **8.7 ± 0.2** | 3.0% | 0 | **19.3 ± 1.6** |
| PSPE v2 amortised | 12.2 ± 4.5 | 12.6 ± 0.6 | 2.2% | 17% | 2.5 |
| unconstrained | 85.7 ± 4.5 | 2.1 ± 0.6 | 61.7% | 100% | 24.4 |

Per-instance minus greedy, per seed: +9.6, +11.7, +15.9 points. Paired t
(df 2) = 6.66; the 5% threshold is 4.30. Local MPS seeds: +10.0, +8.0.

## Reading

- Decision-time planning through the surrogate beats the operational
  heuristic by twelve points of burn reduction at the same budget with no
  violations. The mechanism is visible in the last column: 19% of the
  planner's reduction goes through the U-Net's learned fuel response, against
  6% for greedy. Greedy treats today's risk; the planner treats where the
  model says the fire will be on day three.
- The amortised policy does not work here (12%, under greedy). One CNN over
  thousands of distinct fires in 200 iterations is the wrong tool; a twin
  optimises each decision at decision time, which is what per-instance does.
- On the one-day version of this problem (`runs/ndws_planning_h1_attempt/`)
  greedy is the exact optimum of the separable objective and nothing beats it.
  Planning earns its keep only across days.

## What this does not show

- Intervention effects are in-model. NDWS records what fires did without
  firebreaks; there is no counterfactual to score a plan against. The
  surrogate's forecast is validated against observation; the treatment effect
  rests on the stated action model. The block term is imposed physics; the
  fuel-channel column is the share that does not depend on it.
- One dataset, one action model, one budget level. A budget sweep (Pareto)
  and a second action model (e.g. fuel removal only) are the obvious next
  runs, now cheap on Vista (10 minutes for three seeds).

## Environment defects found on the way

Four attempts before this one, each an environment defect rather than a
planner one: budget scale 30x smaller than the reward (dual could not bind in
200 iterations; fixed with the scale-free error), a surrogate trained on a
quarter of the data (0.17 AUC-PR, not 0.27), an action map whose floor was
above the budget under the saturation wall (linear to squared), and a
penalty-plus-Adam optimiser that froze at a tenth of the budget (replaced by
projected gradient with exact budget projection).
