# The twin loop, closed on real sequential satellite observations

`eval/run_firms_twin.py`, 2026-09-24. Eight large US wildfires from NASA FIRMS
active-fire detections (VIIRS SNPP, science-processed archive), 14 to 20
consecutive days each, gridded to 64x64. Leave-one-fire-out, 3 seeds.
Raw output in `runs/firms_twin/`.

## Why this experiment exists

Every other result in this project forecasts **one step from a given state**.
A digital twin does something harder: it carries a state forward over many
days, and each time the satellite looks it must reconcile what it believed
with what was seen. Until now nothing in this repository tested that, because
NDWS records carry no date, no fire id and no location, so they cannot be
chained into sequences.

FIRMS can. The fires and their observation records:

| fire | days | observed | gaps | growth over window |
|---|---|---|---|---|
| dixie (2021) | 20 | 19 | 1 | 6.6x |
| bootleg (2021) | 20 | 20 | 0 | 1.0x |
| caldor (2021) | 20 | 20 | 0 | 23.2x |
| august complex (2020) | 20 | 20 | 0 | 11.9x |
| creek (2020) | 20 | 20 | 0 | 1.9x |
| cameron peak (2020) | 20 | 16 | 4 | 0.1x (dying) |
| camp (2018) | 14 | 14 | 0 | 0.0x (burned out) |
| mosquito (2022) | 18 | 13 | 0 | 28.0x |

Explosive growth, steady burns, and fires in decline — plus five genuine
satellite gaps, where a day has no overpass. Those days are carried as
`observed=False` rather than as empty fire, because "we did not look" and
"nothing burned" are different facts and a twin that conflates them drifts.

## The three modes

One shared spread model (small U-Net: today's fire and everything burned so
far, to tomorrow's fire), three ways of carrying the state:

- **open loop** — forecast from day 0 and never look again. A pure forecaster.
- **state sync** — each day, replace the belief with what was observed. On days
  with no overpass, keep the model's own prediction.
- **state + model** — also take one gradient step on the spread model from the
  day's observation, so the model tracks this particular fire.

Scored as average precision of the predicted fire against the detections
actually recorded, on observed days only.

## Results

| mode | day +1 | day +2 | day +3 |
|---|---|---|---|
| open loop | 0.073 ± 0.029 | 0.047 ± 0.026 | 0.038 ± 0.021 |
| state sync | 0.608 ± 0.110 | 0.237 ± 0.093 | 0.085 ± 0.047 |
| **state + model** | **0.613 ± 0.112** | **0.288 ± 0.111** | **0.110 ± 0.054** |

Paired across fires (the fires differ by 30x in growth rate, so pooling their
spreads says nothing; what matters is whether the loop helps on the *same*
fire). df = 7, so |t| > 2.36 is p < 0.05:

| comparison | horizon | mean gain | fires improved | paired t |
|---|---|---|---|---|
| state sync vs open loop | day +1 | +0.535 | 8/8 | **15.9** |
| state sync vs open loop | day +2 | +0.190 | 8/8 | 8.6 |
| state sync vs open loop | day +3 | +0.046 | 8/8 | 4.3 |
| model adaptation on top | day +1 | +0.005 | 7/8 | 3.4 |
| model adaptation on top | day +2 | +0.051 | 8/8 | 3.0 |
| model adaptation on top | day +3 | +0.025 | 8/8 | 3.1 |

## Reading

**Syncing to observation is worth 8.4x in forecast skill at one day** (0.608
against 0.073), and it improves every one of the eight fires. This is the twin
property stated plainly: a model that re-anchors to what the sensor saw is not
a marginally better forecaster, it is a different kind of object.

**Adapting the model matters too, and its value grows with lead time.** At day
+1 the gain is negligible in magnitude (+0.005) because the state was just
replaced with ground truth — there is nothing for a better model to add. At day
+2 it is +0.051 (a 21% relative improvement over state sync alone) and at day +3
+0.025 (30% relative). The further the forecast runs from the last observation,
the more the model's own quality dominates, and the more online adaptation pays.
Both halves of the loop are doing work, in different regimes.

**A correction.** Earlier notes in this project, written from a 4-epoch smoke
run, said online model adaptation "adds essentially nothing". That was wrong.
At smoke scale the model was too weak for adaptation to have anything to
improve; with full training and a paired test it is significant at every
horizon and the effect grows with horizon.

## What this establishes, and what it does not

**Establishes.** The observe → estimate → forecast → observe → correct loop,
closed on real sequential satellite data, with both state and model correction
measured separately, including on days when the satellite did not look.

**Does not establish.** Anything about intervention effects. No fire in this
record had a firebreak cut on our instruction, and none ever will — the
counterfactual does not exist in any observational dataset, and switching
domain to floods, heat or rainfall does not change that. The planning results
on NDWS remain in-model, and that limit is unchanged by this experiment.

So the honest scope of the twin claim: **the sync half is demonstrated on real
data; the act half is not.**
