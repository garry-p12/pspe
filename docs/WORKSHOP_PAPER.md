# Planning Interventions Inside a Digital Twin, With a Safety Guarantee You Can State

**A framework for constrained intervention planning in PDE-governed hazards, and two
working twins that test it.**

---

## Abstract

Climate digital twins are good at forecasting and poor at *deciding*. A twin that
predicts a flood is useful; a twin that recommends which levee to build, under a
fixed budget, with a stated bound on how wrong it might be, is a different kind of
instrument. The gap is not simulation fidelity — it is that planning inside a
learned or reduced-order model inherits that model's errors, and nobody says by how
much.

We contribute a safety margin for decisions made inside an imperfect simulator. The
margin is a conformal quantile of the **realised cost distribution** — not of model
error, and not of ensemble disagreement, both of which we show fail. On a
reaction–diffusion hazard it cuts constraint violations from 14.1% to 5.0% (20
seeds, paired difference −9.1 points, 95% CI [−11.8, −5.5], p = 0.0002) at **no cost
in objective value**, and it beats a published model-based safe-RL method by 7.0
points (95% CI [+4.0, +10.0]).

We then test whether this transfers to real hazards by building two twins. A
**wildfire twin** over 607 observed fires (WildfireSpreadTS) closes an
assimilate–forecast–plan loop and reaches the same burned-area reduction as an
operational heuristic with **half the crews**. A **flood twin** over Copernicus DEM
terrain, driven by river discharge, coastal surge or rainfall, plans levee
placements against a real budget and reports a guaranteed floor alongside its
estimate. Both run in an interactive portal that works anywhere on Earth.

We also report what does not work, including three results of our own that we had to
retract during development and one pre-registered prediction that failed.

---

## 1. The problem: twins that forecast but cannot advise

An operational question sounds like this. *We have A$20M. There are four places we
could raise a levee. Which combination reduces flooding most, and how confident
should we be before we pour concrete?*

Answering it requires three things a forecast alone does not provide:

1. **A search over interventions**, not a single prediction. Four sites at several
   heights is thousands of combinations; each needs a hydrodynamic run.
2. **A budget constraint that actually binds.** A recommendation that quietly
   exceeds the budget is not a recommendation.
3. **An error bound on the recommendation itself.** Planning happens in a surrogate
   because the real solver is too slow to search with. The surrogate is wrong. By
   how much, and in which direction?

Point 3 is where most decision-support tooling stops, and it is the one that decides
whether an engineer can act on the output. Our contribution is a way to answer it
that requires no distributional assumption and no tuned constant — just a stated
failure rate.

### What "digital twin" has to mean for this to be useful

We use the term in its strong sense: a model that is **kept synchronised with
observations** and used to **evaluate actions before taking them**. A one-way
forecast is not a twin. Both of our applications close that loop:

| stage | wildfire twin | flood twin |
|---|---|---|
| **Perceive** | VIIRS active-fire detections | Sentinel-1 SAR, same-orbit differencing |
| **Simulate** | learned spread model, adapted daily | LISFLOOD-FP local-inertial solver |
| **Plan** | firebreak allocation under a crew budget | levee heights under a currency budget |
| **Explain** | per-measure contribution | leave-one-out counterfactual attribution |

---

## 2. Technical contribution

### 2.1 The failure being fixed

Planning under a constraint in a learned model usually proceeds by a Lagrangian:
maximise reward, penalise constraint cost, let a dual multiplier find the trade-off.
This works when the cost estimate is right. Inside a surrogate it is not, and the
failure is specific: **the dual converges, reports the constraint as satisfied, and
the real system violates it anyway.** The multiplier is balancing against a number
that is biased.

The obvious fix is to inflate the cost estimate by how wrong the model is. We tried
that, and it made things worse — violations rose from 14.5% to 18.2%. The reason is
structural and is our main conceptual point:

> Under an *expectation* constraint, what carries a trajectory over the limit is not
> how wrong the model is on average. It is the **spread of realised cost from
> episode to episode**. A margin built on model error bounds the wrong
> distribution.

The same argument predicts that **ensemble disagreement** will also fail, since it
too estimates model wrongness rather than outcome spread. We test that below.

### 2.2 The margin

Let `c` be the realised episode cost in the real system and `g` the surrogate's
estimate. We calibrate on matched pairs from a periodic real-system probe and take a
split-conformal quantile of the residual `c − g`. Adding that quantile to the
constraint limit gives, under exchangeability of the residuals,

    P[ realised cost > limit ] ≤ δ

with **δ chosen and stated**, not tuned. Two properties matter operationally. The
quantile refuses to return a value when calibration data is insufficient
(`n ≥ 1/δ − 1`), rather than extrapolating — this caught a real bug in our own
portal, described in §5.3. And the margin is computed from outcomes, so it needs no
assumption about the surrogate's error structure.

### 2.3 Does it work

Reaction–diffusion hazard, 20 seeds, budget constraint, δ = 0.1. Four arms differing
only in which mechanism is active:

| arm | violating % | 95% CI | objective (IQM) |
|---|---|---|---|
| no margin | 14.1 | [9.1, 17.3] | −6.917 |
| margin only | 14.1 | [9.1, 17.3] | −6.917 |
| real-system probe only | 10.9 | [6.4, 15.5] | −6.850 |
| **probe + conformal margin** | **5.0** | **[0.9, 8.2]** | **−6.884** |

Seed-paired against the no-margin arm: **−9.1 points, 95% CI [−11.8, −5.5]**,
15/20 seeds improved, paired t = −4.59, p = 0.0002. Objective value is
statistically unchanged — the safety is not bought with performance.

The decomposition is the part worth keeping. **The margin alone does nothing**
(0.0 points, 0/20 seeds changed): a conformal quantile with no probe data to
calibrate against is identically zero. **The probe alone is not enough** (−4.5
points, interval touches zero). The two together are significant and exceed the sum
of the parts. Safety here is not a component you bolt on; it is a loop that must be
closed.

### 2.4 Against a published safe-RL baseline

We transplant **CAP** [Ma et al., AAAI 2022] — a model-based safe-RL method that
inflates cost by ensemble disagreement with an adaptively tuned coefficient — onto
our own planner, so that the *only* difference is how model error is corrected for.
Both arms get identical planner, dual, testbed, limit and **real-sample budget
(8,320 training and 1,920 probe transitions)**.

We do the same for **SMBPO** [Thomas et al., NeurIPS 2021], which corrects by
*pessimism* instead — the maximum over the ensemble, over a truncated imagination
horizon, with no adaptation.

| method | corrects using | violating % | vs ours | p |
|---|---|---|---|---|
| no margin | — | 14.1 | — | — |
| CAP | ensemble disagreement, **adaptive mean + kσ** | 12.0 | +7.0 [+4.0, +10.0] | 0.0003 |
| SMBPO | ensemble disagreement, **unadapted max** | 12.2 | +7.2 [+4.0, +10.1] | 0.0014 |
| **ours** | **conformal quantile of realised cost** | **5.0** | — | — |

CAP's adaptive coefficient settled between 0.50 and 1.46, so its mechanism was
active, not inert. Neither baseline separates from applying **no margin at all**
(CAP 0.4σ; SMBPO −1.9 points, p = 0.75).

**The two baselines are indistinguishable from each other** — CAP − SMBPO =
−0.2 points, 95% CI [−2.8, +2.5], p = 0.83 — and that is the result worth having.
They summarise disagreement about as differently as one can: an adapted second
moment against an unadapted maximum. They land 0.2 points apart. So:

> **How you summarise model disagreement does not matter. Which distribution you
> compute it on does.** Both describe *how wrong the model is*; under an
> expectation constraint that is not what carries a trajectory over the limit.

We registered the opposite prediction before running SMBPO — that its cruder,
unadapted statistic would fare *worse* than CAP's. It did not, and dropping that
embellishment leaves a cleaner and more falsifiable claim: any third
disagreement-based correction should land near 12%, and any method that bounds
realised outcomes should land near 5%.

### 2.5 Why decision-time planning, not a trained policy

A reasonable objection is that one could train a policy and skip the search. We
tested the strongest version of that — distilling our planner's own solutions into a
compact policy via behaviour cloning, the recipe the literature identifies as
effective:

| arm | burned-area reduction |
|---|---|
| decision-time planner | **39.0%** |
| distilled from the planner | 18.3% |
| policy trained from scratch | 10.4% |

Distillation recovers **47%** of the advantage — a real gain over training from
scratch, and a concession that our earlier framing overstated the case. But **21
points survive it.** The mechanism is visible in the diagnostics: the planner draws
16.0 points of its effect through the learned fire-spread dynamics, the distilled
policy only 1.0. A single forward pass learns *where fires generally go*; it does
not reason about *how this fire's fuel state responds*. For rare, high-consequence,
heterogeneous events — which is the climate-hazard case — decision-time search buys
something a policy does not.

---

## 3. Application 1: a wildfire twin over 607 observed fires

### 3.1 Setup

**WildfireSpreadTS** [Gerard et al., NeurIPS 2023 Datasets & Benchmarks] provides
607 fires and 13,607 daily multi-channel images. We use year-wise
cross-validation — folds of 176 / 74 / 201 / 156 fires from 2018–2021 — because the
operationally honest question is whether a model fitted on past seasons works on a
new one, and fire regimes shift between years.

The loop runs daily: ingest today's detections, correct the model's state, adapt the
model, forecast tomorrow, plan firebreaks.

### 3.2 Assimilation works, and we checked it against a real filter

Skill at forecasting the next day's fire extent, 606 fires:

| mode | day +1 | day +2 | day +3 |
|---|---|---|---|
| free-running forecast | 0.019 | 0.011 | 0.009 |
| + state synchronisation | 0.379 | 0.118 | 0.025 |
| + model adaptation | **0.381** | **0.126** | **0.027** |

State synchronisation beats a free-running forecast by +0.361 at day +1 on 595/606
fires (Cohen's d = 1.55). That is a large effect and also a *textbook* one — keeping
a model on the rails with observations is what data assimilation is. We say so
rather than presenting it as a finding.

The honest question is whether our simple correction is defensible against the
classical machinery. So we ran a **32-member Ensemble Kalman Filter** with the
observation-error term swept, not fitted (VIIRS ships no per-cell error variance,
and choosing one that flatters the result is exactly the trap):

| observation error σ_o | implied Kalman gain | day +1 skill |
|---|---|---|
| 0.05 | 0.979 | 0.329 |
| 0.15 | 0.837 | 0.127 |
| 0.30 | 0.562 | 0.032 |
| our method (gain 1) | 1.000 | **0.379** |

**No setting of the filter beats ours, and skill tracks the implied gain
monotonically.** The interpretation matters more than the win: on this system the
observations are far more reliable than the forecast, which drives the optimal gain
toward 1. Our method is not a crude stand-in for a filter — it is approximately the
filter's own optimum, and now we know why rather than having assumed it.

Model adaptation on top of synchronisation is the genuinely novel row and it is
**small**: +0.0016 at day +1 (helping 62% of fires, d = 0.20) and +0.0075 at day +2
(76%, d = 0.44). We report it at that size.

### 3.3 Planning: the same outcome with half the crews

Firebreak allocation on held-out fires, against an operational
forecast-then-treat heuristic, at matched daily crew budget:

| crew budget/day | heuristic | ours | ratio |
|---|---|---|---|
| 1% | 7.5% | **14.3%** | **1.91×** |
| 2% | 12.8% | 26.2% | 2.05× |
| 3% | 24.1% | 36.5% | 1.52× |
| 5% | 36.1% | 53.0% | 1.47× |
| 8% | 55.0% | 69.4% | 1.26× |

**The advantage is largest where crews are scarcest** — ours at a 1% budget matches
the heuristic at nearly 2%, roughly half the crews for the same outcome. That is the
regime real agencies operate in.

### 3.4 Why it works, tested where theory says it must fail

The claimed mechanism is that planning over a *multi-day* horizon beats
greedy-today allocation, because today's optimum ignores where the fire will be
tomorrow. That mechanism makes a falsifiable prediction: **at a one-day horizon the
advantage must vanish**, since the one-day problem is separable over cells and
treating the highest-risk cells within budget is its exact optimum.

| horizon | heuristic | ours | gap | 95% CI |
|---|---|---|---|---|
| 1 day | 25.1 | 25.6 | +0.6 | **[−0.2, +1.7]** |
| 2 days | 24.3 | 33.6 | +8.5 | [+7.0, +10.4] |
| 3 days | 22.1 | 35.4 | +12.1 | [+10.0, +14.8] |
| 5 days | 18.0 | 34.0 | **+14.3** | [+10.9, +18.2] |

At one day the interval **contains zero and is bounded** — whatever the advantage
is, it is at most 1.7 points. From two days on it excludes zero with every seed
agreeing in sign, and the heuristic actively *degrades* (25.1 → 18.0) because it
keeps optimising for today.

*A result that is absent exactly where theory requires and present exactly where it
predicts is stronger evidence for a mechanism than any single large number.*

---

## 4. Application 2: a flood twin that plans levees anywhere

### 4.1 Setup

Terrain from **Copernicus DEM GLO-30**; hydrodynamics from a **LISFLOOD-FP-style
local-inertial solver**, batched over candidate plans so a scenario library can be
built on demand; forcing from one of three drivers, selected by what the site
actually has.

### 4.2 The forcing has to be physical, and this is where we got it wrong twice

Two corrections we consider instructive for anyone building similar tooling.

**Rainfall alone recommends nothing.** Our first configuration forced the domain
with uniform rainfall. Across four candidate levee sites the best single measure
reduced flooding by **0.4%** — correctly, because a levee does nothing against water
falling behind it. The tool was not broken; the scenario was. **Flood interventions
are only meaningful against a directional driver.**

**A boundary condition sets a state, not a volume.** Injecting the Cedar Rapids
design discharge (6,036 m³/s — the right order against the 2008 event's ~5,400) as a
volumetric source into a channel's few cells produced water **112 m deep**, a number
with no physical meaning: mass arrived faster than the cells could spread it.
Holding a *stage* instead, via Manning's normal depth with no free parameters,

    h = (Q·n / (W·√S))^(3/5)  →  10.2 m stage, 11.7 m peak

The consequences for the recommendation are total:

| forcing | best single site | best plan | verdict |
|---|---|---|---|
| rainfall | 0.4% | — | "no measure worth building" |
| river, as a volume | 7.6% | 8.5% | physically void (112 m deep) |
| **river, as a stage** | **16.2%** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

### 4.3 Validated against satellite observation, including the uncomfortable part

We validate on the Richmond NSW 2022 flood using Sentinel-1 SAR, holding permanent
water separately so that rivers are not scored as model error.

- Our fast solver reproduces the reference hydrodynamic model at **CSI 0.979**.
- The reference model reproduces the **satellite** at **CSI 0.535**.

The second number is the one that belongs in an operational discussion, and it is
not flattering. A 0.98 model-to-model agreement is a statement about our numerics;
a 0.53 model-to-observation agreement is a statement about how well anyone's
hydrodynamics matches a real flood at 30 m. We report both, because quoting only
the first is how decision-support tools acquire unearned authority. **One validated
event is also one event** — this is a demonstration of a validation *method*, not an
established skill score.

---

## 5. The portal: the loop, operable

All of the above runs behind an interactive portal. This section exists because a
framework that only appears in a results table has not been shown to be usable.

### 5.1 It runs the actual loop, anywhere

| stage | endpoint | what happens |
|---|---|---|
| Perceive | `POST /observe` | most recent Sentinel-1 pass, same-orbit differencing, permanent water separated |
| Simulate | solver | local-inertial hydrodynamics on Copernicus DEM |
| Plan | `POST /api/plan` | budget-constrained allocation, milliseconds |
| Explain | — | leave-one-out counterfactual attribution per measure |
| Margin | `pspe.plan.margins` | split-conformal, 30 calibration runs |

Terrain, weather, river discharge and road networks are fetched on demand for **any
location on Earth** — Copernicus DEM, Open-Meteo, and OpenStreetMap via Overpass —
so the twin is not a diorama of one pre-built city. Roads are real OSM geometry, so
the planner can reason about which access routes an intervention protects.

### 5.2 One click, end to end

Budget A$20M →

- **search** over site/height combinations — 12 of 36 fit the budget,
- **recommend**: *levee at site 3, 3.0 m and site 5, 1.0 m — A$18.2M of A$20M*,
- **report**: estimated **52%** reduction, and **at least 39%, 90% of the time**,
  with the margin stated as *"±13 points, from 30 held-out solver runs. Write the
  business case against the lower figure."*
- **attribute**: site 3 contributes **+43%** here against 47% alone; site 5
  contributes **+6%** against 9% — in each case 4 points of what it would achieve
  alone is already covered by the rest of the plan,
- **warn**: *"not selected: site 1 (−9%), site 0 (−9%) — the model measures these
  as deepening flooding at the settlement rather than reducing it"*,
- **caveat**, shown in the interface: *"the plan was chosen inside a surrogate.
  Adopting it should be conditional on a full hydrodynamic run of this exact
  height vector."*

Two details there are the contribution made visible rather than asserted. The
interface tells an engineer to **write the business case against the lower
figure**, which is what a stated failure rate is *for*. And it reports two
candidate levees as actively harmful — the backwater effect — rather than
quietly ranking them last.

That sequence is the contribution made operable: plan in the surrogate, bound the
error against reality, state what you would still verify before committing.

### 5.3 The bug that justifies the whole margin apparatus

Our own portal displayed a confidence it had no data to support. It showed
`1.645 · RMSE · √(1 + 1/n)` — a normal-approximation interval — described in its own
code as "the product-facing form of the framework's safety margin". It was not. Two
errors compounded: the fit was tuned on **six** multi-measure runs where 90%
coverage requires at least nine, and a Gaussian places the 90% point at 1.64×RMSE
where the empirical residuals are heavier-tailed.

Rebuilt honestly at 30 calibration points, the margin is **±20.2 points against the
±10.5 displayed — the tool was overstating its confidence by about 2×.**

The conformal quantile would have **refused to return a number** at six points,
which is the correct behaviour and the reason we prefer it. A normal approximation
always returns something. For decision-support software, a method that declines to
answer when it lacks data is not a limitation; it is the feature.

---

## 6. What this enables, and what we are not claiming

### Enables

**A reusable pattern for decision-capable climate twins.** The margin needs only
matched pairs of (surrogate estimate, realised outcome) from a periodic probe of the
real system. It makes no assumption about the simulator — learned, reduced-order or
full-physics — and none about the error distribution. Any twin that can occasionally
check itself against reality can carry a stated failure rate on its
recommendations.

**Search where a policy will not do.** §2.5 quantifies what decision-time planning
buys over a trained policy on real fire data: 21 points that survive the best
distillation recipe we could implement. For heterogeneous, rare, high-consequence
events, that is the argument for keeping a solver in the loop.

**A testable standard for "our assimilation is good enough."** §3.2's EnKF sweep is
a template: rather than asserting a simple correction suffices, run the classical
filter with its error term swept and show where the optimum sits.

### Not claiming

- **No fire in our archive had a firebreak cut on our instruction.** Planning is
  validated against held-out observations inside the model, not against a controlled
  field trial. The intervention arm is unvalidated in reality, and no retrospective
  dataset can fix that.
- **One validated flood event**, at CSI 0.53 against satellite.
- **The guarantee is marginal, not per-step**: it bounds the probability that a
  trajectory's total cost exceeds the limit, not the cost at every timestep.
- **Exchangeability is assumed** across a learning run, and we test but do not prove
  it.
- The adaptation benefit in §3.2 is **small** (d = 0.20 at day +1). We report the
  effect size rather than the p-value alone.

### On the methodology, briefly

During development we retracted three results of our own: a distillation number that
was an artefact of a loss function collapsing to "treat nothing"; a baseline
comparison invalidated by a 4.7% extrapolation bias four times larger than the
mechanism it was meant to test; and a statistical claim whose interval, once
computed over the correct unit, spanned its own effect size. All three initially
pointed in the direction that favoured us, which is why we now treat a
claim-supporting result as a trigger for an extra check rather than a stopping
point. We think that discipline is worth more to this community than any individual
number in this paper.

---

## Figures

| figure | file | shows |
|---|---|---|
| 1 | `docs/figures/paper/fig8_twin_loop.png` | the assimilate–forecast–plan loop |
| 2 | `docs/figures/paper/fig4_wildfire.png` | budget Pareto and horizon ablation |
| 3 | `docs/figures/paper/figA1_sar_validation.png` | Sentinel-1 validation, permanent water separated |
| 4 | `docs/figures/paper/fig10_flood_precondition.png` | why rainfall recommends nothing |
| 5 | `runs/portal_shots/45_plan_result.png` | **the recommendation, its guaranteed floor, and per-measure attribution** |
| 6 | `runs/portal_shots/42_plan_options.png` | budget slider and ranked options over the 2022 event |
| 7 | `runs/portal_shots/44_roads.png` | live OpenStreetMap road network |
| 8 | `runs/portal_shots/40_forecast.png` | live forecast panel (Open-Meteo) |

All portal figures captured 2026-09-28 against the current interface via
`portal/scripts/poster-shots.mjs` and `plan-result.mjs` (1600×1000 at
deviceScaleFactor 2). The earlier `figA5_portal_appraisal` and
`figA6_portal_anywhere` are **not** used here: they embed screenshots from before
the interface was rebuilt and no longer show the tool as it is.

## Data and reproducibility

WildfireSpreadTS (Zenodo 8006177); Next Day Wildfire Spread; Copernicus DEM GLO-30;
Sentinel-1 RTC and ESA WorldCover via Microsoft Planetary Computer; Open-Meteo;
OpenStreetMap via Overpass. All figures and tables are regenerated by scripts in
`eval/` and `scripts/` from cached inputs.
