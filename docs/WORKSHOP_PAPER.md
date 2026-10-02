# From Forecast to Decision: What a Climate Digital Twin Needs to Advise Under a Budget

*Workshop / AI-for-science submission. This manuscript owns the **digital-twin
systems** contribution. The conformal calibration method is summarised in §2 and
owned by a separate paper (`ICML_PAPER.md`); its derivations are
`THESIS_AND_RESULTS.md` Part 10. Nothing in §2 is reproduced in full here.*

---

## Abstract

Climate digital twins forecast well and **decide** poorly. A twin that predicts a
flood is useful; a twin that recommends which levee to build, under a fixed
budget, with a stated bound on how wrong it might be, is a different instrument.

We ask what it takes to close that gap, and answer it by building two twins and
an operable portal around a five-stage loop — **perceive, simulate, plan,
calibrate, explain**. A **wildfire twin** over **607 observed fires**
synchronises to satellite detections, plans crew allocation multiple days ahead
through learned spread dynamics, and reaches **14.3% burned-area reduction on a
1% daily crew budget where the operational heuristic needs 2% to reach 12.8%** —
the advantage vanishing to +0.6 points (n.s.) at a one-day horizon, exactly where
a greedy allocation is optimal. A **flood twin** over Copernicus terrain plans
levees anywhere on Earth against a real budget, reports a guaranteed floor beside
its estimate, and **refuses to report a confidence its calibration set cannot
support**.

The result we did not anticipate is the paper's centre. Routing an evacuation
against the **peak-depth field that every flood twin publishes** finds **no route
at all on 96 of 96 decisions** where a time-resolved solve gets through on 40 and
an oracle on 82. A peak-depth map is the union of states that never coexist in
time, so a decision taken against it is taken against a scenario that never
happens.

We report what does not work with the same prominence: our own calibration method
contributes **nothing** to discrete route choice, ensemble filtering converts
refusals into departures at **2:1 odds rather than for free**, and 41 measurement
defects are catalogued with the check that caught each.

---

## 1. The question a forecast cannot answer

*We have A$20M. There are six places we could build. Which, how high, and how
wrong might you be?*

A forecast says what happens if nothing changes. This question is about what
changes, under a budget, with a consequence attached to being wrong. Answering it
needs five things closed into a loop, and the gap between a forecasting twin and
a deciding twin is that most twins have two of them.

| stage | what it must do | where it fails if absent |
|---|---|---|
| **Perceive** | synchronise to observation, not merely initialise from it | an open-loop rollout invents a fire that is not there |
| **Simulate** | be fast enough to search over, honest about where it is wrong | a planner cannot search a model that takes a day |
| **Plan** | optimise under a real constraint, not a soft penalty | the budget is advisory and gets exceeded |
| **Calibrate** | state how wrong the recommendation might be | the number is unfalsifiable and unusable |
| **Explain** | say what was decided, in words that can be checked | the operator cannot audit the advice |

This paper is about stages 1, 2, 3 and 5 as engineering, and about what stage 4
buys and does not buy once the other four exist.

**Figure 1** `docs/figures/paper/fig8_twin_loop.png` — the loop, with validated
and model-based paths distinguished.

---

## 2. The calibration stage, in brief

We summarise rather than derive, because this is the other paper's contribution.

Planning inside a learned model under an **expectation** constraint has a
calibration mismatch: the dual controller is fed cost measured in the model while
violation is realised under true dynamics, so the dual converges and the real
system violates. Inflating the estimate by the model's own error makes it worse
(14.5% → 18.2%), and that is structural: a margin measurable from model-error
uncertainty alone cannot control violation probability uniformly in the spread of
**realised** cost. The admissible correction is to conformalise the error around
the quantity the dual actually controls, calibrated on matched pairs from a
periodic probe of the real environment. On a reaction–diffusion hazard this cuts
violations **14.1% → 5.0%** at no objective cost; the derivation, the closed-form
failure boundary, and comparisons against three published disagreement-based
methods are in `ICML_PAPER.md`.

Two properties are what the rest of this paper uses.

**It states a floor, not a point.** The planner reports an estimate and a
guaranteed lower bound at a chosen failure rate, which is what an engineer writes
a business case against.

**It refuses.** A split-conformal quantile at level `δ` needs at least
`1/δ − 1` calibration points; below that no order statistic carries the
guarantee, and the estimator returns nothing rather than extrapolating. With 30
held-out solver runs the highest honest confidence is **96.8%**, so a request for
99% is declined rather than fabricated. §6.4 is the bug that taught us to want
this.

---

## 3. Twin one: wildfire over 607 observed fires

### 3.1 Perceive — synchronising to the satellite

VIIRS active-fire detections drive an ensemble Kalman update of the burned-state
field. The design question is not whether assimilation helps but **how much error
inflation it needs**, and rather than assert a value we ran the classical filter
with its error term swept and showed where the optimum sits.

The benefit is **small and reported as such**: Cohen's d = **0.20** at day +1. We
give the effect size rather than the p-value alone, because at 607 fires almost
anything is significant.

Two data defects found at this stage are worth naming, because both degrade
quietly:

- **"No detection" is not "not observed."** Treating absence as absence would
  have deleted **37.4%** of days — precisely the days on which fires go out.
- **A downsample cropped the raster margin**, losing up to **59%** of a fire's
  detections, smoothly and only for some fires, leaving every downstream number
  plausible. Nothing short of a conservation check finds that.

### 3.2 Simulate and plan — crews, several days ahead

A learned spread model is fast enough to search over, and the planner allocates a
daily crew budget across the fire perimeter multiple days ahead.

**The systems headline is the budget curve.** Burned-area reduction, 3 seeds:

| daily crew budget | operational heuristic | planner | ratio |
|---|---|---|---|
| **1%** | 7.5 ± 1.4 | **14.3 ± 0.5** | **1.91×** |
| **2%** | **12.8 ± 2.0** | 26.2 ± 0.7 | **2.05×** |
| 3% | 24.1 ± 3.8 | 36.5 ± 1.0 | 1.52× |
| 5% | 36.1 ± 5.0 | 53.0 ± 1.1 | 1.47× |
| 8% | 55.0 ± 5.5 | 69.4 ± 0.6 | 1.26× |

> **The planner at a 1% budget (14.3%) beats the heuristic at a 2% budget
> (12.8%).** Roughly half the crews for a better outcome, and the advantage is
> **largest where crews are scarcest** — which is the operational regime that
> matters.

**Figure 2** `docs/figures/paper/fig4_wildfire.png` — budget Pareto and horizon
ablation.

### 3.3 Tested where it must fail

At a one-day horizon there is nothing for a planner to reason about: a greedy
allocation is optimal and the advantage should vanish.

| horizon | heuristic | planner | gap | t |
|---|---|---|---|---|
| 1 day | 24.8 ± 1.1 | 25.4 ± 1.7 | **+0.6** | **1.40 — n.s.** |
| 2 days | 24.2 ± 3.6 | 32.6 ± 3.4 | +8.5 | 10.74 |
| 3 days | 22.0 ± 4.4 | 34.1 ± 4.4 | +12.1 | 11.08 |
| 5 days | 17.7 ± 4.3 | 32.0 ± 5.8 | **+14.3** | 8.99 |

It does vanish, and grows monotonically with the horizon over which sequential
reasoning can act. A result that survives its own falsification test is worth
more than one never exposed to it.

An earlier version of this claim was measured on 8 fires at t = 3.0. At 607 the
benefit is **a third the size** and holds on 76% of them. The 8-fire number was
not wrong; it was underpowered, and we re-ran it rather than publish it.

### 3.4 Why a solver stays in the loop

The obvious objection is to train a policy and skip the search. We tested the
strongest version — distilling our own planner's solutions, the recipe the
literature identifies as effective:

| arm | burned-area reduction |
|---|---|
| decision-time planner | **39.0%** |
| distilled from the planner | 18.3% |
| policy trained from scratch | 10.4% |

Distillation recovers **47%** of the advantage — a real gain, and a concession
that our earlier framing overstated the case. But **21 points survive it**, and
the diagnostics say why: the planner draws 16.0 points of its effect through the
learned spread dynamics, the distilled policy 1.0. A forward pass learns where
fires generally go; it does not reason about how *this* fire's fuel state
responds. For rare, heterogeneous, high-consequence events, that is the argument
for keeping a solver in the loop.

An earlier distillation number was an artefact of a loss collapsing to "treat
nothing" — the MSE-optimal constant in allocation space — and is retracted.

---

## 4. Twin two: flood, levees, and roads anywhere on Earth

### 4.1 Perceive and simulate

Copernicus DEM GLO-30 terrain fetched on demand, a LISFLOOD-FP-style
local-inertial solver batched over candidate plans, and forcing from river
discharge, coastal surge or rainfall — whichever the site actually has. The twin
therefore runs on terrain that is nowhere in the repository.

**Getting the forcing physical changed the answer completely**, and we got it
wrong twice:

| forcing | best single site | best plan | verdict |
|---|---|---|---|
| rainfall | 0.4% | — | "no measure worth building" |
| river, as a volume | 7.6% | 8.5% | physically void (112 m deep) |
| **river, as a stage** | **16.2%** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

Rain falling across the whole box gives a levee nothing to block, and the tool
correctly recommends nothing — a real answer. Injecting discharge as a volume
makes water arrive faster than the grid can spread it and reached 112 m depth, a
number with no physical meaning; a boundary sets a **stage**, which Manning's
normal depth gives with no free parameters. Separately, DEM nodata filled with
sea level produced a **247 m trench** ringing every inland domain, draining the
catchment into it.

**Figure 3** `docs/figures/paper/fig10_flood_precondition.png`.

### 4.2 Validated against satellite, including the uncomfortable part

Richmond NSW 2022, Sentinel-1 SAR, permanent water held separately so rivers are
not scored as model error.

- our fast solver against the reference hydrodynamic model — **CSI 0.979**
- the reference model against the **satellite** — **CSI 0.535**

The second number belongs in an operational discussion and is not flattering.
0.98 is a statement about our numerics; 0.53 is a statement about how well
*anyone's* hydrodynamics matches a real flood at 30 m. We report both, because
quoting only the first is how decision-support tools acquire unearned authority.
**One validated event is also one event** — a demonstration of a validation
*method*, not an established skill score.

**Figure 4** `docs/figures/paper/figA1_sar_validation.png`.

### 4.3 Plan — levees against a real budget

Every candidate option is solved in one batched timestep loop, so a ranked list
of plans costs about what a single scenario costs. The planner then allocates
greedily against the budget, reports per-measure attribution, and flags measures
that make things **worse**:

> **A$20M** → site 3 (3.0 m) + site 5 (1.0 m), A$18.2M of A$20M
> → **estimated 52%, at least 39% nine times in ten**
> → *"Write the business case against the lower figure."*
> → site 3 **+43%** (47% alone) · site 5 **+6%** (9% alone) — measures overlap
> → **rejected:** site 1 (−9%), site 0 (−9%), measured as **deepening** flooding
>   at the settlement rather than reducing it

**Figure 5** `runs/portal_shots/45_plan_result.png`.

Two engineering consequences of the calibration stage are visible here. The
interface instructs an engineer to write the case against **39%**, not 52% — that
is what a stated failure rate is *for*. And the recommended plan is **solved
rather than predicted**: one extra bed through the same loop, 28.7 s → 32.1 s,
which immediately caught the surrogate reporting a **111.7%** reduction where the
solve gives **99.81%** — past the physical maximum, at a configuration where the
margin was unattainable so nothing flagged it.

---

## 5. The result we did not expect: a peak-depth map cannot route an evacuation

Levees are decided months ahead. On the night the question is a trip: *I am here,
I need to get there, what do I do.*

### 5.1 The system

Two places in. The twin derives a box containing both with room to detour, picks
a resolution that keeps the solve interactive, fetches terrain, solves the flood
across it, pulls the road network from OpenStreetMap, builds a routable graph —
junctions from coincident vertices, verified at **287.4 km against a published
288.0 km**, 98% of junctions in one component — and routes. Lismore to Coraki,
20 km, ~4,120 road segments over 1,346 km, **104 s end to end**.

The flood is carried as a **schedule**, not a peak: instantaneous depth at 13
times across the run, and a road is usable for a crossing only if it stays under
the vehicle's depth for the whole time the vehicle is on it. Speeds are free-flow
by road class; a road is judged by its **deepest** point, because that is where a
vehicle stops.

### 5.2 The hero result

Over an eight-member forcing ensemble (×0.40 to ×1.60 of a stated design
discharge; peak network cut spanning 55–190 km), leave-one-out with each member
in turn as reality, on the **12 of 523** candidate trips the members disagree
about — **96 decisions**. An arm succeeds only if the traveller is **never on a
road whose true depth exceeds what the vehicle can cross while they are on it**.

| what the operator is routed on | got through | caught by water | no route offered |
|---|---|---|---|
| **the peak-depth field** | **0 / 96** | 0 | **96** |
| the time-resolved flood, leaving now | **40 / 96** | 7 | 49 |
| + our conformal margin | 40 / 96 | 7 | 49 |
| re-planning on members consistent with what is observed | **54 / 96** | 14 | 28 |
| oracle: plans on the storm that happens | **82 / 96** | 0 | 14 |

> **A peak-depth map is the union of states that never coexist in time.** Route
> on it and it refuses roads that are dry when the driver passes them: on this
> network it reports that **nobody can leave**, on every one of 96 decisions,
> where an oracle gets 82 out and a time-resolved solve 40. The peak field is
> what essentially every flood twin publishes, including ours, which is why this
> is the most transferable finding in the paper.

On a single trip the same effect is a **3h30 deadline**: leaving now the route is
22 minutes and dry; leaving at the last possible departure it is 25.5 minutes on
a **different road**, because by then Coraki Road is under **1.0–2.9 m** where
the direct line crosses it. Neither the deadline nor the detour is computable
from a peak.

**Figure 6** `runs/portal_shots/96_trip_river.png`.

### 5.3 Two results that argue against us

**Our own calibration contributes nothing here.** The route-level conformal
margin is **exactly 0.000 m on all 96 decisions**. The route the central forecast
chooses is dry under every member, so the residual of the reported scalar is
identically zero: what differs between members is not *how deep the chosen road
gets* but **which roads exist**. The decision is discrete and the margin is
continuous. The method's own feasibility statement cuts both ways — it says which
distribution a margin must use, not that a margin exists for every decision.

**Observation helps as a 2:1 bet, not a free lunch.** Filtering the ensemble on
observed depths — Perceive feeding Plan, with a *stated* 0.02 m tolerance and no
training — converts **21 of the forecast's 49 refusals into departures: 14 get
through, 7 are caught.** Nobody who was arriving stops arriving and nobody who
was caught is saved. Reporting this as "+14 arrivals" would hide the seven, and
being trapped on a flooding road is not the same outcome as having stayed home.

**And the routing is exact dynamic programming, not a learned policy.** We say so
because dressing Dijkstra as planning is the overclaim this paper otherwise
argues against. Label-setting on arrival time is optimal here **only because
waiting is permitted**: edge availability is non-monotone once the flood recedes,
and without holding, a two-road test case differs by more than a factor of two
(300 minutes against 127.5).

---

## 6. The portal: the loop, operable

A framework that only appears in a results table has not been shown usable.
Everything above runs behind an interactive interface that works anywhere on
Earth.

### 6.1 Every option's water, and the trace that produced it

The planner solves every option it offers, so selecting a ranked row redraws the
flood from that option's **own solved field** — the comparison the list exists
for. It previously moved a radio button and nothing else: the fields were
computed and discarded. The budget allocation is shown as a **trace**, each 0.5 m
step and what it bought, so an engineer can see where the marginal return fell
off rather than being handed a height vector.

**Figure 7** `runs/portal_shots/81_option_picked.png` ·
**Figure 8** `runs/portal_shots/50_budget_trace.png`

### 6.2 The flood as a body of water

This terrain has 220 m of relief over 32 km while the water reaches 19 m deep:
the **water** is the three-dimensional object in the scene, and painting it flat
hides the one quantity with real vertical extent. Each cell becomes a column on
its own ground at its own depth, so a levee reads as a wall holding a volume back
rather than a marker beside a blue shape.

**Figure 9** `runs/portal_shots/60_water_solid.png` against
`62_water_flat.png` — same event, as a volume and as paint.

### 6.3 The trade-off is a control, and the limits have a page

`/methodology/margin` makes the fraction of achievable gain you will give back
for a guarantee a **slider**, and shows the crossing move as you drag it, because
it is a decision and not a constant. `/methodology/limits` writes what the tool
cannot show **into the tool**.

**Figure 10** `53_margin_f_low.png` + `53_margin_f_high.png` ·
**Figure 11** `54_limits.png`

### 6.4 The bug that justifies the calibration stage

Our own portal displayed a confidence it had no data to support:
`1.645 · RMSE · √(1 + 1/n)`, a normal-approximation interval described in its own
code as "the product-facing form of the framework's safety margin". It was not.
Two errors compounded: the fit used **six** multi-measure runs where 90% coverage
requires at least **nine**, and a Gaussian places the 90% point at 1.64×RMSE
where the honest order statistic is roughly twice that. **The displayed band was
about half the width it should have been.**

A normal-approximation interval always returns something. The conformal quantile
**refuses** — and with 30 held-out runs it declines a request for 99% and says
how many more runs that would need. That is not a detail of the method; it is the
reason to prefer it, and it is why stage 4 is in the loop at all.

**Figure 12** `runs/portal_shots/52_refusal.png`

### 6.5 An explanation the service will not ship unless it matches the plan

The written brief is generated from the same record the map is drawn from, and it
is **parseable**, so faithfulness is computed on every response and a brief that
does not round-trip is **withheld**. The margin's refusal discipline, applied to
prose, and auditable from outside the service because the score travels in the
response.

Its first live run found the guard's own blind spot. A trip that correctly
returned *no route* shipped, at faithfulness 1.000: *"Every way through is under
more water than 25 cm. Cut: Casino Street under 147 cm. **0 km of 212 km is
impassable.**"* The last sentence contradicts the first — the brief quoted the
cut length at t = 0, which is structurally zero because every run starts dry, and
the check was reading only the departure call. Both fixed; the old sentence now
scores 0.750 and is withheld.

### 6.6 What the interface could not express

Chasing one figure found a capability gap worth more than the figure. The trip
view had no way to **state** a design discharge, so it inherited bankfull from
DEM geometry — **234 m³/s** on the Wilsons River, which the largest storm preset
takes to under a thousand against a flood of record of **3,000–4,000**. Every
trip the interface could model was therefore a **1.4%-of-network** event with a
dry route and no deadline, described by a brief that scored 1.000 faithfulness
because it correctly described a non-event. The numbers in §5 came from a command
line the interface could not reproduce.

It now takes a discharge, with the interface stating that the estimate is a 30 m
elevation inference and not a gauged rating. We record this because the gap was
invisible from inside the results: every number was correct.

---

## 7. What this enables, and what we are not claiming

### Enables

**A stated failure rate on any twin that can occasionally check itself** — the
recipe needs matched pairs from a periodic probe and nothing about the error
distribution.

**Decisions a map cannot support.** The same flood, the same network: the
standard peak-depth output reports no route on 96 of 96 decisions where a
time-resolved solve gets 40 out and an oracle 82.

**Search where a policy will not do** — 21 points survive the best distillation
recipe we could implement, on real fire data.

**A testable standard for "our assimilation is good enough"** — run the classical
filter with its error term swept and show where the optimum sits.

### Not claiming

- **No fire in our archive had a firebreak cut on our instruction.** The
  intervention arm is unvalidated in reality and no retrospective dataset can fix
  that.
- **One validated flood event**, at CSI 0.53 against satellite.
- **The guarantee is marginal, not per-step.**
- **The evacuation deadline is one realisation.** The ensemble spans *how big the
  flood is* and nothing else — not timing, not spatial pattern, not solver error.
  And it reproduces from the stated coordinates, not from the place names:
  geocoding "Lismore NSW" returns a point 0.6 km away, which shifts the derived
  box enough that the channel detector finds nothing and the driver falls back to
  rainfall.
- **The calibration contributes nothing to route choice** (§5.3), and a trained
  policy over the same ensemble is an obvious arm we have not run.
- **Exchangeability fails on one of two testbeds** in the method paper's
  experiments, with coverage attained regardless.
- The model-adaptation benefit is **small** (d = 0.20 at day +1).

### On the methodology

We retracted three results of our own during development: a distillation number
that was an artefact of a collapsed loss, a baseline comparison invalidated by an
extrapolation bias four times the effect it tested, and a statistical claim whose
interval, recomputed over the correct unit, spanned its own effect size. **All
three initially pointed the way that favoured us**, which is why a
claim-supporting result is now a trigger for an extra check.

**41 measurement defects are catalogued with the check that caught each, and 47
rules derived from them.** Four are worth a workshop audience's time because each
is a *systems* failure invisible from inside the results:

1. **A downsample that lost up to 59% of a fire's detections** — continuously,
   only for some fires, every downstream number plausible. Only a conservation
   check finds it.
2. **An interface displaying 90% confidence from six calibration runs** where
   nine are required (§6.4).
3. **A coverage rate averaged over configurations in which no plan existed**,
   which moved a headline by four points.
4. **Figures cited as current that predated the interface rebuild** — including
   the figure an earlier draft called its most important. Every portal figure
   here now carries a capture date, because an interface is rebuilt more often
   than a paper is written.

And one we chose not to exploit: a **deadline** was available for Figure 6 at
8,000 m³/s (7h00) or 20,000 (3h30), but the service applies 2.5× the stated peak,
so those figures would show 20,000 and 50,000 m³/s on a river whose flood of
record is 3,000–4,000. The published figure is the plausible event, states that
it contains **no deadline**, and names what was rejected. Choosing the larger
flood would have been every defect in the catalogue committed on purpose.

---

## Figures

| # | file | shows | captured |
|---|---|---|---|
| 1 | `docs/figures/paper/fig8_twin_loop.png` | the five-stage loop | 2026-09-25 |
| 2 | `docs/figures/paper/fig4_wildfire.png` | budget Pareto and horizon ablation | 2026-09-25 |
| 3 | `docs/figures/paper/fig10_flood_precondition.png` | why rainfall recommends nothing | 2026-09-25 |
| 4 | `docs/figures/paper/figA1_sar_validation.png` | Sentinel-1 validation, permanent water separated | 2026-09-27 |
| 5 | `runs/portal_shots/45_plan_result.png` | the recommendation, its floor, attribution, harmful measures | **2026-10-01** |
| 6 | `runs/portal_shots/96_trip_river.png` | a trip across a flood cutting 157 km of 1,309; route clear, **no deadline**, discharge stated not estimated | **2026-10-01** |
| 7 | `runs/portal_shots/81_option_picked.png` | selecting a ranked option redraws the water | 2026-10-01 |
| 8 | `runs/portal_shots/50_budget_trace.png` | the greedy allocation, step by step | **2026-10-01** |
| 9 | `runs/portal_shots/60_water_solid.png` + `62_water_flat.png` | the flood as a volume against the flood as paint | **2026-10-01** |
| 10 | `runs/portal_shots/53_margin_f_low.png` + `_high.png` | the trade-off as a control | **2026-10-01** |
| 11 | `runs/portal_shots/54_limits.png` | what the tool cannot show, inside the tool | **2026-10-01** |
| 12 | `runs/portal_shots/52_refusal.png` | the margin refusing a level it cannot support | 2026-10-01 |
| 13 | `runs/portal_shots/44_roads.png` | live OpenStreetMap road network, scored for passability | **2026-10-01** |
| 14 | `runs/portal_shots/91_route_district.png` | the route card: no way out at car depth, 213 km cut | 2026-10-01 |

**The capture column is not bookkeeping.** An interface is rebuilt more often
than a paper is written, so a figure without a date is an unfalsifiable claim
about software. `figA5_portal_appraisal` and `figA6_portal_anywhere` from earlier
drafts are **not** used: they embed a superseded interface. `95_trip.png` is
retained and unused — it is the rainfall fallback described in §6.6.

## Data and reproducibility

WildfireSpreadTS (Zenodo 8006177); Next Day Wildfire Spread; Copernicus DEM
GLO-30; Sentinel-1 RTC and ESA WorldCover via Microsoft Planetary Computer;
Open-Meteo; OpenStreetMap via Overpass; FloodCastBench as the reference
hydrodynamic solution.

Every result is produced by a script in `eval/` with seeds, limits and `δ` as
command-line arguments; cluster scripts in `scripts/tacc/`. The portal's own
checks are `portal/scripts/verify-*.mjs`, including a parity test holding the
browser's router against the Python one the paper's numbers come from — **14 of
14 quantities identical** — and `refresh-shots.mjs`, `shoot-methodology.mjs` and
`shoot-trip.mjs`, which make every figure above repeatable.

The calibration method is `ICML_PAPER.md`; its derivations, the defect catalogue
and the provenance of every number are `THESIS_AND_RESULTS.md`.
