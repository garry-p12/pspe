# Planning Inside a Digital Twin, With a Safety Margin You Can State

*Workshop paper. ~45% method, the rest two working twins and a portal. The
structural result is stated here and **derived in full** in
`THESIS_AND_RESULTS.md` Part 10, which also carries every number's provenance;
this paper spends its space on what the result enables.*

---

## Abstract

Climate digital twins forecast well and **decide** poorly. A twin that predicts
a flood is useful; a twin that recommends which levee to build, under a fixed
budget, with a stated bound on how wrong it might be, is a different instrument.
The gap is not simulation fidelity — it is that planning inside a learned or
reduced-order model inherits that model's errors, and nobody says by how much.

We contribute a safety margin for decisions made inside an imperfect simulator,
calibrated on the **realised cost distribution** rather than on model error or
ensemble disagreement. This is forced rather than preferred: no margin
measurable from the model-error distribution alone attains a stated level
uniformly in the spread of realised cost. Model-error margins are the right
instrument for a *pointwise* guarantee and the wrong one under an *expectation*
constraint, and the constraint picks which distribution you need.

On a reaction–diffusion hazard the margin cuts constraint violations from
**14.1% to 5.0%** (20 seeds, paired −9.1 points, 95% CI [−11.8, −5.5],
p = 0.0002) at **no cost in objective value**, and beats a published model-based
safe-RL method by 7.0 points (95% CI [+4.0, +10.0]). Three published
disagreement-based corrections land at 11.99%, 12.20% and 13.17%, none better
than no margin at all — the theorem's prediction, pre-registered for one of them.

We then build two twins. A **wildfire twin** over 607 observed fires closes an
assimilate–forecast–plan loop and matches an operational heuristic's burned-area
reduction with **half the crews**. A **flood twin** over Copernicus terrain plans
levees anywhere on Earth against a real budget, and — the result we did not
expect — shows that routing an evacuation on the peak-depth field every flood
twin publishes finds **no route at all** on 96 of 96 decisions where a
time-resolved solve gets through.

We also report five original claims of our own that did not survive, three
retracted results, and the place where our own margin contributes nothing.

---

## 1. The problem: twins that forecast but cannot advise

An operational question sounds like this. *We have A$20M. There are six places
we could build. Which, how high, and how wrong might you be?*

A forecast cannot answer it. It says what will happen if nothing changes. The
question is about what changes, under a budget, with a consequence attached to
being wrong.

**What "digital twin" has to mean for this to be useful.** Not a dashboard over
a simulation. Four things, closed into a loop:

| stage | what it must do |
|---|---|
| Perceive | synchronise to observation, not just initialise from it |
| Simulate | be fast enough to be searched over, and honest about where it is wrong |
| Plan | optimise a decision under a real constraint, not a soft penalty |
| Explain | say what it decided and why, in words that can be checked |

The gap this paper addresses sits between Simulate and Plan: a planner searching
inside a model inherits that model's errors *silently*.

---

## 2. Technical contribution

### 2.1 The failure being fixed

Constrained planning in a learned model is usually a Lagrangian: maximise
reward, penalise constraint cost, let a dual find the trade-off. It works when
the cost estimate is unbiased. Inside a surrogate it is not, and the failure is
specific: **the dual converges, reports the constraint satisfied, and the real
system violates it anyway.** Measured: 7.3% of evaluations over the limit while
every internal diagnostic looks correct.

The obvious fix is to inflate the estimate by how wrong the model is. We tried
it and violations rose from **14.5% to 18.2%**. The reason is our main conceptual
point, and it needs its scope attached, because the unqualified version is both
less accurate and less interesting:

> A model-error margin is the **right** instrument for a pointwise guarantee —
> bound the error at each state and you bound the state. It is the **wrong**
> instrument under an *expectation* constraint, where what carries a trajectory
> over the limit is not how wrong the model is on average but the **spread of
> realised cost from episode to episode**. Two different distributions; the
> constraint picks which one you need.

And it is a theorem, not a preference. Write `R(q; σ)` for realised violation
when margin `q` meets realised-cost spread `σ`.

> **Proposition (separation).** Let `Q` be any margin rule that is a measurable
> function of the model-error distribution alone. Then `R` is strictly
> increasing in `σ`, tends to ½ as `σ → ∞` for symmetric error, and exceeds δ
> for every `σ > (Q − b)/z_δ`. Hence **no margin measurable with respect to
> model error alone attains level δ uniformly in σ.**

The proof is three lines (§10.3 of `THESIS_AND_RESULTS.md`, which derives
every result in this paper from one setup). For Gaussian error it closes to
`R(ρ) = 1 − Φ(z_δ/ρ)` with `ρ = σ/ε`, which predicts measured violation to
within **0.003** across two orders of magnitude in ρ — and predicts that
requesting a *stricter* guarantee from the wrong distribution makes matters
worse, the error growing from 1.2× at δ = 0.30 to 3.7× at δ = 0.02.

The same proposition covers **ensemble disagreement**, which is also a function
of model wrongness. We test it rather than resting on the theory.

### 2.2 The margin

Let `ĝ` be the quantity the dual controls. Calibrate on matched pairs
`s_i = c_i − ĝ` from a periodic probe of the true environment and take a
split-conformal quantile. Adding it to the limit gives

```
P[ realised cost > limit ] ≤ δ
```

with **δ chosen and stated**, not tuned. Two properties matter operationally.
The quantile **refuses** to return a value when calibration data is insufficient
(`n ≥ 1/δ − 1`) rather than extrapolating — this caught a real bug in our own
portal (§5.6). And it is computed from outcomes, so it assumes nothing about the
surrogate's error structure.

### 2.3 Does it work

Reaction–diffusion hazard, 20 seeds, δ = 0.1:

| arm | violating evaluations | objective |
|---|---|---|
| no margin | 14.1% | — |
| model-error margin | 18.2% | — |
| **probe + conformal margin** | **5.0%** | **no loss** |

Paired **−9.1 points, 95% CI [−11.8, −5.5], p = 0.0002**, objective
statistically indistinguishable. The margin removes a bias; it does not buy
safety with performance.

A second family (diffusion–advection) goes 0% → 0%: nobody violates, so the
margin cannot help and must not hurt. That is a regression check, **not** a
second validation. A third (shallow water) needed its cost limit calibrated
against a trained reward-greedy anchor (limit 1.1437); there the margin cuts
33.2% → 16.8% over 20 seeds, paired 95% CI [−24.7, −8.0], and over the three
quarters of seeds where a plan exists at δ = 0.1 the rate is 12.1%, 95% CI
[7.1, 17.1] — containing the level without confirming it. On the other quarter
the margin consumes the entire budget and the task is infeasible, which we found
by reading per-seed data after an aggregate looked bad.

**The honest summary across families: the mechanism replicates on three; the
stated rate is attained on one, trivially satisfied on a second, and not
rejected on the third.** Not "validated on three families".

### 2.4 Against three published corrections

All implemented inside *our* planner, so the only difference is how model error
is corrected for:

| method | violating | beats no margin? |
|---|---|---|
| no margin | 14.1% | — |
| conformal ensemble disagreement | 11.99% | no |
| SMBPO | 12.20% | no |
| SafeDreamer | 13.17% | no |
| **ours** | **5.00%** | — |

Within 1.2 points of each other, none separating from no margin. **We
pre-registered 12% for SafeDreamer before running it**; it returned 13.17%. They
are not three methods that happened to lose — they are three instances of the
class the theorem excludes.

Two of our own implementations were wrong first: SMBPO's truncated prefix cost
rescaled linearly on a non-linear cost, a −4.7% bias four times the pessimism
term that made the arm net *optimistic*; and SafeDreamer's cost critic shadowing
the trainer's own `critic`, scoring the wrong head. Both errors initially
favoured us.

### 2.5 The assumption, tested where it fails

Split conformal needs exchangeable calibration scores, and ours are collected
during a learning run. Three seeds per testbed, 320 scores, 160/160 split:
**on rdf exchangeability is violated** — lag-1 serial dependence at p
indistinguishable from zero on every seed, with drift and distribution-shift
tests both non-rejections. Dependent, not drifting. **Coverage holds anyway**,
0.0854 against δ = 0.1 (dar: 0.0771). Dependence of this kind inflates the
quantile's variance without biasing it, which is why the beyond-exchangeability
bound is the right frame and why adaptive conformal inference is unnecessary
here — demonstrated rather than assumed.

### 2.6 Why decision-time planning, not a trained policy

We tested the strongest version of the objection — distilling our planner's own
solutions into a compact policy, the recipe the literature identifies as
effective:

| arm | burned-area reduction |
|---|---|
| decision-time planner | **39.0%** |
| distilled from the planner | 18.3% |
| policy trained from scratch | 10.4% |

Distillation recovers **47%** — a real gain, and a concession that our earlier
framing overstated the case. But **21 points survive it**. The mechanism is in
the diagnostics: the planner draws 16.0 points of its effect through the learned
spread dynamics, the distilled policy 1.0. A forward pass learns where fires
generally go; it does not reason about how *this* fire's fuel state responds.

---

## 3. Application 1: a wildfire twin over 607 observed fires

### 3.1 Setup

WildfireSpreadTS, 607 fires. A learned spread model, an ensemble Kalman update
against observation, and decision-time planning of crew allocation under a daily
budget.

### 3.2 Assimilation works, and we checked it against a real filter

Rather than assert that a simple correction suffices, we ran the classical
filter with its error term swept and showed where the optimum sits. The
adaptation benefit is **small and reported as such**: d = 0.20 at day +1. We
give the effect size, not the p-value alone.

**Figure 1** `docs/figures/paper/fig8_twin_loop.png` — the
assimilate–forecast–plan loop.

### 3.3 Planning: the same outcome with half the crews

At a 3-day planning horizon and a 3% daily crew budget, planning through the
learned spread model beats the operational forecast-then-treat heuristic by
**12 points** of burned-area reduction — equivalently, it matches the
heuristic's reduction with **half the crews**.

**The operating point belongs in the sentence.** The gap runs **+0.6 at one day**
— where theory requires it to vanish — to **+14.3 at five days**, and **+6.8 at
a 1% budget** to **+16.9 at 5%**. Both sweeps are in the appendix; the headline
is the 3-day row of the horizon ablation, not a separate experiment.

**Figure 2** `docs/figures/paper/fig4_wildfire.png` — budget Pareto and horizon
ablation.

### 3.4 Tested where theory says it must fail

The horizon sweep is the mechanism test: at a one-day horizon there is nothing
for a planner to reason about and the advantage should vanish. It does (+0.6).
A result that survives its own falsification test is worth more than one that
was never exposed to it.

An earlier version of this claim was measured on 8 fires with t = 3.0. At 607
fires the same benefit is **a third the size** and holds on 76% of them. The
8-fire number was not wrong; it was underpowered, and we re-ran it rather than
publish it.

---

## 4. Application 2: a flood twin that plans levees anywhere

### 4.1 Setup

Copernicus DEM GLO-30 terrain, a LISFLOOD-FP-style local-inertial solver batched
over candidate plans, and forcing from river discharge, coastal surge or
rainfall — whichever the site actually has. Terrain and forecast are fetched on
demand, so the tool works on terrain that is nowhere in the repository.

### 4.2 The forcing has to be physical, and we got it wrong twice

| forcing | best single site | best plan | verdict |
|---|---|---|---|
| rainfall | 0.4% | — | "no measure worth building" |
| river, as a volume | 7.6% | 8.5% | physically void (112 m deep) |
| **river, as a stage** | **16.2%** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

Rain falling across the whole box gives a levee nothing to block — the tool
correctly recommends nothing, which is a real answer. Injecting discharge as a
volume made water arrive faster than cells could spread it and reached 112 m
depth, a number with no physical meaning. A boundary sets a **stage**, not a
volume; Manning's normal depth gives it with no free parameters.

Also found and fixed: DEM nodata filled with sea level created a 247 m trench
ringing every inland domain, draining the catchment into it.

**Figure 3** `docs/figures/paper/fig10_flood_precondition.png` — why rainfall
recommends nothing.

### 4.3 Validated against satellite, including the uncomfortable part

Richmond NSW 2022, Sentinel-1 SAR, permanent water held separately so rivers are
not scored as model error.

- Our fast solver reproduces the reference hydrodynamic model at **CSI 0.979**.
- The reference model reproduces the **satellite** at **CSI 0.535**.

The second number belongs in an operational discussion and is not flattering. A
0.98 model-to-model agreement is a statement about our numerics; a 0.53
model-to-observation agreement is a statement about how well anyone's
hydrodynamics matches a real flood at 30 m. We report both, because quoting only
the first is how decision-support tools acquire unearned authority. **One
validated event is also one event** — a demonstration of a validation *method*,
not an established skill score.

**Figure 4** `docs/figures/paper/figA1_sar_validation.png`.

---

## 5. The portal: the loop, operable

A framework that only appears in a results table has not been shown usable.

### 5.1 One click, end to end

> **Budget A$20M** · 12 of 36 options fit
> → **Recommend:** levee at site 3 (3.0 m) + site 5 (1.0 m), A$18.2M of A$20M
> → **Estimated 52% · at least 39%, 90% of the time**
> → *"Margin ±13 points, from 30 held-out solver runs. Write the business case
>   against the lower figure."*
> → **Attribution:** site 3 +43% (47% alone) · site 5 +6% (9% alone)
> → **Warns:** site 1 (−9%), site 0 (−9%) — measured as *deepening* flooding
> → **Caveat on screen:** the plan was chosen inside a surrogate; adopting it
>   should be conditional on a full hydrodynamic run of this exact height vector

Two details are the contribution made visible rather than asserted. The
interface tells an engineer to **write the business case against the lower
figure**, which is what a stated failure rate is *for*. And it reports two
candidate levees as actively harmful — a backwater effect — rather than quietly
ranking them last.

**Figure 5** `runs/portal_shots/45_plan_result.png` — the single most important
figure here.

### 5.2 Every option's water, not just the winner

The planner solves every option it offers in one batched timestep loop, so a
ranked list costs about what one scenario costs. Selecting a row now redraws the
flood from that option's own solved field — the comparison the list exists for.
It previously moved a radio button and nothing else: the fields were computed
and discarded.

The recommended plan is now **solved rather than predicted** — one extra bed
through the same loop, 28.7 s → 32.1 s — which immediately caught the surrogate
reporting a **111.7%** reduction where the solve gives **99.81%**: past the
physical maximum, at a configuration where the margin was unattainable so nothing
flagged it.

**Figure 5b** `runs/portal_shots/81_option_picked.png`.

### 5.3 What else the interface does, because the honesty is in the detail

Four things in the portal exist specifically because a number without them
misleads, and they are worth listing because each is a design consequence of
§2.1–§2.2 rather than decoration.

**The flood is drawn as a body of water, not a blue picture.** The terrain has
220 m of relief over 32 km (1:146, flatter than it sounds once projected) while
the water reaches 19 m deep: the water is the three-dimensional object in the
scene, and painting it flat hides the one quantity with genuine vertical extent.
Each cell becomes a column standing on its own ground at its own depth, so a
levee reads as a wall holding a volume back rather than a marker beside a blue
shape. The flat paint remains available, which is what makes the comparison
legible.

**Figure 6** `runs/portal_shots/60_water_solid.png` against
`runs/portal_shots/62_water_flat.png` — the same event, drawn as a volume and as
paint.

**The allocation is shown as a trace, not a verdict.** The greedy search spends
the budget in 0.5 m steps and the interface shows each step and what it bought,
so an engineer can see where the marginal return fell off rather than being
handed a height vector. **Figure 7** `runs/portal_shots/50_budget_trace.png`.

**The margin has its own page, with the trade-off as a control.**
`/methodology/margin` makes `f` — the fraction of the achievable gain you are
willing to give back for a guarantee — a slider, and shows the crossing move as
you drag it. The point of exposing it is that `f` is a decision and not a
constant. **Figure 8** `runs/portal_shots/53_margin_f_low.png` and
`53_margin_f_high.png` at the two ends of the range.

**And a page for what the tool cannot show.** `/methodology/limits` is written
into the interface rather than left to the paper: one validated event, a marginal
rather than per-step guarantee, an unvalidated intervention arm.
**Figure 9** `runs/portal_shots/54_limits.png`.

### 5.4 Can we get out of this town, and by when

Levees are decided months ahead. On the night the question is a trip: *I am here,
I need to get there, what do I do.*

Give the portal two places and it derives everything else — a box containing both
with room to detour, at a resolution that keeps the solve interactive — fetches
terrain, solves the flood, pulls the road network from OpenStreetMap, and routes.
Lismore to Coraki, 20 km, ~4,120 road segments over 1,346 km, **104 s** end to
end. The forcing is a **stated design discharge** (3,000 m³/s scaled 2.5× by the
severe-storm preset), not today's gauge reading.

The flood is carried as a **schedule**, not a peak: instantaneous depth at 13
times across the run, and a road is usable for a crossing only if it stays under
the vehicle's depth for the whole time the vehicle is on it. That one change
decides the answer:

| what the operator is told | result |
|---|---|
| routed against the **peak** depth field | **no route exists** |
| routed against the **schedule**, leaving now | 22 min, 24.8 km, no standing water |
| the same trip at the last possible departure | 25.5 min, 27.7 km, **a different road** |
| how long the trip stays possible | **3h30** |

A peak-depth map — the standard output of every flood twin, ours included — is
the union of states that never coexist. Routing on it refuses roads that are dry
when the driver passes them, and tells this operator there is **no way out of
Lismore**. There is, for three and a half hours. Leave at the deadline and the
route is a different road, because by then Coraki Road is under **1.0–2.9 m**
where the direct line crosses it.

**And it is not one trip.** Over an eight-member forcing ensemble (×0.40 to
×1.60 of the design discharge, peak network cut spanning 55–190 km),
leave-one-out, on the 12 trips of 523 candidates that the members disagree
about — 96 decisions:

| arm | got through | caught by water | no plan |
|---|---|---|---|
| leave now, on the mean forecast | 41.7% | 7 | 49 |
| **route on the peak field** | **0.0%** | 0 | **96** |
| time-dependent on the mean forecast | 41.7% | 7 | 49 |
| + conformal margin on the route scalar | 41.7% | 7 | 49 |
| **re-plan on members consistent with what is observed** | **56.2%** | 14 | 28 |
| oracle: plan on the member that happens | 85.4% | 0 | 14 |

An arm succeeds only if the traveller is **never on a road whose true depth
exceeds what the vehicle can cross while they are on it**; judging a plan by
whether it looked safe when it was made is how this kind of experiment flatters
itself.

**The peak field fails 96 times out of 96**, against an oracle that gets through
85% of the time. This is the most transferable result in the paper, because the
peak field is what essentially every flood twin publishes.

**Our own margin contributes nothing here, and we say so.** It is exactly
0.000 m on all 96 decisions: the route the central forecast chooses is dry under
every member, so the residual of the reported scalar is identically zero. What
differs between members is not *how deep the chosen road gets* but **which roads
exist**. The decision is discrete and the margin is continuous. A feasibility
theorem cuts both ways — it says which distribution a margin must use, not that
a margin exists for every decision.

**Observation helps, as a 2:1 bet rather than a free lunch.** Filtering the
ensemble on observed depths — Perceive feeding Plan, stated tolerance, no
training — converts **21 of the central forecast's 49 refusals into departures,
of which 14 get through and 7 are caught**. Nobody who was arriving stops
arriving; nobody who was caught is saved. Quoting "+14.5 points" would hide the
seven, and being trapped on a flooding road is not the same outcome as having
stayed home.

The routing itself is a time-dependent shortest path solved exactly, with
waiting permitted — **not** a learned policy, and we say so because dressing
Dijkstra as planning is the overclaim this paper otherwise argues against. Three
PSPE stages do the work: Simulate supplies the schedule, Plan solves the search,
Explain emits the brief. **Perceive is not in the single-trip loop**, and
claiming "all four stages" would be a claim about a diagram.

**Figure 10** `runs/portal_shots/96_trip_river.png` — Lismore Base Hospital to
Coraki, river driver at a **stated** 3,000 m³/s (the service applies the severe
preset's 2.5×, so 7,500 m³/s held as a 4.0 m stage on the north edge). **157 km
of the 1,309 km network is impassable at the worst of this flood (12%)**, and
the trip is 22 minutes over 23.8 km meeting no standing water, with **no
deadline** — this route runs on ground the flood does not reach.

Three things about this figure are deliberate, and the third is the reason the
earlier attempts were discarded. The discharge is **stated rather than
estimated**, because bankfull from a 30 m DEM on this reach is 234 m³/s and the
flood it implies is not the one anyone plans against. The caption says *no
deadline*, because there is none: the route is clear at every departure in the
window. And **a deadline would have been available at 8,000 m³/s** (7h00, 28% of
the network cut) or at 20,000 (3h30, 42%) — but the service multiplies the
stated peak by 2.5, so those figures show 20,000 and 50,000 m³/s on a river
whose flood of record is of order 3,000–4,000. Choosing a larger flood to
obtain a more impressive caption is the failure mode this project is otherwise
about, so the figure shows the honest event and says what it does not contain.

§5.4's 3h30 deadline is measured on a different pair — Lismore town centre to
Coraki at the same stated discharge — whose box the channel detector only
reaches from coordinates the geocoder does not return for "Lismore NSW". That
asymmetry is a property of the detector on a 130 m grid, not of the method, and
is recorded here rather than hidden by a screenshot of a bigger flood.

### 5.5 An explanation the service will not ship unless it matches the plan

The route brief is generated from the same record the map is drawn from, and it
is **parseable**, so faithfulness can be computed: the text is scored against
the route it claims to describe and **withheld** if it disagrees. That is the
margin's refusal discipline applied to prose. The score travels in the response,
so the guarantee is auditable from outside the service.

Its first live run found the guard's own blind spot. A trip that correctly
returned *no route* shipped, at faithfulness 1.000: *"Every way through is under
more water than 25 cm. Cut: Casino Street under 147 cm. **0 km of 212 km is
impassable.**"* The last sentence contradicts the first — the brief was quoting
the cut length at t = 0, which is structurally zero because every run starts
dry, and the check was only reading the departure call. Both fixed; the old
sentence now scores 0.750 and is withheld.

### 5.6 The bug that justifies the whole margin apparatus

Our own portal displayed a confidence it had no data to support:
`1.645 · RMSE · √(1 + 1/n)`, a normal-approximation interval described in its own
code as "the product-facing form of the framework's safety margin". It was not.
Two errors compounded: the fit was tuned on **six** multi-measure runs where 90%
coverage requires at least **nine**, and a Gaussian places the 90% point at
1.64×RMSE where the honest order statistic is roughly twice that. The displayed
band was about half the width it should have been.

A normal-approximation interval always returns something. The conformal
quantile **refuses**. That is not a detail of the method; it is the reason to
prefer it.

**Figure 11** `runs/portal_shots/52_refusal.png`.

---

## 6. What this enables, and what we are not claiming

### Enables

**A stated failure rate on any twin that can occasionally check itself.** The
recipe needs matched pairs from a periodic probe and nothing about the error
distribution.

**Search where a policy will not do.** 21 points survive the best distillation
recipe we could implement, on real fire data.

**Decisions a map cannot support.** The same flood, the same network: the
standard peak-depth output says *no route out* where a time-resolved solve finds
one with three and a half hours of margin, on 96 of 96 decisions. Anything
decided against a union of states that never coexist is decided against a
scenario that never happens.

**A testable standard for "our assimilation is good enough."** Run the classical
filter with its error term swept and show where the optimum sits.

### Not claiming

- **No fire in our archive had a firebreak cut on our instruction.** Planning is
  validated against held-out observations *inside* the model. The intervention
  arm is unvalidated in reality and no retrospective dataset can fix that.
- **One validated flood event**, at CSI 0.53 against satellite.
- **The guarantee is marginal, not per-step**: it bounds the probability that a
  trajectory's total cost exceeds the limit, not the cost at every timestep.
- **Exchangeability is tested, and on one of two testbeds it fails** — lag-1
  dependence, p ≈ 0, three seeds — with coverage attained on both regardless. We
  report this rather than the easier sentence, because it is the first thing a
  conformal reviewer checks.
- **The evacuation deadline is one realisation, not a forecast distribution.**
  "3h30" is conditional on the storm arriving as modelled, and the ensemble in
  §5.4 spans *how big the flood is* and nothing else — not timing, not spatial
  pattern, not the solver's own error.
- **The margin contributes nothing to route choice** (§5.4), and a trained
  policy over the same ensemble is an obvious arm we have not run.
- The model-adaptation benefit is **small** (d = 0.20 at day +1).

### On the methodology, briefly

We retracted three results of our own during development: a distillation number
that was an artefact of a loss collapsing to "treat nothing"; a baseline
comparison invalidated by a 4.7% extrapolation bias four times the mechanism it
was meant to test; and a statistical claim whose interval, recomputed over the
correct unit, spanned its own effect size. **All three initially pointed the way
that favoured us**, which is why a claim-supporting result is now a trigger for
an extra check rather than a reason to stop.

39 measurement defects are catalogued with the check that caught each, and 45
rules derived from them. The one to remember degrades data *continuously*: a
downsample that lost up to **59%** of a fire's detections, smoothly and only for
some fires, leaving every downstream number plausible. Nothing short of a
conservation check finds it.

---

## Figures

| # | file | shows | captured |
|---|---|---|---|
| 1 | `docs/figures/paper/fig8_twin_loop.png` | the assimilate–forecast–plan loop | 2026-09-25 |
| 2 | `docs/figures/paper/fig4_wildfire.png` | budget Pareto and horizon ablation | 2026-09-25 |
| 3 | `docs/figures/paper/fig10_flood_precondition.png` | why rainfall recommends nothing | 2026-09-25 |
| 4 | `docs/figures/paper/figA1_sar_validation.png` | Sentinel-1 validation, permanent water separated | 2026-09-27 |
| 5 | `runs/portal_shots/45_plan_result.png` | the recommendation, its guaranteed floor, attribution | **2026-10-01** |
| 5b | `runs/portal_shots/81_option_picked.png` | selecting a ranked option redraws the water | 2026-10-01 |
| 6 | `runs/portal_shots/60_water_solid.png` + `62_water_flat.png` | the flood as a volume against the flood as paint | **2026-10-01** |
| 7 | `runs/portal_shots/50_budget_trace.png` | the greedy allocation, step by step | **2026-10-01** |
| 8 | `runs/portal_shots/53_margin_f_low.png` + `53_margin_f_high.png` | `f` as a control, and the crossing moving with it | **2026-10-01** |
| 9 | `runs/portal_shots/54_limits.png` | what the tool cannot show, written into the interface | **2026-10-01** |
| 10 | `runs/portal_shots/96_trip_river.png` | a trip across a flood cutting 157 km of 1,309; route clear, **no deadline**, discharge stated not estimated | **2026-10-01** |
| 11 | `runs/portal_shots/52_refusal.png` | the margin refusing a level it cannot support | 2026-10-01 |
| 12 | `runs/portal_shots/44_roads.png` | live OpenStreetMap road network | **2026-10-01** |
| 13 | `runs/portal_shots/40_forecast.png` | live forecast panel (Open-Meteo) | **2026-10-01** |
| 14 | `runs/portal_shots/91_route_district.png` | the route card: no way out at car depth, 213 km cut | 2026-10-01 |

**The capture column is not bookkeeping.** Figures 5, 6, 7, 8, 9, 12 and 13
were re-shot on 2026-10-01 because the versions cited in earlier drafts predated
the per-option water, the route card and the methodology pages — and figure 5 is
the one this paper calls its most important. A paper citing a screenshot of a
superseded interface is making a claim about a tool that no longer exists.
`figA5_portal_appraisal` and `figA6_portal_anywhere` are **not** used here for
the same reason.

`95_trip.png` is retained but unused: it shows a box with no detected
through-channel, so that run fell back to rainfall and flooded nothing. Figure
10 replaces it from a run whose discharge is stated. No figure in this paper is
now a screenshot of an interface or a scenario other than the one its caption
describes.

## Data and reproducibility

WildfireSpreadTS (Zenodo 8006177); Next Day Wildfire Spread; Copernicus DEM
GLO-30; Sentinel-1 RTC and ESA WorldCover via Microsoft Planetary Computer;
Open-Meteo; OpenStreetMap via Overpass; FloodCastBench as the reference
hydrodynamic solution. Every result is produced by a script in `eval/` with
seeds, limits and δ as command-line arguments. Cluster scripts in
`scripts/tacc/`. The portal's own checks are in `portal/scripts/verify-*.mjs`,
including a parity test that holds the browser's router against the one the
paper's numbers come from: **14 of 14 quantities identical**.
