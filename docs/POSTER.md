# POSTER — Digital Twins That Decide, Not Just Forecast

*A0 portrait, four columns. Companion to the **systems** paper
(`WORKSHOP_PAPER.md`): ~15% method, the rest the two twins, the routing result
and the portal. The calibration method is a separate paper (`ICML_PAPER.md`);
derivations and the defect catalogue are `THESIS_AND_RESULTS.md`.*

---

## TITLE BAR (full width, top)

# Digital Twins That Decide, Not Just Forecast
### Perceive → Simulate → Plan → Calibrate → Explain, on 607 fires and a flooded road network

*Authors · affiliation · QR to the live portal and the repo*

> **One line, readable from three metres:**
> A twin that forecasts a flood is useful. A twin that tells you **which levee to
> build for A$20M, how wrong it might be, and whether anyone can still drive out**
> is a different instrument.

---

# COLUMN 1 — The gap, the loop, and the one thing we calibrate

## A. The question a forecast cannot answer

> *"We have A$20M. There are six places we could build. Which, how high, and how
> wrong might you be?"*

A forecast says what happens if nothing changes. This is about what changes,
under a budget, with a consequence for being wrong.

## B. Five stages, and most twins have two

| **Perceive** | synchronise to observation, not just initialise from it |
|---|---|
| **Simulate** | fast enough to search over; honest about where it is wrong |
| **Plan** | optimise under a real constraint, not a soft penalty |
| **Calibrate** | state how wrong the recommendation might be |
| **Explain** | say what was decided, in words that can be checked |

**FIGURE 1** — `docs/figures/paper/fig8_twin_loop.png`

## C. The calibration stage, briefly

*(the full method is the companion paper — this poster is about what it buys)*

Plan in a fast surrogate, then **bound the error against reality**: collect
matched pairs of *(predicted, actually happened)* from an occasional check
against the real system and take a conformal quantile.

> ### P[ real cost > limit ] ≤ δ
> **δ chosen and stated — not tuned.**

**The non-obvious part.** Inflating your estimate by *how wrong the model is*
**made things worse** (violations 14.5% → 18.2%). That instrument is right for a
*pointwise* guarantee and wrong for a *budget*: what pushes you over a budget is
not average model error, it is the **spread of real outcomes**. Bound that, and
violations fall **14.1% → 5.0%** at no cost in objective.

🔑 **The property this poster actually uses:** when there is not enough
calibration data, the method **refuses to return a number**. With 30 held-out
simulations the highest honest confidence is **96.8%** — ask for 99% and it
declines and says how many more runs it would need. A normal-approximation
interval always returns something. Column 4 has the time that saved us.

---

# COLUMN 2 — Wildfire twin · 607 observed fires

**Perceive:** VIIRS detections → ensemble Kalman update of the burned state.
**Simulate:** learned spread dynamics. **Plan:** crew allocation, several days
ahead, under a daily budget.

## The systems headline

| daily crew budget | operational heuristic | planner |
|---|---|---|
| ### 1% | 7.5 ± 1.4 | ### 14.3 ± 0.5 |
| ### 2% | ### 12.8 ± 2.0 | 26.2 ± 0.7 |
| 3% | 24.1 ± 3.8 | 36.5 ± 1.0 |
| 5% | 36.1 ± 5.0 | 53.0 ± 1.1 |
| 8% | 55.0 ± 5.5 | 69.4 ± 0.6 |

> ### The planner on a 1% crew budget (14.3%) beats the heuristic on 2% (12.8%).
> Half the crews for a better outcome — and the advantage is **largest where
> crews are scarcest**, which is the regime that matters.

**FIGURE 2** — `docs/figures/paper/fig4_wildfire.png`

## Tested where it must fail

At a one-day horizon a greedy allocation is optimal, so a planner should buy
nothing:

| horizon | gap over heuristic | |
|---|---|---|
| 1 day | **+0.6** | **not significant** |
| 2 days | +8.5 | t = 10.74 |
| 3 days | +12.1 | t = 11.08 |
| 5 days | **+14.3** | t = 8.99 |

> **It vanishes exactly where theory requires**, then grows with the horizon over
> which sequential reasoning can act. A result that survives its own
> falsification test is worth more than one never exposed to it.

*An earlier version of this claim was measured on 8 fires at t = 3.0. At 607 the
benefit is a third the size and holds on 76%. We re-ran it rather than publish
it.*

## And a policy will not do instead

| decision-time planner | **39.0%** |
|---|---|
| distilled from the planner | 18.3% |
| trained from scratch | 10.4% |

Distillation recovers **47%** — a real gain we initially understated — but **21
points survive it**. The planner draws 16.0 points of its effect through the
learned dynamics; the distilled policy, 1.0.

## Two data defects worth your time

- ❌ **"No detection" is not "not observed."** Treating absence as absence would
  have deleted **37.4%** of days — exactly the days fires go out.
- ❌ **A downsample cropped the raster margin**, losing up to **59%** of a fire's
  detections — continuously, only for some fires, every downstream number
  plausible. **Only a conservation check finds that.**

---

# COLUMN 3 — Flood twin · levees anywhere on Earth

Copernicus DEM GLO-30 · local-inertial solver batched over candidate plans ·
forcing from river discharge, coastal surge or rainfall — whichever the site has.
Terrain fetched on demand, so it runs where no data was prepared.

## One click, end to end

> **A$20M** → site 3 (3.0 m) + site 5 (1.0 m), A$18.2M of A$20M
> → ## estimated 52% · at least 39%, nine times in ten
> → *"Write the business case against the lower figure."*
> → **warns:** site 1 (−9%), site 0 (−9%) — measured as **deepening** flooding

**FIGURE 3** — `runs/portal_shots/45_plan_result.png`

> **Point here when someone asks what the maths is for.** The interface tells an
> engineer to write the case against **39%**, not **52%**. That sentence is the
> contribution, in the place where it matters.

## Getting the physics right changed the answer completely

| forcing | best plan | verdict |
|---|---|---|
| rainfall | — | "no measure worth building" |
| river, as a volume | 8.5% | **physically void** (112 m deep) |
| **river, as a stage** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

A boundary sets a **stage**, not a volume. Also fixed: DEM nodata filled with sea
level put a **247 m trench** around every inland domain.

**FIGURE 4** — `docs/figures/paper/fig10_flood_precondition.png`

## Validated against satellite — including the uncomfortable number

- our solver vs the reference hydrodynamic model — **CSI 0.979**
- the reference model vs **Sentinel-1** — **CSI 0.535**

> **The second number is the one that belongs in an operational discussion.**
> 0.98 is about our numerics. 0.53 is about how well *anyone's* hydrodynamics
> matches a real flood at 30 m. Quoting only the first is how decision-support
> tools acquire unearned authority. **One validated event is also one event.**

**FIGURE 5** — `docs/figures/paper/figA1_sar_validation.png`

---

# COLUMN 4 — On the night: a peak-depth map cannot route an evacuation

Levees are decided months ahead. During the flood the question is a trip. Type
two places: the twin derives the box, solves the flood, pulls the roads, builds a
routable graph (**287.4 km against a published 288.0**, 98% of junctions in one
component) and routes. **20 km · ~4,120 segments · 104 s end to end.**

The flood is carried as a **schedule**, not a peak — depth at 13 times, and a
road is usable only if it stays crossable for the whole time you are on it.

## THE RESULT

Eight forcing members, leave-one-out, on the **12 of 523** trips the members
disagree about — **96 decisions**. Success = **never on a road whose true depth
exceeds what the vehicle can cross while you are on it.**

| routed on | got through |
|---|---|
| ### the peak-depth field | ### 0 / 96 |
| the time-resolved flood, now | **40 / 96** |
| + our conformal margin | 40 / 96 |
| re-planning on what is observed | **54 / 96** |
| oracle — plans on the storm that happens | **82 / 96** |

> # A peak-depth map is the union of states that never coexist in time.
> Route on it and it refuses roads that are dry when the driver passes them: on
> this network it reports that **nobody can leave**, on all 96 decisions, where
> an oracle gets **82** out. **The peak field is what essentially every flood
> twin publishes — including ours.** A decision taken against it is taken
> against a scenario that never happens.

On a single trip the same effect is a **3h30 deadline**: 22 minutes and dry
leaving now; 25.5 minutes on a **different road** at the last departure, because
by then Coraki Road is under **1.0–2.9 m** where the direct line crosses.

**FIGURE 6** — `runs/portal_shots/96_trip_river.png`
*Caption: discharge **stated** at 3,000 m³/s (applied at 7,500). 157 km of 1,309
impassable at peak; trip 22 min, meets no water, so **no deadline** — this route
is on ground the flood does not reach.*

## Two results that argue against us, printed at the same size

> **Our own calibration does nothing here.** Exactly **0.000 m on all 96
> decisions**. The route the forecast picks is dry under every member, so the
> residual is identically zero: what differs is not *how deep the chosen road
> gets* but **which roads exist**. Discrete decision, continuous margin. The
> method's feasibility statement cuts both ways.

> **Observation helps as a 2:1 bet.** Filtering the ensemble on observed depths
> converts **21 of the forecast's 49 refusals into departures: 14 get through, 7
> are caught.** Nobody arriving stops; nobody caught is saved. "+14 arrivals"
> would hide the seven, and being trapped on a flooding road is not the same
> outcome as having stayed home.

**And the routing is exact dynamic programming, not a learned policy.** We will
not dress Dijkstra as planning. It is optimal **only because waiting is
permitted** — availability is non-monotone once the flood recedes, and without
holding a test case differs by more than 2× (300 min against 127.5).

## The bug that justifies the calibration stage

Our own portal displayed a confidence it had no data to support:
`1.645 · RMSE · √(1 + 1/n)` — a normal-approximation interval, described in its
own code as "the product-facing form of the framework's safety margin".

**Six** calibration runs where 90% coverage needs **nine**, and a Gaussian 90%
point at 1.64×RMSE where the honest order statistic is roughly twice that. **The
band was about half the width it should have been.**

> A normal-approximation interval always returns something. **The conformal
> quantile refuses.**

**FIGURE 7** — `runs/portal_shots/52_refusal.png`

## And the explanation is checked before it ships

The written brief is **parseable**, so it is scored against the route it
describes and **withheld if it disagrees** — the refusal discipline applied to
prose. Its first live run caught the guard's own blind spot: *"every way through
is under more water than 25 cm … 0 km of 212 km is impassable"* scored 1.000,
because the check read only the departure call. Fixed; that sentence now scores
0.750 and is withheld.

## What the interface could not express

Chasing one figure found a gap worth more than the figure. The trip view could
not **state** a design discharge, so it inherited bankfull from a 30 m elevation
model — **234 m³/s** here, against a flood of record of **3,000–4,000**. Every
trip it could model was a **1.4%**-of-network event, described by a brief scoring
1.000 faithfulness because it correctly described a non-event. **Every number was
correct and the system was useless**, which is the kind of failure invisible from
inside the results.

## What we are NOT claiming

*(same size as the results — not small print)*

- ❌ **No fire had a firebreak cut on our instruction.** The intervention arm is
  **unvalidated in reality**; no retrospective dataset fixes that.
- ❌ **One** validated flood event, at CSI 0.53 against satellite.
- ❌ The guarantee is **marginal, not per-step**.
- ❌ The **deadline is one realisation**. The ensemble spans *how big* the flood
  is and nothing else — not timing, not pattern, not solver error. And it
  reproduces from **coordinates, not place names**: geocoding "Lismore NSW"
  lands 0.6 km away, enough to move the box off the channel.
- ⚠️ The calibration **contributes nothing to route choice**, and a trained
  policy over the same ensemble is an arm we have not run.
- ⚠️ Model-adaptation benefit is **small** (d = 0.20 at day +1).

We **retracted three of our own results** during development — a distillation
number from a collapsed loss, a baseline comparison invalidated by an
extrapolation bias 4× the effect it tested, and a statistical claim whose
interval spanned its own effect size. **All three initially favoured us.**

**54 defects catalogued, 64 rules derived.** And one we chose not to exploit: a
deadline was available for Figure 6 at 8,000 m³/s or 20,000, but the service
applies 2.5× the stated peak, so those show 20,000 and 50,000 on a river whose
record is 3,000–4,000. **Picking the larger flood for a better caption would have
been every defect in the catalogue, committed on purpose.**

## Data

WildfireSpreadTS (Zenodo 8006177) · Next Day Wildfire Spread · Copernicus DEM
GLO-30 · Sentinel-1 RTC + ESA WorldCover (Planetary Computer) · Open-Meteo ·
OpenStreetMap/Overpass · FloodCastBench (reference hydrodynamics).
**Live portal + full results: QR in the title bar.**

---

## PRODUCTION NOTES (not printed)

- **This poster is the systems paper's companion, not the method paper's.**
  Column 1C is deliberately 15% of the space. If a visitor wants the theorem,
  point them at `ICML_PAPER.md`; do not grow column 1 to accommodate them.
- **Column 4 is what to read in 30 seconds.** The 0/96 row and the two
  results-against-us are the poster's credibility.
- Set these at equal weight: **14.3% on 1% crews**, **0 / 96**, **39% not 52%**.
  Do not let the method number dominate the applications.
- **A cited screenshot is a claim about the tool's present state, so every portal
  figure carries a capture date.** An interface is rebuilt more often than a
  poster is printed; three figures in an earlier draft showed a superseded tool,
  including the one Column 3 calls most important.
- Figures re-shot 2026-10-01: 3 (plan result), 6 (trip), 7 (refusal), plus the
  water, budget-trace, margin-explorer and limits panels held in reserve if space
  allows. `figA5`/`figA6` are excluded — superseded interface. `95_trip.png` is
  the rainfall fallback and is **not** used.
- Every number here appears in `WORKSHOP_PAPER.md`; provenance for each is
  `THESIS_AND_RESULTS.md` §10.14. If a number changes, change it in all three.
- The "NOT claiming" block is not padding. It is why a visitor believes the
  **0 / 96**.
