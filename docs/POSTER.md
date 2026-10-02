# POSTER — Digital Twins That Decide, Not Just Forecast

*A0 portrait, four columns. ~15% method, the rest the two twins and the portal.
Numbers here must match `WORKSHOP_PAPER.md`; the proofs and the full derivation
are `THESIS_AND_RESULTS.md` Part 10.*

---

## TITLE BAR (full width, top)

# Digital Twins That Decide, Not Just Forecast
### Budget-constrained intervention planning for floods and wildfires, with a stated error bound

*Authors · affiliation · QR to the live portal and the repo*

> **One-line version, set large enough to read from three metres:**
> A twin that forecasts a flood is useful. A twin that tells you **which levee
> to build, for A$20M, and how wrong it might be** is a different instrument.

---

# COLUMN 1 — The gap, and the one idea that closes it

## A. The operational question a forecast cannot answer

> *"We have A$20M. There are six places we could build. Which, how high, and
> how wrong might you be?"*

A forecast says what happens if nothing changes. The question is about what
changes, under a budget, with a consequence attached to being wrong.

**And the failure mode is quiet.** Plan inside a learned model and the planner
inherits that model's errors invisibly:

> ### The dual converges. Its own diagnostics say the constraint is satisfied.
> ### The real system violates it.

Measured: **7.3% of evaluations** over the true limit while every internal
number looks correct. Found only by periodically probing the real environment —
which a self-consistent training loop never does.

## B. What we mean by "twin"

Not a dashboard over a simulation. Four things, closed into a loop:

| **Perceive** | synchronise to observation, not just initialise from it |
|---|---|
| **Simulate** | fast enough to search over, honest about where it is wrong |
| **Plan** | optimise under a real constraint, not a soft penalty |
| **Explain** | say what it decided, in words that can be checked |

**FIGURE 1** — `docs/figures/paper/fig8_twin_loop.png`

## C. The technical idea (the only equation on this poster)

Plan in a fast surrogate, then **bound the error against reality**. Collect
matched pairs of *(what the surrogate predicted, what actually happened)* from
an occasional check against the real system, and take a conformal quantile of
the difference:

> ### P[ real cost > limit ] ≤ δ
> **with δ chosen and stated — not tuned.**

**The insight that makes it work.** The obvious approach is to inflate your
estimate by *how wrong the model is*. **We tried it and it made things worse**
(violations 14.5% → 18.2%). That instrument is right for a *pointwise* guarantee
and wrong for a *budget*: what pushes you over a budget is not average model
error, it's the **spread of real outcomes**. Bound that instead.

**And it is forced, not preferred.** No margin computed from model error alone
can hold a stated level however wide the outcome spread gets — a three-line
separation result. For Gaussian error it closes to

> ### R(ρ) = 1 − Φ(z_δ / ρ)   ·   ρ = (outcome spread) / (model error)

**one dimensionless number predicts the failure** — matching measurement to
within **0.003** across two orders of magnitude in ρ — and predicts that asking
for a *stricter* guarantee from the wrong distribution makes it worse
(1.2× error at δ = 0.30 → 3.7× at δ = 0.02).

**FIGURE 2** — `docs/figures/paper/fig0_calibration.png`
*Caption: measured violation against the closed form, with the δ-sweep inset.*

🔑 **One consequence worth the poster space:** when there isn't enough
calibration data, the method **refuses to return a number**. A
normal-approximation interval always returns something. See Column 4 for the
time that saved us.

---

# COLUMN 2 — Does it work, and against what

## The headline

**Reaction–diffusion hazard · 20 seeds · δ = 0.1**

| arm | violating | objective |
|---|---|---|
| no margin | 14.1% | — |
| inflate by model error | 18.2% | — |
| ### ours | ### 5.0% | ### no loss |

**Paired −9.1 points · 95% CI [−11.8, −5.5] · p = 0.0002**

The objective is statistically unchanged. **The margin removes a bias; it does
not buy safety with performance.**

**FIGURE 3** — `docs/figures/paper/fig2_rdf_violations.png`

## Three published methods, and a prediction we registered first

All three reimplemented **inside our planner**, so the only difference is how
model error is corrected for.

| method | violating | beats no margin? |
|---|---|---|
| no margin | 14.1% | — |
| conformal ensemble disagreement | 11.99% | **no** |
| SMBPO | 12.20% | **no** |
| SafeDreamer | 13.17% | **no** |
| ### ours | ### 5.00% | — |

> **They land within 1.2 points of each other and none beats doing nothing.**
> That is the theorem's prediction, not three unlucky baselines — all three are
> functions of model wrongness, which the separation result excludes. **We
> pre-registered 12% for SafeDreamer before running it.** It returned 13.17%.

**Two of these we got wrong first, and both errors favoured us:** SMBPO's
truncated cost rescaled linearly on a non-linear quantity (a −4.7% bias, 4× the
pessimism term, making the arm net *optimistic*), and SafeDreamer's cost critic
shadowing the trainer's own, scoring the wrong head.

## How far it generalises — stated exactly

| family | result |
|---|---|
| reaction–diffusion | **14.1% → 5.0%**, level attained |
| diffusion–advection | 0% → 0% — nobody violates: a **regression check**, not a validation |
| shallow water | 33.2% → 16.8%; on the ¾ of seeds where a plan exists, **12.1%, CI [7.1, 17.1]** — contains δ without confirming it |

> **The mechanism replicates on three families. The stated rate is attained on
> one, trivially satisfied on a second, and not rejected on the third.** We do
> not say "validated on three families", because it isn't.

## The assumption, tested where it breaks

Conformal needs exchangeable calibration scores; ours come from a learning run.
**On rdf exchangeability is violated** — lag-1 serial dependence at p ≈ 0 on
every seed, with drift and distribution-shift tests both non-rejections.
Dependent, not drifting.

> **And coverage holds anyway: 0.0854 against δ = 0.1.** Dependence of this kind
> inflates the quantile's variance without biasing it. Adaptive conformal
> inference is **not needed here** — demonstrated, not assumed.

---

# COLUMN 3 — Two twins, one of them surprising

## Wildfire · 607 observed fires

Learned spread model + ensemble Kalman update + decision-time planning of crews
under a daily budget.

> ### Same burned-area reduction as the operational heuristic, with HALF the crews
> **+12 points at a 3-day horizon and a 3% daily crew budget**

**The operating point is part of the claim.** The gap runs **+0.6 at one day** —
where theory says it must vanish, and it does — to **+14.3 at five days**; and
**+6.8 at a 1% budget** to **+16.9 at 5%**.

**FIGURE 4** — `docs/figures/paper/fig4_wildfire.png`

**And a policy will not do instead.** Distilling our own planner into a compact
policy recovers **47%** of the advantage — a real gain we initially understated —
but **21 points survive it**. The planner draws 16.0 points of its effect through
the learned dynamics; the distilled policy, 1.0.

## Flood · plans levees anywhere on Earth

Copernicus DEM GLO-30 · local-inertial solver batched over candidate plans ·
forcing from river discharge, coastal surge or rainfall — whichever the site has.

> **Budget A$20M** → **site 3 (3.0 m) + site 5 (1.0 m)**, A$18.2M of A$20M
> → ## Estimated 52% · at least 39%, 90% of the time
> → *"Write the business case against the lower figure."*
> → **Warns:** site 1 (−9%), site 0 (−9%) — measured as **deepening** flooding

**FIGURE 5** — `runs/portal_shots/45_plan_result.png` ← **the single most
important figure on this poster**

> **Point here when someone asks what the maths is for.** The interface tells an
> engineer to write the business case against **39%**, not **52%**. That sentence
> is the whole contribution, in the place where it matters.

## Getting the physics right changed the answer completely

| forcing | best plan | verdict |
|---|---|---|
| rainfall | — | "no measure worth building" |
| river, as a volume | 8.5% | **physically void** (112 m deep) |
| **river, as a stage** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

A boundary sets a **stage**, not a volume. Also fixed: DEM nodata filled with
sea level put a **247 m trench** around every inland domain.

## Validated against satellite — including the uncomfortable number

- our solver vs the reference hydrodynamic model — **CSI 0.979**
- the reference model vs **Sentinel-1** — **CSI 0.535**

> **The second number is the one that belongs in an operational discussion.**
> 0.98 is a statement about our numerics. 0.53 is a statement about how well
> *anyone's* hydrodynamics matches a real flood at 30 m. Quoting only the first
> is how decision-support tools acquire unearned authority. **One validated event
> is also one event.**

**FIGURE 6** — `docs/figures/paper/figA1_sar_validation.png`

---

# COLUMN 4 — The night of the flood, and what we won't claim

## Levees are decided months ahead. On the night the question is a trip.

Type two places. The portal derives the box, solves the flood across it, pulls
the roads, and routes. **Lismore → Coraki · 20 km · ~4,120 road segments · 104 s
end to end.**

The flood is carried as a **schedule**, not a peak — depth at 13 times, and a
road is usable only if it stays crossable for the whole time you're on it.

| what the operator is told | answer |
|---|---|
| routed on the **peak depth map** | ### no route exists |
| routed on the **time-resolved flood**, now | **22 min**, 24.8 km, dry |
| same trip, last possible departure | 25.5 min, **a different road** |
| how long it stays possible | ## 3h30 |

> **This is the poster's argument in one row.** A peak-depth map is every flood
> twin's standard output — ours included — and it is the **union of states that
> never coexist**. Route on it and it refuses roads that are dry when the driver
> passes them: it says *nobody gets out of Lismore*. They do, for three and a
> half hours. Leave at the deadline and the road is different, because by then
> Coraki Road is under **1.0–2.9 m** where the direct line crosses.

## And it holds over an ensemble, not one storm

Eight forcing members (×0.40–×1.60 of the design discharge; peak network cut
spanning **55–190 km**), leave-one-out, on the 12 trips of 523 the members
disagree about — **96 decisions**. Success = **never on a road whose true depth
exceeds what the vehicle can cross while you're on it.**

| arm | got through |
|---|---|
| route on the **peak field** | ### 0.0%  (0 of 96) |
| leave now, mean forecast | 41.7% |
| time-resolved, mean forecast | 41.7% |
| + our conformal margin | 41.7% |
| **re-plan on what is observed** | **56.2%** |
| oracle (plans on the storm that happens) | 85.4% |

**FIGURE 7** — ⚠️ **NEEDS RE-CAPTURE.** `runs/portal_shots/95_trip.png` exists
but shows the wrong case: that box had no detected through-channel, so the driver
fell back to rainfall and nothing flooded. Shoot it from a river-driven run:

```
curl -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{
  "origin":      {"lat": -28.8135, "lon": 153.2773, "name": "Lismore"},
  "destination": {"lat": -28.9956, "lon": 153.2906, "name": "Coraki"},
  "driver": "river", "river_q_m3s": 3000, "rain_mm_h": 50,
  "storm_hours": 6, "run_hours": 12, "vehicle_depth_m": 0.25}'
```

## Two results that argue against us, printed at the same size

> **Our own margin does nothing here.** Exactly **0.000 m on all 96 decisions**.
> The route the central forecast picks is dry under every member, so the residual
> is identically zero: what differs is not *how deep the chosen road gets* but
> **which roads exist**. The decision is discrete, the margin continuous. A
> feasibility theorem cuts both ways — it says which distribution a margin must
> use, not that a margin exists for every decision.

> **Observation helps as a 2:1 bet, not a free lunch.** Filtering the ensemble on
> observed depths converts **21 of the forecast's 49 refusals into departures:
> 14 get through, 7 are caught.** Nobody who was arriving stops; nobody who was
> caught is saved. "+14.5 points" would hide the seven, and being trapped on a
> flooding road is not the same outcome as having stayed home.

**And the routing is exact dynamic programming, not a learned policy.** We will
not dress Dijkstra as planning. Three PSPE stages do the work here — Simulate,
Plan, Explain. **Perceive is not in the single-trip loop** and we do not claim it.

## The bug that justifies the whole apparatus

Our own portal displayed a confidence it had no data to support:
`1.645 · RMSE · √(1 + 1/n)` — a normal-approximation interval, described in its
own code as "the product-facing form of the framework's safety margin".

Two errors compounded: the fit used **six** calibration runs where 90% coverage
needs at least **nine**, and a Gaussian puts the 90% point at 1.64×RMSE where the
honest order statistic is roughly twice that. **The displayed band was about half
the width it should have been.**

> A normal-approximation interval always returns something. **The conformal
> quantile refuses.** That is the reason to prefer it.

**FIGURE 8** — `runs/portal_shots/52_refusal.png`

## And the explanation is checked before it ships

The portal's written brief is **parseable**, so it can be scored against the
route it describes and **withheld if it disagrees** — the margin's refusal
discipline applied to prose. Its first live run caught the guard's own blind
spot: a brief reading *"every way through is under more water than 25 cm … 0 km
of 212 km is impassable"* scored 1.000, because the check was only reading the
departure call. Fixed; that sentence now scores 0.750 and is withheld.

## What we are NOT claiming

*(set plainly, same size as the results — not as small print)*

- ❌ **No fire in our archive had a firebreak cut on our instruction.** The
  intervention arm is **unvalidated in reality** and no retrospective dataset
  can fix that.
- ❌ **One** validated flood event, at CSI 0.53 against satellite.
- ❌ The guarantee is **marginal, not per-step**.
- ❌ The evacuation **deadline is one realisation**; the ensemble spans *how big*
  the flood is and nothing else — not timing, not spatial pattern, not solver
  error.
- ⚠️ **Exchangeability fails on one of our two testbeds** (lag-1, p ≈ 0) — and
  the margin attains δ on both anyway.
- ⚠️ Model-adaptation benefit is **small** (d = 0.20 at day +1).

During development we **retracted three of our own results** — a distillation
number that was an artefact of a collapsed loss, a baseline comparison
invalidated by an extrapolation bias 4× the effect it tested, and a statistical
claim whose interval spanned its own effect size. **All three initially pointed
the way that favoured us.** We now treat a claim-supporting result as a trigger
for an extra check.

**39 measurement defects catalogued, 45 rules derived.** The one to remember
degrades data *continuously*: a downsample that lost up to **59%** of a fire's
detections, smoothly and only for some fires, leaving every downstream number
plausible. Nothing short of a conservation check finds it.

## Technical detail, for those who want it

*(smallest type on the poster — deliberately)*

Margin: `s_i = c_i − ĝ`, `ĝ` the quantity the dual controls; split-conformal
quantile at rank `ceil((n+1)(1−δ))`, refusing below `n ≥ 1/δ − 1`.
Separation (proof and all corollaries: `THESIS_AND_RESULTS.md` Part 10):
`R` strictly increasing in σ, → ½ for symmetric error, `> δ` for
`σ > (q − b)/z_δ`. Routing: label-setting on arrival time with waiting
permitted — required for exactness, because edge availability is non-monotone
once the flood recedes. Solver: local-inertial shallow water, `dt` by CFL,
batched over candidate beds. Portal router parity-checked against the Python
one: **14 of 14 quantities identical**.

## Data

WildfireSpreadTS (Zenodo 8006177) · Next Day Wildfire Spread · Copernicus DEM
GLO-30 · Sentinel-1 RTC + ESA WorldCover (Planetary Computer) · Open-Meteo ·
OpenStreetMap/Overpass · FloodCastBench (reference hydrodynamics).
**Live portal + full results: QR in the title bar.**

---

## PRODUCTION NOTES (not printed)

- **Column 4 is the one to read if a visitor has 30 seconds.** The 0-of-96 row
  and the two results-against-us are the poster's credibility.
- Set the three big numbers at the same weight: **5.0%**, **half the crews**,
  **0 of 96**. Do not let the method number dominate the applications.
- Figures 1–6 and 8 exist. **Figure 7 must be re-captured** (see the command in
  Column 4) — the current file shows the rainfall fallback, not the flood.
- Every number here appears in `WORKSHOP_PAPER.md`; the separation result, its
  proof, and the provenance of each number are `THESIS_AND_RESULTS.md` Part 10
  (§10.14 is the number-to-script table). If a number changes, change it in all
  three.
- The "NOT claiming" block is not padding. It is why a visitor believes the
  5.0%.
