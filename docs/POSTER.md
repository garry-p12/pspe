# POSTER — Digital Twins That Decide, Not Just Forecast

**Layout:** A0 portrait, 4 columns. Reading order is left→right, and each column is
self-contained so a passer-by can start anywhere. Technical content is confined to
Column 1 §C and the small print in Column 4 — roughly 15% of the surface. Everything
else is the two twins and the live portal.

Colour discipline: one accent for *observed* data, one for *modelled*, one for
*recommended intervention*. Use the same three throughout — a viewer should learn
the legend once in Column 1 and never re-read it.

---

## TITLE BAR (full width, top)

# Digital Twins That Decide, Not Just Forecast
### Budget-constrained intervention planning for floods and wildfires, with a stated error bound

**Authors · Affiliation · QR code to the live portal (right-aligned, 40mm)**

> **One-line takeaway, set large:**
> *We plan interventions inside an imperfect simulator and tell you how wrong the
> recommendation might be — validated on 607 observed wildfires and a satellite-checked
> flood.*

---

# COLUMN 1 — The gap, and the one idea that closes it

## A. The operational question a forecast cannot answer

> **"We have A$20M. There are four places we could raise a levee.
> Which combination, and how confident should we be before we pour concrete?"**

A forecast tells you the flood is coming. It does not tell you what to build.
Answering the question above needs three things forecasts don't provide:

| | need | why it's hard |
|---|---|---|
| 1 | **Search over interventions** | thousands of combinations, each a hydrodynamic run |
| 2 | **A budget that binds** | a plan that overspends isn't a plan |
| 3 | **An error bound on the advice** | you searched in a *surrogate*; it's wrong; by how much? |

**Point 3 is where decision-support tools usually stop — and it's the one that
decides whether an engineer can act.**

## B. What we mean by "twin"

A one-way forecast is not a twin. Ours stay synchronised with observations and are
used to **evaluate actions before taking them**.

```
   OBSERVE ──▶ SIMULATE ──▶ PLAN ──▶ EXPLAIN
  satellite    physics /    budget-   what each
  detections   learned      limited   measure
      ▲        model        search    contributed
      └──────── corrected daily ◀───────┘
```

**FIGURE 1** — `docs/figures/paper/fig8_twin_loop.png`
*Caption: the loop, closed. Observations correct the model each day; the corrected
model is what the planner searches in.*

## C. The technical idea (the only equation on this poster)

Plan in a fast surrogate, then **bound the error against reality**. Collect matched
pairs of *(what the surrogate predicted, what actually happened)* from an occasional
check against the real system, and take a conformal quantile of the difference:

> ### P[ real cost > limit ] ≤ δ
> **with δ chosen and stated — not tuned.**

**The insight that makes it work.** The obvious approach is to inflate your estimate
by *how wrong the model is*. **We tried it and it made things worse** (violations
14.5% → 18.2%). Under a budget-style constraint, what pushes you over the limit is
not average model error — it's the **spread of real outcomes**. Bound that instead.

🔑 **One consequence worth the poster space:** when there isn't enough calibration
data, the method **refuses to return a number**. A normal-approximation interval
always returns something. See Column 4 for the time that saved us.

---

# COLUMN 2 — Wildfire twin · 607 observed fires

**Data:** WildfireSpreadTS — 607 fires, 13,607 daily images. **Year-wise**
cross-validation (train on past seasons, test on a new one), because fire regimes
shift between years.

## The headline

> # Same burned-area reduction as the operational heuristic, with **half the crews**

| crew budget / day | operational heuristic | **ours** | ratio |
|---|---|---|---|
| **1%** | 7.5% | **14.3%** | **1.91×** |
| 2% | 12.8% | 26.2% | 2.05× |
| 3% | 24.1% | 36.5% | 1.52× |
| 5% | 36.1% | 53.0% | 1.47× |
| 8% | 55.0% | 69.4% | 1.26× |

**The advantage is largest where crews are scarcest** — which is the regime agencies
actually operate in. Ours at 1% ≈ the heuristic at 2%.

**FIGURE 2** — `docs/figures/paper/fig4_wildfire.png`
*Caption: left, the budget curve — the gap widens as crews get scarcer. Right, the
horizon test below.*

## Why it works — tested where theory says it must fail

Our claim: planning over *several days* beats optimising for *today*, because
today's best move ignores where the fire will be tomorrow.

**That prediction is falsifiable.** At a one-day horizon the advantage must
**vanish** — the one-day problem is separable, and treating the highest-risk cells
is already its exact optimum.

| horizon | gap vs heuristic | 95% CI | verdict |
|---|---|---|---|
| **1 day** | **+0.6** | **[−0.2, +1.7]** | **absent, as required** |
| 2 days | +8.5 | [+7.0, +10.4] | present |
| 3 days | +12.1 | [+10.0, +14.8] | present |
| 5 days | **+14.3** | [+10.9, +18.2] | present, 5/5 seeds |

> *A result that is absent exactly where theory requires it to be, and present
> exactly where it predicts, is stronger evidence than any single large number.*

Note the heuristic **degrades** as the horizon grows (25.1 → 18.0): it keeps
optimising for today.

## Assimilation, checked against the classical method

We didn't just assert our simple correction was good enough. We ran a **32-member
Ensemble Kalman Filter** with the observation-error term **swept, not fitted**:

| observation error | implied gain | day-+1 skill |
|---|---|---|
| 0.05 | 0.98 | 0.33 |
| 0.15 | 0.84 | 0.13 |
| 0.30 | 0.56 | 0.03 |
| **ours** | **1.00** | **0.38** |

**No filter setting beats ours — and now we know why.** On this system satellite
detections are far more reliable than the forecast, which drives the optimal gain
toward 1. Our method isn't a crude stand-in for a filter; it's approximately the
filter's own optimum.

**FIGURE 3** — `docs/figures/paper/figA3_wildfire_perception.png`
*Caption: detections in, corrected fire state out. Median-performing patch, not the
best one.*

---

# COLUMN 3 — Flood twin · plans levees anywhere on Earth

**Terrain:** Copernicus DEM GLO-30. **Physics:** LISFLOOD-FP-style local-inertial
solver, batched over candidate plans. **Forcing:** river discharge, coastal surge, or
rainfall — whichever the site actually has.

## One click, end to end

> **Budget A$20M** · 12 of 36 options fit
> ↓
> **Recommend:** levee at site 3 (3.0 m) + site 5 (1.0 m) — **A$18.2M of A$20M**
> ↓
> # Estimated 52%
> # At least 39%, 90% of the time
> *"Margin ±13 points, from 30 held-out solver runs.*
> ***Write the business case against the lower figure."***
> ↓
> **Attribution:** site 3 **+43%** (47% alone) · site 5 **+6%** (9% alone) —
> in each case 4 points of what it would do alone is already covered by the rest
> of the plan
> ↓
> **Warns:** *site 1 (−9%), site 0 (−9%) — "the model measures these as deepening
> flooding at the settlement rather than reducing it"*
> ↓
> **Caveat, on screen:** *"the plan was chosen inside a surrogate. Adopting it
> should be conditional on a full hydrodynamic run of this exact height vector."*

**FIGURE 4** — `runs/portal_shots/45_plan_result.png` ← **the single most
important figure on this poster**
*Caption: the recommendation, its guaranteed floor, the margin it came from, and
what each measure contributed.*

> **Point at this when someone asks what the maths is for.** The interface tells
> an engineer to write the business case against **39%**, not **52%**. That
> sentence is the whole contribution, in the place where it matters.

## Getting the physics right changed the answer completely

Two corrections we made during development — both worth showing, because both would
have produced confident nonsense.

| forcing | best site | best plan | verdict |
|---|---|---|---|
| rainfall only | **0.4%** | — | *"no measure worth building"* |
| river, as a volume | 7.6% | 8.5% | **physically void — 112 m deep** |
| **river, as a stage** | **16.2%** | **21.3%** (floor 20.8%) | 17 held-out combinations |

**Rainfall alone recommends nothing** — correctly! A levee does nothing against water
falling behind it. *Flood interventions are only meaningful against a directional
driver.*

**A boundary sets a state, not a volume.** Pouring in 6,036 m³/s as a source term
made water arrive faster than cells could spread it → **112 m deep**. Holding a
*stage* via Manning's normal depth, no free parameters:

    h = (Q·n / (W·√S))^(3/5)  →  10.2 m stage, 11.7 m peak

**FIGURE 5** — `docs/figures/paper/fig10_flood_precondition.png`
*Caption: the same four sites under rainfall and under a river stage. Only one of
these is a planning problem.*

## Validated against satellite — including the uncomfortable number

Richmond NSW 2022, Sentinel-1 SAR, permanent water held separately so rivers aren't
scored as model error:

| comparison | CSI |
|---|---|
| our fast solver **vs reference model** | **0.979** |
| reference model **vs satellite** | **0.535** |

**We show both on purpose.** The 0.98 is a statement about our numerics. The **0.53
is the one that belongs in an operational conversation** — it's how well anyone's
hydrodynamics matches a real flood at 30 m resolution. Quoting only the first is how
tools acquire unearned authority.

**FIGURE 6** — `docs/figures/paper/figA1_sar_validation.png`

---

# COLUMN 4 — The portal, honesty, and what's next

## Live, and it works anywhere

Terrain, weather, river discharge and **real OpenStreetMap road networks** fetched on
demand for **any location on Earth**. Not a diorama of one pre-built city.

| stage | endpoint |
|---|---|
| Perceive | `POST /observe` — Sentinel-1, same-orbit differencing |
| Simulate | local-inertial solver on Copernicus DEM |
| Plan | `POST /api/plan` — budget-constrained, **milliseconds** |
| Explain | leave-one-out attribution per measure |

**FIGURE 7** — `runs/portal_shots/44_roads.png`
*Caption: live OpenStreetMap road network — the planner can reason about which
access routes an intervention protects.*

**FIGURE 8** — `runs/portal_shots/42_plan_options.png`
*Caption: the 2022 event, the budget slider, and the ranked options — 20.6 km of
288 km of road cut, 185 km² flooded if nothing is built.*

**FIGURE 9** — `runs/portal_shots/40_forecast.png`
*Caption: the live forecast panel, reading Open-Meteo at capture time.*

> ### 📱 QR → try it yourself
> *(large QR, ~60mm, with "point your phone here" in plain type)*

## The bug that justifies the whole apparatus

**Our own portal displayed a confidence it had no data to support.** It showed a
normal-approximation interval, described in its own source as "the product-facing
form of the framework's safety margin". It wasn't.

- Tuned on **6** calibration runs; 90% coverage needs at least **9**
- A Gaussian puts the 90% point at 1.64×RMSE; the real residuals are heavier-tailed

Rebuilt honestly at 30 points:

> ## ±20.2 actual vs ±10.5 displayed
> ### The tool was overstating its confidence **2×**

The conformal method **would have refused to return a number** at 6 points. The
normal approximation always returns something.

**For decision-support software, a method that declines to answer when it lacks data
is not a limitation. It is the feature.**

## What we are NOT claiming

*(set this block plainly, same size as the results — not as small print)*

- ❌ **No fire in our archive had a firebreak cut on our instruction.** Planning is
  validated against held-out observations *inside* the model. **The intervention arm
  is unvalidated in reality**, and no retrospective dataset can fix that.
- ❌ **One** validated flood event, at CSI 0.53 against satellite.
- ❌ The guarantee is **marginal, not per-step**: it bounds total trajectory cost, not
  every timestep.
- ⚠️ The model-adaptation benefit is **small** (d = 0.20 at day +1). We report the
  effect size, not just the p-value.

During development we **retracted three of our own results** — a distillation number
that was an artefact of a collapsed loss, a baseline comparison invalidated by an
extrapolation bias 4× larger than the effect it was testing, and a statistical claim
whose interval spanned its own effect size. **All three initially pointed the way
that favoured us.** We now treat a claim-supporting result as a trigger for an extra
check.

## Technical detail, for those who want it

*(smallest type on the poster — deliberately)*

Reaction–diffusion benchmark, 20 seeds, δ = 0.1:

| arm | violating % | 95% CI |
|---|---|---|
| no margin | 14.1 | [9.1, 17.3] |
| margin alone | 14.1 | [9.1, 17.3] |
| real-system probe alone | 10.9 | [6.4, 15.5] |
| **probe + conformal** | **5.0** | **[0.9, 8.2]** |

Paired −9.1 points, 95% CI [−11.8, −5.5], p = 0.0002, **no cost in objective
value**.

Against two published model-based safe-RL methods at matched sample budget:

| method | corrects using | violating % |
|---|---|---|
| no margin | — | 14.1 |
| CAP | adaptive mean + kσ of ensemble spread | 12.0 |
| SMBPO | unadapted max over ensemble | 12.2 |
| **ours** | **conformal quantile of realised cost** | **5.0** |

Both lose by ~7 points (p = 0.0003, 0.0014). **Neither separates from using no
margin at all.** And CAP vs SMBPO: p = 0.83 — indistinguishable, despite
summarising disagreement about as differently as possible. *How you summarise
model disagreement doesn't matter; which distribution you compute it on does.*
We predicted SMBPO would do worse than CAP. It didn't — and dropping that
prediction left a cleaner claim.

**Margin alone does nothing** (0/20 seeds) — a conformal quantile with no probe data
is identically zero. Safety is a *loop*, not a component.

**Decision-time search vs a trained policy:** planner 39.0%, distilled from the
planner 18.3%, trained from scratch 10.4%. Distillation recovers 47% — **21 points
survive it.** The planner draws 16.0 points through learned fire dynamics, the
distilled policy 1.0.

## Data

WildfireSpreadTS (Zenodo 8006177) · Next Day Wildfire Spread · Copernicus DEM GLO-30
· Sentinel-1 RTC + ESA WorldCover (Planetary Computer) · Open-Meteo · OpenStreetMap

---

## PRODUCTION NOTES (not printed)

**Surface budget.** Technical content = Column 1 §C + Column 4 small print ≈ **15%**.
Columns 2 and 3 (the two twins) + portal ≈ **70%**. Framing/caveats ≈ 15%.

**If you must cut**, in order: (1) the EnKF table in Column 2, (2) Figure 9,
(3) the horizon table's 3-day row. **Never cut** the "What we are NOT claiming" block
or the 2× overstatement box — at a climate venue those are what earn trust, and
they're what will start conversations with practitioners.

**The three numbers a viewer should leave with:**
1. **1.91×** — half the crews, same outcome
2. **0.53** — how well hydrodynamics actually matches a satellite
3. **2×** — how badly our own tool overstated its confidence before we fixed it

**Booth staffing.** Have the portal live on a laptop with a location the visitor
picks. The "works anywhere" claim lands when someone names their own catchment and
it builds a twin in front of them. Have a fallback screenshot set for no-WiFi:
`runs/portal_shots/`.

**Figure files.** From `docs/figures/paper/` (PDFs of the same names exist and are
preferred for print): `fig8_twin_loop`, `fig4_wildfire`, `figA3_wildfire_perception`,
`fig10_flood_precondition`, `figA1_sar_validation`. From `runs/portal_shots/`, all
captured **2026-09-28** against the current interface at 3200×2400:
`45_plan_result`, `42_plan_options`, `44_roads`, `40_forecast`.

**Do NOT use** `figA5_portal_appraisal`, `figA6_portal_anywhere`, `09_observe`, or
anything numbered below 15 in `runs/portal_shots/`: they predate the interface
rebuild and show a tool that no longer exists. `figA5`/`figA6` are composites whose
own timestamps look recent but which embed screenshots from 2026-09-27 15:16.
Regenerate with `portal/scripts/poster-shots.mjs` and `plan-result.mjs`.
