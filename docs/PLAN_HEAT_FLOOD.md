# Plan: heat and flood, and what a digital-twin claim can rest on

Draft for review, 2026-09-25. **Nothing here has been run.**

## 0. The one finding that reorders this plan

I drafted this expecting heat to be the strong domain and flood the weak one.
Checking the datasets reversed it.

**FloodCastBench** (Nature Sci Data 2025, with FloodCast, arXiv 2403.12226)
predicts **continuous water depth** by solving the **shallow-water equations**
on a staggered finite-difference grid following LISFLOOD-FP (θ=0.7, α=0.7), over
14-day events (Pakistan 2022, UK 2015, Australia 2022, Mozambique 2019), with an
explicit fidelity ladder **across events**: Pakistan 2022 and Mozambique 2019 at
**480 m**, Australia 2022 at **30 m** and UK 2015 at **60 m**, all at a 300 s
timestep, in TIFF. (Correction to my first draft, which said 480 m vs 360 m
within one event — that is FloodCast's *downscaling transfer* experiment, not the
dataset's own fidelity split.) One 20.1 GB zip on Zenodo, CC-BY-4.0:
`https://zenodo.org/records/11431853`.

Three consequences, each of which matters more than the hazard label:

1. **Flood has a solver, so flood has counterfactuals.** Every result in the
   current paper carries §6.2: no observational record contains what would have
   happened under our intervention. NDWS cannot refute a firebreak. HeatCast
   cannot refute a tree. But an accepted shallow-water solver *can* be run with
   a levee in place. That is not observational validation, and the write-up must
   never call it that — but it is validation against an independent physics model
   rather than against our own perturbed surrogate, which is what the wildfire
   experiment was reduced to.
2. **The fidelity axis is already in the dataset.** Our ρ = σ/ε diagnostic needs
   a surrogate-error axis; we built one synthetically and had to add weaker
   levels by hand when the sweep's ρ never crossed 1. A 480 m event and a 30 m
   event give that axis from published design choices rather than from noise we
   inject ourselves.
3. **It is the same mathematical object we already handle.** Shallow water is in
   `pspe/simulate/solvers.py` and `pspe/envs/pde_env.py` today. The flood work is
   largely swapping synthetic initial and boundary conditions for a real DEM and
   a real rainfall forcing — not a new stack.

**HeatCast** (arXiv 2608.07640) is real and useful but different in kind:
**monthly** 30 m Landsat tiles, 124 U.S. cities, 2013–2025/06, with LST,
elevation, RGB, three indices, albedo, quality masks and Local Climate Zone
labels, fixed temporal splits and published baselines (Earthformer **7.74 K
RMSE**, CNN+LSTM 10.42 K). Note that my earlier assumption of daily cadence was
wrong: it is monthly, which rules out a daily assimilation loop but matches the
timescale a canopy intervention actually acts on.

So the division of labour is not "two hazards, same experiment twice":

| | **heat (HeatCast)** | **flood (FloodCastBench / GFF)** |
|---|---|---|
| field | LST, dense, continuous | water depth, dense, continuous |
| cadence | monthly, ~150 steps/city | 300 s, 14-day events |
| breadth | 124 cities, real observations | 4 events, solver + real forcing |
| counterfactual | **none** — same gap as fire | **solver** — interventions evaluable |
| answers | perception and forecasting | planning and the margin |

---

## 1. What already constrains this

Not written from a clean slate. Four things the project has established, each of
which would otherwise get re-discovered at cost:

* **§9.8** — "state sync beats open loop" is textbook data assimilation; ours is
  nudging with gain 1. The flood-twin literature confirms this bluntly: the
  Alzette proof-of-concept (arXiv 2505.08553) runs a **particle filter** over
  **LISFLOOD-FP** with Sentinel-1 and GloFAS, and reports improvement over
  open-loop as its result. Repeating sync-vs-open-loop on a new hazard adds
  nothing. A real DA baseline is not optional.
* **§9.10** — the "better reconstruction, worse decision" finding may be an
  artefact of a **sparse** field (fire at 1.3% prevalence makes "assume nothing
  burns" nearly free). LST and water depth are dense. This is the open question
  the project already flagged as highest value.
* **Proposition 2** — a margin is usable only when `b + z_δ·σ_agg < f·s`. The
  wildfire task failed it after four configurations. Screen before building.
* **Defect 12** — `swe` was retired because actuation moved the objective inside
  a **7% achievable band**, so methods could not separate. This is the sharpest
  risk here, because FloodCastBench *is* shallow water. The difference is real
  terrain and a real levee rather than a synthetic actuator, and levees do move
  inundation — but that is a hypothesis to measure in Phase 0, not to assume.

And the positioning this implies. The flood-twin DA work closes the loop and
stops. **Nobody puts constrained intervention planning with a calibrated safety
margin on top of that loop.** That is the same gap the current paper claims, now
on a hazard where it can be checked against a solver.

---

## 1a. Phase 0 work already done (2026-09-25)

**Vista is unavailable** — the MFA session expired, and I will not script TOTP.
Phase 0 is small enough to run locally; Phases 1 and 3 will need the allocation
back.

**A new solver, `pspe/simulate/flood.py`.** Reusing `ShallowWaterTransport` was
not an option, and inspecting it explains defect 12 rather than merely recording
it. That testbed is *linearised with no topography at all* —
`h_t = -H(u_x + v_y) - c_h h + s` — and its control adds an equal source to the
surface height everywhere. There is no geometry for an intervention to act on, so
of course methods ranked inside a 7% band.

The new module implements the scheme **FloodCastBench itself uses as ground
truth**: the LISFLOOD-FP local-inertial form, with Manning friction, the upwind
interface depth `h_f = max(h_i+z_i, h_j+z_j) - max(z_i, z_j)` that gives correct
wetting and drying, an adaptive CFL timestep, and free-outflow boundaries by
normal depth. Our reality model is therefore the accepted scheme, not an
approximation of it.

**Why a levee has leverage where the `swe` actuator had none.** Momentum acts on
the *free surface* `h + z`. Raising `z` does not add or remove water; it changes
where water can go, and `h_f` collapses to zero across a crest the flood has not
topped. That is a gating action on the flow, not a uniform source.

**Verification** (all three needed; the second caught a real bug):

| test | result |
|---|---|
| lake at rest over uneven bed | depth drift **0.000**, free-surface std 1.2e-7 m — topography enters correctly |
| closed-domain mass conservation | **+0.00001%** over 600 steps |
| dam break spreads downhill | peak 8.71 m, drains to 1.4 of 768 through an open edge |

The mass test initially failed: the dam break **gained 6.4% volume**, because
`clamp(min=0)` on a cell an explicit flux had over-drained *creates* water. Fixed
with a per-cell flux limiter that scales every outflow by the fraction of depth
actually available. Worth recording as a measurement defect in its own right —
an unlimited explicit scheme silently manufactures the quantity under constraint.

**Scenario design, and a discarded first attempt.** My first terrain put the
settlement in an isolated depression. It flooded from **rain falling inside the
depression**, which no levee can prevent, while channel discharge never reached
it (town depth 0.000 m at rain = 0, 0.303 m at rain > 0 regardless of inflow).
That would have been as artificial as `swe`. The replacement is a floodplain
separated from its channel by a berm with three low gaps, which is how real levee
systems fail and are repaired. The town now floods *from the channel*:

| channel inflow | exposure-weighted damage | town max depth |
|---|---|---|
| 1e-3 | 0.0526 | 0.318 m |
| 2e-3 | 0.0930 | 0.461 m |
| 4e-3 | 0.1665 | 0.680 m |
| 8e-3 | 0.3095 | 1.166 m |

**Gate 0 result: PASSED, at 99.8%.** Full record in §5.6 of the thesis doc. Three
attempts were needed; the first two failed on faults in my own scenario, not on
flood control:

| attempt | span | what was wrong |
|---|---|---|
| 1 | **1.63%** | forcing submerged the 2.2 m berm under 9–12 m of water (≈44,000 m³/s sustained) |
| 2 | — | hydrograph injected *across* the berm; settlement depression overlapped berm and channel |
| 3 | **99.77%** single site, 100% all six | geometry asserted separate, inflow inside the channel, forcing in the over-topping window |

Two conclusions worth more than the gate itself:

* **Defect 12 was about the actuator, not the physics.** `swe` and the flood
  testbed are the *same PDE family*. Changing the control from an additive source
  to a geometric one took the achievable band from 7% to 99.8%. The `swe`
  retirement was right about `swe` and wrong as a claim about shallow-water
  control.
* **Two of six available levee sites make flooding *worse*** (−5.09%, −2.06%).
  Both sit downstream of the settlement, where a levee impedes drainage and backs
  water up — the known levee backwater effect. A planner that treats actuators as
  uniformly beneficial actively harms what it was deployed to protect. That is the
  paper's thesis, in a domain where a solver can demonstrate it.

**Design consequence for Phase 3.** At a 6.0 m budget the task is trivial: 1.0 m
at the dominant gap already gives +99.74%, because that gap has 0.37 m of
freeboard. The informative regime is a **tight budget under hydrograph
uncertainty** — commit levee heights before the flood magnitude is known, and let
the conformal margin bound the shortfall. That is the real levee-design problem
(freeboard under flood-frequency uncertainty), and where a calibrated margin earns
its place.

---

## 2. Phases and gates

### Phase 0 — screening, before anything is built (3–4 days)

Three numbers decide whether the rest is worth running.

1. **Span.** Run the do-nothing / best-achievable gap for a realistic flood
   intervention (levee segment, retention allocation) on one FloodCastBench
   event. **If it lands inside ~7% the way `swe` did, flood planning is dead for
   the same measured reason** and we say so and stop. This is the single most
   informative day in the plan.
2. **Precondition.** `--calibrate-only` on both hazards: report `f`, `b`,
   `σ_agg`, `s` and the verdict on `b + z_δ·σ_agg < f·s`. The runner already
   takes the flag.
3. **ρ range.** Confirm the 480/360 fidelity pair straddles ρ = 1, or find the
   pair that does. Last time the sweep's ρ never crossed 1 and levels had to be
   added after the fact.

**Gate 0.** Span fails → drop flood planning, report it as a second worked
example of defect 12. Precondition fails → drop the margin claim on that hazard
and report it as a second worked example of Proposition 2. Phase 1 proceeds
regardless: it needs neither a margin nor a large span.

### Phase 1 — the dense-field test (1–2 weeks). *Highest value, lowest risk.*

Pre-registered, written before the run:

> On a sparse field, filling unobserved cells made the decision worse in 0 of 9
> configurations. On a **dense** field, does it invert?

Port `eval/run_ndws_partial.py` — which already carries the four beliefs
(`blind` / `persist` / `perceive` / `perceive-hard`), `--rates`, and the
plan-on-belief-score-on-truth protocol — from a binary mask to a continuous
field: beliefs become regression, AP becomes RMSE. Run on HeatCast under its
own quality masks (real cloud gaps, not synthetic), then repeat on flood depth.

**Prediction, recorded now:** imputation should *help* on a dense field, because
"assume nothing" has no meaning for a temperature or a depth — there is no cheap
prior to fall back on. If it helps, §9.10 becomes a statement about *when*
belief-filling pays, with sparsity as the boundary, tested on two dense fields
and one sparse one. If it does not invert, the original finding is far more
general than we thought, which is the more interesting outcome.

One complication to handle honestly: HeatCast reports that the **eight non-LST
channels alone forecast better than LST history** (7.72 K vs 8.15 K). If
covariates carry the signal, imputing LST may not matter much either way, and
the test needs the covariate-only arm included or it will mis-attribute.

Either outcome is publishable. That is the test for running an experiment.

### Phase 2 — twin loop against a real DA baseline (1–2 weeks)

Open loop / state sync / state+model, **plus a particle filter or EnKF analysis
step**, on the same sequences. On flood this is a direct comparison against the
Alzette design; on heat it is monthly assimilation across 124 cities, which is
broader than any DA study but at a cadence where assimilation matters less.

**Gate 2.** If our sync trails the filter, report that. "A learned surrogate with
crude nudging recovers X% of a particle filter's benefit at Y% of its cost" is a
statement a DA audience can use. Another 8× over free-running is not.

### Phase 3 — constrained planning against the solver (2 weeks, only if Gate 0 passed)

Flood only, because only flood has the solver. Plan interventions against the
learned surrogate; **evaluate by running the shallow-water solver with the
intervention applied**; hold a depth or population-exposure limit with the
conformal margin. Protocol as §5.1: budget Pareto, horizon ablation, and the
fidelity-transfer check the 480/360 pair gives for free.

This is the strongest version of the paper's claim available anywhere in the
project, because the reality model is an accepted solver rather than our own
surrogate with noise added.

---

## 3. What this cannot establish

- **That the interventions would work in the world.** A solver is not reality.
  It is a validated physics model, and a levee result inherits every assumption
  LISFLOOD-FP makes. This is weaker than a field trial and stronger than §6.2;
  the write-up must say which it is, in those terms.
- **Anything about heat interventions.** No solver, no counterfactual. Heat's
  contribution is perception and forecasting only, and claiming otherwise would
  repeat the wildfire mistake.
- **A daily flood assimilation loop from observations.** Sentinel-1 revisits
  every 6–12 days; an event is seen ~9 times. The 300 s cadence is the solver's,
  not the satellite's, and conflating the two would be dishonest.
- **An unqualified digital twin.** Property 4 of §6 — correction from
  intervention outcomes — stays unmet. A solver loop tests the planner against
  physics, not the twin against consequences.

---

## 4. Cost

| phase | effort | compute | kills |
|---|---|---|---|
| 0 screening | 3–4 days | hours | Phase 3, possibly |
| 1 dense-field test | 1–2 weeks | ~1 day Vista | nothing |
| 2 twin loop + filter | 1–2 weeks | ~1 day | nothing |
| 3 planning vs solver | 2 weeks | ~2 days | — |

Reusing `pspe/simulate/solvers.py` and `pde_env.py` is what keeps Phase 3 at two
weeks; a new hydrodynamic stack would be two months.

**The risk worth naming.** Phase 0 may kill Phase 3 exactly as it killed `swe`,
and for a measurable reason. If so the contribution is Phases 1 and 2 — a
dense-field result across 124 cities and a DA comparison — which is a strong
section or a workshop paper, not a second main-track paper. Running Phase 3
without screening costs a day to learn the same thing and risks reporting
numbers from an infeasible task, which is what happened on wildfire.

---

## 5. Decisions I need

1. **Flood-first, or heat-first?** I recommend **flood-first**: it is the only
   place the intervention counterfactual is reachable, and it reuses the existing
   PDE stack. Heat is broader and safer but inherits the permanent gap.
2. **Is a solver-validated intervention result worth having**, given it is
   explicitly not observational validation? My read is yes, and that it is the
   single biggest upgrade available to the paper's central claim — but it is a
   judgement about what reviewers will accept, which is yours.
3. **Phase 3 if the precondition or the span fails?** I would not run it, and
   would report the failure as evidence. Cheaper, and honest either way.
4. **Second paper, or strengthen the current one?** Phase 2 is worth two weeks
   only for the former.
