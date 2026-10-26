module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
const SHOTS = "/private/tmp/claude-501/-Users-guruprasadparasnis-pspe/38a1c11f-e34e-4a54-8c7a-26eb95373499/scratchpad/shots";
return [

H1("8 · The portal"),

P("A Next.js front end over a FastAPI service, both running the real thing. Every step the portal takes posts to the service, which advances the actual environment — the same solver, the same budget arithmetic, the same reward — and returns its state. Nothing in the browser is an approximation of the model, which is why none of the displayed numbers needs a disclaimer about differing from the real ones."),

H2("8.1 Why the portal is part of the research, not a demo"),

P("Building it found defects that no amount of offline analysis had surfaced. Three are worth stating because of *how* they were found."),

table(
  ["#", "Defect", "How it was caught"],
  [["36", "A ranked list of solved options whose selection reached nothing but its own radio button — every option's depth field was computed and all but the baseline discarded", "**Using it**"],
   ["22", "A page header naming a constant region: the tool said \"Richmond Valley\" over a map of Iowa", "Reading the first rendering of the interface"],
   ["29", "Plan, margin and attribution keyed to a precomputed library, so the framework worked in one valley and everywhere else was a bare solve", "Entering another location and finding only a levee slider"]],
  [5, 55, 40]),

QUOTE("§5.14s records writing the user manual and finding **five controls that did nothing**. An interface is a test harness that exercises code paths in combinations no unit test enumerates, and it is read by somebody who does not already know what the answer should be."),

H2("8.2 The explanation panels, and the problem they solve"),

P("The gym page originally showed a single sorted list of policies with a small red note beside the infeasible ones, and a sidebar of prose. It was accurate and nearly unreadable. It was rebuilt into three columns."),

...figure(SHOTS + "/panels.png",
  "The rebuilt explanation panels. Left: what the decision is, in four labelled fields. Centre: every policy, split by whether it stayed within the allowance. Right: the finding, and a per-claim breakdown of how real the inputs are.",
  { maxHeight: 400 }),

H3("8.2.1 Three structural decisions"),

NUM("**Every percentage is a share of doing nothing** (§7.3), computed by one function the service and the page both call, so a live run lands on the same scale as the ladder."),
NUM("**Affordable and unaffordable are separate lists, not a column.** The best-scoring policy is usually one you may not run; sorted into a single list above the rest, it reads as though it won. \"Went over the allowance, so they don't count\" is a heading rather than a footnote."),
NUM("**The caveat is split into claims that can be weighed one at a time** — thirty-three rows across the fourteen environments, each marked REAL or MADE UP. Run together as prose, a reader who skims takes either \"it is real\" or \"it is made up\" from an entry that says both."),

WHY("A design constraint that overrode the mockup", "The mockup used green for the affordable list. The portal's stylesheet documents why the palette is achromatic — the colour on screen belongs to the water, where it carries depth — with a single red spent only on harm. An unaffordable policy is harm, so red fits; green would have been a new hue against a stated rule. The structure of the mockup was adopted and its colour was not."),

H3("8.2.2 Two edge cases the mockup did not cover"),

P("`PDEControl-swe-v0` has **nothing affordable at all** — not even doing nothing, because the allowance can be spent by the fluid regardless of the dials. The affordable list is omitted entirely and the headline reads \"Nothing stayed within it\"."),

P("`FloodEvacuate-v0`'s best affordable policy **is** doing nothing, so the generic sentence — \"it leaves 100% of the harm of doing nothing\" — is true and circular. That case gets its own sentence: *not acting is the best anyone has managed here; every approach that does less harm costs more than you are allowed to spend.*"),

H2("8.3 Two implementations of one guarantee"),

P("§5.14c records the portal's margin and the framework's being **two implementations of the same formula**, and a parity check between them found four defects at once."),

table(
  ["#", "Defect"],
  [["42", "`conformal_quantile`'s guard used `round(1/δ)−1`, under-counting when `1/δ` is not an integer. At δ = 0.30, 0.12 and 0.08 the rank clamp returned the sample maximum — a band with no guarantee at all"],
   ["43", "The portal's mirror derived δ as `1 − conf/100`, and floating point made it demand 10 runs where 9 suffice, declining 90% at n = 9"],
   ["44", "The script producing the portal's `surrogate.json` held **private copies** of the saturating form and of the rearranged threshold, so the displayed value came from unchecked code and the refusal message disagreed with the refusal test"],
   ["45", "De-duplicating the above shadowed the new shared function with a bool local, so every caller got `TypeError: 'bool' object is not callable`"]],
  [6, 94]),

WHY("The rule", "Two numbers are compared only after checking they were produced by the same code. A version skew between runs is indistinguishable, in the output, from a finding about the systems being compared. The deeper point is that a guarantee implemented twice is a guarantee implemented zero times."),

H2("8.4 Operational details that turned out to matter"),

BULLET("**The service module is named `gym_routes.py`, not `gym.py`**, because `app.py` runs with `service/` on `sys.path` and a module called `gym` would shadow the Gymnasium package."),
BULLET("**Screenshots must be taken against `next start`, not `next dev`** — or, as now, with `allowedDevOrigins` set. The dev server's HMR connection was blocked as cross-origin from the loopback address, React never hydrated, and every captured screenshot read \"Loading…\". The block is silent in the page."),
BULLET("**The street solve is the longest request in the service** — a parent catchment at 40 m then the block at 10 m, thirteen snapshots across eighteen hours. Cold, it runs past the five minutes a browser will wait for response headers, and until the run was shared every retry started a *second* solve of the same block, so the reader saw \"could not open this street\" and made it slower by asking again."),

H1("9 · Forty-five defects, and the rules that followed"),

P("This chapter exists because the corrections are a larger part of the contribution than the original claims were. The defects divide into two kinds, and the second is harder."),

QUOTE("The first eighteen are failures of **measurement protocol** — the code is wrong, or the comparison is unfair. Defects 19 onward are mostly failures of **belief about data**: what a number in a file means. Those are harder, because the code is correct and the output looks right."),

H2("9.1 The defects most worth remembering"),

table(
  ["#", "Defect", "Why it is instructive"],
  [["25", "Downsampling cropped the raster margin, losing up to **59%** of a fire's detections", "It degrades data **continuously** rather than breaking it, affects only some fires, and leaves every downstream number plausible. Only a conservation check finds it."],
   ["21", "119 km² of apparent model over-prediction was the sea and the river channel", "Caught by **drawing the map and looking at it**. Headline CSI moved 0.378 → 0.535."],
   ["34", "A distilled policy learned the MSE-optimal constant and would have **confirmed our own amortisation claim**", "A null that agrees with your hypothesis is the most dangerous kind, because nothing prompts you to check it."],
   ["39", "Coverage averaged over seeds where the margin consumed the entire limit", "A result was written up as *missing* a level it was in fact not rejected at. Found by reading `effective_limit` per seed after the aggregate looked bad."],
   ["31", "DEM nodata filled with sea level, producing a 247 m trench around every inland domain", "Caught by a terrain analysis reporting a coast 1,500 km inland. Physically absurd output is the cheapest diagnostic there is."],
   ["37", "Headline reduction read from a surrogate extrapolated past the physical maximum — 111.7% against a solved 99.81%", "The margin was unattainable at that point, so no guard fired. Found by solving the recommended plan to answer an unrelated question."]],
  [5, 45, 50]),

H2("9.2 The standing rules"),

P("Each rule exists because something specific forced it. The parenthetical is what."),

...[
 "A baseline starts from the same initial condition as the method, or the comparison reports an initialisation. *(Four safe-RL baselines reported as structurally failing, which failed from their initialisation.)*",
 "Constrained comparisons are scored at matched constraint, always.",
 "Any result depending on a hand-specified component is re-run with that component removed.",
 "A margin or bound names the distribution it bounds, and that distribution is the one that produces the failure.",
 "Any metric comparing a generated artefact to a target ships with a permutation control. *(The faithfulness claim, withdrawn.)*",
 "A testbed is calibrated for the variation a metric needs before the metric runs on it.",
 "A testbed is admitted for method comparison only if the span between doing nothing and the best achievable return is wide relative to the differences being measured — and that span is reported. *(swe, admitted on a 7% band.)*",
 "A belief, estimate or reconstruction is evaluated on the **decision**, not on its own reconstruction metric. *(AP 0.594 vs 0.006, decision better in 0 of 9.)*",
 "A calibration sample is exchangeable with the quantity compared against the limit: same estimator, same unit, same aggregation.",
 "Before fitting a correction, measure whether the system has room to apply it. Infeasible and miscalibrated are indistinguishable in an outcome metric.",
 "A sweep reports the value of its own axis at both endpoints, so a range that cannot reach the regime under test is visible before the jobs run.",
 "A threshold verdict carries its own uncertainty, and is flagged when the estimate sits within that uncertainty of the threshold.",
 "Two numbers are compared only after checking they were produced by the same code.",
 "A metric's resolution is checked against the effect size before the run: a rate targeting δ needs enough evaluations that δ is not between two adjacent attainable values.",
 "A defect found in a cheap configuration is re-checked at the configuration that will actually run before its cost is stated.",
 "A scheme that cannot violate the constrained quantity cannot be used to study violating it; verify conservation before trusting a solver. *(6.4% water gained from a clamp.)*",
 "Verify geometry separation numerically before running any physics. *(Inflow straddled the berm.)*",
 "When a testbed cannot separate planners, suspect the actuator's mechanism before concluding the domain is uncontrollable. *(7% → 99.8% on changing the actuator.)*",
 "An intervention's physical dimensions are specified in physical units, not in cells.",
].map((t) => NUM(t)),

H2("9.3 The four newest rules, from the gym work"),

...[
 "Report the paired comparison on **both** the pool and the held-out set. `FloodDefend` +0.21 → +0.59 and `FloodBuyout` +7.57 → +0.05 are indistinguishable before a split and opposite after it.",
 "Check a data source's resolution against the decision before adopting it. ERA5 understated a convective storm's peak five-fold against AORC, and a 10 km model cannot express a decision about city blocks.",
 "When a proposed input does not exist, drop the part of the specification that needed it and say so, rather than inventing a plausible substitute.",
 "A category with two values will meet a third case, and the rendering will be wrong before the assertion is — `dynamics` rendered measured data as \"guessed by a model\".",
 "Prose that states a configuration is a claim, and it drifts silently when the registered default differs from the research default. All three `PDEControl` entries described \"sixteen turns\" while `gym.make` gave eight; the ladder was never wrong, which is exactly why the error survived — no number disagreed with any other number, only with the sentence beside them.",
 "A percentage needs a denominator a reader can name. Anchoring on the worst rung meant \"100%\" was \"the worst anyone happened to try\", which moves when somebody adds a row.",
].map((t) => NUM(t)),

];
};
