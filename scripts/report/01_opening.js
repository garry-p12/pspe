module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("1 · What this document is"),

P("This report describes everything built in the PSPE project: what each piece does, **why it was built that way rather than some other way**, what was tried and rejected, and how every headline number was arrived at. It is written to be read end to end by somebody who has seen the work only in fragments."),

P("Three things make it different from the paper and the slide deck."),

BULLET("**It states the choices, not only the outcomes.** Where a design could reasonably have gone two ways, the report says which way it went, what the alternative was, and what measurement settled it. A result with no stated alternative is a result nobody can argue with, which is not the same as a result that is right."),
BULLET("**It includes what failed.** Forty-five numbered defects were found during this project, a majority of them in our own code and our own published numbers. They are catalogued here with the measurement that exposed each one, because the corrections are a larger part of the contribution than the original claims were."),
BULLET("**It separates measured from modelled from assumed, at every number.** Wherever a figure depends on an assumption — a floor height, a calibration multiplier, a cost rate — the assumption is named at the point of use rather than folded into the number."),

H2("1.1 The one-paragraph version"),

QUOTE("Model-based decision systems fail silently. Every component's own metric can improve while the decision it feeds degrades or the constraint it protects is violated, because each metric is computed inside the model's own account of the world. This project is (a) a catalogue of those failures with the measurement that exposes each, and (b) a set of mechanisms that close the loop on real measurement — probe the environment, synchronise to observation, score on the decision — which recover the lost performance and, in the constrained case, carry a stated failure rate."),

P("Everything else in this document is either evidence for that paragraph, machinery that implements it, or an honest statement of where it does not hold."),

...figure("docs/figures/pspe_silent_failure.png",
  "The failure shape, instantiated nine times. Each row is a component whose own metric said it was working; the right column is the measurement that disagreed. Three of these were found after the thesis was written, by applying its own rule to itself.",
  { maxHeight: 420 }),

H2("1.2 What is actually shipped"),

P("Four things exist and run, and it is worth separating them because they are at very different stages of maturity."),

table(
  ["Artefact", "What it is", "State"],
  [
    ["The safe-RL method", "A split-conformal margin calibrated on probes of the real system, inside a constrained policy-optimisation loop with a hybrid gradient and a PID dual.", "Validated on one PDE family, replicated on three, with the level attained on two."],
    ["The flood twin", "A live digital twin of Onion Creek, Austin: real terrain, real roads, 27 live river gauges, a shallow-water solver, and five decision views.", "Runs. Validated against FEMA floodplain and Sentinel-1 observation, with the disagreement reported."],
    ["The wildfire twin", "The same loop on fire: NASA FIRMS detections, a learned spread model, and crew allocation over 1,689 held-out fire days.", "Runs. Planning beats the operational rule by 12 points at a 3-day horizon."],
    ["The simulation gym", "Fourteen Gymnasium environments wrapping both twins as constrained decision problems, each with a budget, a ladder of every policy ever run, and a held-out scenario split.", "Runs. Eight of the fourteen have had no learned policy on them at all."],
  ],
  [16, 50, 34]),

...figure("docs/deck_bundle/images/slide-06-what-it-offers.png",
  "What the framework offers, in three properties: it learns from far less real-world data, it states its own failure rate in advance, and it refuses when its calibration set cannot support the level asked for.",
  { maxHeight: 260 }),

H2("1.3 The five original claims, and what happened to them"),

P("The project began from a proposal with five headline claims. **None survives in its original form.** That is stated first rather than buried, because the thesis this report defends is not a repair of the old one — it is a different claim that the evidence actually supports."),

table(
  ["#", "Original claim", "Outcome"],
  [
    ["1", "Joint training of Perceive/Simulate/Plan/Explain under one objective beats a disaggregated pipeline", "**Refuted** on three PDE families. The decision never changed: t = +0.66, +1.00, and identical."],
    ["2", "The Eq. 8 mixing rule improves the hybrid gradient over a variance-only rule", "**Degenerate.** Measured B² ≈ 0.0013, so both rules select the same α (0.992). Return t = +0.22."],
    ["3", "Time-averaged constraint violation tends to zero", "**False as stated** on the PDE testbeds. Holds only on a toy CMDP with a closed-form optimum."],
    ["4", "Faithful natural-language explanations via a faithfulness term", "**Withdrawn.** A permutation control shows the briefs carry zero state-specific information."],
    ["5", "Cross-PDE-family transfer", "**Direction only.** Magnitudes into the wave family have σ ≈ mean."],
  ],
  [5, 50, 45]),

WHY("Why this is at the front", "A reader who finds a refuted claim on page sixty, after sixty pages written as though it held, stops trusting the other fifty-nine. Putting the refutations first costs nothing and makes the rest readable."),

H2("1.4 How to read a result in this document"),

P("Every empirical claim here carries four things, and if any is missing the claim is weaker than it looks."),

NUM("**The sample size and the unit.** A paired comparison over twelve episodes and one over six hundred fires are different claims. Several results in this project moved substantially when the sample grew — the twin loop's benefit fell to about a third of its eight-fire value at 607 fires."),
NUM("**Whether it was measured on data the method was fitted to.** Scenario pools in the gym are split train and eval under a fixed seed, and every quoted figure comes from the held-out half. Two headline comparisons that looked solid on the full pool did not survive the split."),
NUM("**The cost, not only the return.** On several environments a table of returns alone ranks a random policy first, because the random policy spends several times the budget. A return without its spend is not a result."),
NUM("**What is measured, what is modelled, and what is assumed.** These are three different epistemic states and this report never renders them identically."),

RULE(),

P("Dates in section references are the dates the finding was recorded in the master record (`docs/THESIS_AND_RESULTS.md`), which is the primary source for everything in this report. Where this document and that one disagree, that one is correct and this one is stale."),

];
};
