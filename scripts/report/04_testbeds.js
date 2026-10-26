module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("4 · Results on the controlled testbeds"),

H2("4.1 The three PDE families, and what each is for"),

P("Three families, and they are **not** three independent confirmations. Saying so was an audit finding against our own draft."),

table(
  ["Family", "PDE", "Channels", "Limit d", "Role"],
  [["`dar`", "diffusion–advection–reaction", "1", "0.936", "Primary testbed. Constraint does not bind at baseline — a **regression check** that the margin costs nothing where it is not needed."],
   ["`rdf`", "reaction–diffusion front", "2", "3.26", "**The validation.** Baseline violates 14.1%; the margin separates at p = 0.0002."],
   ["`swe`", "shallow water", "3", "0.192 → 1.1437", "Retired, then recalibrated. Originally could not rank planners at all."]],
  [10, 26, 10, 14, 40]),

H3("4.1.1 Why swe was retired, and what rescued it"),

P("The first version of `swe` ranked every method inside a **7% achievable band** — the span between doing nothing and the best achievable return was so narrow that method differences were noise. A testbed that cannot separate planners cannot be used to compare them."),

P("The diagnosis was initially that the domain was uncontrollable. That was wrong. Changing the **actuator's mechanism**, on the same equations, moved the span from 7% to 99.8%. The rule that followed: when a testbed cannot separate planners, suspect the actuator before concluding the domain is uncontrollable."),

P("A second correction followed. A best-of-N action search had been drawing **fresh initial conditions per rollout**, manufacturing a 29.4% span on `swe` where the published sweep found none — selection on lucky initial states, not controllability. That is defect 33, and it had admitted a testbed on a span that did not exist."),

H2("4.2 The headline: violations cut roughly threefold"),

P("Reaction–diffusion, 20 seeds, four arms. This is the single most informative experiment in the methods half, because the two middle bars say the obvious story is wrong."),

table(
  ["Arm", "Runs violating the limit"],
  [["No correction", "**14.1%**"],
   ["Margin alone (computed the standard way)", "**14.1%** — identical"],
   ["Real-world probing alone", "**10.9%**"],
   [{ cells: ["**Probe + conformal margin**", "**5.0%**"], opts: { fill: "EAF2EC" } }][0]],
  [55, 45]),

P("Paired, −9.1 points, 95% CI [−11.8, −5.5], p = 0.0002, **at no cost in objective value**."),

...figure("docs/figures/paper/fig6_ablation.png",
  "The four-arm ablation in full. The probe is a measurement and the margin is the estimator that consumes it; each is nearly worthless alone.",
  { maxHeight: 320 }),

QUOTE("A safety margin on its own does nothing at all. It only works when it is calibrated against real measurements. The two do not add — they compose, because they are two halves of one object: the probe is a **measurement**, the margin is the **estimator that consumes it**. A measurement nobody acts on and an estimator with nothing to act on are each worth about nothing."),

WHY("Why the margin-alone arm being exactly identical matters", "Not approximately identical — identical to four decimal places. That is the expected behaviour and a useful internal check: with no probe there are no matched pairs, so there is nothing to calibrate and the margin is correctly inert. If it had moved slightly, something would have been wrong with the implementation."),

H2("4.3 Replication across families, stated precisely"),

P("The mechanism replicates on all three. **The stated rate does not**, and the distinction is the point of the project."),

table(
  ["Family", "Mechanism replicates?", "Level attained?"],
  [["`rdf`", "Yes — 14.1% → 5.0%", "**Yes**, 5.0% against δ = 0.1"],
   ["`dar`", "Yes, trivially — nobody violates", "Trivially, at 0%"],
   ["`swe`", "Yes — paired −15.91 points, CI [−19.77, −12.05], 80 seeds", "**Not rejected**, not attained: 12.41%, CI [9.35, 15.47]"]],
  [12, 44, 44]),

H3("4.3.1 The swe number that changed twice"),

P("This figure moved twice under audit, and both moves are instructive."),

NUM("At 20 seeds it read **16.8%** against δ = 0.1 and was written up as missing the level."),
NUM("At 80 seeds, defect 39 was found: the rate had been **averaged over seeds where the margin consumes the entire limit**. On the 52 of 80 seeds where a plan exists at all, the rate is **12.41%, CI [9.35, 15.47]** — an interval containing δ = 0.1. Caught by reading `effective_limit` per seed after the aggregate looked bad."),
NUM("The excess that remains is an estimate of the exchangeability gap, **Δ̂ = 0.024, CI [−0.006, 0.055]** — the degradation theory predicts for calibrating against a distribution an on-policy planner is still changing."),

P("The honest sentence, after all three corrections: *the mechanism replicates on three families; the rate is attained on rdf, trivially satisfied on dar, and on swe is consistent with but trending above the level.*"),

WHY("Why not just report the 12.41%", "Because the first two numbers were published internally before the third existed. A report that shows only the final figure teaches nothing about how it was obtained, and the sequence — aggregate looks bad, check the per-seed feasibility, find the pooling defect — is the reusable part."),

H2("4.4 Against published safe RL, at a matched real-sample budget"),

...figure("docs/figures/paper/fig11_matched_budget.png",
  "Improvement over doing nothing, at a matched number of real-system interactions. The dashed line is the null policy. On the right-hand problem three of the four baselines cannot be distinguished from it.",
  { maxHeight: 330 }),

WHY("Why matched budget, and not convergence", "Model-free safe-RL methods are sample-hungry by design. Comparing at convergence would compare our method against theirs at a sample count no operator would pay for, which flatters us in a way that does not transfer. Fixing the number of real interactions is the comparison an operator faces."),

P("We return more than all four on both testbeds. The trade is stated rather than hidden: **we cross the constraint more often than the cautious methods do.** They are cautious largely by being inert — a policy that does nothing satisfies every constraint perfectly — which is why both axes appear on the plot. Constraint satisfaction alone is not evidence of a working method."),

H3("4.4.1 A baseline failure we had to fix before reporting it"),

P("One of the baselines initially failed in a way that matched our predicted direction, which is exactly when to be suspicious. SMBPO's truncated prefix cost was being **rescaled linearly, on a cost that accrues non-linearly** — a −4.7% bias, four times the size of the pessimism term it was supposed to carry, making the arm net *optimistic* so it violated more than baseline. Fixed before the comparison ran. That is defect 35."),

P("A related earlier correction: four constrained-RL baselines were reported as \"structurally failing\" when in fact they failed **from their initialisation**. Standing rule 1 followed: a baseline starts from the same initial condition as the method, or the comparison reports an initialisation."),

...figure("docs/deck_bundle/images/reserve-R1-null-floor.png",
  "The null floor. Several baselines do not beat doing nothing, which is why every comparison in this project is anchored on the null policy rather than on another method.",
  { maxHeight: 260 }),

H2("4.5 δ is a width, not a dial"),

P("A natural expectation is that tightening δ buys safety. It does not."),

P("Sweeping δ from 0.30 down to 0.02 costs return monotonically and **buys no reduction in realised violation**, because the margin is already covering the distribution and tightening it only widens the band. The counter-intuitive consequence predicted in §2.4.2 — that relative error `R/δ` *rises* as δ tightens — is visible in the same sweep."),

WHY("Why this is worth a figure", "Because δ looks like a safety knob and is not one, and a practitioner who treats it as one will pay return for nothing. A parameter is not a dial until its response curve is measured — standing rule 66."),

H2("4.6 Exchangeability: rejected, and coverage holds anyway"),

P("Split conformal assumes the calibration scores and the test score are exchangeable. An on-policy planner is still changing its own distribution, so this assumption is suspect by construction. It was tested rather than assumed."),

table(
  ["Testbed", "Exchangeability test", "Coverage"],
  [["`dar`", "Holds", "Attained"],
   ["`rdf`", "**Rejected on 20 of 20 seeds**", "**Holds anyway**, 5.0% against δ = 0.1"]],
  [16, 46, 38]),

P("The violation is *negative* autocorrelation — scores alternate rather than trend — and negative autocorrelation does not degrade the conformal bound in the direction that matters. The falsifying experiment, stated so somebody can run it, is a hazard whose scores **trend** rather than alternate; that is in future work."),

];
};
