module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("2 · The failure: why a satisfied budget is violated"),

P("This chapter is the technical core. It explains the failure mode, proves that the obvious correction cannot work, gives the failure in closed form, and states precisely where that closed form stops being useful."),

H2("2.1 The setting"),

P("The problem is a constrained Markov decision process: maximise discounted reward subject to expected cumulative cost staying under a limit."),

...CODE([
  "max   E[ Σ γ^t r ]     subject to     E[ Σ c ] ≤ d",
  " π",
]),

P("Reward is tracking a target field without using more actuator effort than necessary. Cost is charged whenever the field exceeds a safety threshold, or actuator effort exceeds its own budget. The limit `d` is placed at **35% of the way from doing nothing to acting greedily**."),

WHY("Why 35%", "A constraint that never activates measures nothing. Early runs placed the limit where doing nothing already satisfied it, and every method then scored identically because the constraint was decorative. The limit is set from a trained reward-greedy anchor so that it binds, and the anchor run is reported alongside it. On `swe` this took a dedicated calibration job: the greedy anchor sits at 3.2647 and do-nothing at 0.0017, putting the limit at 1.1437 and replacing an earlier lower bound of about 0.083 that did not bind."),

P("Note the expectation on the cost side. **The standard CMDP constrains a mean.** Everything that follows is downstream of that one symbol."),

H2("2.2 The decomposition"),

P("A realised episode cost is three things, not one:"),

...CODE([
  "c_i  =    μ      +      b        +     σ ξ_i",
  "      what the      hidden        run-to-run",
  "      dual sees     bias          spread",
]),

P("The dual controller is a feedback loop whose **sensor is the simulator**. It drives μ, the simulator's mean cost, down to the limit `d`, and then correctly reports that the constraint is satisfied — correctly, *about the simulator*. The bias `b` is whatever the surrogate systematically gets wrong. The spread `σξ` is the policy's own run-to-run variation, which exists even under a perfect model because the environment is stochastic."),

H3("2.2.1 The type mismatch"),

P("Here is the centre of the whole project, and it is a mismatch of **kinds**, not of magnitudes."),

QUOTE("The optimiser controls a **mean**. The guarantee an operator wants is about a **quantile** — \"at most one run in ten breaks the budget\". A mean and a quantile differ by the spread. Nothing in the training loop can notice, because the loop only ever observes the mean it is already driving to zero error."),

P("Write the requirement out. For at most a fraction δ of runs to violate, a margin `q` (aim at `d − q` instead of `d`) must satisfy:"),

...CODE(["q  ≥  b  +  z_δ σ          (z_δ ≈ 1.28 for δ = 0.1)"]),

P("The margin has to grow with σ. **That single line is the problem**, because σ is a property of the policy and the environment, and no amount of improving the model reduces it."),

H2("2.3 Why the obvious fix cannot work"),

P("The natural correction is to inflate the cost estimate by how wrong the model is. Ensembles, error bars, pessimism bonuses, disagreement penalties — all of these estimate model error. **None of them estimates σ.**"),

WHY("The intuition", "Aim versus scatter. If you are shooting at a target and your sights are biased, measuring the bias and correcting it works. If your shots scatter, correcting the sights does nothing — you need the size of the group, and no amount of inspecting the rifle tells you that. Every published uncertainty method in model-based RL inspects the rifle."),

P("The consequence is that a fixed margin is fine at one spread and broken at another, with no change to the margin itself:"),

table(
  ["Spread σ", "(q − b)/σ", "Violation rate"],
  [["0.1", "2.0", "**2.3%** — fine"],
   ["1.0", "0.2", "**42%** — broken"]],
  [30, 30, 40], { numericFrom: 1 }),

P("And this is confirmed empirically against the field's own methods. CAP, SMBPO and SafeDreamer all land at roughly the rate you get with **no margin at all** — 12 to 13% against 14.1%. They are not badly implemented; the margin they compute is in the wrong units."),

...figure("docs/deck_bundle/images/slide-19-safety-violations.png",
  "The ablation that isolates the mechanism. A margin computed the standard way is not merely weak — it is identical to no correction at all, to four significant figures.",
  { maxHeight: 330 }),

...figure("docs/figures/pspe_margin_choice.png",
  "The choice of what to calibrate on. Calibrating against model error measures the wrong object; calibrating against the realised overshoot measures the one that violates.",
  { maxHeight: 340 }),

H2("2.4 The separation theorem"),

P("The informal argument above is made precise. The statement is a **separation result**: it says no member of a whole class of corrections can work, rather than that particular ones happen not to."),

QUOTE("**Theorem.** No margin that is a measurable function of the model-error distribution alone controls violation probability uniformly in σ, the spread of realised cost. Realised violation is strictly increasing in σ and tends to ½ for symmetric error, so every such margin fails beyond a computable threshold."),

P("The proof is three lines. With `t(σ) = (q − b)/σ` strictly decreasing and the error CDF `F` strictly increasing, the violation rate `R = 1 − F(t)` increases in σ; `t → 0⁺` gives the limit of ½; and `R > δ` exactly when `t < z_δ`."),

WHY("Why the brevity is the point", "The theorem is not deep. What was missing was not a proof technique but the **statement** — the recognition that the class of corrections everyone reaches for is characterised by what it measures, and that this characterisation is already enough to rule it out. The hypotheses matter and are recorded separately: full support gives strictness, symmetry is used only to evaluate the limit, and the requirement δ < 1 − F(0) is not a removable technicality."),

H3("2.4.1 Where ρ = 1 comes from"),

P("Calibrating on model error gives the matched-pair recipe `q = b + z_δ ε`, where ε is the spread of the model error. What is actually required is `b + z_δ σ`. These coincide **exactly** when ε = σ. Define"),

...CODE(["ρ  =  σ / ε"]),

P("and the crossover sits at ρ = 1 by construction, not by fitting. Above it the recipe under-margins; below it, over-margins."),

H3("2.4.2 The closed form"),

P("For Gaussian error the failure rate is available in closed form:"),

...CODE(["R(ρ)  =  1  −  Φ( z_δ / ρ )"]),

P("Two properties worth stating aloud. `R(1) = δ` identically for every δ — the recipe is *exactly* calibrated at the crossover. And `R → ½` as ρ grows, because the margin becomes negligible beside the spread, so you are aiming at the limit with symmetric noise and half the runs land above it."),

P("There is also a counter-intuitive, testable consequence: the **relative** error `R/δ` rises as δ tightens, because margin and requirement scale by the same factor while the tolerance shrinks. At ρ = 1.4, `R/δ` runs from 1.18× at δ = 0.30 to 3.56× at δ = 0.02."),

H2("2.5 Validating the closed form"),

P("The curve was derived first and measured against second, with no fitted parameters. The experiment is a controlled CMDP where σ and ε are set by hand, δ = 0.1, and each point is 600 trials; each trial draws 400 calibration samples, computes the margin, and evaluates on 20,000 fresh samples."),

table(
  ["ρ = σ/ε", "Measured", "Predicted", "|error|"],
  [["0.10", "0.0000", "0.0000", "0.0000"],
   ["0.50", "0.0043", "0.0052", "0.0009"],
   ["0.80", "0.0539", "0.0546", "0.0007"],
   ["**1.00**", "**0.0990**", "**0.1000**", "0.0010"],
   ["1.25", "0.1528", "0.1526", "0.0002"],
   ["2.00", "0.2582", "0.2608", "**0.0026**"],
   ["4.00", "0.3734", "0.3743", "0.0009"],
   ["10.0", "0.4487", "0.4490", "0.0003"]],
  [22, 26, 26, 26], { numericFrom: 1 }),

...figure("docs/figures/paper/fig0_calibration.png",
  "The closed form against measurement. The curve was derived first and has no fitted parameters; the inset shows relative error R/delta rising as delta tightens, which is the counter-intuitive prediction of Corollary 2.",
  { maxHeight: 300 }),

P("**Maximum error 0.003**, across two orders of magnitude in ρ, and the transition lands where the theory puts it. Source: `runs/margin_synthetic/results.json`, reproduced with `eval/run_margin_coverage.py`."),

WHY("What the 0.003 is not", "It is the accuracy with which we predict how badly the **inadmissible** recipe fails. It is not an accuracy figure for our method, and it is the most likely misreading of the result. Saying \"our experiments match the formula to 0.003\" without that clause invites a reader to hear it as a performance claim."),

H3("2.5.1 The residual is systematic, not noise"),

P("The measured value sits **below** the predicted one in six of the seven non-zero rows. Random sampling error would scatter both ways, so this is a small systematic conservatism rather than Monte-Carlo noise, and it has an identifiable cause: the formula uses the true Gaussian quantile `z_δ`, while the experiment estimates a quantile from 400 calibration points, and a finite-sample conformal quantile is slightly conservative by construction."),

P("This is worth stating because it is stronger than attributing the gap to noise. A residual that is understood is better evidence than a residual that is merely small."),

H2("2.6 Where the closed form stops transferring"),

P("Both quantities were then measured on the real planner, at every probe, with an arm running the excluded recipe alongside. Twenty seeds, reaction–diffusion:"),

table(
  ["Quantity", "Measured"],
  [["σ_ep — spread of realised episode cost", "**0.454**"],
   ["ε — spread of the matched-pair model error", "**0.0265**"],
   ["ρ = σ/ε", "**median 17.9**, range [7.4, 91.1], **> 1 on 20 of 20 seeds**"]],
  [55, 45]),

P("So the separation result's precondition holds by an order of magnitude, and the mechanism is visible: the matched pair is taken on common random numbers, so the policy's own spread cancels and ε collapses. The excluded recipe's margin is consequently **0.059 of budget where ours applies 0.583 — 9.9× too small** (paired difference +0.5245, CI [+0.4766, +0.5768]), and it recovers only about an eighth of what ours does."),

QUOTE("**But the closed form does not transfer.** `R(ρ)` predicts 0.470 at the measured ρ; the arm violates 0.105 — a **4.5× over-prediction**."),

P("The reason is in the derivation. `R(ρ)` assumes a dual that drives the model's cost exactly to `d` and tends to ½ as ρ grows. A real planner's violation rate is bounded above by its own no-margin rate (14.09%), because a negligible margin degenerates the arm toward the unmargined planner rather than toward the asymptote."),

WHY("The lesson, stated as a rule", "**A theorem's conclusion and its closed form transfer independently.** The qualitative statement — this statistic is insufficient, so a margin built from it must fail — survives contact with a learned planner and is confirmed by a margin 10× too small. The curve, exact to 0.003 where its assumptions hold, is an upper bound the real system does not approach. We report it as such."),

];
};
