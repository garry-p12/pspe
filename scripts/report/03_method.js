module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("3 · The method, component by component"),

P("Four stages, and the contribution is not in any of them individually — it is in the correction attached to each. The pattern repeats: find the reference the stage is scoring itself against, and replace it with something measured outside the system."),

...figure("docs/deck_bundle/images/slide-05-system-architecture.png",
  "The loop. The top row is a conventional model-based pipeline; the green row is the claim. Each green box replaces a self-defined reference with a measurement. The red dashed box is the one correction that cannot be made.",
  { maxHeight: 400 }),

H2("3.1 Perceive — state from observation, under occlusion"),

P("The input is partial: satellites pass on an orbit, cloud intervenes, and a field is observed on some cells and not others. The job is a state estimate the planner can use."),

H3("3.1.1 The negative result that shaped it"),

P("The obvious thing is to impute the unobserved cells as well as possible, and the obvious evaluation is reconstruction accuracy. Both are wrong here, and the measurement that showed it is worth stating in full."),

table(
  ["What was measured", "Value"],
  [["Reconstruction average precision, imputed vs raw masked input", "**0.594 vs 0.006** — roughly 100× better"],
   ["Configurations in which the better belief produced a better decision", "**0 of 9**"]],
  [62, 38]),

P("A belief that matches the hidden state far better planned no better in a single configuration out of nine. The mechanism was identified afterwards rather than filed as an anomaly: **the estimator inflates area.** On a sparse field, a reconstruction optimised for a pixel-wise loss spreads probability mass outward, which raises reconstruction score and simultaneously makes the planner treat ground that will not burn."),

WHY("The rule this forced", "A belief, estimate or reconstruction is evaluated **on the decision**, not on its own reconstruction metric. This is standing rule 8, and it is the one most likely to generalise outside this project."),

H2("3.2 Simulate — the learned surrogate"),

P("A Fourier Neural Operator. The field is transformed to frequency space, the twelve lowest modes are kept and mixed by learned weights, and the result is transformed back; a plain 1×1 convolution carries the fine detail that truncation discards."),

...CODE(["v_{l+1}  =  σ( W v_l  +  F⁻¹( R · F v_l ) )"]),

WHY("Why twelve modes is safe here", "The fields are smooth and advective, so energy sits at low wavenumbers. The truncation is not free — it is precisely the source of the bias term `b` that the next two sections have to handle — but it is bounded and it buys a differentiable forward model fast enough to plan through."),

P("Training has three terms: next-step prediction, multi-step rollout stability, and consistency with the governing equation. The third matters because a model trained on one-step accuracy alone will drift when stepped on its own output, and the planner steps it sixteen times."),

H3("3.2.1 What the surrogate is measured on"),

P("Published-benchmark skill on PDEBench shallow water, and on real fire data at the level of the forecaster published with the NDWS dataset. **Not** at the level of the best model anybody has built, which is stated explicitly: forecast skill is an input to the planning problem, not the contribution. A better surrogate improves every downstream number and changes nothing in the method."),

...figure("docs/figures/pspe_mod_simulate.png",
  "The Simulate module. A Fourier Neural Operator gives a differentiable forward model fast enough to plan through; its truncation error is exactly the bias term the planner then has to handle.",
  { maxHeight: 320 }),

H2("3.3 Plan — the hybrid policy gradient"),

P("There are two ways to get a gradient through a learned model and they fail in opposite directions."),

table(
  ["Estimator", "Property", "Failure mode"],
  [["Pathwise `g_pw`", "Differentiate straight through the simulator", "Low noise, but the simulator's error enters as a **systematic bias B** in the direction you move"],
   ["Likelihood ratio `g_lr`", "Try actions, do more of what scored well", "No simulator bias, but **variance V_lr** is enormous"]],
  [18, 34, 48]),

P("They are combined, and the weight is not a hyperparameter:"),

...CODE(["g  =  α g_pw  +  (1 − α) g_lr",
         "",
         "            V_lr  −  Cov",
         "α*  =  ───────────────────────────────",
         "        B²  +  V_pw  +  V_lr  −  2 Cov"]),

WHY("Why this is the interesting part", "α* is the closed-form minimiser of the mean-squared error of the combined estimator, and **every term in it is measured on the real system**. So the weight adapts on its own: when the surrogate is accurate, B² is small and α moves toward the pathwise gradient; when it drifts, α falls back toward the unbiased one. Trust in the simulator becomes a measured quantity rather than a setting somebody chose."),

H3("3.3.1 The claim about this rule that did not hold"),

P("The proposal claimed the full rule, including the B² term, improves on a variance-only rule. **It is degenerate in practice.** Measured B² ≈ 0.0013, so both rules select α = 0.992 and the returns are indistinguishable (t = +0.22 at twenty seeds). The rule is kept because it is correct and costs nothing, but it is not a contribution and is not presented as one."),

H2("3.4 The dual — a price on cost"),

P("The constraint is enforced by a Lagrangian with a price λ per unit of cost, set by a PID controller:"),

...CODE(["max   r  −  λ c",
         " π"]),

P("Proportional: cost above the limit, raise the price. Integral: above it for a while, raise it more. Derivative: cost climbing fast, react early. The integral term is what removes steady-state error, which is why the budget gets hit at all."),

QUOTE("**And here is the flaw the whole project is about.** Read what this controller is differencing: *the simulator's* cost against the limit. It is a well-tuned controller on a miscalibrated instrument. It will drive the reading to the setpoint, and the reading is wrong by `b` plus the spread."),

...figure("docs/deck_bundle/images/reserve-R2-mixing-rule.png",
  "The mixing rule measured. The full rule and the variance-only rule select the same weight because the measured bias term is small — reported as degenerate rather than as a contribution.",
  { maxHeight: 300 }),

H2("3.5 The fix — calibrate on the real overshoot"),

P("Probe the real system periodically. Each probe episode yields a score:"),

...CODE(["s_i  =  c_i  −  μ̂",
         "",
         "q̂  =  s_(k),     k = ⌈(n + 1)(1 − δ)⌉",
         "",
         "n  <  1/δ − 1    ⟹   refuse"]),

...figure("docs/deck_bundle/images/slide-07-the-estimator.png",
  "The estimator. Each probe episode yields a real overshoot; the k-th order statistic of those overshoots is the margin, and below a certain count the required order statistic does not exist and the code refuses.",
  { maxHeight: 340 }),

H3("3.5.1 Why the score contains both missing terms"),

P("`c_i` is a real outcome and `μ̂` is what the dual believed that episode would cost. **The difference is, by construction, everything the dual did not know about** — the bias and the spread, already combined, already sampled from the real system. Nothing has to be estimated separately and added; the overshoot *is* the sum."),

H3("3.5.2 Why no distributional assumption is needed"),

P("Because the calibration scores and a future score are exchangeable, the future score's rank among them is uniform. Taking the k-th order statistic with `k = ⌈(n+1)(1−δ)⌉` gives coverage of at least 1−δ in finite samples, for **any** distribution. The Gaussian in Chapter 2 was for the *analysis* of why the wrong recipe fails; it is not used by the method."),

H3("3.5.3 Why it refuses, and what the threshold means"),

P("When `n < 1/δ − 1`, the required `k` exceeds `n`: **the order statistic the guarantee needs is not in the sample.** There is no conservative fallback, only extrapolation past the largest observation, which is exactly the move the rest of this project is about not making. At δ = 0.1 that bites below 9 runs; at δ = 0.05, below 19."),

WHY("Why a refusal is a result", "A displayed band that cannot carry its stated level is worse than no band, because it is acted on. Defect 27 was exactly this: the portal claimed 90% confidence from 6 calibration runs, understating the honest margin by about 2×. It was caught by calling the conformal function, which refuses — the portal had been computing the quantile itself."),

H2("3.6 Explain — and the claim that was withdrawn"),

P("The fourth stage produces a natural-language brief with a conformal certificate on its faithfulness. The certificate holds. **The faithfulness claim does not**, and this is the clearest single instance of the project's own thesis turned on itself."),

table(
  ["Measurement", "Result"],
  [["Faithfulness F(b) against a post-hoc control", "0.47 vs 0.18, t = +21 — apparently decisive"],
   ["The same briefs scored against **another state's** features", "**Indistinguishable.** The briefs carry zero state-specific information."]],
  [50, 50]),

P("The permutation control is the entire finding. A metric comparing a generated artefact to a target will reward generic fluency, and without a control that scores the artefact against the *wrong* target there is no way to tell fluency from information. Two further corrections followed: the post-hoc control turned out to be an untrained generator (making the comparison meaningless in the other direction too), and the apparent plateau in the faithfulness term was the token cap, not convergence."),

WHY("Why this is kept in the report", "It would be easy to drop a withdrawn module. It stays because the permutation control is now standing rule 5 — any metric comparing a generated artefact to a target ships with one — and because the project's credibility rests on the catalogue being complete rather than flattering."),

];
};
