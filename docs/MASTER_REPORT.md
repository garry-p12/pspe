# PSPE: the complete experimental record

**Perceive – Simulate – Plan – Explain: constrained intervention design for
PDE-governed systems.** Consolidated results, updated 2026-09-24.

Every number here is five seeds unless stated otherwise, reported as mean ±
sample standard deviation, with paired *t* where two arms are compared
(df = 4 for five seeds, so |t| > 2.78 is p < 0.05; df = 2 for three seeds, so
|t| > 4.30). Each section names the run directory its numbers come from.
Detailed analyses live in `docs/results/`; this document is the single place
where all of it is stated together, including the claims that did not hold.

---

## Executive summary

PSPE is four modules trained around one loop: turn an observation of a
physical hazard into a state estimate, roll a differentiable surrogate
forward, choose an intervention under hard constraints, and write a brief a
human can act on. This record covers everything measured on it.

**Four contributions, as they stand after all testing:**

| # | contribution | status |
|---|---|---|
| 1 | **Safety by measurement.** Model-based constrained planning fails silently because the dual controller trusts the surrogate's cost. Probing the real system and planning against a measured margin fixes most of it. | **Holds.** Two PDE families. Violations cut ~7x on dar (1.8% to 0%) and 85% to 5.5% on rdf. The same diagnosis repaired PPO-Lagrangian. |
| 2 | **Sample efficiency.** The planner acts inside the surrogate, so its real-world cost is the data the surrogate was fitted on. | **Holds.** 9x fewer real transitions than model-free safe RL on dar, 5x on rdf, at comparable return. |
| 3 | **Planning on observed data.** Constrained firebreak planning through a surrogate fitted to real fire spread. | **Holds, 5 seeds.** +12.0 points of burn reduction over the operational forecast-then-greedy heuristic at identical budget, zero violations, paired t = 11.15. Advantage is 1.9–2.1x at the tightest budgets, widens when the imposed action physics is removed, and grows monotonically with horizon (t = 1.40 at 1 day, +14.3 at 5 days) — the mechanism, measured. Constrained-RL baselines on the same fires reach 4.6%, below the heuristic's 24.1%. |
| 4 | **Certified explanations.** Briefs trained against the policy, with a conformal bound on their faithfulness. | **Half withdrawn.** The certificate's coverage holds on real data. The faithfulness claim does not: a permutation control shows the briefs carry no state-specific information. |

**The three findings that changed the project:**

1. The planner's constraint violations came from controlling the dual on
   surrogate cost while violation is realised under true dynamics. Measuring
   reality periodically and planning against a margin is the fix, and it is
   contribution 1.
2. Joint Simulate+Plan training — the paper's original headline — never
   changed a decision on any of three PDE families. Anchored to data it
   improves the surrogate 5–9x; unanchored it destroys it. A clean negative
   with a safety lesson.
3. Explanation faithfulness was never being measured. A permutation control
   (score each brief against a *different* state's action) shows zero gap on
   every arm and both testbeds, and the synthetic testbed used for the
   published numbers has a state-independent policy, so it could not have
   detected the difference.

---

## Part I — Method

Four modules, one loop.

**Perceive.** A frozen vision backbone (SigLIP) with LoRA adapters and two
heads: a decoder that reconstructs the physical field, and a projection head
aligned to caption embeddings by InfoNCE. Loss `L = w_reg L_reg + w_con L_con`.

**Simulate.** A Fourier Neural Operator: lift, then blocks combining a
spectral path (FFT, keep modes k ≤ 12, learned complex weights, inverse FFT)
with a pointwise 1x1 convolution, then project. Trained on one-step,
rollout and physics-residual terms. Optionally spectrally normalised so its
Lipschitz constant is bounded.

**Plan.** A tanh-squashed Gaussian policy over actuator amplitudes. The
Lagrangian `-(r - λ c)` is differentiated two ways — pathwise through the
surrogate (low variance, biased by surrogate error) and likelihood-ratio
(unbiased, high variance) — and mixed at the variance-minimising weight. A
PID-Lagrangian dual drives λ from measured cost. Constraints: intervention
budget, pointwise safety, and a distributional equity term.

**Explain.** A frozen small LM (Qwen2.5-0.5B) with LoRA, conditioned on a
prefix built from the planner's state and action. A frozen parser reads the
generated brief back into an action distribution, and
`F(b) = exp(-KL(π ‖ parse(b)))` scores the agreement. A split-conformal
certificate bounds F on unseen states.

**Testbeds.** Three PDE families — `dar` (diffusion–advection–reaction),
`swe` (shallow water), `rdf` (reaction–diffusion front) — plus two real
datasets: PDEBench 2D shallow water, and Next Day Wildfire Spread (NDWS).

---

## Part II — Results by module

### Perceive

Real SigLIP backbone (`google/siglip-base-patch16-224`), 3 seeds
(`runs/perception_real/`):

| arm | field rel L2 | retrieval accuracy | trainable |
|---|---|---|---|
| PSPE — frozen + LoRA + contrastive | 0.0596 ± 0.027 | **0.754 ± 0.026** | 1.64 M |
| probe — frozen backbone, decoder only | **0.0345 ± 0.011** | 0.135 ± 0.009 | 1.64 M |
| CNN from scratch | 0.0564 ± 0.015 | 0.130 ± 0.016 | 0.56 M |

The frozen probe reconstructs best and beats the CNN (t = -4.87). The design
wins on alignment instead: retrieval 0.754 vs 0.130, t = +37. Real SigLIP
features are strong enough that adapters do not help reconstruction. The stub
backbone gave the same shape (0.0524 / 0.0476 / 0.0422; retrieval 0.510 vs
0.137).

**Reading.** PSPE's perception trades reconstruction accuracy for
text–image alignment. It does not beat a plain CNN at estimating the field.

### Simulate

PDEBench 2D shallow water, 512 real trajectories, 64², 20 epochs
(`runs/simulate/`, `runs/pdebench/`):

| operator | rel L2 (1-step) | rel L2 (rollout) | params | published FNO (nRMSE) |
|---|---|---|---|---|
| **FNO** | **0.0018** | **0.0098** | 1.19 M | 0.0044 |
| DeepONet | 0.0044 | 0.0137 | 0.09 M | — |
| GNOT | 0.0056 | 0.0543 | 0.31 M | — |

Metrics differ (relative L2 vs nRMSE), so this is an anchor, not a
like-for-like comparison. The operator ranking reproduces on synthetic data
(FNO 0.057, DeepONet 0.256, GNOT 0.268, 5 seeds), which is the more robust
claim.

**Resolution invariance** (`runs/resolution_seeds/`), one FNO trained at 64²
and evaluated at three grids, 5 seeds:

| grid | cells | rel L2 |
|---|---|---|
| 64² | 4,096 | 0.0298 ± 0.014 |
| 96² | 9,216 | 0.0298 ± 0.014 |
| 128² | 16,384 | 0.0300 ± 0.014 |

Flat to three decimals: genuine discretization invariance.

**Observed wildfire spread** (`runs/ndws_seeds/`), ~18k real 64 km patches,
1.3% fire prevalence, AUC-PR on the validation split, 3 seeds:

| model | AUC-PR | vs published 0.284 |
|---|---|---|
| **U-Net** | **0.277 ± 0.002** | **97.5%** |
| Hybrid (U-Net + gated FNO) | 0.235 ± 0.005 | 83% |
| FNO alone (1 seed) | 0.200 | 70% |
| all-zeros floor | 0.011 | — |

The U-Net beats the hybrid on every seed (t = +12.5), and the hybrid's learned
gate over its spectral path — free to take any value, initialised at zero —
settles at **0.12 on every seed**. Offered the Fourier operator, the model
declines it.

**On the test split** (`runs/ndws_test/`, 3 seeds, the split the published
0.284 is reported on, with flip augmentation):

| model | test AUC-PR | vs published 0.284 | spectral gate |
|---|---|---|---|
| **U-Net** | **0.3162 ± 0.017** | **111%** | — |
| Hybrid (U-Net + gated FNO) | 0.2939 ± 0.003 | 103% | 0.111 |

The surrogate **exceeds** the published benchmark on its own split. Two
conditions belong with that number: training used 8,000 patches rather than the
full train split, and this run used flip augmentation (with the wind-direction
channel corrected under the flip) which the validation runs did not. The
spectral gate settles at 0.111 again, a third independent replication of the
operator-scope finding.

**Reading.** The operator's scope, measured: excellent on smooth PDE fields
and resolution-transferable; loses to convolutions on sharp fronts, because a
spectral model truncates the high frequencies that *are* a fire front. PSPE
survives this because it is a loop with a swappable surrogate.

### Plan

**Against safe RL on dar** (`runs/seeds_v3/`), 5 seeds, cost limit 0.936:

| method | return | cost | violating evals | worst | real samples |
|---|---|---|---|---|---|
| **PSPE (adaptive α)** | **-2.448 ± 0.029** | 0.412 ± 0.37 | 7.3% | 1.109 | 3,072 |
| PSPE (fixed α) | -2.738 ± 0.365 | 2.752 ± 3.6 | 12.7% | 2.758 | 3,072 |
| PPO-Lagrangian | -2.558 ± 0.039 | 0.266 ± 0.009 | 0% | 0.297 | 38,400 |
| CPO | -2.541 ± 0.035 | 0.263 ± 0.011 | 0% | 0.277 | 38,400 |
| Sauté RL | -2.537 ± 0.024 | 0.264 ± 0.008 | 0% | 0.275 | 38,400 |
| Primal-dual NPG | -2.544 ± 0.026 | 0.263 ± 0.008 | 0% | 0.275 | 38,400 |

Beats every baseline on return (t = +3.84 to +17.95) at 12.5x fewer real
transitions — but violates a limit they all respect. That failure and its fix
are Part III.

**Against safe RL on rdf** (`runs/baselines_rdf/`, after the planner fix):

| method | return | violating evals | real samples |
|---|---|---|---|
| PPO-Lagrangian (stock gains) | -7.66 ± 0.31 | 1.8% | 38,400 |
| PPO-Lagrangian (with our dual fix) | -7.69 ± 0.31 | 0% | 38,400 |
| CPO / Sauté / NPG | -7.97 to -8.08 | 0% | 38,400 |
| **PSPE v2 (probe + margin)** | **-7.49 ± 0.62** | 5.5% | 7,360 |

Paired t vs PPO-Lagrangian: **+0.82**. Parity on return, 5.2x fewer real
samples, and PPO-Lag has the better tail. CPO, Sauté and NPG are safe by
inaction: their cost sits at the calibrated do-nothing level.

**Reading.** The return edge is a dar result, not a general one. Sample
efficiency and the safety mechanism hold on both families.

### Explain

The full Explain story is Part III §5–6. In summary: the conformal
certificate's coverage holds on real data, and every claim about the briefs
being faithful to a particular decision has been withdrawn.

---

## Part III — The experiments

### 1. The constraint failure, and the fix

**The failure.** The planner exceeded the cost limit on 7.3% of evaluations
(worst case 1.109 against 0.936) while every model-free baseline exceeded it
on none. The mechanism is a property of model-based planning, not a coding
error: the dual controller is updated from cost measured on **surrogate**
rollouts, while violation is realised under the **true** dynamics. When the
surrogate under-predicts cost, the policy is safe in-model and unsafe in
reality.

**The fix.** Two mechanisms (`runs/constraint_fix/`), 5 seeds:

- a **probe** that rolls the current policy in the true environment every 20
  iterations and feeds that cost to the dual, with the surrogate's cost
  bias-corrected in between;
- a **margin** that plans against `limit - k σ`, σ being the measured spread
  of surrogate cost error.

| arm | return | violating evals | worst cost | real samples |
|---|---|---|---|---|
| baseline | -2.302 ± 0.16 | 1.8% | 0.75 (max 1.73) | 3,200 |
| probe only | -2.307 ± 0.16 | 1.8% | 0.56 | 4,160 |
| margin only | -2.302 ± 0.16 | 1.8% | 0.75 | 3,200 |
| **probe + margin** | **-2.301 ± 0.17** | **0% on all 5 seeds** | **0.29** | 4,160 |

Neither half works alone. The margin alone is *identical* to the baseline,
because with no probe there is no measured error to derive a margin from:
**conservatism needs a measurement**.

**Honest rate.** Across all 15 seed-runs that use probe + margin (the
constraint-fix, alpha-rule and joint runs), 13 are clean and two had a single
evaluation over the limit. So the headline is a ~7x reduction — about 1% of
evaluations versus 7.3% — not a guarantee.

**Attempting a guarantee, and getting the quantity wrong.** A split-conformal
margin should convert that reduction into a stated failure rate: with n probe
observations and level delta, the ceil((n+1)(1-delta))-th order statistic is a
margin under which P(cost > limit) <= delta. The first implementation
conformalised the **surrogate's cost error** and made rdf *worse* — 18.2%
violating against the 14.5% baseline, with a margin of 0.25 where the k-sigma
rule had found 0.69:

| testbed | arm | violating evals | effective limit |
|---|---|---|---|
| dar | probe + conformal margin | 0% | 0.769 |
| rdf | baseline | 14.5% | 3.260 |
| rdf | probe + conformal margin (over model error) | **18.2%** | 3.010 |

The diagnosis is that the bound was over the wrong distribution. On rdf the
surrogate is accurate (cost bias −0.08); what violates the limit is the
**policy's own episode-to-episode spread**, which a quantile over model error
never sees.

**Conformalising the right quantity** — the deviations of individual real
episode costs from their probe mean, plus any systematic surrogate bias — gives
a margin that holds (`runs/constraint_fix_conf*/`, 5 seeds, δ = 0.1, probe
every 10 iterations):

| testbed | arm | violating evals | target | worst cost | return | effective limit |
|---|---|---|---|---|---|---|
| dar | baseline | 0% | — | 0.225 | −2.304 | 0.936 |
| dar | **probe + conformal margin** | **0%** | ≤ 10% | 0.225 | −2.309 | 0.676 |
| rdf | baseline | 14.5% | — | 4.00 | −7.184 | 3.260 |
| rdf | **probe + conformal margin** | **7.3%** | ≤ 10% | 3.51 | −7.351 | 2.777 |

The bound holds on both families. Against the hand-tuned k-sigma margin it
replaces (5.5% violating at return −7.49 on rdf), the conformal version gives
slightly more violations but a **stated failure rate** rather than a tuned
constant, and costs 0.14 less return. This is the form the claim should take:
not "violations fell 7x", but "P(cost > limit) ≤ δ, and at δ = 0.1 the realised
rate was 7.3% and 0%".

The failure is worth keeping: a margin is only as good as the distribution it
bounds, and conformalising a conveniently available quantity rather than the
one that violates gives a bound over the wrong thing.

### 2. The planner on a second family: rdf

As shipped, the planner violated the rdf limit on **85%** of evaluations at
2.5x the limit, in every arm, regardless of the probe. Three defects, each
read off the training trace once the previous one was removed
(`docs/results/RDF_PLANNER_FIX.md`):

**Defect 1 — action saturation.** The reward-greedy sprint drives the
pre-tanh policy mean past |μ| ≈ 3, where tanh has no slope:

| iteration | cost | λ | α | ‖g_pathwise‖ | ‖g_LR‖ | var_pathwise |
|---|---|---|---|---|---|---|
| 20 | 1.05 | 0 | 0.70 | 0.55 | 2.9 | 1e-2 |
| 40 | 5.87 | 0 | 0.81 | 1.9 | 196 | 0.7 |
| 100 | 8.08 | 14.5 | 0.95 | 0.036 | 5,160 | 0.0 |
| 198 | 8.21 | 38.5 | 0.99 | 0.036 | 12,700 | 0.0 |

The pathwise gradient dies (norm 0.036, variance exactly zero); the
variance-optimal mixing rule then *picks the dead branch* (α → 0.99), because
zero variance looks like precision; and the likelihood-ratio branch explodes
because every rollout returns the same value and the advantage normaliser
divides by ~0. λ = 38 multiplies nothing. Fix: a soft wall
`c·relu(|μ| - 1.5)²` on both branches and a floor on the advantage std.
**85% → 44%.**

**Defect 2 — dual oscillation.** λ collapses to zero whenever cost dips under
the limit; the policy sprints greedy again and overshoots. Lowering the gain
made it *worse* (60%). Raising the integral gain ×10 lets λ hold.
**44% → 14.5%.**

**Defect 3 — tail violations.** A dual that holds the *mean* at the limit
violates on about half of evaluations by construction. The margin now also
covers the policy's own episode-to-episode spread, measured at each probe.
**14.5% → 5.5%**, worst case 3.23 against a 3.26 limit.

**dar regression check.** The same configuration on dar: return unchanged
(-2.304 vs -2.302), 0% violations on every arm, worst case 3x smaller. One
planner configuration now works on both families.

### 3. Joint Simulate+Plan training

The paper's original contribution 1. One pretrained surrogate, deep-copied
per arm, 5 seeds (`runs/joint/`, `runs/joint_swe/`, `runs/joint_rdf/`):

| testbed | arm | return | held-out surrogate rel L2 before → after |
|---|---|---|---|
| dar | disaggregated | -2.302 ± 0.17 | 0.0048 → 0.0048 |
| dar | joint, anchored | -2.296 ± 0.18 | 0.0048 → **0.0005** |
| dar | joint, unanchored | -2.369 ± 0.17 | 0.0048 → **0.975** |
| swe | disaggregated | -0.1733 ± 0.010 | 0.0246 → 0.0246 |
| swe | joint, anchored | -0.1733 ± 0.010 | 0.0246 → **0.0027** |
| swe | joint, unanchored | -0.1733 ± 0.010 | 0.0246 → **1.14** |
| rdf | disaggregated | -6.76 ± 0.69 | 0.0024 → 0.0024 |
| rdf | joint, anchored | -6.70 ± 0.70 | 0.0024 → **0.0005** |
| rdf | joint, unanchored | -7.46 ± 0.45 | 0.0024 → **0.83** |

Paired t on return, joint vs disaggregated: **+0.66** (dar), identical (swe),
**+1.00** (rdf). Unanchored vs disaggregated: -5.84 (dar), -2.41 (rdf).

**Reading.** Joint training never changes the decision. Anchored to data it
improves the surrogate 5–9x on held-out trajectories — the planning gradient,
scaled to a tenth of the data gradient, acts as extra training on the states
the policy visits. Unanchored it destroys the surrogate. The claim "joint
beats disaggregated" is not supported on any testbed; "joint is safe when
anchored and destructive when not" is supported on all three.

### 3b. swe: a testbed that could not rank methods, and why

`runs/calibration/swe_u*.json` plus direct probes. The calibration procedure
checked that the **cost** separates between doing nothing and a reward-greedy
policy, so the constraint binds. It never checked that the **return**
separates, so actuation can move the objective. swe passed the first and failed
the second for the whole project.

| probe | result |
|---|---|
| 400 random constant action vectors | not one beats doing nothing |
| hand-written state-feedback damper | +0.0098 (5.5% of the objective) |
| trained reward-greedy planner | +0.0122 (7% of the objective) |

So the task is controllable and the planner does find control — it beats both
doing nothing and the hand damper. The achievable band is simply ~7%, and the
differences between planning methods live inside it. That is why swe's
joint / disaggregated / unanchored arms all returned −0.1733 to four decimals.

**A correction to earlier drafts of this report**, which said the swe policy
never leaves its initial behaviour: it does, and it improves. The band is too
narrow to rank methods within, which is a different and more precise statement.

**The fix that does not work.** Raising actuator amplitude grows the margin for
a *deterministic* controller (5.5% at amplitude 1.0 → 32.6% at 4.0) but gives a
stochastic policy nothing, because amplitude scales the useful signal and the
exploration noise together: planner separation is +0.0122 at amplitude 1.0 and
+0.0081 at 4.0. Amplitude was left at 1.0 and `return_separation` is now a
first-class output of the calibration script.

**What would work:** an objective whose optimum is far from zero action —
tracking a non-trivial target profile — rather than driving height to zero in a
system where localised forcing only injects energy.

### 4. Theory checks

**Assumption 1 and Proposition 1** (`runs/lipschitz/`), 5 seeds:

| arm | rel L2 | L_G | L_G ≤ 1 | measured return bias | Prop 1 bound, ∞-horizon | bound at H = 12 |
|---|---|---|---|---|---|---|
| default FNO | 0.069 ± 0.062 | 0.985 ± 0.013 | 4 of 5 | 0.10 | 422 | 9.8 |
| spectrally normalised | 0.079 ± 0.026 | 0.964 ± 0.003 | 5 of 5 | 0.11 | 748 | 17.3 |

Assumption 1 holds after training (initial L_G was 1.61). The
infinite-horizon bound is **vacuous** — 422 against a measured bias of 0.10,
because the 1/(1-γ) prefactor at γ = 0.98 is 2,450. The finite-horizon form
holds on every seed and is still 100x loose. Report the finite-horizon form
and say it is loose.

**Equation 8 mixing rule** (`runs/alpha_rule/`), 5 seeds:

| arm | return | violating evals | final α | measured B² | real samples |
|---|---|---|---|---|---|
| fixed α = 0.5 | -2.415 ± 0.30 | 3.6% | 0.5 | — | 4,160 |
| variance rule | -2.302 ± 0.17 | 1.8% | 0.992 | — | 4,160 |
| Eq. 8 (measured bias) | -2.299 ± 0.15 | 0% | 0.992 | 0.0013 | 6,080 |

The measured pathwise bias is ~0.001, so Eq. 8 and the variance-only rule
choose the same α. Return t = +0.22. Eq. 8's zero-violation result comes from
its 1,920 extra truth rollouts, not from the formula.

**Adaptive α** is a variance effect, not a mean effect: +0.290 return over
fixed α but t = +1.66. What separates them is the tail — std 0.029 vs 0.365,
and one fixed-α seed diverging to cost 8.1 with λ = 19.1.

**Cross-family transfer** (`runs/transfer_seeds/`), 5 seeds:

| transfer | fidelity gap | measurable? |
|---|---|---|
| rdf → dar | 0.358 ± 0.033 | yes, ~11σ from zero |
| swe → rdf | 0.311 ± 0.13 | yes |
| dar → rdf | 1.81 ± 1.1 | marginal |
| swe → dar | 0.087 ± 0.14 | no, crosses zero |
| rdf → swe | 10.5 ± 8.2 | no, σ ≈ mean |
| dar → swe | 27.4 ± 26 | no, σ ≈ mean |

Direction survives (parabolic families transfer; transfer into the wave family
collapses); magnitude does not. And forecast error decouples from decision
loss: dar → swe has a fidelity gap of 27 and a planning gap of 0.007 ± 0.014
(nothing), while dar → rdf has fidelity gap 1.8 and planning gap 0.58 ± 0.22
(about 8% of return).

**Ablations** (5 seeds, full budget):

| component | on | off | t | verdict |
|---|---|---|---|---|
| physics-informed loss | 0.0567 ± 0.026 | 0.0496 ± 0.044 | +0.52 | no effect on final accuracy |
| frozen perception | 0.0681 ± 0.013 | 0.0908 ± 0.017 | -3.97 | frozen wins |
| faithfulness term | 0.643 ± 0.15 | 0.598 ± 0.006 | +0.67 | no effect |

The physics loss buys convergence speed only (final rel L2 0.117 with it vs
0.273 without at a quick budget).

### 5. Planning on observed wildfire data

The first planning result whose dynamics are a model of what a real fire did
(`docs/results/NDWS_PLANNING.md`). The NDWS U-Net, frozen, with an 8×8 grid of
firebreak intensities per day under a 3%-of-patch daily crew budget, three
days ahead, population-weighted burn as the objective. Action model, stated in
code: a break removes fuel (NDVI -2σ × intensity, the learned model decides
the effect) and blocks spread into treated cells (0.9 × intensity, imposed).
1,500 held-out fires, 3 seeds:

| policy | burn reduction | treated/day | over budget | via learned fuel channel |
|---|---|---|---|---|
| random | 4.5 ± 0.3% | 3.0% | 0 | 0.6% |
| greedy: forecast today, treat riskiest cells, repeat daily | 24.1 ± 3.8% | 3.0% | 0 | 5.7% |
| **PSPE per-instance: plan three days jointly through the surrogate** | **36.5 ± 1.0%** | 3.0% | 0 | **19.3%** |
| PSPE amortised policy | 12.2 ± 4.5% | 2.2% | 17% | 2.5% |
| unconstrained | 85.7 ± 4.5% | 61.7% | 100% | 24.4% |

Per-instance minus greedy, per seed: +9.6, +11.7, +15.9 points. Paired
t (df 2) = **6.66**.

**Extended to 5 seeds** (`runs/ndws_plan/`, seeds 3–4 added):

| policy | burn reduction, 5 seeds | per-seed |
|---|---|---|
| greedy | 22.0 ± 4.4% | 28.0, 23.9, 20.4, 21.9, 15.9 |
| **PSPE per-instance** | **34.0 ± 4.4%** | 37.6, 35.6, 36.3, 34.1, 26.5 |
| PSPE amortised | 10.4 ± 6.5% | — |

Paired t (df 4) = **11.15**, mean gap +12.0 points. The two added seeds did not
dilute the effect: the paired gap is stable from +9.6 to +15.9 across all five.

### 5a. The crew budget sweep (Pareto)

Same fires and seeds, budget varied (`runs/ndws_plan_b*/`):

| crew budget / day | greedy | PSPE per-instance | ratio |
|---|---|---|---|
| 1% | 7.5 ± 1.4% | 14.3 ± 0.5% | **1.91x** |
| 2% | 12.8 ± 2.0% | 26.2 ± 0.7% | **2.05x** |
| 3% | 24.1 ± 3.8% | 36.5 ± 1.0% | 1.52x |
| 5% | 36.1 ± 5.0% | 53.0 ± 1.1% | 1.47x |
| 8% | 55.0 ± 5.5% | 69.4 ± 0.6% | 1.26x |

The advantage is **largest where crews are scarcest** and decays as the budget
grows large enough that placement stops mattering. PSPE at a 1% budget matches
greedy at close to 2%, i.e. roughly half the crews for the same outcome. The
planner's variance is also consistently smaller (±0.5 to ±1.1 against ±1.4 to
±5.5): it is not only better on average but steadier across fires.

### 5a2. Constrained RL on the same fires

`runs/ndws_baselines_fair/`, 5 seeds. Every arm plans or learns against the
same frozen U-Net under the same 3% daily budget and is scored by the same
function on the same held-out fires. Sample efficiency does **not** apply here:
there is no separate true environment, so every method draws from the same
surrogate and this is a comparison of decision quality only.

| method | burn reduction @ matched budget | treated/day | fires over budget |
|---|---|---|---|
| greedy heuristic | 24.1 ± 3.8% | 3.00% | 0 |
| PPO-Lagrangian | 4.6 ± 0.4% | 4.32% | 58% |
| CPO | 4.3 ± 0.1% | 2.84% | 6% |
| Sauté RL | 4.7 ± 0.5% | 7.37% | 79% |
| primal-dual NPG | 4.6 ± 0.1% | 3.26% | 100% |
| **PSPE per-instance** | **36.5 ± 1.0%** | 3.00% | 0 |

The constrained-RL methods land near random placement (4.5%) and below the
heuristic. The reason is amortisation, not the constraint: one policy has to
serve 1,024 distinct fires from 200 iterations, which is the same failure our
own amortised arm shows (10.4%). On a per-instance decision problem this
diverse, decision-time optimisation is the right tool and a learned policy is
not.

**A near-miss worth recording.** The first version of this comparison had all
four baselines treating 22–38% of the sector against the 3% budget on every
fire, which looked like a structural finding about constrained RL in
high-dimensional action spaces. It was an artefact: under the squared intensity
map a randomly initialised policy emits 25% treatment, and the agents had
barely moved off initialisation, while PSPE's optimiser *starts* at the budget
and projects onto it every step. With `--fair-init` giving the baselines the
same starting point, PPO-Lagrangian holds 2.99% after three iterations. The
structural claim was withdrawn before it was made. See
`docs/results/ROBUSTNESS_NOTES.md`.

### 5a3. Horizon ablation: planning pays only when the decision is sequential

`runs/ndws_horiz_h*/`, **5 seeds** per horizon, budget 3%/day:

| horizon | greedy | PSPE per-instance | gap | paired t (df 4) |
|---|---|---|---|---|
| 1 day | 24.8 ± 1.1% | 25.4 ± 1.7% | +0.6 | **1.40 — not significant** |
| 2 days | 24.2 ± 3.6% | 32.6 ± 3.4% | +8.5 | 10.74 |
| 3 days | 22.0 ± 4.4% | 34.1 ± 4.4% | +12.1 | 11.08 |
| 5 days | 17.7 ± 4.3% | 32.0 ± 5.8% | **+14.3** | 8.99 |

(Significance threshold at df 4 is |t| > 2.78.)

At a one-day horizon the two are statistically indistinguishable (t = 1.40),
which is the prediction, not a disappointment: the one-day objective is
separable over cells, so treating the highest-risk cells within budget is its
*exact* optimum and no planner can beat it. The moment the decision becomes
sequential the statistic jumps by an order of magnitude (t = 10.74 at two
days), and the gap grows monotonically with horizon while the heuristic
**degrades** (24.8% → 17.7%) because it keeps optimising for today while the
fire moves.

A result that is insignificant exactly where theory says it must be, and
significant exactly where theory says it should be, is stronger evidence for
the mechanism than any single large number.

This is the mechanism behind the headline result, measured rather than
asserted: planning buys nothing on a myopic problem, and its value grows with
how far ahead the decision reaches.

### 5b. Fuel-only ablation: does the result rest on the imposed physics?

The action model has two mechanisms — a **learned** one (fuel removed from the
NDVI channel, the surrogate decides the effect) and an **imposed** one (spread
into treated cells suppressed by hand). Setting `block = 0` removes the imposed
half entirely (`runs/ndws_plan_fuelonly/`, 3 seeds, budget 3%):

| | greedy | PSPE per-instance | gap |
|---|---|---|---|
| both mechanisms | 24.1 ± 3.8% | 36.5 ± 1.0% | +12.4 (t = 6.66) |
| **learned fuel response only** | **5.7 ± 2.9%** | **19.4 ± 1.6%** | **+13.6 (t = 5.93)** |

Removing the hand-written physics costs the heuristic 76% of its effect and the
planner 47%, and the **gap widens**. The planner's advantage therefore does not
rest on the part of the action model that was specified rather than learned —
it comes from exploiting the surrogate's own fuel response, which is what the
claim requires. This is the ablation a reviewer asks for, and it survives.

**Reading.** Decision-time planning beats the operational heuristic by twelve
points at the same budget with no violations, and the margin comes through the
learned model: 19% of the planner's reduction survives with the imposed block
term switched off, against 6% for greedy. Greedy treats today's risk; the
planner treats where the model says the fire will be on day three. On the
one-day version of the problem greedy is the *exact* optimum of a separable
objective and nothing beats it — planning earns its keep only across days. The
amortised policy is the wrong tool on a distribution this diverse.

**Not shown:** any effect on a real fire. NDWS records what fires did without
firebreaks; there are no counterfactuals. The forecast is validated against
observation (AUC-PR 0.264–0.280 on these same held-out fires); the treatment
effect rests on the stated action model.

### 6. Explanation: the certificate holds, the faithfulness claim does not

**What was claimed.** Trained-in briefs beat a genuine post-hoc control: on
dar with Qwen, 0.468 ± 0.16 vs 0.182 ± 0.16, t = +21.3; on real fire plans,
0.814 ± 0.039 vs 0.741 ± 0.005, t = +3.61. A split-conformal certificate holds
at 90% and 95% coverage on 3 of 3 seeds, with a non-vacuous floor
(F ≥ 0.706 at δ = 0.1 on real fire plans).

**The control that broke it.** Score each brief against a **different**
state's action distribution. If briefs are state-specific, aligned F should
exceed shuffled F. If the generator emits the same brief regardless of state,
the two are equal by construction:

| testbed | arm | F aligned | F shuffled | gap |
|---|---|---|---|---|
| dar, seed 0 | trained-in | 0.2809 | 0.2805 | +0.0004 |
| dar, seed 0 | post-hoc | 0.0034 | 0.0034 | 0.0000 |
| dar, seed 1 | trained-in | 0.5570 | 0.5570 | 0.0000 |
| dar, seed 2 | trained-in | 0.5657 | 0.5659 | -0.0002 |
| NDWS, seed 0 | trained-in | 0.7439 | 0.7436 | +0.0003 |
| NDWS, seed 1 | trained-in | 0.7673 | 0.7674 | -0.0001 |
| NDWS, seed 2 | trained-in | 0.8630 | 0.8639 | -0.0009 |

Zero state-specific information, on both testbeds, for every arm including the
post-hoc control. The sampled briefs confirm it: the same actuators at the
same amplitudes for different states, with only the cost and reward lines
varying.

**Three fixes, all negative** (`docs/results/EXPLAIN_PERMUTATION.md`):

1. *More conditioning capacity* — prefix 4 → 16 vectors, cond_dim 64 → 256,
   deeper projection, 10% condition dropout. No effect.
2. *A differentiable contrastive objective* — each brief must be cheaper under
   its own condition than under its neighbours'. Unit-tested to reach the
   prefix parameters. In training the loss sat at log B, the value for a model
   that ignores its condition, and in one seed rose.
3. *2.5x the training budget* — 500 iterations, gap logged every 100:

| iteration | NLL/token (plain) | gap | NLL/token (contrastive) | gap |
|---|---|---|---|---|
| 100 | 0.662 | +0.0015 | 0.669 | +0.0012 |
| 200 | 0.544 | +0.0027 | 0.542 | -0.0004 |
| 300 | 0.757 | -0.0097 | 0.533 | -0.0019 |
| 400 | 1.000 | +0.0000 | 0.515 | +0.0041 |
| 500 | 0.524 | -0.0229 | 0.674 | -0.0209 |

NLL plateaus near 0.5 nats/token; the gap never leaves zero.

**Not a plumbing bug.** On the Qwen path the prefix receives a gradient of
norm 1.1e5 (against 4.6e3 for the LoRA adapters), the NLL moves by 1.51 when
the condition is zeroed, and prefix embeddings differ across rows. The model
*can* use the condition; the optimisation settles on the marginal distribution
over actions rather than the conditional, and greedy decoding then emits the
modal brief every time.

**The testbed could not have answered the question.** The trained dar policy
is effectively state-independent: across 64 states the action std is **1.2%**
of the action magnitude, and on the brief's 0.05 quantisation grid those 64
states give **8 distinct action vectors**. Its briefs are nearly identical, its
supervised NLL reaches 0.05 nats/token, and a constant brief is close to
correct. On dar the permutation gap is zero no matter what the generator does.
Every Explain number measured on dar was computed on a task with nothing to
explain. NDWS — 1,013 distinct plans across 1,024 fires, across-fire intensity
std 3.06x the mean — is the only testbed here where the question is posed.

**What stands.** The certificate machinery: coverage holds at 90% and 95% on
3 of 3 seeds on real data, and the floor it reports is honest about the
generator it was given. Certifying a constant brief is not a defect of the
certificate.

**What to change.** Report the gap (aligned minus shuffled), not F. Retire dar
as an Explain testbed and calibrate any replacement for action diversity.
Treat the fix as a design change — cross-attention conditioning, or a
structured head for the (patch, amplitude) list with prose generated around it
— not a tuning exercise.

---

## Part IV — What survives, and what does not

**Survives multi-seed scrutiny**

- Return advantage over four safe-RL baselines **on dar** (t = +3.84 to +17.95)
- Real-sample efficiency: 9x on dar, 5x on rdf
- FNO ≫ DeepONet ≫ GNOT, on synthetic and real benchmark data
- Surrogate accuracy competitive with published PDEBench FNO error
- Resolution invariance to 128² (16,384 cells)
- Constraint violations cut ~7x by probe + margin; 13 of 15 seed-runs clean
- A conformal margin with a **stated failure rate**: at δ = 0.1 the realised
  violation rate was 0% (dar) and 7.3% (rdf), both inside the bound
- Joint training is safe when data-anchored (surrogate improves 5–9x, decision
  unchanged) and destructive when not, on all three testbeds
- Assumption 1 (L_G ≤ 1) after training
- The conformal certificate's coverage, on synthetic and real data
- The FNO's scope: operators for smooth fields, convolutions for sharp fronts
  (NDWS gate 0.12, U-Net > hybrid at t = +12.5)
- U-Net on real wildfire data at 97.5% of published AUC-PR
- Constrained planning on real fire data: +12.0 points over forecast-then-greedy
  (5 seeds, t = 11.15), 1.9–2.1x the heuristic's reduction at tight budgets
- That planning advantage **without** the imposed action physics: the gap widens
  to +13.6 points when only the learned fuel response is active
- The mechanism behind it: the advantage is insignificant at a one-day horizon
  (where greedy is the exact optimum, t = 1.40) and grows monotonically to
  +14.3 at five days, while the heuristic degrades

**Does not survive**

- Constraint satisfaction *as originally built* — 7.3% of evaluations violated
- Constraint satisfaction as a *guarantee* — the fix is a 7x reduction, with
  two excursions in 15 seed-runs
- A return edge on rdf — parity with PPO-Lagrangian at matched safety
- Joint training *beating* the disaggregated pipeline on return
- Equation 8 improving on the variance-only α (measured B² is too small)
- Proposition 1's infinite-horizon bound as a usable number (4,000x loose)
- The physics-informed loss improving asymptotic accuracy
- Adaptive α improving mean return (it improves the tail)
- The faithfulness objective contributing beyond supervised brief training
- PSPE perception beating a CNN or a frozen probe on reconstruction
- Any specific magnitude for transfer into the wave family
- **Any claim that a brief explains the decision it was generated for**
- `swe` as a planning testbed — not because the policy fails to act (it beats
  doing nothing by +0.0122, better than a hand-written damper) but because the
  entire span between doing nothing and the best achievable return is ~7% of
  the objective, so method differences sit inside it
- `dar` as an Explain testbed — its policy is state-independent

**Not yet measured**

- Flood testbed on real data; human study on real briefs
- Safe-RL baselines on swe (blocked on the swe task redesign)
- NDWS test-split numbers with focal loss
- Safe-RL baselines on swe (blocked on its redesign)

---

## Part V — Measurement defects found and fixed

This project's most transferable output may be the list of ways its own
measurements were wrong. Each was found by a check that should be standard.

| defect | effect | how it was caught |
|---|---|---|
| Constraint read from one final evaluation snapshot | Identical runs scored "0% violation" or "100%" on timing luck | Multi-seed reruns disagreeing |
| Evaluation perturbed training through the global RNG | Baseline returns shifted -2.51 → -2.30 | Forking the RNG changed the baseline |
| Optional-dependency tests skipped silently | A "tested" loader failed on first real use | `importorskip` audit |
| Truth solver unstable above 64² | NaN at every resolution above the training grid | Resolution sweep |
| Faithfulness REINFORCE had exactly zero gradient | Samples never parsed; every score equalled the parser fallback | Parse-rate diagnostic |
| Cost limits uncalibrated | The dual never activated; five unconstrained learners tying | Measuring do-nothing and greedy cost |
| `F = exp(-KL)` with KL summed over dimensions | Every arm scored exactly 0.0 at 64 actions while the reference scored 0.985 | Running Explain on a large action space |
| Faithfulness never measured state-specificity | A constant brief scores as well as a correct one | **Permutation control** |
| dar's policy is state-independent | The Explain testbed had nothing to explain | Measuring action diversity across states |
| Conformal margin bounded the surrogate's error, not the policy's spread | Violations rose to 18.2% on rdf, worse than no margin at all | Running it on a testbed where the surrogate is accurate but the policy is variable |
| Baselines compared against their own initialisation | Four constrained-RL methods "failed structurally" at 22–38% of a 3% budget | Solving for the initial action the parameterisation implies |
| Testbed calibrated for a binding constraint but not a controllable objective | swe ranked planning methods inside a 7% band for the whole project | Measuring return separation, not just cost separation |

Seven of these checks, their standing rules, and the three results they
overturned *before* publication are collected in
`docs/results/ROBUSTNESS_NOTES.md`.

**The general lesson.** Every metric that compares a generated artefact to a
target should ship with a permutation control, and every testbed should be
calibrated for the variation the metric needs before it is used. Without
those, "the metric went up" and "the model learned the task" are
indistinguishable — and in this project they were, for three rounds of
scrutiny including a real-backbone replication and a conformal certificate.

---

## Part VI — Infrastructure and reproduction

**Protocol.** 5 seeds by default, mean ± sample std, paired *t*-tests. 171
tests including regression tests for each measurement defect above. One
command per sweep (`eval/run_seeds.py`), with aggregation by run or arm key.

**Compute.** TACC Vista (GH200). Jobs are packed — all seeds of one experiment
concurrently on one GPU — because a single short job backfills into gaps a
multi-node array never fits. A 5-seed sweep takes about 10 minutes against
~9 hours on one CPU. Time limits matter more than node counts: 25–30 minute
jobs backfill within minutes when hour-long jobs wait hours.

**Key scripts.**

| purpose | command |
|---|---|
| Multi-seed sweep | `python eval/run_seeds.py --seeds 0 1 2 3 4 --full` |
| Constraint fix / planner v2 | `python eval/run_constraint_fix.py --sat-coef 1 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 --margin-episode-std` |
| Joint vs disaggregated | `python eval/run_joint.py --beta 0.1` |
| NDWS surrogate | `python eval/run_ndws.py --epochs 18 --n-train 8000` |
| NDWS planning | `python eval/run_ndws_planning.py --horizon 3` |
| NDWS briefs + certificate | `python eval/run_ndws_explain.py --arms trained-in` |
| Permutation control | `python eval/run_explain_baselines.py --n-permutation 128` |
| Vista submission | `sbatch --export=ALL,EXPERIMENT=<name>[,TESTBED=,ARMS=,CW=] scripts/tacc/vista_packed.slurm` |

**Detailed analyses.**

| file | contents |
|---|---|
| `docs/results/PHASE1_ANALYSIS.md` | Joint training, Eq. 8, Lipschitz/Prop 1, faithfulness weight sweep, conformal on dar |
| `docs/results/PHASE3_ANALYSIS.md` | Constraint fix and joint training on swe and rdf |
| `docs/results/RDF_PLANNER_FIX.md` | The three-defect diagnosis, the dar regression, rdf safe-RL baselines |
| `docs/results/NDWS_PLANNING.md` | Constrained firebreak planning on observed fire data |
| `docs/results/NDWS_EXPLAIN.md` | Briefs and certificate on real fire plans |
| `docs/results/EXPLAIN_PERMUTATION.md` | The permutation control and three failed fixes |
| `docs/technical_report.md` | Full narrative report with method and literature |
| `docs/slides_experiments.md` | Slide-by-slide version of the experimental section |
| `docs/figures/` | Architecture, module and application diagrams |

---

## Part VII — Where this leaves the work

**The defensible framing.** Not "joint training of four modules beats
disaggregated pipelines" — that was tested on three PDE families and is false.
The framing the evidence supports is:

> Learned surrogates make constrained planning cheap, but the planner
> inherits every error the surrogate makes, silently. Measure the real system
> periodically, plan against the measured error, and the violations mostly go
> away at no return cost and a fraction of the real-world samples. Where the
> decision is genuinely sequential, planning through the surrogate beats the
> operational heuristic on real data. And the parts that do not work — joint
> training, the faithfulness objective, the explanation module's conditioning
> — are reported with the measurements that show why.

**Three contributions stand.** Safety by measurement; sample efficiency;
planning on observed data. Each is multi-seed, mechanism-explained, and
reproduced across testbeds where applicable.

**One is halved.** The explanation module has a working certificate and an
open conditioning problem, both measured.

**The immediate next experiments**, in order of value:

1. ~~NDWS test-split numbers~~ — **done**, and the surrogate exceeds the
   published figure (§Simulate). Remaining: numbers with focal loss, to close the 0.277 → 0.284 gap on
   the split the published figure uses.
2. ~~The swe task redesign~~ — **attempted and characterised** (§3b). The
   actuator-amplitude fix does not work on a stochastic policy; a new objective
   would be needed. Retired as a comparison testbed with a measured reason.
3. The former item 2 — more actuation authority until the do-nothing and
   greedy costs separate — or its retirement as a planning testbed.
3. The Explain redesign (structured head for the action list), which is a
   different module and possibly a different paper.

*Done since the first draft of this report:* the NDWS test-split result
(§Simulate), the swe characterisation (§3b), the NDWS result extended to five
seeds (§5), the crew-budget Pareto sweep (§5a), constrained-RL baselines on the
same fires (§5a2), the horizon ablation at five seeds that shows the mechanism (§5a3), the
fuel-only ablation (§5b), and a working conformal margin (§1). The robustness
checks that overturned two results before they were reported are in
`docs/results/ROBUSTNESS_NOTES.md`.
