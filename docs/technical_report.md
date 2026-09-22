# PSPE: Perceive–Simulate–Plan–Explain

**A technical report on constrained intervention design for PDE-governed systems**

Status: 2026-09-16. Every number here comes from a run in this repository. Where
a claim did not survive its own multi-seed rerun, that is stated in the results
rather than omitted — several did not, and those are among the more useful
findings.

---

## 1. What this repository does

PSPE is a research stack for deciding **where and how hard to intervene** in a
spatially extended physical system that evolves under a partial differential
equation, when the intervention is subject to hard constraints and the decision
has to be explainable afterwards.

It is built as four modules that form one loop:

| module | job | implementation |
|---|---|---|
| **Perceive** | indirect observation (imagery) → physical field | frozen vision backbone + LoRA adapters + field decoder |
| **Simulate** | differentiable reduced-order forward model | in-tree FNO, with DeepONet and GNOT as alternatives |
| **Plan** | constrained policy over actuator fields | hybrid pathwise/likelihood gradient + PID-Lagrangian dual |
| **Explain** | natural-language brief for each decision | LM + LoRA, frozen parser, faithfulness objective |

The loop matters more than any module: perception feeds the state estimate the
surrogate rolls forward, the surrogate provides the differentiable dynamics the
planner exploits, and the explainer is trained against the planner's own action
distribution rather than describing it after the fact.

## 2. The problem

Consider a hazard field — a pollutant plume, a flood wave, a spreading front —
governed by a PDE on a 2D domain. A decision-maker can apply a limited,
spatially distributed intervention at each step. Three things make this hard,
and all three are what the design responds to:

1. **The state is not observed directly.** What exists is imagery or sparse
   sensors; the field driving the dynamics has to be inferred.
2. **The action is constrained, and the constraint is not soft.** There is a
   budget on total intervention, a pointwise safety ceiling on exposure, and —
   in the formulation, though off by default in the comparisons — a
   distributional equity requirement that harm not concentrate in one
   sub-region.
3. **An unexplained intervention is not actionable.** A recommendation to flood
   a specific district or ground a specific area needs a defensible account of
   why, in terms a domain expert can check.

Formally this is a constrained partially observed MDP (Altman, 1999): maximise
expected return subject to expected cost constraints, with the state inferred
from observations. The planner acts inside a *learned* model of the dynamics,
which is what makes it sample-efficient — and, as the results show, is also the
source of its safety failure mode.

## 3. Literature and what it leaves open

**Neural operators.** Li et al. (ICLR 2021) introduced the Fourier Neural
Operator, which learns mappings between function spaces and is therefore
discretization-invariant in principle. Lu et al. (Nat. Mach. Intell. 2021)
proposed DeepONet from a different approximation-theoretic direction, and Hao et
al. (ICML 2023) generalised to transformer-based operator learning with GNOT.
These give fast surrogates but say nothing about *acting* through them under
constraints.

**Benchmarks.** Takamoto et al. (NeurIPS 2022 D&B) released PDEBench with
published baseline errors across PDE families. Without it, a surrogate can only
be scored against its author's own solver, which cannot detect a bias shared
between model and reference. This report uses it for exactly that reason.

**Constrained RL.** Achiam et al. (ICML 2017) gave CPO's trust-region approach
to constrained policy improvement; Stooke et al. (ICML 2020) reframed the
Lagrangian multiplier as a PID controller to damp the oscillation plain dual
ascent produces; Sootla et al. (ICML 2022) enforced constraints through state
augmentation in Sauté RL. All are model-free: they establish safety through
direct environment interaction, at a sample cost this work is trying to avoid.

**Explanation.** Post-hoc explanation of a fixed policy is the dominant paradigm
— generate a rationale after the decision. The alternative pursued here is to
train the explanation jointly with the policy against a faithfulness objective,
so the brief is optimised to imply the policy that produced it.

**Climate and weather ML.** FourCastNet (Pathak et al., 2022), GraphCast (Lam et
al., Science 2023) and ClimaX (Nguyen et al., ICML 2023) established that
learned surrogates can match or beat numerical weather prediction at forecast
tasks, with WeatherBench (Rasp et al.) as the standard scoring ground. These are
*forecasting* systems. The open gap they leave — and this repository's target —
is the step from prediction to **constrained intervention**: not what the
atmosphere will do, but what to do about it under a budget.

## 4. Method

### 4.1 Perceive
A frozen vision backbone (`google/siglip-base-patch16-224`, or an in-tree
patch-transformer stub for offline work) with LoRA adapters (Hu et al., ICLR
2022), a convolutional field decoder, and a projection head trained with a
contrastive term against weak text descriptions. Only adapters and heads train.

### 4.2 Simulate
An in-tree FNO trained with a data term, a physics-residual term evaluated
against the same solver operators that generate the data, and a multi-step
rollout-consistency term. Gradient checkpointing is available; a channel-padded
variant lets one surrogate accept any of the three families' state arities,
which is what makes cross-family transfer measurable at all.

### 4.3 Plan
The policy is a Gaussian over actuator amplitudes. Two gradient estimators are
combined per step:

* **pathwise** — differentiate the Lagrangian objective through the surrogate;
  low variance, biased by surrogate error;
* **likelihood-ratio** — REINFORCE (Williams, 1992) with a critic baseline;
  unbiased, high variance.

They are mixed with a coefficient α adapted from the fold-wise variance ratio of
the two gradients, so α → 1 (pathwise) when the surrogate's gradients are
reliable relative to the score-function noise.

The constraint is enforced by a PID-Lagrangian dual (Stooke et al., 2020):
`λ = max(0, K_p·e + K_i·I + K_d·max(0, Δe))` where `e` is the smoothed
constraint violation. A second, independent dual handles the equity constraint
when enabled.

### 4.4 Explain
An LM with LoRA generates a brief conditioned on the state and the chosen
action. A **frozen** parser maps the brief back to an implied action
distribution, and faithfulness is `F(b) = exp(−D_KL(π ‖ π̂_b))`, bounded in
(0, 1]. The exponential form is a deliberate deviation from the linear
`1 − KL`: the linear form is unbounded below, so one catastrophic sample
dominates a REINFORCE batch.

## 5. Experimental protocol

**Multi-seed by default.** Every comparison below is 5 seeds (3 for the real-
backbone runs), reported as mean ± sample standard deviation, with paired
*t*-statistics on 4 degrees of freedom. |t| > 2.78 is p < 0.05. Runs are paired
by seed because the arms share initial conditions and data.

**Hardware.** NVIDIA GH200 nodes on TACC Vista, one seed per array task, so a
5-seed sweep costs one seed's wall-clock. A full-budget Phase 2 sweep takes ~9.5
minutes on 5 nodes against an estimated 9 hours on a laptop CPU.

**Budgets.** "Full" is 20 surrogate epochs / 200 planner iterations / 200
baseline iterations at 64² resolution, horizon 12, 9 actuators.

### Two measurement bugs that had to be fixed first

Both inflated earlier tables, and neither was visible at single-seed scale.

1. **Constraint satisfaction was read from one end-of-training evaluation.** The
   planner's cost excursions decay within an evaluation interval, so two runs
   with identical behaviour were recorded as "violation 0.0" and "violation 1.0"
   depending only on where the last snapshot landed. Runs now report
   `violating_eval_fraction` and `cost_max_over_run` across the whole run.
2. **Evaluation perturbed the training it measured.** `env.step` draws from the
   global RNG, so adding periodic evaluation to the baselines moved their mean
   return from −2.51 to −2.30. Both evaluators now fork the RNG and use a fixed
   set of evaluation initial conditions, with regression tests asserting that a
   run's result is independent of its evaluation schedule.

A third, smaller instance: the PDEBench loader's tests used
`pytest.importorskip("h5py")` while `h5py` was declared in neither
`requirements.txt` nor `pyproject.toml`, so the loader path silently *skipped*
on machines without it and the suite still reported green. It failed on first
contact with a GPU node. Now declared.

## 6. Datasets

| dataset | kind | role |
|---|---|---|
| `dar` — diffusion–advection–reaction | synthetic, in-tree solver | primary testbed; 1 state channel |
| `swe` — shallow water | synthetic, in-tree solver | wave-family testbed; 3 channels |
| `rdf` — reaction–diffusion front | synthetic, in-tree solver | second parabolic family; 2 channels |
| **PDEBench 2D shallow-water** | **independent solver, published baselines** | external validation of the surrogate (6.2 GB, DaRUS) |
| EuroSAT (Sentinel-2 → NDVI) | real satellite imagery | perception on real imagery — loader built, run pending |

The three synthetic testbeds are generated by the repository's own solvers, and
the physics-residual loss uses those same operators. That is a closed loop: it
cannot detect an error shared by solver and surrogate. PDEBench exists in this
report to break that loop.

## 7. Results

### 7.1 Surrogate accuracy on a real benchmark

512 real 2D shallow-water trajectories from PDEBench, 64², 20 epochs:

| operator | rel L2 (1-step) | rel L2 (rollout) | params | PDEBench published FNO (nRMSE) |
|---|---|---|---|---|
| **FNO** | **0.0018** | **0.0098** | 1.19 M | 0.0044 |
| DeepONet | 0.0044 | 0.0137 | 0.09 M | — |
| GNOT | 0.0056 | 0.0543 | 0.31 M | — |

The FNO's rollout error is ~2.2× the published figure and its one-step error is
below it. Because PDEBench reports normalised RMSE and this repository reports
relative L2, this is an **order-of-magnitude anchor, not a like-for-like
comparison**. Read conservatively: the surrogate is competitive on a community
benchmark, which is a materially stronger statement than agreeing with its own
solver.

It also independently reproduces the operator ranking found on synthetic data
(FNO ≫ DeepONet ≫ GNOT), which is the more robust of the two claims.

### 7.2 Constrained planning against safe-RL baselines

5 seeds, `dar`, 200 iterations, calibrated cost limit 0.936:

| method | return | episode cost | violating evals | worst cost |
|---|---|---|---|---|
| **PSPE hybrid (adaptive α)** | **−2.448 ± 0.029** | 0.412 ± 0.37 | **7.3%** | 1.109 |
| PSPE hybrid (fixed α) | −2.738 ± 0.365 | 2.752 ± 3.6 | 12.7% | 2.758 |
| PPO-Lagrangian | −2.558 ± 0.039 | 0.266 ± 0.009 | 0% | 0.297 |
| CPO | −2.541 ± 0.035 | 0.263 ± 0.011 | 0% | 0.277 |
| Sauté RL | −2.537 ± 0.024 | 0.264 ± 0.008 | 0% | 0.275 |
| Primal-dual NPG | −2.544 ± 0.026 | 0.263 ± 0.008 | 0% | 0.275 |

**Return: the advantage is real.** Paired by seed, PSPE beats every baseline —
t = +3.84 (PPO-Lag), +4.87 (CPO), +17.95 (Sauté), +4.52 (PD-NPG), df = 4. And
it does so on **12.5× fewer real environment transitions**: the planner acts
inside the surrogate, whose entire real-sample cost is the 3,072 transitions it
was fitted on, against 38,400 for each model-free baseline.

**Safety: the advantage is not free, and the paper's claim does not hold.** The
planner exceeds the constraint on 7.3% of evaluations, worst case 1.109 against
a 0.936 limit. Every baseline violates on **zero** evaluations. So part of the
return advantage is bought by running past the limit that the baselines respect.
Corollary 1 of the proposal (time-averaged violation → 0) holds on a toy CMDP
with a closed-form optimum but **not** on the PDE testbed at this budget.

The mechanism is visible in the logs and is a property of model-based planning,
not a coding error: the dual controller is updated from the cost measured on
**surrogate** rollouts, while violation is measured on the **true** dynamics.
When the surrogate under-predicts cost, the policy is safe in-model and unsafe
in reality.

**The fix, and its cost.** Two mechanisms, tested separately and together over
5 seeds (`runs/constraint_fix/`): a *probe* that rolls the current policy in the
true environment every 20 iterations and feeds that cost to the dual, with the
surrogate's cost bias-corrected in between; and a *margin* that plans against
`limit − 2σ`, σ being the measured spread of surrogate cost error.

| arm | return | violating evals | worst cost | real samples |
|---|---|---|---|---|
| baseline | −2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 (max 1.73) | 3,200 |
| probe only | −2.307 ± 0.16 | 1.8% | 0.56 ± 0.48 | 4,160 |
| margin only | −2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 | 3,200 |
| **probe + margin** | **−2.301 ± 0.17** | **0% (all 5 seeds)** | **0.29 ± 0.15** | 4,160 |

Violations are eliminated at no measurable return cost, for 960 additional real
transitions — the planner still uses 9.2× fewer real samples than the model-free
baselines. Neither mechanism works alone: probing alone leaves the excursions,
and the margin alone is *identical* to the baseline, because with no probe there
is no measured error to derive a margin from. Conservatism needs a measurement.

**Adaptive α is a variance effect, not a mean effect.** Adaptive beats fixed by
+0.290 return, but paired t = +1.66 — not significant. What separates them is
the tail: 0.029 std vs 0.365, 7.3% violating evaluations vs 12.7%, and one
fixed-α seed diverging to cost 8.1 with λ = 19.1. The defensible claim is that
adaptive α **prevents dual divergence**, not that it improves return.

### 7.3 Ablations (5 seeds, full budget)

| ablation | on | off | paired t | verdict |
|---|---|---|---|---|
| physics-informed loss (rel L2) | 0.0567 ± 0.026 | 0.0496 ± 0.044 | +0.52 | **no effect** |
| frozen vs fine-tuned perception | 0.0681 ± 0.013 | 0.0908 ± 0.017 | −3.97 | **frozen wins** |
| faithfulness loss (F(b)) | 0.643 ± 0.15 | 0.598 ± 0.006 | +0.67 | **no effect** |

**The physics-informed loss does not improve final accuracy.** At smoke scale
(32², 3 epochs) physics-on scores 0.063 against physics-off at 0.177 — a 2.8×
gap that looks decisive. At 64² for 20 epochs the gap is gone, and physics-on is
nominally worse. The honest claim is that the residual term buys **convergence
speed at small budgets**, not asymptotic accuracy.

**Freezing the perception encoder beats fine-tuning it** (t = −3.97), the
opposite of the expected direction — measured on the stub backbone.

**The faithfulness objective does not separate** at stub-LM scale; the apparent
gain comes from one seed of five.

Operator comparison on synthetic data, same seeds: FNO 0.057 ± 0.026, DeepONet
0.256 ± 0.004, GNOT 0.268 ± 0.040.

### 7.4 Cross-family transfer

One channel-padded surrogate per family, rolled out on every family, 5 seeds:

| pair | transfer gap | measurable? |
|---|---|---|
| rdf → dar | 0.358 ± 0.033 | **yes** — ~11σ from zero |
| swe → rdf | 0.311 ± 0.13 | yes |
| dar → rdf | 1.81 ± 1.1 | marginal |
| swe → dar | 0.087 ± 0.14 | **no** — crosses zero |
| rdf → swe | 10.5 ± 8.2 | **no** — σ ≈ mean |
| dar → swe | 27.4 ± 26 | **no** — σ ≈ mean |

The qualitative story survives: transfer between the two parabolic families is
finite, and transfer *into* the wave family collapses — shallow water is fast,
oscillatory and momentum-coupled, which a parabolic surrogate has never seen.
But the magnitudes into `swe` are **not measurements**: the standard deviation
equals the mean, so the single-seed "gap ≈ 38" was one draw from a distribution
spanning an order of magnitude. Two of six pairs are consistent with no transfer
penalty at all. Quote the direction, never the number.

### 7.5 Section 7.2 controls: what the design buys

**Perception** (5 seeds, stub backbone, one decoder and dataset across arms):

| arm | field rel L2 | retrieval acc | trainable |
|---|---|---|---|
| PSPE — frozen + LoRA + contrastive | 0.0524 ± 0.008 | **0.510 ± 0.058** | 164 k |
| probe — frozen, decoder only | 0.0476 ± 0.003 | 0.133 ± 0.007 | 115 k |
| **CNN — from scratch, regression only** | **0.0422 ± 0.004** | 0.137 ± 0.018 | 563 k |

The from-scratch CNN reconstructs the field *better* than the method (t = +2.17),
and so does the adapter-free probe. What the method wins overwhelmingly is
retrieval accuracy — 0.510 vs 0.137, t = +14.3. The design buys **text–image
alignment**, not reconstruction accuracy, and trades a little accuracy for it.
The probe arm is what isolates LoRA's contribution; the freeze-vs-fine-tune
ablation cannot, because both of its arms carry adapters.

**Explanation** (5 seeds, 200 iterations, each against its own trained planner):

| arm | F(b) | vs post-hoc |
|---|---|---|
| trained-in (faithfulness objective) | 0.168 ± 0.13 | **t = +5.53** |
| no-faithfulness | 0.229 ± 0.21 | t = +2.56 |
| post-hoc (generator never trained on this policy) | 0.101 ± 0.11 | — |

**Trained-in beats the post-hoc control decisively** — the central Explain claim,
tested against a real control for the first time. On real Qwen2.5-0.5B (3
seeds) the gap widens: trained-in **0.468 ± 0.16** vs post-hoc **0.182 ± 0.16**,
t = +21.3.

But the faithfulness *term* is not what does it. On Qwen, trained-in and
no-faithfulness are identical to four decimals (0.4676 vs 0.4676). That number
first looked like a bug, and partly was: at the generation temperature the
REINFORCE samples never parsed, so every score equalled the parser fallback and
the term had exactly zero gradient. Fixed with low-temperature rollouts and a
self-critical baseline — after which the gradient is real (parse rate rises to
100%, |reinforce| ≈ 0.04) and the result is *still* identical. The term's
contribution is two orders of magnitude below the supervised term at its
default weight. What buys faithfulness is training the generator on the
policy's briefs at all; whether the objective can add to that needs a weight
sweep it has not had.

Note that `use_faithfulness=False` is **not** the post-hoc baseline, though it is
the obvious candidate: that arm has still seen the policy through the supervised
term. Using it as the control would have credited the faithfulness objective
with the entire trained-in advantage.

F(b) falls to 0.10–0.23 against a *trained* policy from ~0.6 against an untrained
one: briefs describing a policy that actually actuates are much harder to keep
faithful.

### 7.6 Resolution generalization

An FNO trained at 64² and evaluated at 64 / 96 / 128² (4,096 / 9,216 / 16,384
cells), 5 seeds: **0.0298 ± 0.014 / 0.0298 ± 0.014 / 0.0300 ± 0.014**. Flat to
three decimals — discretization invariance, not grid memorisation, and the
proposal's "tens of thousands of grid cells" tested rather than asserted.

This sweep first returned NaN at every resolution above 64². The cause was the
*truth solver*, not the surrogate: a fixed time step whose diffusion number
exceeded the explicit-scheme stability limit at finer grids. Micro-steps now
scale with (grid/64)².

### 7.7 Real backbones

**Perception on SigLIP** (`google/siglip-base-patch16-224`, 3 seeds, the first
runs with `backbone_is_stub: false`):

| arm | field rel L2 | retrieval acc |
|---|---|---|
| probe — frozen SigLIP + decoder | **0.0345 ± 0.011** | 0.135 |
| PSPE — LoRA + contrastive | 0.0596 ± 0.027 | **0.754 ± 0.026** |
| CNN from scratch | 0.0564 ± 0.015 | 0.130 |

Same shape as the stub result: the frozen probe reconstructs best (beats the
CNN, t = −4.87), and LoRA + contrastive trades reconstruction for alignment
(t = +37 on retrieval). Real SigLIP features are strong enough that adapters do
not help reconstruction.

**Planning transfer** (5 seeds): forecast error and decision loss decouple.
dar → swe has a fidelity gap of 27 and a planning gap of 0.007 ± 0.014 —
nothing — while dar → rdf has fidelity gap 1.8 and planning gap **0.58 ± 0.22**,
about 8% of return. The fidelity number does not predict the decision-relevant
one.

### 7.8 Observed wildfire spread (NDWS)

Next Day Wildfire Spread: ~18k real 64×64 km patches of US fires, day-t mask
plus eleven drivers → day-t+1 mask. Fire prevalence 1.3%; 4.4% of cells
unlabelled and masked out rather than zero-filled. AUC-PR on the validation
split, 3 seeds:

| model | AUC-PR | vs published 0.284 |
|---|---|---|
| **U-Net** | **0.2769 ± 0.002** | **97.5%** |
| hybrid — U-Net + gated FNO | 0.2347 ± 0.005 | 82.7% |
| FNO alone (1 seed) | 0.1995 | 70% |
| all-zeros floor | 0.011 | — |

The U-Net beats the hybrid on every seed (paired t = +12.5). The hybrid's gate
over its spectral path — free to take any value, initialised at zero — settles
at **0.12 ± 0.01** on all three seeds: given the choice, the model keeps the
Fourier operator nearly off. This is the operator's scope limit, measured: a
spectral model truncates the high frequencies that *are* a fire front. The same
FNO reaches 0.0098 relative L2 on shallow water.

Three pipeline defects had to be fixed before this number meant anything —
unnormalised driver channels spanning four orders of magnitude, a class weight
of 77 that made the model predict fire everywhere, and a `predict_delta`
residual across a 12-channel input stack — and they account for most of the
gap between the first run (0.035) and this one. The remaining gap to the
published 0.284, and to the 0.34 state of the art, is architectural.

### 7.9 Theory checks and joint training (Vista, Phase 1 and 2a)

Full tables in `runs/PHASE1_ANALYSIS.md`. dar, paper budget, probe +
margin on in every planning arm.

**Joint vs disaggregated — the paper's central comparison.** 5 seeds, one
pretrained FNO copied per arm:

| arm | return | held-out surrogate rel L2 before → after | drift |
|---|---|---|---|
| disaggregated (frozen surrogate) | −2.302 ± 0.17 | 0.0048 → 0.0048 | 0 |
| **joint, anchored (β = 0.1)** | −2.296 ± 0.18 | 0.0048 → **0.0005** | −0.004 |
| joint, unanchored | −2.369 ± 0.17 | 0.0048 → **0.975** | +0.970 |

Joint vs disaggregated on return: t = +0.66 — no difference. The same three
outcomes reproduce on swe and rdf (`runs/PHASE3_ANALYSIS.md`): joint return
identical to disaggregated (t = +1.00 on rdf; identical to four decimals on
swe), anchored surrogate error down 5–9× on held-out data, unanchored surrogate
destroyed (rel L2 0.83–1.14). Two testbed caveats from the same runs: on swe
the policy never leaves its initial near-zero behaviour at the calibrated
actuation scale (every arm, every fix, identical return −0.1733), so swe
currently benchmarks surrogates, not planners; on rdf the dual fails —
cost sprints to the reward-greedy value (8.0 vs limit 3.26) in the first 50
iterations and λ = 37 cannot pull it back, an action-saturation failure of the
planner that probe + margin does not touch. The anchored
joint surrogate does *not* get captured; its held-out error falls 9× on every
seed, because the planning gradient at one tenth of the data gradient acts as
extra training on the states the policy visits. The unanchored arm is the
failure mode run on purpose: the surrogate stops predicting the data (rel L2
0.975) and return falls (t = −5.84). So "joint beats disaggregated" does not
hold here; "joint is safe when anchored and destructive when not" does.

**Eq. 8 mixing rule.** Measured pathwise bias B² ≈ 0.001, so the Eq. 8 α and
the variance-only α coincide (0.992). Return t = +0.22. The Eq. 8 arm was the
only one with zero violating evaluations on all 5 seeds, at 1,920 extra truth
transitions — a tail effect of the extra rollouts, not of the formula.

**Assumption 1 and Proposition 1.** Trained FNO: L_G = 0.985 ± 0.013 (≤ 1 on
4 of 5 seeds; init 1.61); spectral normalisation gives 0.964 on all seeds for
+0.01 rel L2. Measured return bias 0.10. Prop 1's infinite-horizon bound is
422 (vacuous, prefactor 2,450 at γ = 0.98); the finite-horizon form (H = 12)
is 9.8 and holds on every seed, 100× loose.

**Faithfulness weight sweep on Qwen** (3 seeds): w = 1 → 0.468, w = 10 →
0.366, w = 100 → 0.218 (≈ post-hoc), w = 1000 → 0.399; no-term 0.468. The
term is null at its default and harmful above it. Dropped from the method.

**Conformal certificate on Qwen** (3 seeds, n_cal 200, n_test 400): holds at
δ = 0.1 and 0.2 on all three seeds (coverage 0.878 / 0.893 / 0.895 at nominal
0.90); fails at δ = 0.05 on one seed (0.928 vs 0.95, p = 0.03). Certified
floors track the generator: F ≥ 0.28 on the weak seed, F ≥ 0.56 on the others.

**Constraint fix across all probe + margin runs.** With 15 seed-runs of the
fixed configuration now available (constraint_fix, alpha_rule, joint), 13 are
clean; two seeds each had one evaluation over the limit (0.98 and 1.71 vs
0.936). The "0% on all 5 seeds" of §7.2 was one draw. Honest rate: about 1%
of evaluations vs 7.3% before the fix — a 7× reduction, not a guarantee.

### 7.10 The planner on a second family: rdf (Experiments 1 and 2)

Full detail in `runs/RDF_PLANNER_FIX.md`. On rdf the planner as shipped
violated the limit on 85% of evaluations at 2.5× the limit, in every arm,
regardless of the probe. Three defects, read off the training trace in
sequence:

1. **Action saturation.** The reward-greedy sprint drives the pre-tanh
   policy mean past |μ| ≈ 3; tanh has no slope there, the pathwise gradient
   dies (‖g‖ 0.036, variance exactly 0), the variance rule picks the dead
   branch (α → 0.99, since zero variance looks like precision), and the LR
   branch explodes (‖g‖ 1e4) because every rollout returns the same value
   and the advantage normaliser divides by ~0. Fix: a soft wall on |μ| and a
   floor on the advantage std. 85% → 44%.
2. **Dual oscillation.** λ collapses to zero whenever cost dips under the
   limit, the policy sprints, overshoots. Lowering the gain made it worse
   (60%); raising the integral gain ×10 lets λ hold. 44% → 14.5%.
3. **Tail violations.** A dual holding the *mean* at the limit violates on
   half the evaluations by construction; the margin now includes the policy's
   own episode cost spread, measured on the probe. 14.5% → **5.5%**, worst
   case 3.23 vs the 3.26 limit.

The same configuration on dar: return unchanged, 0% violations, worst case 3×
smaller. One planner configuration now works on both families.

**Against the baselines on rdf** (5 seeds, 38,400 real transitions each):
PPO-Lagrangian −7.66 at 1.8% violations, and −7.69 at **0%** once given the
same integral-gain fix; CPO/Sauté/NPG −7.97 to −8.08 at 0%, but with cost at
the do-nothing level — safe by inaction. PSPE v2 (probe + margin): −7.49 at
5.5% on 7,360 real transitions. Paired t vs PPO-Lag: **+0.82** — parity, and
PPO-Lag has the better tail. Without the margin PSPE beats every baseline
(t +2.1 to +3.8) but at 14.5% violations, a different operating point.

So the planning claim, honestly stated: a return edge on dar (t 3.8–18),
parity on rdf, 5–9× fewer real samples on both, and a safety mechanism whose
failure modes are measured and whose fix also repaired PPO-Lagrangian.

## 8. What survives, and what does not

**Survives multi-seed scrutiny**

* the planner's return advantage over all four safe-RL baselines **on dar**
  (t = 3.84–17.95); on rdf it is parity with PPO-Lagrangian (t = +0.82)
* real-sample efficiency: 9× on dar, 5× on rdf, from the model-based design
* FNO ≫ DeepONet ≫ GNOT, on both synthetic and real benchmark data
* surrogate accuracy competitive with published PDEBench FNO error
* trained-in explanation beating a genuine post-hoc control
* the direction of cross-family transfer

* constraint violation cut ~7× by probe + margin (13 of 15 seed-runs clean,
  ~1% of evaluations vs 7.3%), at no return cost and 9.2× fewer real samples
* joint training is safe when data-anchored: surrogate held-out error falls
  5–9× on all three testbeds, decision unchanged; unanchored, it destroys the
  surrogate on all three (§7.9)
* Assumption 1 (L_G ≤ 1) after training; the conformal faithfulness
  certificate at 90% on real Qwen, 3 of 3 seeds
* resolution invariance to 128² (16,384 cells), 5 seeds
* trained-in explanation on a real LM: 0.468 vs 0.182 post-hoc, t = +21
* the FNO's scope: operators for smooth fields, convolutions for sharp fronts
  (NDWS gate 0.12, U-Net > hybrid at t = +12.5)

**Does not survive**

* constraint satisfaction *as originally built* — 7.3% of evaluations violated
  because the dual was controlled on surrogate cost; fixed above
* the physics-informed loss improving asymptotic accuracy
* adaptive α improving mean return (it improves the tail instead)
* the faithfulness objective contributing beyond supervised brief training —
  confirmed on Qwen with a working gradient; contribution is negligible at the
  default weight
* PSPE perception beating a CNN or frozen probe on reconstruction (stub *and*
  SigLIP); it wins alignment instead
* any specific magnitude for transfer into the wave family
* the FNO as a universal surrogate — it loses to a U-Net on observed fire spread
* joint training *beating* the disaggregated pipeline on return (t = +0.66
  dar, +1.00 rdf, identical on swe)
* swe as a planning testbed (policy never moves)
* rdf's dual at the original gains — fixed (§7.10); the return edge on rdf
  did not survive the fix (parity with PPO-Lag at matched safety)
* Eq. 8 improving on the variance-only α (B² is too small to matter)
* Proposition 1's infinite-horizon bound as a useful number (4,000× loose)
* constraint satisfaction as a guarantee — probe + margin is a 7× reduction

## 9. Application to climate and earth-science problems

The testbeds are deliberately generic PDEs, but each maps onto a real class of
intervention problem, and the mapping is what makes the negative results matter.

**Wildfire and smoke.** A reaction–diffusion front under advection is the
standard reduced model for fire spread and smoke plumes. The intervention is
retardant drops or controlled burns; the budget is aircraft sorties; the
pointwise safety constraint is population exposure. The perception module's job
— satellite imagery to a continuous field — is exactly the Sentinel-2 → index
task the EuroSAT path implements.

**Flood control.** Shallow water is the governing system for riverine and
coastal flooding. The intervention is reservoir release scheduling or barrier
operation; the constraint is that no district exceeds a depth threshold. This is
where the **equity constraint** is not optional: releasing water optimally in
aggregate while concentrating harm in one floodplain is the failure mode
distributional constraints exist to prevent, and the repository implements it as
a second dual with its own multiplier.

**Air quality.** Diffusion–advection–reaction with a source term is the
workhorse model for pollutant dispersion. The intervention is traffic
restriction or industrial curtailment; the budget is economic cost; the
constraint is a concentration ceiling.

**Irrigation and drought.** Soil-moisture dynamics under diffusion with
extraction, with NDVI as the observable — the exact perception target already
implemented.

### What would have to be true for this to transfer

Three conditions, and the results above bear directly on all three:

1. **The surrogate must be trustworthy where it is used to plan.** The safety
   failure found here — dual controlled on surrogate cost, violation realised
   under true dynamics — is not an artifact of a toy problem. It is the generic
   risk of model-based planning, and in a flood or exposure setting a 7.3%
   violation rate on a safety constraint would be disqualifying. Any deployment
   needs either a surrogate-error-aware constraint margin, dual updates driven
   by real-environment cost, or both.
2. **Constraint limits must be calibrated to the system, not chosen.** Limits
   here are set from measured reference points — the cost of doing nothing and
   the cost a reward-greedy policy incurs — because guessed limits produced
   testbeds that were variously never-binding and over-tight, which silently
   turned the comparison into an unconstrained one.
3. **Explanations must be checked against behaviour, not read for
   plausibility.** F(b) falls sharply once the policy actually does something,
   which is precisely when a decision-maker would rely on the brief. A
   plausible-sounding brief for a genuinely complex policy is the dangerous
   case, and the trained-in vs post-hoc gap is the only evidence here that
   joint training helps at all.

The honest summary for an earth-science audience: the *forecasting* component is
competitive on a public benchmark, the *planning* component is more
sample-efficient than model-free constrained RL and gets better return, and the
*safety* component does not yet meet the standard that a real deployment would
demand. The last of those is a specific, reproducible, open problem rather than a
vague caveat — one seed reproduces it deterministically.

## 10. Reproduction

```bash
make data                  # generate the three synthetic testbeds
make seeds                 # Phase 2 comparison, 5 seeds, mean ± std
make perception-baselines  # Section 7.2 perception controls
make explain-baselines     # trained-in vs no-faithfulness vs post-hoc
make transfer              # cross-family matrix
make pdebench              # real benchmark (needs the 6.2 GB download)
```

On a SLURM cluster, `scripts/tacc/driver.sh` runs the whole queue as job arrays,
one seed per node, and aggregates the results.

Raw results live under `runs/` and are gitignored; the tables above are the
committed record. `docs/proposal_deltas.md` tracks every place the written
proposal and this code disagree, with the resolution for each.
