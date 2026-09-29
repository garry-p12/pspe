# PSPE — thesis, complete results, and the state of the climate-twin framework

**Constrained intervention design for PDE-governed systems.**
Definitive record, 2026-09-24. Supersedes the framing in the original proposal.

Every number below is multi-seed with paired significance tests; each section
names the run directory and the analysis file it comes from. Claims that were
made and then withdrawn are kept, with the measurement that withdrew them,
because in this project those were often more informative than the claims.

**Last updated 2026-09-24**, with §3.5 (the derived diagnostic and the
controlled coverage study), §5.5 (the margin on real fire dynamics, and the two
preconditions it exposed), and five further entries in the measurement-defect
catalogue. Results still pending are marked as such in place rather than
omitted; `paper/REVISION_NOTES.md` §8e pre-registers what each would mean.

**Part 9 positions every result against the current literature** and downgrades
several of them; inline notes through Parts 3 to 7 flag the nearest prior work
where each claim is made. Read §8.1 and §9.1 together — they are the two
scorecards, one against our own baselines and one against the field. The long
form of the audit is [`RELATED_WORK.md`](RELATED_WORK.md).

---

## Part 0 — The thesis

### 0.1 What the proposal claimed, and what happened to it

| # | original claim | outcome |
|---|---|---|
| 1 | Joint training of Perceive/Simulate/Plan/Explain under one objective beats a disaggregated pipeline | **Refuted** on three PDE families. Decision never changed (t = +0.66, +1.00, identical). |
| 2 | Eq. 8 mixing rule `α* = V_L/(B² + V_p + V_L)` improves the hybrid gradient | **Degenerate.** Measured B² ≈ 0.0013, so Eq. 8 and the variance-only rule select the same α (0.992). Return t = +0.22. |
| 3 | Corollary 1: time-averaged constraint violation → 0 | **False as stated** on the PDE testbeds. Holds only on a toy CMDP with closed-form optimum. |
| 4 | Faithful natural-language explanations via `F(b)` | **Withdrawn.** Permutation control shows zero state-specific information. |
| 5 | Cross-PDE-family transfer | **Direction only.** Magnitudes into the wave family have σ ≈ mean. |

Five headline claims; none survives in its original form. That is the honest
starting point, and it is why the thesis below is not a repair of the old one.

### 0.2 The thesis the evidence actually supports

> **Model-based decision systems fail silently.** Every component's own metric
> can improve while the decision degrades or the constraint is violated,
> because each metric is computed inside the model's account of the world. The
> contribution is (a) a catalogue of these failures with the measurement that
> exposes each, and (b) mechanisms that close the loop on real measurement —
> probe the environment, synchronise to observation, score on the decision —
> which recover the lost performance and, in the constrained case, carry a
> stated failure rate.

Seven independent instances of the same failure shape were measured. Several
have precedent, named in the right-hand column of §9.1; the contribution is the
catalogue and the corrections, not the discovery of the genre.

| the component looked correct by… | but in fact… | the measurement that exposed it | §  |
|---|---|---|---|
| surrogate episode cost; dual satisfied | 7.3% of evaluations violated the true limit | probe the true environment periodically | 3.1 |
| one-step forecast skill | open-loop rollout invents a fire that does not exist | sync to observation; compare against open loop | 5.2 |
| faithfulness `F(b) = 0.47` vs post-hoc `0.18`, t = +21 | briefs carry **zero** state-specific information | permutation control (score against another state) | 4.4 |
| cost separates between do-nothing and greedy | swe ranked methods inside a 7% achievable band | **return** separation, not just cost separation | 3.4 |
| four constrained-RL baselines "structurally fail" | they failed from their initialisation | start baselines where the method starts | 5.4 |
| reconstruction AP 0.594 vs 0.006 (≈100×) | decision worse in **0 of 9** configurations | score the belief on the decision, not on reconstruction | 5.3 |
| margin cuts violations ≈7× | no stated failure rate; and the first conformal attempt made rdf worse | conformalise the distribution that actually violates | 3.2 |
| our solver matches the reference at CSI 0.979 | the reference itself matches the satellite at 0.535 | score the reference against observation, not only the model against the reference | 5.7 |
| the twin loop improved 8 of 8 fires, t = 3.0 | at 607 fires the same benefit is a third the size and holds on 76% | run the paired test at a sample size that can be wrong | 5.2a |

**Three of these were found after the thesis was written, by applying its own
rule to itself.** §5.7 scored the *reference* rather than only our agreement
with it. §5.2a re-ran a finished result at 75× the sample size. §5.8 asked why
the perception failure happened rather than filing it. Each one moved a number
the project had already published, and all three moved it down. The catalogue
is not a list of other people's mistakes.

The positive results share one shape: **measure the quantity that actually
matters — real cost, observed state, decision outcome — and the system works.**

And the negative results share a sharper one. It is not that components are
uninformative; it is that **the map from component metric to decision quality is
not monotone, and in three measured cases it is inverted.** A belief that
matches the hidden state better plans worse (§5.3, mechanism in §5.8). A margin
fitted to the model's error raises violations above using no margin at all
(§3.2). An ensemble-disagreement penalty that is demonstrably active buys
nothing (CAP, §3.2). Each of these is a method that a component metric would
have certified.

![The seven silent failures and the measurement that exposed each](figures/pspe_silent_failure.svg)

*Figure 1. Each row is a component whose own metric said it was working. The
right column is the measurement that disagreed. Regenerate with
`python scripts/make_thesis_figures.py`.*

### 0.3 The paper I would write

> **Constrained planning with learned dynamics: silent failure and
> measurement-based correction.**
>
> A planner that optimises inside a learned surrogate inherits every error that
> surrogate makes, and inherits them invisibly: the dual controller enforcing a
> budget is fed cost measured *in the model*, while violation is realised under
> *true* dynamics. We characterise this failure, give a split-conformal margin
> that converts it into a stated failure rate, and validate on three PDE
> families and on wildfire intervention over real fire records, where planning
> through a learned spread model beats the operational forecast-then-treat
> heuristic by 12 points of burned-area reduction at matched crew budget.

Technical core: the conformal margin (§3.2). Empirical core: the wildfire
results with the horizon mechanism (§5.1). Robustness core: the failure
catalogue (§0.2, §7).

> **Revised after the literature audit.** Each failure in the catalogue turns
> out to have precedent — silent violation under model-based constraints is
> named in safe RL, joint training restates objective mismatch, and the
> metric-versus-outcome divergence is being reported concurrently in our own
> application domain. A reviewer who knows that literature reads the catalogue
> as careful replication rather than a headline. **§9.13 states the narrower
> paper that survives Part 9**, led by the conformal margin's negative result
> against the field's default instantiation. Venue read in §8.3.

---

## Part 1 — Formalism

![PSPE as it now stands, with validated and model-based paths distinguished](figures/pspe_system.svg)

*Figure 2. The combined architecture. Green paths carry real measurement back
into the loop and are validated against observations; the dashed red path is
model based and, for the reason given in §6.2, cannot be validated on any
observational record. The original Fig. 1 drew joint training as a
distinguishing feedback path; §4.3 removed it.*

### 1.1 The control problem

A field `u(x,t)` on a spatial domain `Ω ⊂ ℝ²` evolves under a PDE with a
control term:

```
∂u/∂t = 𝒩[u] + ℬ(a(t)),        u(·,0) = u₀,        x ∈ Ω
```

`𝒩` is the (nonlinear) spatial operator — diffusion, advection, reaction, or
shallow-water dynamics — and `ℬ` maps an actuator vector `a ∈ [-1,1]^K` to a
forcing field through a Gaussian basis:

```
ℬ(a)(x) = Σ_{k=1..K}  A · a_k · exp( −‖x − c_k‖² / 2σ² )
```

with `A` the actuator amplitude (§3.4 shows why this parameter decides whether
a testbed can rank planners at all).

Discretised on an `N × N` grid with step `Δt`, the true transition is
`u_{t+1} = F(u_t, a_t)`; the learned surrogate is `G_θ ≈ F`.

### 1.2 The constrained objective

A CMDP over a horizon `H`:

```
maximise    J_R(π) = 𝔼_π [ Σ_{t=0..H−1} γ^t r(u_t, a_t) ]
subject to  J_C(π) = 𝔼_π [ Σ_{t=0..H−1} c(u_t, a_t) ]  ≤  d
```

with tracking reward and cost

```
r(u,a) = −w_track ‖u_{[ch]} − u*‖²_Ω  −  w_act ‖a‖²
c(u,a) = Σ_x relu( |u_{[ch]}(x)| − u_max )  +  relu( mean|a| − budget )
```

The equity constraint `g₂` penalises concentration of harm across sub-regions
and is enforced by a second dual.

**Limits are calibrated, not chosen.** With `c₀` the cost of doing nothing and
`c_g` the cost a reward-greedy policy incurs,

```
d = c₀ + 0.35 · (c_g − c₀)
```

so a constrained learner must give up a real share of greedy behaviour. §3.4
shows this calibration is necessary but *not sufficient*.

### 1.3 The hybrid policy gradient

The Lagrangian per step is `ℒ = −(r − λ c)`. Two estimators of `∇_φ 𝔼[ℒ]`:

```
pathwise (reparameterised, through G_θ):
    g_pw = ∇_φ  ℒ( rollout(G_θ, π_φ) )                 low variance, biased by ‖G_θ − F‖

likelihood ratio (score function, with critic baseline):
    g_lr = −𝔼[ ∇_φ log π_φ(a|u) · Â ],   Â = (R − V(u))/σ_R    unbiased, high variance
```

mixed as `g(α) = α g_pw + (1−α) g_lr`. Minimising MSE of the mixture,

```
MSE(α) = α²(B² + V_p) + (1−α)² V_L + 2α(1−α) Cov

α* = (V_L − Cov) / (B² + V_p + V_L − 2 Cov)                              (Eq. 8)
```

where `B = ‖𝔼[g_pw^{G_θ}] − 𝔼[g_pw^{F}]‖` is the pathwise bias from surrogate
error. Setting `B = 0` recovers the variance-only rule that shipped.

**Measured:** `B² = 0.0013 ± 0.0015`, so `α*` is 0.992 under both rules and the
distinction is empirically void (§4.2).

### 1.4 The dual

A PID-Lagrangian controller on the constraint error `e_k = Ĵ_C,k − d`:

```
λ_k = clip( K_p e_k + K_i I_k + K_d max(0, e_k − e_{k−1}),  0,  λ_max )
I_k = max(0, I_{k−1} + e_k)
```

with `Ĵ_C` an EMA of measured episode cost. §3.3 shows `K_i` is the gain that
decides whether the controller holds or oscillates.

### 1.5 The surrogate

A Fourier Neural Operator. Per block, with `ℱ` the 2-D DFT and modes truncated
at `k ≤ k_max = 12`:

```
v_{ℓ+1}(x) = σ(  W v_ℓ(x)  +  ℱ^{-1}[ R_ℓ · ℱ[v_ℓ] ](x)  )
```

`R_ℓ ∈ ℂ^{k_max × k_max × w × w}` are learned spectral weights; `W` is a 1×1
convolution carrying the high frequencies the truncation discards. Trained on

```
ℒ_sim = ‖G_θ(u,a) − u'‖²  +  β_roll Σ_{h≤H} ‖G_θ^h(u,a) − u_{t+h}‖²  +  β_phys ‖ℛ[G_θ(u,a)]‖²
```

Optional spectral normalisation projects each `R_ℓ` to bound the operator norm,
giving the Lipschitz constant needed by Assumption 1 (§4.1).

### 1.6 Explanation and its certificate

A brief `b` is parsed by a frozen deterministic parser into an action
distribution `π̂_b`, and scored

```
F(b) = exp( −KL( π_φ(·|u) ‖ π̂_b ) )
```

**Two corrections to this definition, both forced by measurement:**

1. The proposal writes `F = 1 − KL`, which is unbounded below and unusable as
   a REINFORCE reward. The exponential form is bounded in (0,1] and agrees to
   first order at small KL.
2. KL **summed** over action dimensions makes F incomparable across action
   spaces and sends it to 0 above ~16 dimensions. On 64-patch fire plans every
   arm scored exactly 0.0 while the reference brief scored 0.985. Per-dimension
   KL, `KL/K`, is the only form that can be compared.

Split-conformal certificate: with calibration scores `s_i = 1 − F(b_i)` on `n`
exchangeable samples and level `δ`,

```
ŝ = s_(⌈(n+1)(1−δ)⌉)          ⟹          ℙ[ F(b) ≥ 1 − ŝ ]  ≥  1 − δ
```

**The control this definition needs** (§4.4): the permutation statistic

```
Δ = 𝔼_i[ F(b_i ; π_i) ] − 𝔼_i[ F(b_i ; π_{σ(i)}) ],     σ a derangement
```

`Δ ≈ 0` means the brief carries no state-specific information, whatever `F` is.

### 1.7 The twin loop

A belief `û_t`, a model `G_θ`, an observation operator `𝒪` that returns a
partially observed field with a validity mask `m_t`:

```
forecast:    ũ_{t+1} = G_θ(û_t, a_t)
observe:     (y_{t+1}, m_{t+1}) = 𝒪(u_{t+1})
sync:        û_{t+1} = m_{t+1} ⊙ y_{t+1} + (1 − m_{t+1}) ⊙ Φ(ũ_{t+1}, y_{t+1})
adapt:       θ ← θ − η ∇_θ  ℓ( G_θ(û_t, a_t),  y_{t+1} )      [on observed cells]
```

`Φ` is the imputation rule for unobserved cells. Three choices are compared in
§5.3, and the open-loop baseline is `m ≡ 0` for all `t > 0` — never look again.

---

## Part 2 — Assets

### 2.1 Testbeds

| id | PDE | state channels | calibrated limit `d` | admissible for planning? |
|---|---|---|---|---|
| `dar` | diffusion–advection–reaction | 1 | 0.936 | **yes** — the primary testbed |
| `rdf` | reaction–diffusion front | 2 | 3.26 | **yes**, after the planner fix (§3.3) |
| `swe` | shallow water | 3 | 0.192 | **no** — achievable band ≈7% (§3.4) |

### 2.2 Real datasets

| dataset | content | size | used for |
|---|---|---|---|
| **PDEBench** 2-D shallow water | solver trajectories | 512 trajectories | surrogate benchmark (§4.5) |
| **NDWS** Next Day Wildfire Spread | ~18k real US fire days, day-t mask + 11 remote-sensing drivers, 64 km patches | 2.3 GB | forecasting, planning, partial observation |
| **FIRMS** VIIRS active fire | 8 large wildfires, 14–20 consecutive days, real detections | downloaded per fire | the sequential twin loop (§5.2) |

NDWS drivers are themselves real remote-sensing products (MODIS NDVI, gridded
meteorology, VIIRS/MODIS fire masks). FIRMS supplies what NDWS structurally
cannot: **sequences of the same fire**, because NDWS records carry no date, no
fire id and no location.

---

## Part 3 — The safety contribution

### 3.1 The failure

`runs/seeds_v3/`, `docs/results/RDF_PLANNER_FIX.md`

> **Prior work.** This failure is named in the model-based safe-RL literature:
> cost estimates are error-prone when computed under a learned model, and
> Lagrangian methods degrade at low cost thresholds because of that estimation
> error [Ma et al., 2022; Jayant & Bhatnagar, 2022; Huang et al., 2024]. We
> should not present it as a discovery. What is ours is the probe-plus-margin
> decomposition and the conformal instantiation (§3.2). See §9.2.

The dual is updated from cost measured on **surrogate** rollouts; violation is
realised under **true** dynamics. Where `G_θ` under-predicts cost, the policy is
safe in-model and unsafe in reality, and nothing in the loop notices.

dar, 5 seeds, `d = 0.936`:

| method | return | cost | violating evals | worst | real samples |
|---|---|---|---|---|---|
| **PSPE (adaptive α)** | **−2.448 ± 0.029** | 0.412 ± 0.37 | **7.3%** | 1.109 | 3,072 |
| PSPE (fixed α) | −2.738 ± 0.365 | 2.752 ± 3.6 | 12.7% | 2.758 | 3,072 |
| PPO-Lagrangian | −2.558 ± 0.039 | 0.266 ± 0.009 | 0% | 0.297 | 38,400 |
| CPO | −2.541 ± 0.035 | 0.263 ± 0.011 | 0% | 0.277 | 38,400 |
| Sauté RL | −2.537 ± 0.024 | 0.264 ± 0.008 | 0% | 0.275 | 38,400 |
| Primal-dual NPG | −2.544 ± 0.026 | 0.263 ± 0.008 | 0% | 0.275 | 38,400 |

Beats every baseline on return (t = +3.84 to +17.95) at **12.5× fewer real
transitions** — but violates a limit they all respect. Part of the advantage
was bought.

> **A caveat that must be stated.** Every baseline here is **model-free**
> [Achiam et al., 2017; Stooke et al., 2020; Sootla et al., 2022; Ding et al.,
> 2020]. A model-based method using fewer real transitions than model-free
> methods is close to tautological. Against a model-based safe-RL baseline —
> SMBPO [Thomas et al., 2021], SafeDreamer [Huang et al., 2024], CAP [Ma et
> al., 2022] — the sample-efficiency claim may not survive at all. This is the
> largest gap in the baseline set (§9.2).

### 3.2 The fix, and the bound

**Mechanism.** A *probe* rolls the current policy in the true environment every
`N` iterations and feeds that cost to the dual; between probes the surrogate's
cost is corrected by the running bias. A *margin* plans against a tightened
limit `d_eff < d`.

`runs/constraint_fix/`, 5 seeds:

| arm | return | violating evals | worst | real samples |
|---|---|---|---|---|
| baseline | −2.302 ± 0.16 | 1.8% | 0.75 (max 1.73) | 3,200 |
| probe only | −2.307 ± 0.16 | 1.8% | 0.56 | 4,160 |
| margin only | −2.302 ± 0.16 | 1.8% | 0.75 | 3,200 |
| **probe + margin** | **−2.301 ± 0.17** | **0% (5/5 seeds)** | **0.29** | 4,160 |

**Neither half works alone.** Margin-only is *identical* to baseline, because
with no probe there is no measured error from which to derive a margin.
Conservatism requires a measurement.

**Honest rate.** Across all 15 seed-runs that use probe+margin, 13 are clean and
two had a single excursion. So ≈1% of evaluations versus 7.3% — a ~7×
reduction, **not a guarantee**.

**The conformal margin.** Conformal prediction [Vovk et al., 2005; Lei et al.,
2018; Angelopoulos & Bates, 2023] has been applied to safe control in a growing
line of work [Lindemann et al., 2023; ACP+CBF, arXiv 2503.17678], but **almost always
by conformalising the error between the learned dynamics model and the true
system** and pushing that bound into a barrier or Lyapunov constraint. §3.2's
failed first attempt is that recipe, and it makes things worse here. We
conformalise a different quantity.

Let `e_i = c_i − c̄` be deviations of individual real
episode costs from their probe mean, and `b̂ = max(0, bias)` the surrogate's
systematic cost bias. With `n` such deviations and level `δ`:

```
q = e_(⌈(n+1)(1−δ)⌉)              d_eff = d − ( q + b̂ )
```

Under exchangeability of the deviations, `ℙ[ c > d ] ≤ δ`.

`runs/constraint_fix_conf*/`, 5 seeds, δ = 0.1:

| testbed | arm | violating | target | worst | return | `d_eff` |
|---|---|---|---|---|---|---|
| dar | baseline | 0% | — | 0.225 | −2.304 | 0.936 |
| dar | **probe + conformal** | **0%** | ≤10% | 0.225 | −2.309 | 0.676 |
| rdf | baseline | 14.5% | — | 4.00 | −7.184 | 3.260 |
| rdf | **probe + conformal** | **7.3%** | ≤10% | 3.51 | −7.351 | 2.777 |

The bound holds on both families. Against the hand-tuned `kσ` margin it
replaces (5.5% at return −7.49 on rdf): slightly more violations, a **stated
failure rate** instead of a tuned constant, and 0.14 better return.

> **Superseded by §3.2b (2026-09-28).** This table is five seeds and a point
> estimate. At **20 seeds** the same configuration gives **14.1% → 5.0%**, a
> seed-paired **−9.1 points, 95% CI [−11.8, −5.5]**, p = 0.0002, 15/20 seeds
> improved — and **no return penalty** (IQM −6.884 against −6.917). Cite the
> 20-seed numbers; the five-seed table could not separate its own effect from
> zero (§3.2a).

![The probe, the conformal margin, and the measured violation rates](figures/pspe_safety.svg)

*Figure 3. Left: why a satisfied dual still violates, and the margin that fixes
it. Right: measured violation rates on the reaction diffusion front as each
planner defect of §3.3 is removed, ending with the conformal margin against the
rate it states.*

> **Positioning, sharpened.** Calling this "the model is accurate and the
> policy is variable" names the symptom. The cause is structural, and stating
> it turns an empirical curiosity into a rule:
>
> > **Matched-pair conformalisation is correct when the controlled quantity is
> > per instance, and undercovers when the dual controls an expectation.**
>
> Residuals `c_i − g_i` cancel whatever scatter is common to both terms. If the
> controller holds `g_i` at the limit for each instance — which is what a
> control barrier function does, pointwise, in the setting the recipe was
> developed for — that cancellation is exactly right. In a CMDP the dual holds
> `E[c]`, so `g_i` is not the controlled quantity, and the per-instance scatter
> that actually breaches `d` cancels out of the calibration set. The recipe is
> sound where it came from and unsound here, for a reason that can be written
> down in advance.
>
> **This also simplifies the method.** Conformalising `s_i = c_i − ĝ` — realised
> cost minus the quantity the dual controls — covers bias and spread in a single
> order statistic, needs no separate `b̂` term, and the condition of Prop. 1
> becomes exactly what the dual already does. Implemented as the default in
> `pspe/plan/margins.py`, with `model_error` and `episode` selectable for the
> head-to-head; both this and the wildfire experiment call that one module.
>
> Two caveats still belong in any write-up: the bound is marginal where
> [*Proactive Safety Constraints*, arXiv 2609.08080] gives a per-step
> cumulative one, and exchangeability across a learning run is assumed rather
> than tested — adaptive conformal inference [Gibbs & Candès, 2021] exists
> because that assumption fails under exactly this drift. The `residual` form
> needs *one* exchangeable sample where the `episode` form needed an
> exchangeable sample plus an estimated bias, so the assumption is weaker than
> it was. See §9.3.

**A failed first attempt, kept because it is instructive.** Conformalising the
*surrogate's cost error* made rdf **worse** (18.2% vs a 14.5% baseline, margin
0.25 where 0.69 was needed). On rdf the surrogate is accurate (bias −0.08);
what violates the limit is the policy's own episode-to-episode spread, which a
quantile over model error never sees. **A margin is only as good as the
distribution it bounds.**

### 3.2a What the margin claim survives under a bootstrap (2026-09-28)

§9.11 cites [Agarwal et al., 2021] against reporting a point estimate over a
handful of seeds, and then §3.2 does exactly that. §8.4 item 9 was to fix it.
`eval/metrics.py` already had `iqm`, `bootstrap_ci` and `seed_report`; what was
missing was applying them. `eval/run_bootstrap_report.py` now does, resampling
over **seeds**, because that is the unit of independence — two episodes from one
training run share a policy.

The first thing the script found was a **provenance ambiguity in the table
above**. Two different violation numbers live in every summary: the fraction of
periodic evaluations during the run whose episode cost exceeded the limit
(`eval/violating_eval_fraction`), and the final evaluation's rate
(`violation_rate`). On rdf they disagree — 14.5% against 17.5% for baseline —
and only the first reproduces §3.2, so **the published numbers are the during-run
evaluation fraction.** That is now stated in the script rather than left for a
reader to infer.

`runs/bootstrap_report/`, 5 seeds, δ = 0.1, IQM with a 95% percentile bootstrap:

| testbed | arm | violating % (mean) | IQM | 95% CI | final-eval % |
|---|---|---|---|---|---|
| dar | baseline | 0 | 0 | [0.0, 0.0] | 0 |
| dar | probe + conformal | 0 | 0 | [0.0, 0.0] | 0 |
| rdf | baseline | 14.5 | 15.2 | [6.1, 24.2] | 17.5 |
| rdf | margin only | 14.5 | 15.2 | [6.1, 24.2] | 20.0 |
| rdf | probe only | 14.5 | 15.2 | [9.1, 18.2] | 25.0 |
| rdf | **probe + conformal** | **7.3** | **6.1** | **[0.0, 15.2]** | **0** |

And the comparison that matters, differenced **within** each seed (seed 5 is hard
for every arm, so the unpaired interval is mostly across-seed spread that cancels
inside a seed):

| comparison | Δ violating pts (IQM) | 95% CI | improved | worse |
|---|---|---|---|---|
| rdf, margin − baseline | 0.0 | [0.0, 0.0] | 0/5 | 0/5 |
| rdf, probe − baseline | 0.0 | [−12.1, +12.1] | 1/5 | 1/5 |
| rdf, **probe + conformal − baseline** | **−6.1** | **[−21.2, +6.1]** | **3/5** | **1/5** |

**At five seeds the headline safety improvement does not separate from zero.**
The direction is right and the effect is the largest of the three arms, but the
interval spans +6.1 points, three seeds improve, one gets worse and one ties.
§3.2's "14.5% → 7.3%" is a point estimate whose uncertainty is roughly the size
of the effect.

> **Resolved at twenty seeds — see §3.2b. The effect is real and larger than
> published: −9.1 points, 95% CI [−11.8, −5.5], 15/20 seeds improved, paired
> t = −4.59, p = 0.0002.** Five seeds could not see it. The analysis below was
> correct about the evidence available and wrong to be read as evidence against
> the claim; it was evidence of insufficient power, which is a different thing.

Two things do survive, and they are worth separating from the thing that does
not:

1. **The coverage claim is not the comparative claim.** What §1.2 derives is
   `ℙ[c > d] ≤ δ` — a statement about the margin's *validity*, not about beating
   an unmargined baseline. 7.3% mean and 6.1% IQM sit under δ = 10%, so the
   bound is satisfied as stated. The honest caveat is the upper interval: 15.2%
   exceeds δ, so five seeds cannot confirm coverage either, only fail to refute
   it.
2. **It is never worse on the final policy.** On `violation_rate` the baseline
   is [0, 0, 0, 0, 0.875] and probe+conformal is [0, 0, 0, 0, 0] — one seed
   fixed, four ties, none broken. That is a weaker statistical claim than the
   paper made and a cleaner qualitative one.

The cost is return: IQM −7.435 against −7.304, intervals overlapping heavily.

**What this changes in the write-up.** The abstract cannot say the margin
*reduces* violations on rdf; it can say the margin *holds its stated rate* on
both families, and that where the unmargined planner violated on one seed in
five the margined one did not. The 12-point and +14.3-point claims flagged in
§8.4 item 9 need the same treatment before submission — see §8.4.

> Why this is a finding and not just a caveat: the retired `swe` testbed (§3.4)
> was carried for a whole project because nobody measured whether its return
> separated. This is the same class of error one level up — a comparative claim
> carried because nobody measured whether the *difference* separated. The
> machinery to catch it had been sitting in `eval/metrics.py` unused.

### 3.2b Twenty seeds: the margin claim holds, and is stronger than published (2026-09-28)

§3.2a found the five-seed margin comparison could not separate from zero, so the
comparison was re-run at **20 seeds** — same configuration as the published
`cfixconf` arm, all twenty in one fresh tree, Vista job 1032033, 51 minutes.
`runs/constraint_fix_conf_rdf_20/`, `runs/bootstrap_report_20/`.

| arm | violating % (mean) | IQM | 95% CI | final-eval % | return (IQM) |
|---|---|---|---|---|---|
| baseline | 14.1 | 12.7 | [9.1, 17.3] | 15.0 | −6.917 |
| margin only | 14.1 | 12.7 | [9.1, 17.3] | 15.6 | −6.917 |
| probe only | 10.9 | 11.8 | [6.4, 15.5] | 12.5 | −6.850 |
| **probe + conformal** | **5.0** | **4.5** | **[0.9, 8.2]** | **1.2** | −6.884 |

Seed-paired against baseline, resampling the paired differences over seeds:

| comparison | Δ pts (IQM) | 95% CI | improved | worse | excludes 0 |
|---|---|---|---|---|---|
| margin − baseline | 0.0 | [0.0, 0.0] | 0/20 | 0/20 | no |
| probe − baseline | −4.5 | [−8.2, 0.0] | 10/20 | 3/20 | no |
| **probe + conformal − baseline** | **−9.1** | **[−11.8, −5.5]** | **15/20** | **1/20** | **yes** |

Paired t = **−4.59**, p = **0.0002**; Wilcoxon signed-rank p = **0.0008**.

**The claim holds, and the published numbers understated it.** 14.1% → 5.0% at 20
seeds against the published 14.5% → 7.3% at five, and the paired effect is −9.1
points where five seeds put it at −6.1. The baseline reproduces (14.1 against
14.5), so the difference is in the treated arm: five seeds happened to sample its
worse tail.

**And it now costs nothing in return.** IQM −6.884 against the baseline's −6.917 —
marginally *better*, well inside overlapping intervals. The five-seed run
suggested a price (−7.435 against −7.304); at 20 seeds there is none. The
"slightly more violations for a stated failure rate and 0.14 better return"
trade-off in §3.2 can be restated as **fewer violations at no cost in return**.

**The decomposition is the part worth keeping.** The three rows separate cleanly
and each says something:

* **margin alone does literally nothing** — 0.0 points, 0/20 seeds changed. The
  conformal quantile needs probe data to calibrate against; without it the margin
  is identically zero, not merely small.
* **probe alone is not enough** — −4.5 points but the interval touches zero and
  3/20 seeds get worse. Real-environment data by itself does not fix the dual.
* **the two together are significant** — and the effect exceeds the sum of the
  parts, which is the interaction §3.2 argues for.

That decomposition was invisible at five seeds, where all three arms straddled
zero and could not be told apart.

**What this costs and what it buys.** The paper cannot cite the five-seed table
any more; §3.2's headline moves to 14.1% → 5.0% over 20 seeds with an interval.
In exchange the central safety claim goes from *unsupported* to *supported at
p = 0.0002 with no return penalty*, and the mechanism decomposes. Every other
five-seed comparison in this document is now suspect for the same reason and
should be re-run at 20 before submission — §8.4 item 9.

> The lesson is not "the margin works after all." It is that **§3.2a and §3.2b are
> the same measurement at different n, and only one of them could have been
> published honestly.** Five seeds produced a point estimate that overstated
> nothing and supported nothing; twenty produced a claim. The cost of finding out
> was 51 minutes of one GPU.

### 3.2c Which other claims have the same power problem (2026-09-28)

§3.2b showed a real effect that five seeds could not see. The obvious question is
which *other* five-seed comparisons in this document are in the same position, and
it is answerable for free wherever per-seed arm summaries survive. Seed-paired,
bootstrapped over seeds, on `eval/violating_eval_fraction`:

| claim | section | ref % → arm % | Δ pts | 95% CI | crosses 0 |
|---|---|---|---|---|---|
| `constraint_fix_sat_rdf` probe+margin | §3.3 | 43.6 → 32.7 | −12.1 | [−30.3, **+12.1**] | **yes** |
| `alpha_rule` Eq. 8 vs fixed α | §4.2 | 3.6 → 0.0 | 0.0 | [−12.1, 0.0] | yes |
| `alpha_rule` variance rule vs fixed | §4.2 | 3.6 → 1.8 | 0.0 | [−12.1, **+6.1**] | **yes** |
| `constraint_fix` dar probe+margin | §3.2 | 1.8 → 0.0 | 0.0 | [−6.1, 0.0] | yes |

**Two different things are on that list and they need different treatment.**

*Comparative* claims — "our arm beats the baseline by Δ" — are the ones with a
real problem. **`constraint_fix_sat_rdf` is the serious case**: §3.3's planner-fix
story rests on 43.6% → 32.7%, an 11-point improvement whose interval spans
+12.1. That is §3.2a's situation exactly, and §3.2b says the honest response is
20 seeds rather than a re-analysis. Same for the `alpha_rule` variance row, which
§4.2 uses to argue Eq. 8 beats the variance rule.

*Descriptive* claims — "our arm reached 0% violations across five seeds" — are
**not** invalidated by a crossing interval, and reporting them as such would be
its own error. The dar rows and `alpha_rule`'s Eq. 8 row are floor effects: the
reference rate is 1.8–3.6%, so at n = 5 no comparative test can exclude zero no
matter how good the arm is. "0 violations in 55 evaluations" remains true and is
what those sections mostly assert. What must *not* be claimed from them is that
the arm is *significantly better* than the reference.

So the audit produces a short, prioritised list rather than a blanket re-run:

1. **`constraint_fix_sat_rdf` at 20 seeds** — §3.3's planner-defect result, an
   11-point comparative claim with an interval spanning zero. Highest value.
2. **`alpha_rule` at 20 seeds** — §4.2's mixing-rule comparison, same shape.
3. Everything else on the list is descriptive at a floor and needs a wording fix,
   not a rerun: state the rate and its sample size, drop any implied comparison.

**Rule 40: distinguish a claim that an arm *achieved* a rate from a claim that it
*beat* another arm.** The first survives a small sample and a wide interval; the
second does not. Most of this document's five-seed rows are the first kind, and
§3.2's headline was the second kind wearing the first kind's clothes.

### 3.3 Three planner defects, found on a second family

rdf, as shipped: **85% of evaluations violated** at 2.5× the limit, in every
arm, regardless of the probe. Each defect was read from the training trace once
the previous was removed.

**Defect 1 — action saturation.** The reward-greedy sprint drives the pre-tanh
mean past `|μ| ≈ 3`, where `tanh'(μ) ≈ 0`:

| iteration | cost | λ | α | ‖g_pw‖ | ‖g_lr‖ | var(g_pw) |
|---|---|---|---|---|---|---|
| 20 | 1.05 | 0 | 0.70 | 0.55 | 2.9 | 1e-2 |
| 40 | 5.87 | 0 | 0.81 | 1.9 | 196 | 0.7 |
| 100 | 8.08 | 14.5 | 0.95 | **0.036** | 5,160 | **0.0** |
| 198 | 8.21 | **38.5** | 0.99 | **0.036** | 12,700 | **0.0** |

Three things compound: the pathwise gradient dies; **the variance-optimal rule
then selects the dead branch** (α → 0.99, because zero variance reads as
precision); and the LR branch explodes because every rollout returns the same
value and `σ_R → 0`. λ = 38 multiplies nothing.

Fix: a soft wall `ρ · relu(|μ| − μ₀)²` on both branches (`μ₀ = 1.5`, where
`tanh'(1.5) = 0.18`) plus a floor on `σ_R`. **85% → 44%.**

**Defect 2 — dual oscillation.** λ collapses to 0 whenever cost dips below `d`;
the policy sprints and overshoots. *Lowering* the gain made it worse (60%).
Raising `K_i` ×10 lets λ hold. **44% → 14.5%.**

**Defect 3 — tail violations.** A dual holding the *mean* at `d` violates on
~half of evaluations by construction. Adding the policy's own episode spread to
the margin: **14.5% → 5.5%**, worst case 3.23 against `d = 3.26`.

**dar regression:** same configuration, return unchanged (−2.304 vs −2.302), 0%
violations, worst case 3× smaller. One configuration for both families.

**The diagnosis transfers.** The same `K_i` change took PPO-Lagrangian on rdf
from 1.8% to 0% violations. The failure belongs to the PID controller, not to
PSPE.

### 3.4 A testbed that could not rank planners

`docs/results/ROBUSTNESS_NOTES.md` §7

Calibration checked that **cost** separates, so the constraint binds. It never
checked that **return** separates, so actuation can move the objective. swe
passed the first and failed the second for the entire project.

| probe | result |
|---|---|
| 400 random constant action vectors | **not one** beats doing nothing |
| hand-written state-feedback damper | +0.0098 (5.5% of objective) |
| trained reward-greedy planner | +0.0122 (7% of objective) |

The task is controllable and the planner does find control — it beats both
doing nothing *and* the hand damper. The achievable band is simply ≈7%, and
method differences live inside it. That is why swe's joint/disaggregated/
unanchored arms all returned −0.1733 to four decimals.

**The fix that fails.** Raising actuator amplitude `A` grows the margin for a
*deterministic* controller (5.5% at A=1 → 32.6% at A=4) but gives a *stochastic*
policy nothing, because `A` scales the useful signal and the exploration noise
equally (separation +0.0122 at A=1, +0.0081 at A=4). A new objective is needed,
not a louder actuator.

**Correction to earlier drafts:** they said the swe policy "never leaves its
initial behaviour." It does, and it improves. The band is too narrow to rank
methods within — a different and more precise statement.

### 3.4a A positive control overturns the diagnosis above (2026-09-28)

§8.4 item 12 is to build a third PDE family, and the sensible starting point was
to revive `swe` with an objective that separates. `eval/run_testbed_calibrate.py`
measures that. Running it against **`rdf` as a positive control** — the family
the whole §3.2 safety story runs on, which demonstrably ranks planners — was
what made this section wrong.

Same probe, 60 random constant action vectors, batch 24, identical initial
conditions across every vector:

| testbed | target | do-nothing | best random | span | directions that improve | cost at best | limit |
|---|---|---|---|---|---|---|---|
| **rdf** (works) | −1.0 | −7.638 | −7.098 | 7.1% | **23/60** | 3.385 | 3.26 |
| **swe** (retired) | 0.0 | −0.1867 | −0.2122 | **−13.7%** | **0/60** | 2.467 | 0.192 |
| swe | 0.1 | −0.6758 | −0.6673 | 1.3% | 3/60 | 5.438 | 0.192 |
| swe | 0.2 | −2.125 | −1.988 | 6.4% | 11/60 | 7.564 | 0.192 |
| swe | 0.3 | −4.534 | −4.265 | 5.9% | 14/60 | 7.564 | 0.192 |

The `swe` row at target 0 reproduces §3.4 exactly — not one of 60 vectors beats
doing nothing, where §3.4 found the same over 400. But **`rdf` scores 7.1%,
which is the same ≈7% band that got `swe` retired.** And on the *trained*
planners already in `runs/`, the retired testbed is the wider of the two:

| testbed | do-nothing | trained baseline | return span | worst cost | limit | **cost / limit** |
|---|---|---|---|---|---|---|
| swe (retired) | −0.1867 | −0.1733 | **7.2%** | 0.1186 | 0.192 | **62%** |
| rdf (works) | −7.638 | −7.304 | **4.4%** | 4.515 | 3.26 | **138%** |

**The retired testbed has the larger return span.** So "return does not separate"
cannot be why `swe` could not rank planners, and §3.4's stated diagnosis needs
correcting. Two statistics do separate the families, and both are in the tables
above:

1. **Whether any improving direction exists at all.** rdf 23/60, swe 0/60. Not
   the *size* of the achievable band but whether actuation helps or hurts. On
   `swe` at target 0 random actuation is *strictly worse* than idling, because
   the field damps to the target on its own and any control pushes it away.
2. **Whether chasing return drives cost to the limit.** rdf's reward optimum
   lands at 3.385 against a 3.26 limit — 104%, right on it. `swe`'s trained
   planner reaches 62% of its limit and never activates the constraint.

And (1) causes (2), which is the actual mechanism: **no improving direction →
the planner learns to idle → cost stays at the do-nothing level → the constraint
never binds → every arm is the same run.** That is why all four `swe` arms return
−0.1733 and 0% violations to four decimals. It was never the narrowness of the
band; it was that the band pointed the wrong way.

**Rule 36: admit a testbed on two measurements, not one — that some fraction of
random actuation *improves* return, and that the reward optimum puts cost near
the limit.** Span magnitude is not the criterion; `rdf` ranks planners on 4.4%.
Any threshold on span alone, applied honestly, rejects the family this project's
headline safety result is built on.

**Where this leaves item 12.** A non-zero target does fix problem (1) — improving
directions go 0/60 → 14/60 at target 0.3, against rdf's 23/60 — so the mechanism
was identified correctly even though my reason for expecting it was wrong (I
predicted the *span* would widen; it did not, staying at 5.9%). What it does not
fix is (2): cost at the reward optimum is 7.564 against a limit of 0.192, forty
times over, because `u_max = 0.04` was calibrated for a target of 0 and a field
held at 0.3 is above the cap everywhere and always. **The remaining work is
recalibrating `u_max`, `budget` and `cost_limit` for the new target** — the same
retuning the `TASK_SPECS` comment records for the original `swe` — so the
constraint binds instead of being violated trivially. Only then is the margin
comparison worth running.

**The cost recalibration failed, and said why.** Sweeping `u_max` over
target + {0.02 … 0.20} with the field traces recorded once, the exposure term
came out **identically zero at every cap**, and the whole cost was actuation
overspend — a constraint decoupled from the PDE state, which is worse than the
one being replaced. The reason is in the field statistics: at target 0.3 the
tracked channel's peak is **0.172** under the reward optimum and 0.150 under
idling. The target is not merely hard, it is *unreachable*, so the residual error
swamps whatever the control achieves and the band stays narrow whatever the cap.

So the binding limit is **actuator authority**, and it is worth measuring rather
than inferring. Highest sustained *mean* level any constant action holds at the
final step (40 random vectors, identical initial conditions):

| horizon | idle mean level | best reachable mean | best peak |
|---|---|---|---|
| 12 | −0.0023 | +0.0130 | +0.1401 |
| 24 | −0.0023 | +0.0281 | +0.1835 |
| 48 | −0.0023 | +0.0584 | +0.2523 |
| 96 | −0.0022 | +0.1185 | +0.2813 |

**Reachable level is linear in horizon** — each doubling roughly doubles it
(×2.16, ×2.08, ×2.03), so `h_damp = 0.02` is far too weak to saturate the
control over these episode lengths. At the horizon 12 that every `swe` result in
this document used, the control can hold a mean level of **0.013**. A target of
0.3 is 23× outside that, and even 0.1 is 7.7× outside it.

That reframes §3.4's conclusion once more. "A different objective, not a louder
actuator" is right that amplitude does not help a stochastic policy, but the
quantity that was missing is neither: it is **horizon**. The objective and the
authority have to be matched, and at horizon 12 no target is both reachable and
large enough to make idling costly.

> Three prediction failures in one experiment, all worth recording. I expected a
> non-zero target to widen the return span: it does not. I expected the probe to
> confirm §3.4: it did, and then the positive control showed §3.4's reasoning
> does not survive contact with the family next door. And I expected the cost
> caps to be the remaining obstacle: they were a symptom, and the obstacle was
> that the target could not be reached at all.

**Matching the objective to the authority revives the testbed.** With the target
set inside what the control can reach and the horizon raised to give it time,
40 random vectors, batch 16:

| testbed | horizon | target | do-nothing | best random | span | improving directions |
|---|---|---|---|---|---|---|
| **rdf** (reference) | 12 | −1.0 | −7.638 | −7.098 | 7.1% | 23/60 = **38%** |
| swe | 96 | 0.04 | −1.586 | −1.368 | 13.8% | 2/40 = 5% |
| swe | 96 | 0.08 | −3.498 | −2.120 | **39.4%** | 12/40 = 30% |
| **swe** | **96** | **0.12** | −6.640 | −3.462 | **47.9%** | 15/40 = **37.5%** |

**`swe` at horizon 96 with a target of 0.12 matches rdf's improving-direction
rate to within half a point and has close to seven times its span.** By both
criteria in Rule 36 it is now the better of the two testbeds, and the thing that
was missing all along was neither the objective nor the actuator amplitude but
the **horizon**: at 96 steps the control has time to reach 0.12, and holding it
there against `h_damp` is work that idling does not do. Target 0.04 stays dead
(2/40) because it sits below what idling already achieves — the original failure,
reproduced at the other end of the range, which is a useful bracket.

**Placing the limit, and the convention nobody wrote down.** At horizon 96 the
do-nothing episode cost is **0.5114** against the shipped limit of 0.192, so
idling itself violates — the limit was defined on 12-step episodes and nothing
about it survives an 8× longer horizon. Recalibrating it turned up something
useful about the three limits already in `TASK_SPECS`:

| testbed | do-nothing cost | reward-greedy cost | limit | where the limit sits |
|---|---|---|---|---|
| dar | 0.229 | 2.249 | 0.936 | **35.00%** |
| swe (old) | 0.108 | 0.347 | 0.192 | **35.15%** |
| rdf | 0.594 | 8.213 | 3.260 | **34.99%** |

**Every shipped limit sits at 35% of the way from doing nothing to the reward
optimum.** That is an exact convention and it was never stated, so a fourth
family had no documented rule to follow. It is now in
`eval/run_testbed_calibrate.py --calibrate-cost`, which traces the fields once
and sweeps the caps in numpy — both cost terms enter only through a relu, so the
solver need not be re-run per candidate.

Sweeping for horizon 96, target 0.12 (`u_max` above the target, three budgets):

| u_max | budget | idle cost | best-random cost | what carries the cost |
|---|---|---|---|---|
| 0.14 | 0.20 | 0.0017 | 20.0644 | actuation overspend |
| 0.14 | 0.35 | 0.0017 | 5.6644 | actuation overspend |
| **0.14** | **0.60** | **0.0017** | **0.2338** | **field exposure** |
| 0.17 | 0.60 | 0.0001 | 0.0641 | field exposure |
| 0.22 | 0.60 | 0.0000 | 0.0035 | cap too high to bind |

The budget choice decides *what the constraint is about*. At 0.20 and 0.35 the
optimum's mean |a| of 0.4066 exceeds the budget and the overspend term carries
essentially all the cost, leaving exposure a rounding error — a constraint on
actuation spend that has stopped coupling to the PDE state, which is the
degenerate failure §3.4a already diagnosed at target 0.3. At 0.60 the budget term
is slack and the cost is genuinely the field overshooting its cap, which is the
constraint worth having: *hold the level without exceeding it*. So
**`u_max = 0.14`, `budget = 0.60`**, and by the 35% rule `cost_limit ≈ 0.083`.

**One honest gap before this family can be used.** That 0.083 is a **lower
bound**, because the upper anchor here is the best of 40 random constant vectors
while `TASK_SPECS` anchors on a *trained* reward-greedy planner, which reaches a
higher cost — and §3.4's own table shows the trained planner beating the best
random vector on `swe`. The limit has to be re-derived from a trained
unconstrained run before the margin comparison goes on this family, and the
script says so in its output rather than leaving the caveat to a reader. The
objective is solved; the calibration is one GPU run from done.

**Defect 33 — the first version of this probe drew fresh initial conditions per
rollout.** `env.reset()` takes a generator; not passing one reseeds the random
field every call, so scoring 120 action vectors scored them on 120 *different*
episode batches. "Best of N" then selects lucky initial states as much as good
actions. It reported a **29.4% span on `swe` at target 0, with random beating
do-nothing** — flatly contradicting §3.4's 400-vector sweep. Fixing the
generator reproduced §3.4 (0/60) and turned the span negative. **Rule 37: a
best-of-N search over actions must score every candidate on identical initial
conditions, and a probe that contradicts an established result is suspect before
the result is.** The contradiction was the bug report.

---

### 3.5 What the margin should bound: the derived answer

`runs/margin_synthetic/`, `pspe/plan/margins.py`, `tests/test_margins.py`

§3.2 established empirically that conformalising surrogate error fails and
conformalising realised episode cost works. Deriving the threshold turns that
from an observation into a rule.

**The algebra.** With `d_eff = d − q` the realised mean sits at `d_eff + b`, so
an episode violates when `n_i > q − b`, and coverage at level `δ` needs

```
q  ≥  b + z_δ · σ            z_δ = Φ⁻¹(1 − δ)
```

The matched-pair residual is `c_i − g_i = b − e_i`, so its quantile is about
`b + z_δ · ε`. **The bias appears on both sides and cancels.** The default
recipe therefore covers exactly when

```
ε  ≥  σ
```

that is, when the model's **per-instance** error is at least the policy's
episode-to-episode spread. The diagnostic is `ρ = σ / ε`, crossover at `ρ = 1`.

*Correction to earlier drafts:* this was first written as `ρ = σ / |bias|`,
which is wrong — bias shifts the requirement and the quantile equally and drops
out. Caught by deriving the threshold rather than reasoning from the symptom.

**Confirmed in a controlled setting.** Bias and spread dialled independently,
nothing learned, 600 calibration draws per point, δ = 0.1:

| ρ = σ/ε | no margin | model error | deviations + bias | **residual** |
|---|---|---|---|---|
| 0.10 | 0.841 | **0.000** | 0.098 | 0.098 |
| 0.50 | 0.841 | **0.004** | 0.097 | 0.097 |
| 0.80 | 0.842 | **0.054** | 0.099 | 0.100 |
| **1.00** | 0.841 | **0.099** | 0.101 | 0.101 |
| 1.25 | 0.842 | 0.153 | 0.100 | 0.100 |
| 2.00 | 0.841 | 0.258 | 0.099 | 0.099 |
| 10.0 | 0.841 | 0.449 | 0.099 | 0.100 |

The default sits at 0.099 exactly at `ρ = 1` and degrades monotonically above
it. Our recipe holds 0.097–0.101 across two orders of magnitude. The no-margin
arm tends to 0.5 as spread grows, which is §4.1's claim — a dual holding the
mean violates on half of evaluations — measured rather than asserted.

**A simpler margin with a cleaner guarantee.** §3.2 conformalises `c_i − c̄` and
adds `b̂ = max(0, bias)` separately, so Prop. 1 needs the dual to hold a
*bias-corrected* mean. Conformalising

```
s_i = c_i − ĝ        ĝ = the quantity the dual controls
q   = s_(⌈(n+1)(1−δ)⌉)        d_eff = d − q
```

covers bias and spread in one order statistic, needs no separate bias term, and
the condition becomes exactly what the dual already does. It **reduces to the
matched-pair recipe when control is per-instance**, which is why the default is
right in the pointwise-control setting it was developed for and wrong in a
CMDP, where the dual holds an expectation.

**Two rules that generalise the finding.** Both were violated by this project's
own experiments before being stated:

1. **Same estimator, same unit, same aggregation.** A margin is valid only if
   its calibration scores are exchangeable with the quantity compared against
   the limit. §3.2's failure (model error vs episode cost) and §5.5's
   (per-fire vs batch-mean) are the same error.
2. **`b < f · s`** — see §5.5.

All three recipes live in `pspe/plan/margins.py`, which both the PDE and
wildfire experiments call, so no comparison can differ by implementation. The
module's unit test is the mechanism in three numbers: with an accurate
per-instance model and a policy that scatters, the model-error margin is 0.017
where 0.662 is needed, and the residual margin returns 0.662 exactly.

**On a real planner: the margin behaves as predicted, the violation rate is
underpowered.** `runs/margin_choice/`, rdf, 5 surrogate qualities × 3 seeds,
15 runs × 4 recipes.

**The result.** As the surrogate improves, the default's margin collapses while
ours does not:

| surrogate rel L2 | `model_error` | `residual` | `episode` |
|---|---|---|---|
| 0.8172 | 0.385 | 0.693 | 0.664 |
| 0.5437 | 0.572 | 0.671 | 0.608 |
| 0.4987 | 0.860 | 1.012 | 1.032 |
| 0.3478 | 0.234 | 0.555 | 0.589 |
| 0.1881 | 0.167 | 0.575 | 0.540 |
| 0.1479 | 0.106 | 0.505 | 0.504 |
| 0.1081 | 0.076 | 0.453 | 0.507 |
| 0.0691 | 0.047 | 0.492 | 0.525 |
| 0.0323 | 0.027 | 0.593 | 0.641 |
| 0.0114 | **0.008** | **0.506** | 0.510 |

A **48× collapse** in the default's margin, 0.385 → 0.008, against a limit that
stays at 3.26. Ours is flat: mean 0.616, sd 0.143, no trend with fidelity.
That is the mechanism stated as directly as it can be — a quantile over model
error shrinks toward nothing as the model improves, while a quantile against
the controlled quantity tracks the policy's spread, which does not improve with
the model.

This evidence is **continuous**, so it is not subject to the resolution limit
below, and it is arguably better than a violation rate: it measures the cause
rather than inferring it from a coarse binary outcome.

**The violation rate, at adequate resolution and power.** The first sweep used
`eval_every 20` — 11 evaluations per run, so every rate was a multiple of 1/11
and δ = 0.1 was not attainable. Rerun at `eval_every 5` and extended to 5 seeds
(`runs/margin_choice_hires/`, 25 runs, 41 evaluations each):

| | `none` | `model_error` | `episode` | `residual` |
|---|---|---|---|---|
| **ρ < 8**, 17 runs, 697 evals | 0.1162 | **0.0718** | 0.0603 | 0.0603 |
| **ρ ≥ 8**, 8 runs, 328 evals | 0.1403 | **0.1402** | 0.0671 | 0.0732 |

| contrast | ρ < 8 | ρ ≥ 8 |
|---|---|---|
| `model_error` vs `residual` | z = +0.86 | **z = +2.78** |
| `none` vs `residual` | **z = +3.68** | **z = +2.78** |

**The claim, confirmed at significance.**

* Where the model's per-instance error dominates (ρ < 8) the default **works**:
  0.0718, inside δ = 0.1, and statistically indistinguishable from ours
  (z = 0.86). The recipe is not wrong in general, and saying so is what makes
  this a condition rather than a complaint.
* Where the policy's spread dominates (ρ ≥ 8) the default is **worth nothing**:
  0.1402 against 0.1403 for applying no margin at all — the same number to
  three decimal places — and above the level it states. Ours holds 0.0732.
  The contrast is z = 2.78.

That the default and the no-margin baseline coincide at high ρ is the sharpest
form of the finding. It is not that the margin is loose; it is that the
quantity it conformalises has stopped carrying information about what breaches
the limit, so tightening by it changes nothing.

Our recipe reduces violations against no margin in **both** regimes
(z = 3.68 and z = 2.78) and holds δ in both.

**On the pre-registration.** `paper/REVISION_NOTES.md` §8e set this as the test
before any of it ran. At 11 evaluations it could not be run and the margin
collapse was reported instead, flagged at the time as a post-hoc substitute.
The test has now been run as written, at adequate resolution and power, and the
pattern it predicted is the pattern observed. The substitution caveat is
withdrawn; the margin-collapse evidence below supports this result rather than
standing in for it.

**Against a model-based safe-RL baseline.** Every other safe-RL baseline in
this project is model-free, which makes "fewer real transitions than model-free
methods" near-tautological for a model-based planner (§9.2). CAP (Ma et al.,
AAAI 2022) is the closest method that also plans in a learned model and also
corrects for it being wrong, inflating the cost estimate by ensemble
disagreement, `c = mean + k·std` over five surrogates, with `k` adapted from the
same periodic real probe we use. `eval/run_cap_baseline.py` transplants that
penalty onto our planner — same dual, testbed, limit, evaluation protocol and
**probe budget** — so only the correction mechanism differs. It is not a
reimplementation of the paper, and the docstring says so.

Matched real-sample budget (6,400 training transitions plus 1,920 probe), rdf,
full-fidelity surrogate:

| method | violating | seeds |
|---|---|---|
| no margin | 0.1317 | 5 |
| model-error conformal | 0.1415 | 5 |
| **CAP (ensemble penalty)** | **0.1199** | 12 |
| **ours (residual)** | **0.0780** | 10 |

| contrast | z |
|---|---|
| CAP vs no margin | **−0.43** |
| ours vs no margin | **+3.68** |
| **ours vs CAP** | **+2.08** |

**CAP provides no measurable benefit over applying no margin at all.** Its
adapted `k` settled between 0.50 and 1.46, so the mechanism was active rather
than inert; ensemble disagreement simply is not the quantity that breaches this
limit, for the same structural reason model error is not. Ours improves
significantly on both.

> **Provenance audit of these three z-values (2026-09-28).** §3.2a found the
> five-seed margin comparison could not separate from zero once differenced
> within seed, so these z-values were checked for the same weakness. **There is
> no two-proportion z-test anywhere in the repository** — no committed script
> produces them, which is the same unrecorded-provenance situation that produced
> the fabricated z-values corrected in §9.15. Reconstructing them from the
> fractions in the table above:
>
> | contrast | published | pooled over evaluations | reproduces |
> |---|---|---|---|
> | ours vs CAP | +2.08 | **+2.08** | exactly |
> | CAP vs no margin | −0.43 | ±0.43 | yes, sign convention only |
> | **ours vs no margin** | **+3.68** | **+2.13** | **no** |
>
> Two things follow. First, **the unit is the evaluation, not the run**: each
> arm's fraction is a whole number of violations over 41 evaluations per run
> (32/410 for ours, 59/492 for CAP, 27/205 and 29/205 for the two five-seed
> arms), and only n = 410 against 492 returns +2.08. Second, **`ours vs no
> margin` cannot be reproduced from its own numbers** under the unit that
> reproduces the other two, and +3.68 (p ≈ 0.0002) against +2.13 (p ≈ 0.03) is
> the difference between decisive and marginal. Until it is re-derived it should
> be reported as +2.13 or not at all.
>
> **The methodological problem is larger than the arithmetic one.** Pooling over
> evaluations treats 41 evaluations inside one training run as 41 independent
> Bernoulli trials. They are the same policy at nearby points on one trajectory,
> so they are strongly dependent and the effective sample size is far closer to
> the number of runs (10 and 12) than to the number of evaluations (410 and 492).
> That inflates every z in this table, by up to √41 ≈ 6.4 in the limit of perfect
> within-run correlation. The correct test differences per-run fractions and
> resamples over runs, exactly as §3.2a does for the five-seed table — and when
> that is done there, the effect vanishes into its own interval.
>
> The per-run fractions survive for CAP (12 values, mean 0.1199) but **not for
> our residual arm**: only the aggregate 0.0780 was kept, in `runs/cap/` and on
> Vista alike. So the run-level test cannot be done retrospectively on this
> table. Vista job **1032033** re-runs the conformal comparison at **20 seeds**
> with per-seed output preserved, which is what settles both this and §3.2a.
>
> **What stands meanwhile.** The *structural* argument — that corrections built
> on model disagreement do not bound the quantity that breaches an expectation
> constraint — does not rest on these z-values. It rests on CAP landing at 0.1199
> against no-margin's 0.1317 (a 0.4σ nothing, on any unit) while its `k` was
> demonstrably active, and it makes a prediction that §9.2a registered in advance
> and job 1031901 is testing on SMBPO. A mechanism that predicts the next
> method's result is worth more than a z-value that cannot be reproduced.

**The run-level test, now possible (2026-09-28).** The obstacle was that our
residual arm kept only its aggregate. §3.2b's 20-seed run supplies 20 per-run
fractions, and CAP's 12 survive, so the comparison can be made at the correct
unit of independence. The two arms are **matched on real samples — 8,320
training and 1,920 probe transitions each** — so only the correction mechanism
and the evaluation density differ:

| test | unit | statistic | p |
|---|---|---|---|
| published | 410 vs 492 **evaluations** | z = +2.08 | ≈ 0.037 |
| Mann-Whitney U | 20 vs 12 **runs** | U = 29.0 | **0.00015** |
| Welch t | 20 vs 12 **runs** | t = −4.39 | **0.00013** |

CAP − ours = **+6.99 points, 95% bootstrap CI [+3.96, +9.98]** over runs.

**Correcting the unit makes the result stronger, not weaker.** That is worth
stating plainly, because the audit above was written expecting the opposite: the
inflation from pooling correlated evaluations was real, but it was more than
offset by our arm improving from 0.0780 at ten runs to **0.0500** at twenty. The
published +2.08 was both computed on the wrong unit *and* an understatement of
the effect.

*Caveat.* This is unpaired — different seeds, different runs — and the two arms
evaluate at different densities (11 evaluations per run against CAP's 41). That
changes the precision of each run's fraction, not its expectation, which Welch's
t accommodates and the rank test sidesteps. A matched-density rerun of CAP at 20
seeds would remove the caveat; the conclusion does not depend on it, since the
gap is 7 points with a lower bound of 4.

**The general form of the finding.** Two mechanistically different uncertainty
corrections — a conformal quantile over model error, and an ensemble-variance
penalty — both reduce to doing nothing on an expectation constraint, while
conformalising against the quantity the dual controls works. The common cause
is that both estimate *how wrong the model is*, and under an expectation
constraint that is not what carries the policy over the limit. This is a
stronger claim than a baseline comparison and is what the abstract should say.

**A caveat in CAP's favour.** It matches on real samples but uses five times the
training compute, the ensemble being five models fitted to the same data. A
compute-matched comparison would be less generous to it; a sample-matched one is
the right choice for a claim about constraint satisfaction rather than
efficiency, and is the one reported.

**ρ > 1 across the entire sweep.** Measured ρ = σ/ε ranges 1.20 to 96.24, and
all 15 runs exceed 1 — even the worst surrogate tested (rel L2 0.82, worse than
predicting zero). **The regime where the default is correct is not reachable on
this testbed.** That matters for the framing: §3.5's threshold is real and
derived, but in a CMDP with any usable surrogate the system sits far above it.
The contribution is better stated as a **correction** — matched-pair
conformalisation is the wrong recipe here — than as a **diagnostic** to choose
between recipes, with the per-instance case noted as the exception that
explains the CBF literature.

**Pending:** a rerun at `eval_every 5` giving ~41 evaluations per run and 615
pooled per arm, enough to resolve a 3-point difference and settle whether the
violation ordering is real; and the stage-2 wildfire arms (§5.5).

### 3.6 A separation theorem, and the transition in closed form (2026-09-28)

§8.5 lists T1.1 (a separation result) and T1.2 (closed-form coverage) as the two
theory items that answer the reviewer's "Proposition 1 is essentially standard
conformal coverage after redefining the residual". Both are derived here, and
T1.2 reproduces §3.5's measured transition to within Monte-Carlo error.

**Setup.** Write the realised episode cost as

    c_i  =  μ + b + σ ξ_i ,     ξ_i ~ F,  E ξ = 0,  scale 1

where `μ` is the quantity the dual holds, `b` the surrogate's bias, and `σ` the
policy's episode-to-episode spread. The surrogate's per-instance estimate
satisfies `c_i − g_i = b − e_i` with `e_i ~ G`, mean 0, scale `ε` (§3.5). A
margin `q` lowers the effective limit to `d_eff = d − q`. Then

    violation  ⟺  σ ξ_i > q − b ,

so the **realised violation rate** of any margin `q` is

    R(q; σ)  =  1 − F( (q − b) / σ )                                    (1)

and coverage at level δ holds iff `q ≥ b + z_δ σ`, with `z_δ = F⁻¹(1 − δ)`.

---

**Proposition 3 (separation).** Let `Q` be any margin rule that is a measurable
function of the *model-error* distribution alone — that is, `q = Q(b, G)`, with
no dependence on `F` or `σ`. If `F` has full support then:

1. `R(q; σ)` is strictly increasing in `σ`;
2. `R(q; σ) → 1 − F(0) = ½` as `σ → ∞`, for symmetric `F`;
3. coverage **fails** for every `σ > (Q(b, G) − b) / z_δ`.

Consequently **no margin measurable with respect to model error alone attains
level δ uniformly in σ.** Attaining δ requires the margin to depend on the
realised-cost distribution.

*Proof.* Immediate from (1): `(q − b)/σ` is decreasing in `σ` and `F` is
increasing, so `R` increases; the limit is `1 − F(0)`; and `R > δ` exactly when
`(q − b)/σ < z_δ`. ∎

The result is not deep, and that is the point — it is the statement that was
missing, not a harder proof of the one already there. It says the *choice of
distribution* is not a modelling preference but a feasibility constraint.

---

**Corollary 3.1 (the default recipe, and where ρ = 1 comes from).** The
matched-pair conformal margin takes `q = Q_{1−δ}(c − g) = b + z^G_δ ε`.
Substituting into (3):

    coverage fails  ⟺  σ  >  (z^G_δ / z^F_δ) · ε

and when `F` and `G` share a shape this is exactly **ρ = σ/ε > 1**. §3.5's
threshold, derived in §3.5 by matching quantiles, is the special case of
Proposition 3 in which the offending margin happens to be a conformal one.

**Corollary 3.2 (closed-form coverage — T1.2).** For Gaussian `F`, the default
recipe's realised violation rate is

    R(ρ)  =  1 − Φ( z_δ / ρ )                                          (2)

Against `runs/margin_synthetic/`, δ = 0.1, 600 calibration draws per point:

| ρ = σ/ε | measured | **predicted by (2)** | error |
|---|---|---|---|
| 0.10 | 0.000 | **0.000** | 0.0000 |
| 0.50 | 0.004 | **0.005** | 0.0012 |
| 0.80 | 0.054 | **0.055** | 0.0006 |
| **1.00** | **0.099** | **0.100** | 0.0010 |
| 1.25 | 0.153 | **0.153** | 0.0004 |
| 2.00 | 0.258 | **0.261** | 0.0028 |
| 10.0 | 0.449 | **0.449** | 0.0000 |

**Maximum error 0.0028 across two orders of magnitude of ρ**, which is the
Monte-Carlo noise of 600 draws. The transition §3.5 observed is not an empirical
regularity to be plotted — it is (2), and the crossing at `ρ = 1` is
`R(1) = 1 − Φ(z_δ) = δ` by construction.

---

**Corollary 3.3 (why CAP and SMBPO are indistinguishable).** CAP penalises by
`mean + k·σ_ens` with `k` adapted from observed violations; SMBPO by the
`max` over ensemble members with no adaptation. Both are functionals of the
**ensemble-disagreement** distribution, which estimates `G` — how wrong the
model is — and neither is a functional of `F`. Both therefore fall under
Proposition 3 and fail once `ρ > 1`.

Note what the Proposition does *and does not* say. It says any such functional
fails; it does **not** rank them. Two different summaries of `G` should land in
the same place, because the theorem is indifferent to which summary is used.

> **This is the prediction §9.2a got wrong, and the theorem would have got
> right.** §9.2a predicted SMBPO would lose to the margin *by more than CAP*,
> reasoning that `max` is cruder than an adapted `mean + kσ`. Measured:
> **11.99% against 12.20%, p = 0.83** — indistinguishable, exactly as
> Proposition 3 implies. The pre-registered prediction contradicted a theorem
> that had not yet been derived. Deriving it first would have produced the
> better prediction, which is an argument for doing the theory before the
> baseline rather than after.

**Corollary 3.4 (what a valid margin must use).** By Proposition 3, any margin
attaining δ uniformly must depend on `F` and `σ`. The recipe of §3.5,
`s_i = c_i − ĝ` with `ĝ` the quantity the dual controls, is measurable with
respect to the realised-cost distribution and is therefore admissible; §3.2b
measures it holding 5.0% against a 14.1% baseline at δ = 0.1.

**What this buys the paper.** The contribution is no longer "we chose a better
conformal score". It is a **feasibility statement**: an entire family of
corrections — every method that inflates by model uncertainty, however
summarised — cannot attain a coverage level under an expectation constraint once
policy spread exceeds model error, and `ρ` says exactly when. CAP and SMBPO are
then not two baselines that happened to lose; they are two instances of a class
the theorem excludes, and their near-identical results are the prediction.

**Assumption, stated plainly.** (1) needs the episode deviation to be a scale
family, `n_i = σ ξ_i` with `ξ` of fixed shape. That is mild, it is testable, and
it is weaker than the Gaussianity used for the closed form (2) — Proposition 3
needs only that `F` is increasing with full support.

## Part 4 — Per-module results

Each module has a panel giving what it does, what held up and what did not.

| module | panel |
|---|---|
| Perceive | [`pspe_mod_perceive.svg`](figures/pspe_mod_perceive.svg) |
| Simulate | [`pspe_mod_simulate.svg`](figures/pspe_mod_simulate.svg) |
| Plan | [`pspe_mod_plan.svg`](figures/pspe_mod_plan.svg) |
| Explain | [`pspe_mod_explain.svg`](figures/pspe_mod_explain.svg) |

![Plan](figures/pspe_mod_plan.svg)

*Figure 4. The Plan panel. The other three follow the same three-band layout:
the mechanism with its equations, then what the measurements supported, then
what they withdrew.*

### 4.1 Theory checks

`runs/lipschitz/`, 5 seeds. `L_G` by power iteration; `ε` the one-step error;
return bias = `|J_R^{G_θ} − J_R^{F}|` for the same policy.

| arm | rel L2 | `L_G` | `L_G ≤ 1` | measured bias | Prop 1 (∞-horizon) | Prop 1 (H = 12) |
|---|---|---|---|---|---|---|
| default FNO | 0.069 ± 0.062 | 0.985 ± 0.013 | 4/5 | 0.10 | **422** | 9.8 |
| spectrally normalised | 0.079 ± 0.026 | 0.964 ± 0.003 | 5/5 | 0.11 | 748 | 17.3 |

Assumption 1 (`L_G ≤ 1`) holds after training (init 1.61). Proposition 1's
infinite-horizon bound

```
|J_R^{G} − J_R^{F}|  ≤  (L_r ε / (1−γ)) · Σ_{h} γ^h L_G^h        →   prefactor 1/(1−γ) = 2450 at γ = 0.98
```

is **vacuous**: 422 against a measured 0.10, a factor of 4,000. The
finite-horizon form holds on every seed and is still ~100× loose. Report the
finite-horizon form and state that it is loose.

### 4.2 The mixing rule

`runs/alpha_rule/`, 5 seeds:

| arm | return | violating | final α | measured B² | real samples |
|---|---|---|---|---|---|
| fixed α = 0.5 | −2.415 ± 0.30 | 3.6% | 0.5 | — | 4,160 |
| variance rule | −2.302 ± 0.17 | 1.8% | 0.992 | — | 4,160 |
| Eq. 8 (measured bias) | −2.299 ± 0.15 | 0% | 0.992 | 0.0013 | 6,080 |

Eq. 8 vs variance: t = +0.22. Its zero-violation result comes from 1,920 extra
truth rollouts, not from the formula. **Adaptive α is a variance effect, not a
mean effect**: +0.290 over fixed α but t = +1.66; what separates them is the
tail (std 0.029 vs 0.365; one fixed-α seed diverged to cost 8.1 with λ = 19.1).

### 4.3 Joint Simulate+Plan training

`runs/joint*/`, 5 seeds, one pretrained surrogate deep-copied per arm:

| testbed | arm | return | held-out surrogate rel L2, before → after |
|---|---|---|---|
| dar | disaggregated | −2.302 ± 0.17 | 0.0048 → 0.0048 |
| dar | joint, anchored (β=0.1) | −2.296 ± 0.18 | 0.0048 → **0.0005** |
| dar | joint, unanchored | −2.369 ± 0.17 | 0.0048 → **0.975** |
| swe | disaggregated | −0.1733 ± 0.010 | 0.0246 → 0.0246 |
| swe | joint, anchored | −0.1733 ± 0.010 | 0.0246 → **0.0027** |
| swe | joint, unanchored | −0.1733 ± 0.010 | 0.0246 → **1.14** |
| rdf | disaggregated | −6.76 ± 0.69 | 0.0024 → 0.0024 |
| rdf | joint, anchored | −6.70 ± 0.70 | 0.0024 → **0.0005** |
| rdf | joint, unanchored | −7.46 ± 0.45 | 0.0024 → **0.83** |

Paired t on return: **+0.66** (dar), identical (swe), **+1.00** (rdf).
Unanchored: −5.84 (dar), −2.41 (rdf).

The planning gradient is scaled to a fraction of the data gradient's norm:

```
∇_θ^{total} = ∇_θ ℒ_data  +  β · (‖∇_θ ℒ_data‖ / ‖∇_θ ℒ_plan‖) · ∇_θ ℒ_plan
```

Without that normalisation the raw planning term (O(1)) swamps the data term
(O(1e-4)) within 8 iterations and held-out error rises 10×.

> **Prior work.** That a more accurate task-agnostic model need not yield a
> better policy is **objective mismatch** [Lambert et al., 2020], with a 2023
> taxonomy sorting the remedies into distribution correction, control-as-
> inference, value equivalence and differentiable planning [*A Unified View*, arXiv 2310.06253].
> Our anchored joint training is the differentiable-planning family; the
> value-equivalence line is [Grimm et al., 2020] and [Farahmand et al., 2017],
> and the decision-focused analogue in optimisation is [Donti et al., 2017;
> Elmachtoub & Grigas, 2022].
>
> **Our data point is the mirror image of the usual one.** Objective mismatch
> says a better model can give a worse policy. We found decision-aware training
> made the surrogate **5–9× better on held-out trajectories** and still never
> moved the decision. Worth reporting as a nuance, not as a discovery. §9.4.

**Verdict:** "joint beats disaggregated" is unsupported on every testbed.
"Joint is safe when anchored and destructive when not" is supported on all
three — anchored, the planning gradient acts as extra regularised training on
the states the policy visits, improving held-out accuracy 5–9×.

### 4.4 Explanation: the certificate holds, the faithfulness claim does not

**What was claimed.** dar/Qwen2.5-0.5B: trained-in 0.468 ± 0.16 vs post-hoc
0.182 ± 0.16, t = +21.3. Real fire plans: 0.814 ± 0.039 vs 0.741 ± 0.005,
t = +3.61. Certificate holds at δ = 0.1 and 0.05 on 3/3 seeds.

**The permutation control.** This is standard practice we initially omitted.
The canonical statement is [Adebayo et al., 2018]: a saliency method that is
"independent both of the model and of the data generating process" fails a
randomisation test and is inadequate, whatever it scores. The chain-of-thought
faithfulness literature runs the same family of controls — counterfactual
perturbation, norm-matched random controls, causal-mediation tests [Turpin et
al., 2023; Lanham et al., 2023]. Our briefs fail exactly the Adebayo test, in
its original form. `docs/results/EXPLAIN_PERMUTATION.md`

| testbed | arm | F aligned | F shuffled | Δ |
|---|---|---|---|---|
| dar, seed 0 | trained-in | 0.2809 | 0.2805 | +0.0004 |
| dar, seed 0 | post-hoc | 0.0034 | 0.0034 | 0.0000 |
| dar, seed 1 | trained-in | 0.5570 | 0.5570 | 0.0000 |
| dar, seed 2 | trained-in | 0.5657 | 0.5659 | −0.0002 |
| NDWS, seed 0 | trained-in | 0.7439 | 0.7436 | +0.0003 |
| NDWS, seed 2 | trained-in | 0.8630 | 0.8639 | −0.0009 |

`Δ = 0` exactly is the signature of a **constant** generator: aligned and
shuffled then use the same multiset of (policy, brief) pairs. The sampled
briefs confirm it — identical actuators at identical amplitudes across
different states.

**Three fixes, all negative.** Wider conditioning (prefix 4 → 16, cond_dim 64 →
256, condition dropout); a differentiable contrastive objective
`−log softmax_j(−NLL(b_i | c_j))` requiring each brief to be cheaper under its
own condition (unit-tested to reach the prefix, sat at `log B` throughout);
2.5× the training budget (NLL plateaus ≈0.5 nats/token, Δ never leaves 0).

**Not a plumbing bug:** on the Qwen path the prefix receives gradient of norm
1.1e5 against 4.6e3 for the LoRA adapters, and NLL moves by 1.51 when the
condition is zeroed.

**The testbed could not have answered the question.** The trained dar policy
varies **1.2%** across states and yields **8 distinct action vectors in 64
states** on the brief's 0.05 quantisation grid. A constant brief is near-optimal
there. Every dar Explain number was measured on a task with nothing to explain.

**What stands:** the certificate machinery. Coverage holds at 90% and 95% on
3/3 seeds on real data, with a non-vacuous floor (F ≥ 0.706 at δ = 0.1). It
reports honestly on a weak generator rather than overstating it.

### 4.5 Simulate

**PDEBench 2-D shallow water**, 512 real trajectories, 64², 20 epochs:

| operator | rel L2 (1-step) | rel L2 (rollout) | params | published FNO (nRMSE) |
|---|---|---|---|---|
| **FNO** | **0.0018** | **0.0098** | 1.19 M | 0.0044 |
| DeepONet | 0.0044 | 0.0137 | 0.09 M | — |
| GNOT | 0.0056 | 0.0543 | 0.31 M | — |

Metrics differ (rel L2 vs nRMSE), so this is an anchor, not like-for-like. The
ranking reproduces on synthetic data (0.057 / 0.256 / 0.268, 5 seeds).

**Resolution invariance**, trained at 64², 5 seeds: 0.0298 / 0.0298 / 0.0300 at
64² / 96² / 128² (4,096 / 9,216 / 16,384 cells). Flat to three decimals.

**Observed wildfire spread (NDWS)**, AUC-PR, prevalence 1.3%:

| model | validation (3 seeds) | **test** (3 seeds) |
|---|---|---|
| **U-Net** | 0.277 ± 0.002 | **0.3162 ± 0.017** |
| Hybrid (U-Net + gated FNO) | 0.235 ± 0.005 | 0.2939 ± 0.003 |
| FNO alone | 0.200 | — |
| all-zeros floor | 0.011 | — |

**Where this sits.** Above the originating baseline (Huot et al., 0.284),
**below the current state of the art**, which is 0.3673 for a single SwinUNETR
with modular augmentations and 0.3790 for a mixed ensemble (arXiv 2609.17763,
3-seed means). Earlier drafts said the surrogate "exceeds the published
benchmark" — true against the 2022 number and misleading against the field, so
it is restated here. Forecast skill is an **input** to the planning result, not
a contribution; we did not set out to optimise it. Conditions: 8,000 training
patches rather than the full split, and flip augmentation (with the
wind-direction channel corrected under the flip). Whether 0.3790 uses an
identical split and protocol is unverified and must be checked before any
comparison table goes into a paper.

Related, and worth citing rather than resisting: *WildfireSpreadBench* (arXiv
2609.22191) argues that on this task the metric outranks the architecture —
the best-AP model flags 4–5× the area that actually burned while ranking fifth
on F1/IoU. That is the same shape as §0.2, measured by someone else on the
forecast rather than the decision. See [`RELATED_WORK.md`](RELATED_WORK.md) §3.

> **Where the operator sits.** FNO [Li et al., 2021], DeepONet [Lu et al.,
> 2021] and GNOT [Hao et al., 2023] on PDEBench [Takamoto et al., 2022] are
> 2021–2023 baselines. The current frontier is transformer PDE foundation
> models — Poseidon [Herde et al., 2024], which needs an order of magnitude
> fewer samples than FNO on 13 of 15 tasks, DPOT [Hao et al., 2024], and BCAT
> [*BCAT*, arXiv 2501.18972]. **The surrogate architecture is not a contribution of
> this work**, and no claim here depends on it being state of the art. §9.5.

**Operator scope, measured.** The hybrid's learned gate over its spectral path
— free to take any value, initialised at zero — settles at **0.12 / 0.111** on
every seed and split. Offered the Fourier operator on sharp fire fronts, the
model declines it. The same FNO reaches 0.0098 on smooth shallow water. A
spectral model truncates the high frequencies that *are* a fire front.

### 4.6 Perceive

Real SigLIP (`google/siglip-base-patch16-224`), 3 seeds:

| arm | field rel L2 | retrieval accuracy | trainable |
|---|---|---|---|
| PSPE — frozen + LoRA + contrastive | 0.0596 ± 0.027 | **0.754 ± 0.026** | 1.64 M |
| probe — frozen, decoder only | **0.0345 ± 0.011** | 0.135 ± 0.009 | 1.64 M |
| CNN from scratch | 0.0564 ± 0.015 | 0.130 ± 0.016 | 0.56 M |

The frozen probe reconstructs best and beats the CNN (t = −4.87). The design
buys **alignment**, not reconstruction: retrieval 0.754 vs 0.130, t = +37.

§5.3 gives Perceive's first *decision-relevant* result on real data.

### 4.7 Cross-family transfer

`runs/transfer_seeds/`, 5 seeds:

| transfer | fidelity gap | measurable? |
|---|---|---|
| rdf → dar | 0.358 ± 0.033 | yes, ~11σ |
| swe → rdf | 0.311 ± 0.13 | yes |
| dar → rdf | 1.81 ± 1.1 | marginal |
| swe → dar | 0.087 ± 0.14 | no — crosses zero |
| rdf → swe | 10.5 ± 8.2 | no — σ ≈ mean |
| dar → swe | 27.4 ± 26 | no — σ ≈ mean |

Direction survives; magnitude does not. And **forecast error decouples from
decision loss**: dar → swe has fidelity gap 27 but planning gap 0.007 ± 0.014
(nothing), while dar → rdf has fidelity gap 1.8 and planning gap 0.58 ± 0.22
(~8% of return).

### 4.8 Ablations

5 seeds, full budget:

| component | on | off | t | verdict |
|---|---|---|---|---|
| physics-informed loss | 0.0567 ± 0.026 | 0.0496 ± 0.044 | +0.52 | no asymptotic effect |
| frozen perception | 0.0681 ± 0.013 | 0.0908 ± 0.017 | −3.97 | frozen wins |
| faithfulness term | 0.643 ± 0.15 | 0.598 ± 0.006 | +0.67 | no effect |

The physics loss buys convergence speed only (0.117 vs 0.273 at a quick budget).

---

## Part 5 — Real-data results

### 5.1 Constrained planning on observed wildfire records

`docs/results/NDWS_PLANNING.md`. NDWS U-Net frozen; 8×8 grid of firebreak
intensities per day; 3%-of-patch daily crew budget; three days ahead;
population-weighted burn as objective. Action model, stated in code:

```
learned:  NDVI channel  ←  NDVI − 2σ · b(x)          the surrogate decides the effect
imposed:  p(x)          ←  p(x) · (1 − 0.9 · b(x))   suppression in treated cells
```

The per-instance planner solves, for each fire,

```
min_{b ∈ [0,1]^{T×K}}   𝔼[ burned_{t+T} · (1 + w·pop) ]        s.t.  mean_k b_{t,k} ≤ B  ∀t
```

by projected gradient through the autoregressive surrogate rollout, with exact
Euclidean projection onto the budget simplex at every step.

> **Prior work on this exact problem.** Firebreak placement has already been
> attacked with deep RL over the **Cell2Fire** simulator [Pais et al., 2021]:
> DQN, Double DQN and Dueling DDQN in *Advancing Forest Fire Prevention*
> (arXiv 2404.08523) and *Deep RL for optimal firebreak placement*
> (Appl. Soft Comput. 2025); with combinatorial optimisation on a landscape
> graph in *A graph-based optimization framework for firebreak planning*
> (2025); and for suppression-resource allocation in *Spatiotemporal Wildfire
> Prediction and RL for Helitack Suppression* (arXiv 2601.14238). The trade-off
> is real and cuts both ways: their
> environment **defines** the intervention, so the counterfactual is exactly
> evaluable, at the price of a simulator gap; ours is fitted to what real fires
> did, and the intervention effect is therefore unvalidated (§6.2). Neither
> dominates. §9.6.

**Headline, 1,500 held-out fires, 5 seeds:**

| policy | burn reduction | treated/day | over budget |
|---|---|---|---|
| random placement † | 4.5 ± 0.3% | 3.0% | 0 |
| greedy: forecast, treat riskiest, repeat | 22.0 ± 4.4% | 3.0% | 0 |
| **PSPE per-instance** | **34.0 ± 4.4%** | 3.0% | 0 |
| PSPE amortised policy | 10.4 ± 6.5% | 2.2% | 17% |
| no budget (reference) † | 85.7% | 61.7% | all |

† The 5-seed extension re-ran only the three contested arms; the two reference
rows are the 3-seed values and are marked rather than silently mixed in.

**Paired t = 11.15** (df 4), mean gap **+12.0 points**. Per-seed gaps run
+9.6 to +15.9, so the two seeds added after the first three did not dilute the
effect.

*Two run families appear in this section and should not be conflated.* The
headline above is `runs/ndws_plan/`, 5 seeds. The Pareto, fuel-only and
constrained-RL comparisons below are `runs/ndws_planning_vista/` and
`runs/ndws_plan_b*/`, 3 seeds, where greedy reads 24.1 ± 3.8 and PSPE
36.5 ± 1.0 (gap +12.4, t = 6.66). The horizon ablation is a third family,
`runs/ndws_horiz_h*/`, 5 seeds. The three agree to within their spreads; each
table states which produced it.

**Budget Pareto** (`runs/ndws_plan_b*/`), 3 seeds:

| crew budget/day | greedy | PSPE | ratio |
|---|---|---|---|
| 1% | 7.5 ± 1.4 | 14.3 ± 0.5 | **1.91×** |
| 2% | 12.8 ± 2.0 | 26.2 ± 0.7 | **2.05×** |
| 3% | 24.1 ± 3.8 | 36.5 ± 1.0 | 1.52× |
| 5% | 36.1 ± 5.0 | 53.0 ± 1.1 | 1.47× |
| 8% | 55.0 ± 5.5 | 69.4 ± 0.6 | 1.26× |

The advantage is **largest where crews are scarcest**. PSPE at 1% matches greedy
at nearly 2% — roughly half the crews for the same outcome.

**The mechanism, measured** (horizon ablation, `runs/ndws_horiz_h*/`, 5 seeds):

| horizon | greedy | PSPE | gap | paired t |
|---|---|---|---|---|
| 1 day | 24.8 ± 1.1 | 25.4 ± 1.7 | +0.6 | **1.40 — n.s.** |
| 2 days | 24.2 ± 3.6 | 32.6 ± 3.4 | +8.5 | 10.74 |
| 3 days | 22.0 ± 4.4 | 34.1 ± 4.4 | +12.1 | 11.08 |
| 5 days | 17.7 ± 4.3 | 32.0 ± 5.8 | **+14.3** | 8.99 |

At one day the two are indistinguishable — **as theory requires**, since the
one-day objective is separable over cells and treating the highest-risk cells
within budget is its *exact* optimum. The statistic jumps an order of magnitude
the moment the decision becomes sequential, and the heuristic **degrades**
(24.8 → 17.7) because it keeps optimising for today.

*A result that is insignificant exactly where theory says it must be, and
significant exactly where it predicts, is stronger evidence for the mechanism
than any single large number.*

**Bootstrapped (2026-09-28).** §8.4 item 9 asked for intervals instead of a
paired *t* over five seeds. This is the claim §9.6 calls "novel as evidence", so
it is the one that most needed them. `eval/run_bootstrap_report.py`, differencing
**within** each seed (same seed, same fire, same budget) and resampling the
paired differences:

| horizon | greedy (IQM) | PSPE (IQM) | gap (IQM) | gap (mean) | 95% CI | PSPE wins |
|---|---|---|---|---|---|---|
| 1 day | 25.1 | 25.6 | 0.4 | +0.6 | **[−0.2, +1.7]** | 4/5 |
| 2 days | 24.3 | 33.6 | 8.2 | +8.5 | **[+7.0, +10.4]** | 5/5 |
| 3 days | 22.1 | 35.4 | 11.5 | +12.1 | **[+10.0, +14.8]** | 5/5 |
| 5 days | 18.0 | 34.0 | 13.9 | +14.3 | **[+10.9, +18.2]** | 5/5 |

**The mechanism claim survives intact, and the interval says it more precisely
than the *t* did.** At one day the interval *contains zero* — the null the theory
demands is a real null, not an underpowered one, and it is bounded: whatever the
one-day advantage is, it is at most 1.7 points. From two days on the interval
excludes zero with every seed agreeing in sign, and the lower bound alone
(+7.0, +10.0, +10.9) exceeds the entire one-day interval. The IQM sits below the
mean at every horizon, so the effect is not carried by one lucky seed.

This is worth contrasting with §3.2a, run the same day with the same tool on the
same number of seeds: **the safety-margin comparison did not separate from zero
and this does.** Two claims reported identically in the old style turn out to
have very different evidential weight, which is the entire argument for the
change of reporting.

![Budget Pareto and horizon ablation on observed wildfire records](figures/pspe_wildfire.svg)

*Figure 5. Left: the advantage is largest where crews are scarcest, reaching
1.91× at a 1% daily budget. Right: the gap is statistically absent at a
one-day horizon, exactly where the objective is separable and the heuristic is
optimal, and opens as the decision becomes sequential.*

**Fuel-only ablation** (`runs/ndws_plan_fuelonly/`, 3 seeds; `block = 0`, removing the imposed physics entirely):

| | greedy | PSPE | gap |
|---|---|---|---|
| both mechanisms | 24.1 ± 3.8% | 36.5 ± 1.0% | +12.4 (t = 6.66) |
| **learned fuel response only** | **5.7 ± 2.9%** | **19.4 ± 1.6%** | **+13.6 (t = 5.93)** |

Removing the hand-written physics costs the heuristic 76% of its effect and the
planner 47% — and **the gap widens**. The advantage rests on the learned model,
not on the action model I specified.

**Against constrained RL on the same fires** (`runs/ndws_baselines_fair/`),
5 seeds. Every arm plans or learns against the same frozen U-Net, under the
same 3% daily budget, scored by the same function on the same held-out fires.

| method | burn reduction | treated/day | fires over budget |
|---|---|---|---|
| greedy heuristic | 24.1 ± 3.8% | 3.00% | 0 |
| PPO-Lagrangian | 4.6 ± 0.4% | 4.32% | 58% |
| CPO | 4.3 ± 0.1% | 2.84% | 6% |
| Sauté RL | 4.7 ± 0.5% | 7.37% | 79% |
| primal-dual NPG | 4.6 ± 0.1% | 3.26% | 100% |
| **PSPE per-instance** | **36.5 ± 1.0%** | 3.00% | 0 |

Note the last two columns before reading the first: three of the four RL arms
**exceed** the budget, Sauté by 2.5×, so this is not a matched-constraint
comparison in their favour. They lose while spending more, which strengthens
the reading rather than weakening it, but the comparison should be described
as it is.

> **The amortisation reading is weaker than it looks.** Planner amortisation
> *works* when the policy is distilled from the planner's own solutions —
> MPO plus behaviour cloning on MPC-generated data [*Evaluating model-based planning and planner
> amortization*, Byravan et al., 2022],
> TD-MPC [Hansen et al., 2022]. We trained a CNN from scratch under a budget
> dual for 200 iterations and did not try distillation, which is the recipe the
> literature identifies as the one that works. The honest claim is about *our*
> amortised arm, not about learned policies in general. §9.7.

The RL methods land near random placement (4.5%) and below the heuristic. The
reason is **amortisation, not the constraint**: one policy must serve 1,024
distinct fires from 200 iterations, the same failure our own amortised arm
shows (10.4%). On a per-instance decision problem this diverse, decision-time
optimisation is the right tool and a learned policy is not.

*Sample efficiency does not apply here* — there is no separate true environment
on NDWS, so every method draws from the same surrogate.

### 5.1a Distilling the planner: two ways to get a meaningless number (2026-09-28)

§9.7 is the most exposed claim in the paper: the literature says a model-based
planner **can** be amortised into a compact policy via behaviour cloning on
planner-generated data [Byravan et al., 2022], and we never tried it — we trained
a CNN from scratch under a budget dual and reported the gap (planner 36.5%,
amortised 10.4%). `eval/run_ndws_distill.py` runs the recipe we skipped. Same
`GaussianFieldPolicy` class as the amortised arm, same width, demonstrations
collected along the planner's own trajectory on training patches, scored on
held-out patches by the same `score_policy`.

Two attempts produced numbers that looked like results and were not. Both are
recorded because the failure modes are opposite and a reader could hit either.

**Attempt 1 — cloned raw actions, overspent the budget 7.6×.** The smoke run
reported a 20.5% reduction while treating **22.8% per day against a 3% budget**.
The intensity map `u = ((a+1)/2)²` is convex, so a small action error becomes a
large overspend, and nothing in an MSE enforces the constraint the planner's
projection enforces. Scoring that against a feasible planner is exactly the
unmatched-constraint comparison §5.1 criticises in the constrained-RL baselines
(three of four exceed the budget), so it cannot be done here either.

**Attempt 2 — projected onto the budget, then collapsed to no treatment.**
`runs/ndws_distill_actionmse_void/`, 3 seeds, kept for the record:

| seed | planner % | distilled % | distilled treated %/day |
|---|---|---|---|
| 0 | 44.63 | **−0.00007** | 6.0e−05 |
| 1 | 33.55 | **0.0** | 1.6e−06 |

The behaviour-cloning loss converged cleanly, 0.358 → 0.064. It learned exactly
the right constant for the wrong objective: with a 3% budget over 64 patches the
planner leaves most patches at zero, so most target actions are
`action_for(0) = −1`, the MSE-optimal constant is −1 everywhere, and
`intensity(−1) = 0`. **The policy learned to treat nothing.** The budget
projection from attempt 1 cannot catch this — it only scales spending *down*.

**Why this one was dangerous.** A 0% distilled result *confirms* our paper's
amortisation claim. §9.7 says the literature runs against that claim, so a null
here is the result I should distrust most, and it arrived looking clean — a
converged loss, a feasible policy, three consistent seeds. Had it gone into the
document as "distillation recovers none of the gap", it would have strengthened
the paper with an artefact of my loss function.

**The fix, and what it preserves.** The planner's solution is an *allocation* of
a fixed budget — it spends its whole 3% (measured: 2.996–3.000%/day) and the
decision is *where*. Normalising the policy's intensities to the budget removes
the degenerate solution: a constant output becomes a **uniform** allocation, which
is the planner's own initialisation, not an empty one, and what the network can
still express is where to concentrate. Training and evaluation then optimise the
same quantity. The network, its width and the tanh → intensity map are unchanged,
so the matched-capacity argument — the whole point of using the amortised arm's
own class — is untouched.

**Rule 38: when a null result would support your own claim, treat it as a bug
report until you have shown the arm can express a non-null answer.** The check is
one line: does the policy spend its budget? Attempt 2 spent 0.002% of it. The
same check catches attempt 1 from the other side, and both now run in the script,
with `over budget` and `treated %/day` reported for *both* arms in every summary
row so the comparison cannot be read without them.

**The result.** 3 seeds, 50 planning steps, 256 demonstration fires, Vista job
1031950. `runs/ndws_distill/`:

| arm | mean % | sd | per-seed |
|---|---|---|---|
| planner (decision-time) | **39.04** | 6.99 | 36.86, 33.41, 46.86 |
| **distilled from planner** | **18.32** | 0.48 | 17.77, 18.49, 18.69 |
| from-scratch amortised (§5.1) | 10.4 | — | — |

**Distillation recovers 46.9% of the planner's advantage: +7.9 points over the
from-scratch arm, and 20.7 points short of the planner.** Both arms spend their
budget exactly (2.994% and 3.000% per day, neither over).

This is the middle outcome of the three the script named in advance, and it is
the one I predicted there — that distillation would recover *some* of the gap but
not all of it, "because a single forward pass cannot reproduce 50 steps of
per-instance projected gradient descent on a fire it has not seen." Recording the
prediction before the run is what makes that worth anything.

**What it costs the paper, and what it buys.** §9.7's warning was half right:
the recipe we skipped does help, and our original framing — a bare planner-vs-
amortised gap — overstated the case by attributing to amortisation what was
partly our training recipe. That has to be conceded. But the literature's claim,
that a model-based planner can be amortised "without performance loss"
[Byravan et al., 2022], does **not** hold on this task: more than half the
advantage survives the recommended recipe. **The defensible claim is now stronger
than the one it replaces**, because it has survived the specific objection
§9.7 raised: decision-time planning retains a large advantage on observed fire
data *even after distillation from the planner's own solutions*.

**A mechanistic difference worth following up.** The `reduction via fuel channel
only` column separates the two arms sharply: the planner gets **15.95** through
the learned model's fuel channel, the distilled policy **0.98**. So the planner is
exploiting the surrogate's learned fire dynamics, while the distilled policy
reproduces mostly the imposed spread block — it has learned *where* fires tend to
go, not *how this fire's* fuel state responds. The seed spreads say the same
thing from another angle: the planner varies a lot across seeds (sd 6.99) because
it adapts to the fires it faces, and the distilled policy barely varies at all
(sd 0.48) because it has converged on one generic allocation. That is a concrete
account of what decision-time planning is buying, and it is more useful to the
paper than the raw gap.

**Still open.** One distillation recipe, not the family: no MPO step
[Byravan et al., 2022 pair BC with it], no DAgger-style correction for the
distribution shift a cloned policy induces, and 256 demonstration fires against
1,024 training fires. A stronger recipe could close more of the gap, and §9.7
should say so rather than claim the question is settled.

### 5.2 The twin loop, closed on real satellite sequences

> **Superseded by §5.2a (2026-09-27).** This section is the eight-fire FIRMS
> version. It has been re-run on WildfireSpreadTS at 607 fires with a stricter
> split, and two of its claims do not survive at that sample size: model
> adaptation's benefit is about a third of what is stated below, and the
> unanimity ("8/8 fires") was small-sample luck. The section is kept because it
> is what the FIRMS pipeline supports and because the correction is the point.

`docs/results/FIRMS_TWIN.md`. Eight large US wildfires, VIIRS active-fire
detections, 14–20 consecutive days, leave-one-fire-out, 3 seeds.

| fire | days | observed | gaps | growth |
|---|---|---|---|---|
| dixie (2021) | 20 | 19 | **1** | 6.6× |
| bootleg (2021) | 20 | 20 | 0 | 1.0× |
| caldor (2021) | 20 | 20 | 0 | 23.2× |
| august complex (2020) | 20 | 20 | 0 | 11.9× |
| creek (2020) | 20 | 20 | 0 | 1.9× |
| cameron peak (2020) | 20 | 16 | **4** | 0.1× (dying) |
| camp (2018) | 14 | 14 | 0 | 0.0× (burned out) |
| mosquito (2022) | 18 | 13 | 0 | 28.0× |

Unobserved days are carried as `observed = False`, never as empty fire:
**"we did not look" and "nothing burned" are different facts**, and a twin that
conflates them drifts.

> **This is the trivial baseline in a mature field.** Replacing the belief with
> the observation is **nudging with gain 1** — no covariance, no observation
> error model, no ensemble, no variational step. Data assimilation runs from
> 4D-Var and the Ensemble Kalman Filter [Evensen, 2003] to latent-space DA with
> learned surrogates [Cheng et al., 2023; *Physically consistent global
> atmospheric DA with ML in latent space*, Sci. Adv. 2026] and deep latent
> particle filters [*Deep Latent Space Particle Filter*, arXiv 2406.02204]. Comparing an assimilating system to a
> free-running forecast is what assimilation *is*; it verifies the loop is
> wired correctly. **Read the 8.4× as a control, not a finding** (§9.8), and
> note the dataset problem in §9.9.

| mode | day +1 | day +2 | day +3 |
|---|---|---|---|
| open loop | 0.073 ± 0.029 | 0.047 ± 0.026 | 0.038 ± 0.021 |
| state sync | 0.608 ± 0.110 | 0.237 ± 0.093 | 0.085 ± 0.047 |
| **state + model** | **0.613 ± 0.112** | **0.288 ± 0.111** | **0.110 ± 0.054** |

Paired across fires (df 7, threshold 2.36):

| comparison | day +1 | day +2 | day +3 |
|---|---|---|---|
| sync vs open | +0.535, 8/8, **t = 15.9** | +0.190, 8/8, t = 8.6 | +0.046, 8/8, t = 4.3 |
| model adaptation on top | +0.005, t = 3.4 | **+0.051, 8/8, t = 3.0** | **+0.025, 8/8, t = 3.1** |

**Syncing to observation is worth 8.4× at one day**, on every fire. Adapting the
model adds +21% at day 2 and +30% at day 3 — negligible at day 1 because the
state was just replaced with ground truth, growing as the model's own quality
starts to dominate. **Both halves of the loop do work, in different regimes.**

![Three ways to carry a state across real satellite days](figures/pspe_twin_loop.svg)

*Figure 6. The open loop drifts because nothing corrects it. State sync
replaces the belief with what was observed, and on days with no overpass every
mode must trust the model. Skill is measured, not schematic: the table beneath
the diagram carries the paired tests.*

*Correction:* earlier notes, written from a 4-epoch smoke run, said model
adaptation "adds essentially nothing." That was wrong.

### 5.2a The same loop at 607 fires, and what does not survive (2026-09-27)

**Why re-run it.** §9.9 states the problem: the twin loop ran on **8 fires**
built from a FIRMS pipeline we wrote, while **WildfireSpreadTS**
[Gerard et al., NeurIPS 2023 D&B] has had **607 fires and 13,607 daily images**
public since 2023. "8 of 8 fires" is honest and badly underpowered, and "why
your own dataset?" is a fair question with no good answer.

**Setup.** `eval/run_firms_twin.py --source wsts`, `pspe/simulate/real/wsts.py`,
`scripts/tacc/vista_wsts_twin.slurm` (job 1029659, 61 min on one GH node).
607 fires, 13,607 days of which **12,916 observed (94.9%)**, three seeds. The
loader emits `firms.FireSequence`, so the loop code is byte-identical between
the two datasets and the comparison below is not confounded by the harness.

**Held out by year, not by fire.** Leave-one-fire-out trains on other fires
from the same season and weather regime, which leaks. The dataset's own
documentation recommends cross-validation across years because the distribution
shifts between them, so that is what runs here: four folds of 176 / 74 / 201 /
156 fires, exactly the dataset's own grouping. It is the stricter split, and it
was expected to bring the headline down.

| mode | day +1 | day +2 | day +3 |
|---|---|---|---|
| open loop | 0.0186 ± 0.0311 | 0.0108 ± 0.0159 | 0.0088 ± 0.0115 |
| state sync | 0.3792 ± 0.2418 | 0.1184 ± 0.1079 | 0.0251 ± 0.0341 |
| **state + model** | **0.3808 ± 0.2432** | **0.1259 ± 0.1130** | **0.0268 ± 0.0360** |

| horizon | comparison | mean gain | 95% CI | fires improved | t | Cohen's d |
|---|---|---|---|---|---|---|
| day +1 | sync vs open | +0.3606 | [0.342, 0.379] | 595/606 | 38.2 | **1.55** |
| day +2 | sync vs open | +0.1076 | [0.100, 0.115] | 559/606 | 28.7 | 1.17 |
| day +3 | sync vs open | +0.0163 | [0.014, 0.018] | 532/606 | 16.9 | 0.69 |
| day +1 | adaptation on top | +0.0016 | [0.0009, 0.0022] | 376/606 | 4.9 | **0.20** |
| day +2 | adaptation on top | +0.0075 | [0.0061, 0.0088] | 463/606 | 10.9 | 0.44 |
| day +3 | adaptation on top | +0.0016 | [0.0012, 0.0020] | 431/606 | 7.8 | 0.32 |

**What survives.** The shape of the §5.2 story. Syncing to observation
dominates at one day, its advantage decays with horizon, and model adaptation
matters *more* at longer horizons because at day 1 the state was just replaced
with ground truth and there is nothing left to fix. Both halves of the loop do
work, in different regimes. Sync-vs-open is if anything stronger here — the
ratio is **20.4× at day +1** against 8.3× on the eight fires, because the open
loop degrades further on a harder split than state sync does.

**What does not survive — 1: the size of the adaptation benefit.** §5.2 reports
"+21% at day 2 and +30% at day 3". Measured on 607 fires the same quantities are
**+6.3% and +6.8%**, roughly a third of the claim. The eight-fire estimate was
not wrong arithmetic; it was eight draws from a distribution whose per-fire
spread is large, taken under a split that leaks season.

**What does not survive — 2: the unanimity.** "8/8 fires" becomes **376/606
(62%)** at day +1 and **463/606 (76%)** at day +2. Adaptation helps on most
fires, not all, and the eight-for-eight was the small sample being kind.

**And a third thing the eight fires could not have shown: significance stopped
meaning importance.** Adaptation at day +1 is t = 4.9 — comfortably significant
— on a mean gain of **+0.0016 against a base of 0.3792, which is 0.4%**, with
d = 0.20. At n = 607 an effect can be certain and negligible at the same time.
The row worth reporting is day +2, where d = 0.44 and the gain is 6.3% of base;
the day +1 row should be reported as *detectably nonzero and operationally
nothing*. **Rule 28: past a few hundred paired samples, report an effect size
and a confidence interval next to every t, because the t alone stops
discriminating.** The eight-fire table could not have taught this — at df 7
nothing negligible was ever going to clear the threshold.

**Still a control, not a finding.** None of this changes §9.8: replacing the
belief with the observation is gain-1 nudging, and comparing it to a
free-running forecast is what assimilation *is*. 607 fires removes the
underpowering and the "why your own dataset?" objection; it does not promote a
wiring check to a result. The genuinely novel row remains **adaptation on top of
state sync**, which is now measured properly and is smaller than advertised.
§8.4 item 3 — a real DA baseline — is still what would make the first three rows
mean something, and **it is now run: §5.2c.**

**And it still says nothing about intervention.** No fire in this archive had a
firebreak cut on our instruction. The dataset switch does not touch §6.2.

### 5.2c The EnKF baseline: gain 1 was the right answer (2026-09-28)

§9.8's concession is blunt: "our state sync is nudging with gain 1 — no
covariance, no observation-error model, no ensemble, no variational step", and
comparing that to a free-running forecast is what assimilation *is*, not a
finding. §8.4 item 3 was to add a real filter or demote the claim. This adds the
filter: `eval/run_firms_twin.py --n-ens 32 --obs-sd ...`, 606 fires, year-wise
CV, 3 seeds, Vista job 1031659, 1 h 29 m. Ensemble forecast spread, observation
error stated rather than assumed away, analysis `K = σ_f²/(σ_f² + σ_o²)` with
perturbed observations.

**`σ_o` is swept, not fitted.** VIIRS ships no per-cell error variance, and
choosing the one that makes the result come out is the failure this document
catalogues. Gain-1 nudging is the `σ_o → 0` corner, which doubles as the
correctness check.

| mode | day +1 | day +2 | day +3 |
|---|---|---|---|
| open loop | 0.0193 ± 0.0313 | 0.0115 ± 0.0161 | 0.0094 ± 0.0119 |
| **state sync (gain 1)** | **0.3785 ± 0.2413** | 0.1216 ± 0.1095 | 0.0257 ± 0.0343 |
| state + model | 0.3805 ± 0.2426 | **0.1291 ± 0.1147** | **0.0273 ± 0.0362** |
| EnKF σ_o = 0.05 | 0.3288 ± 0.2375 | 0.1120 ± 0.1101 | 0.0281 ± 0.0368 |
| EnKF σ_o = 0.15 | 0.1267 ± 0.1365 | 0.0281 ± 0.0465 | 0.0143 ± 0.0244 |
| EnKF σ_o = 0.30 | 0.0317 ± 0.0473 | 0.0119 ± 0.0181 | 0.0094 ± 0.0122 |

Paired across 606 fires at day +1:

| comparison | mean gain | fires improved | paired t |
|---|---|---|---|
| state sync vs open loop | +0.3592 | 594/606 | **38.19** |
| model adaptation vs state sync | +0.0020 | 394/606 | **5.72** |
| EnKF σ_o = 0.05 vs state sync | **−0.0497** | 78/606 | **−20.95** |
| EnKF σ_o = 0.15 vs state sync | −0.2518 | 18/606 | −35.38 |
| EnKF σ_o = 0.30 vs state sync | −0.3468 | 11/606 | −38.65 |

**No setting of the filter beats gain-1 nudging, and skill falls monotonically as
`σ_o` rises.** With a mean ensemble spread of 0.340, the swept `σ_o` correspond to
Kalman gains of **0.979, 0.837 and 0.562** — and day-+1 skill tracks them in
order (0.329, 0.127, 0.032). The best gain in the swept range is the largest one.

**So the answer to §9.8 is that gain 1 is not a naive stand-in for a filter here;
it is approximately the filter's own optimum.** The observations are far more
reliable than the forecast on this system, which drives `K → 1`, and the classical
machinery has nothing left to estimate. That reading is worth more to the paper
than a win would have been: it explains *why* the simple thing was adequate
instead of leaving it as an unexamined shortcut.

**Two honest caveats.** First, the σ_o = 0.05 arm does **not** exactly recover
state sync — it is reliably worse, by 0.0497 with t = −20.95, where the eight-fire
smoke test earlier in this work put the same comparison at t = −0.96 and called it
convergence. At 606 fires that reading does not hold. The gain there is 0.979, so
the gap is not the gain; it is the **perturbed-observation** implementation, which
injects sampling noise a deterministic nudge does not have. A deterministic
(square-root) EnKF would remove that noise floor and is the right follow-up. This
is the same lesson as §5.2a's, arriving again: *an eight-fire paired test agreed
with the hypothesis, and at 75× the sample size the sign was real and the
magnitude was not.*

Second, the sweep bounds the conclusion rather than proving it. No σ_o in
{0.05, 0.15, 0.30} beats gain 1; the claim is not that none could.

**What changes in the write-up.** The first three rows of §5.2a stop being a
wiring check: state sync can now be reported as *the σ_o → 0 corner of a filter
that was actually run*, with the sweep showing the corner is where the skill is.
Model adaptation on top of state sync remains the genuinely novel row (+0.0020 at
day +1, t = 5.72; +0.0075 at day +2, t = 10.92) — small, and now measured against
a real baseline rather than against nothing.

### 5.2b Four defects in the WildfireSpreadTS loader, all caught before a number (2026-09-27)

Every one of these would have produced a plausible-looking result.

**Defect 23 — "no detection" read as "not observed".** The active-fire channel
is NaN almost everywhere and finite only where VIIRS detected fire, so the
obvious observability test, "does this day have any finite fire pixels", marks
every quiet day unobserved. Measured over 890 days across 40 fires: the
reflectance and terrain bands are finite on 100% of pixels on a normal day, the
fire channel on 0.0–0.2%, and **37.4% of days are real observations with zero
fire**. The loop scores observed days only, so this would have silently deleted
a third of the record — specifically the days a fire goes out, which is the part
of carrying state forward that is actually hard, and the part where an open loop
looks worst. Observability now comes from the VIIRS reflectance bands the fire
product is derived from; **2.2%** of days are genuinely missing.
**Rule 29: when a channel encodes absence as NaN, it cannot also tell you
whether anyone looked. Find a channel that is present when the instrument
worked.**

**Defect 24 — the fire band is a clock, not a power.** `FireSequence.frp` is
documented as fire radiative power in MW, and the obvious band to fill it from
is the active-fire channel. Its values run 742 to 2142, the minutes are always a
multiple of six and never reach sixty: these are **HHMM VIIRS granule
acquisition times** (07:42 to 21:42), and I was clipping them at 500 and calling
the result megawatts. The detection *mask* was unaffected, so nothing visible
would have broken. `frp` is now zeros with the reason written next to it, since
the loop never reads it and anything that starts to should fail loudly rather
than regress on clock times.

**Defect 25 — the downsample was deleting fire.** Reshaping into blocks of
`h//grid` by `w//grid` crops whatever does not divide evenly. WSTS rasters are
around 300×250, so at grid 64 that discards roughly a fifth of each axis, and
measured against the native detection count it lost between 0% and **59%** of a
fire's detections depending on where in the frame the fire sat — a fire near an
edge was simply smaller in the data than in the world. Replaced with index
binning over the full extent; recovery ratios now sit at 0.96–1.06, the residual
being cells that hold four rows against five.

**Defect 26 — a one-fire fold.** Year folds keyed on `dates[0][:4]`, and a fire
filed under 2018 can have its first frame on 30 December 2017, which invents a
"2017" fold containing one fire — not a train/test split of anything. Keyed on
the modal year instead, which reproduces the dataset's own directory grouping
exactly: 176 / 74 / 201 / 156.

**The pattern across all four.** None is an algorithmic mistake; each is a wrong
belief about what a number in a file means, and each would have been invisible
in the output. Defect 25 is the one that would have been hardest to catch — it
degrades the data smoothly and only for some fires. **Rule 30: when adopting an
external dataset, verify each channel's semantics against the raw values before
running anything, and check that any resampling conserves the quantity it is
supposed to conserve.** Total detections before and after is a two-line test
that would have caught defect 25 immediately.

### 5.3 Planning under partial observation

`docs/results/NDWS_PARTIAL.md`. Spatially correlated occlusion (low-resolution
noise upsampled and thresholded — a blob over the front, not scattered pixels).
**Plan on the belief, score on the truth.**

Burn reduction (%), PSPE planner, 3 seeds:

| occlusion | blind | persist | perceive | perceive-hard |
|---|---|---|---|---|
| 0% | 41.11 ± 0.75 | — | — | — |
| 15% | **39.72 ± 0.77** | 39.72 ± 0.66 | 38.20 ± 1.15 | 39.44 ± 0.66 |
| 35% | **37.44 ± 0.19** | 37.41 ± 0.29 | 33.30 ± 1.98 | 36.63 ± 0.93 |
| 60% | **34.52 ± 0.56** | 34.44 ± 0.56 | 27.34 ± 2.44 | 33.04 ± 1.10 |

Reconstruction AP on hidden cells:

| occlusion | blind | persist | perceive | perceive-hard |
|---|---|---|---|---|
| 15% | 0.006 | 0.041 | **0.594** | 0.284 |
| 35% | 0.006 | 0.037 | **0.489** | 0.224 |
| 60% | 0.005 | 0.032 | **0.346** | 0.147 |

Paired against `blind`, all 9 (rate, seed) configurations:

| belief | mean change | better in | paired t |
|---|---|---|---|
| persist | −0.04 pts | 5/9 | −1.04 (n.s.) |
| **perceive** | **−4.28 pts** | **0/9** | **−4.33** |
| perceive-hard | −0.86 pts | 1/9 | −2.79 |

**Finding 1 — graceful degradation.** Losing **60% of the observation costs 6.6
points out of 41** (16% relative), and the planner still beats the *full-state*
greedy heuristic (22.2%) by a wide margin.

> **Why this one is worth more than §5.2.** The entire DA and twin stack
> optimises **analysis quality** on the assumption that a better analysis yields
> a better outcome; the Science Advances latent-DA result states it in those terms — latent DA
> improves both analysis quality *and* forecast skill. Neither is the decision.
> Under a hard budget on a sparse field we have a counterexample, and
> *WildfireSpreadBench* (arXiv 2609.22191) finds the same shape on the forecast side of the same
> domain. §9.10.

**Finding 2 — better reconstruction, worse decisions.** The estimator recovers
hidden cells ≈100× better and genuinely changes the plan (Jaccard 0.49 vs
blind), yet the outcome is worse in **0 of 9** configurations. The
`perceive-hard` control splits the cause:

* **Representation (3.4 of 4.3 points).** The surrogate was trained on a
  near-binary mask and reacts badly to a smeared probability field.
* **A real residual (−0.86, t = −2.79).** Under a hard budget, spending on cells
  that *might* be burning takes crews from cells confirmed to be burning.

**The caveat that bounds it:** fire is a **rare-event field** (1.3% prevalence),
so "assume nothing burns where I cannot see" is close to the base rate and
`blind` gets a strong prior for free. On a dense field — flood depth, surface
temperature — this could invert. The claim is *not* "twins should never impute";
it is **on a sparse field under a hard budget, act on what you observe, and
prove reconstruction's value on the decision, not on the reconstruction metric.**

### 5.4 A near-miss worth recording

The first constrained-RL comparison on the fire task had all four baselines
treating 22–38% of the sector against a 3% budget, on every fire — which looked
like a structural finding about constrained RL in high-dimensional action
spaces. It was an artefact: under the squared intensity map a randomly
initialised policy emits `((0+1)/2)² = 0.25`, i.e. 25% treatment, and the agents
had barely moved off initialisation, while PSPE's optimiser *starts* at the
budget and projects onto it every step. With fair initialisation,
PPO-Lagrangian holds 2.99% after three iterations.

**The structural claim was withdrawn before it was ever made.** Four
algorithmically distinct methods failing identically is the signature of a
shared setup artefact, not four independent limitations.

---

### 5.5 The conformal margin on real fire dynamics

`eval/run_ndws_margin.py`, `runs/ndws_calib/`, `runs/ndws_perturb/`

§5.1 enforced the crew budget by exact projection, so it validated the planner
and never exercised the margin. This closes that gap, and the route to it is
most of the result.

**The silent violation reproduces, and is larger than on the PDE testbeds.**
An expectation constraint (population-weighted burn over the settled quarter of
cells) conflicts with the objective (total burn) under a fixed crew budget. A
single multiplier shared across fires is driven by the batch mean under the
planner's model `G`; violation is realised under a better model `F` and
measured per evaluation batch. With no margin:

| seed | violating batches |
|---|---|
| 0 | 0.650 (local) / 0.750 (Vista) |
| 1 | 0.400 |
| 2 | 0.550 |

40–75%, against 14.5% on rdf. The constraint separates independently on two
machines (span 10.5% and 11.8% of the limit).

**A learned second model cannot supply a usable discrepancy here.** Four
(budget, capacity) settings, all infeasible:

| budget | G epochs | span `s` | bias `b` | b/s |
|---|---|---|---|---|
| 3% | 4 | 0.009468 | 0.030637 | 3.2× |
| 8% | 4 | 0.007328 | 0.033054 | 4.5× |
| 8% | 10 | 0.005069 | 0.010344 | **2.0×** |
| 15% | 10 | 0.005492 | 0.026921 | 4.9× |

**More crew makes it worse**, shrinking the absolute span from 0.0095 to
0.0055: with more capacity the objective-optimal and constraint-optimal plans
both drive settled burn toward the same floor, closing the gap between them.
The span only rises as a *percentage* because the limit itself falls. Better
`G` is the lever that works, and it is not enough.

**The precondition this exposes.** Write `s` for the span of the constrained
quantity between the constraint-optimal and objective-optimal policies, `b` for
the model-reality bias, `σ_agg` for the spread of the quantity violation is
declared on, and place the limit at `d = d_best + f·s`. The **margin** must fit
the headroom `f·s`, and the margin is a quantile of the residuals rather than
the bias alone, so the condition is

```
b + z_δ · σ_agg  <  f · s
```

**Correction.** This was first written as `b < f·s`, omitting the spread term,
and `--calibrate-only` was built to test that weaker condition. It is why four
configurations were reported usable and none of them were. At shift 0.01:
headroom 0.003445, estimated bias 0.001509 — "usable" — against an actual
margin of **0.016440**, 4.8× the headroom. The bias was a tenth of the margin;
the spread was the rest.

The lesson generalises: a feasibility check must measure **the quantity that
will be subtracted from the limit**, not one component of it. That is §3.2's
error — conformalising surrogate error rather than realised cost — appearing a
third time, which argues for stating the rule once and applying it everywhere.

Otherwise `d_eff` falls below anything the policy can reach and the constraint
is unachievable however well the margin is calibrated. **Infeasible and
miscalibrated look identical in a violation rate**, which is why this cost four
runs before being named.

`f` is not a free parameter but the trade-off itself: it sets how tightly the
constraint binds *and* how much room a margin has, in opposite directions. The
first version used `f = 0.35`, chosen only to make the constraint bite, leaving
headroom of 4.5% of the limit against a bias of 18%.

The precondition also explains the earlier testbeds retroactively: on dar the
constraint never binds (worst 0.225 against a limit of 0.936) so `s` is
enormous and any margin is trivially usable; on rdf the surrogate is accurate
so `b` is small. Wildfire is the first setting in this project where `b > f·s`.

**Reality as a controlled variable.** Rather than hoping a learned-vs-learned
gap lands where the claim is testable, `F = G` plus a calibrated perturbation:
a logit `shift` setting the systematic bias `b`, and a fixed input-dependent
`scatter` setting the per-instance error `ε` and hence `ρ = σ/ε`. The scatter
term is deterministic in the state — the same fire must produce the same error,
or it is noise rather than model error and the matched-pair recipe would have
nothing to cancel. This is a perturbed model, not an independently trained one,
and any write-up must say so in its first paragraph.

The threshold is reached and bracketed:

| logit shift | span `s` | bias `b` | headroom `f·s` | b/headroom | verdict |
|---|---|---|---|---|---|
| 0.01 | 0.006887 | 0.001509 | 0.003443 | 0.44 | usable |
| 0.02 | 0.006940 | 0.001988 | 0.003470 | 0.57 | usable |
| 0.04 | 0.007086 | 0.003029 | 0.003543 | 0.85 | usable |
| 0.08 | 0.007389 | 0.005134 | 0.003694 | **1.39** | infeasible |
| 0.16 | 0.007887 | 0.009718 | 0.003943 | **2.46** | infeasible |

**A comparison that did not hold up.** Two sweeps of the same configuration
disagreed at shift 0.08 — Vista reading `b = 0.005134` (infeasible) and local
`b = 0.001826` (usable). This was initially recorded here as evidence that the
shift-to-bias mapping is model-specific. It is not evidence of that: the Vista
sweep ran a single-batch estimator and the local sweep an eight-batch average,
because the averaged version was synced forty minutes after the Vista job
finished. **Two estimators were compared and the difference attributed to
hardware.** A like-for-like rerun is queued; until it reports, nothing is
claimed here about reproducibility across machines.

**The rerun settles it: estimator variance, not hardware.** Running the same
code on both:

| shift | Vista, 1 batch (old) | Vista, 8 batches | local, 8 batches |
|---|---|---|---|
| 0.04 | 0.85 usable | **0.18** usable | **0.43** usable |
| 0.08 | **1.39 infeasible** | **0.65** usable | **0.45** usable |

With matched code the two machines agree — same side of the threshold, same
order of magnitude — and the single-batch estimate is the outlier. The
withdrawal above was correct, and this is the evidence for it rather than an
inference.

**A caveat the rerun exposes.** The bias standard deviations are large relative
to the means: 0.000640 ± 0.001667 at shift 0.04, where the deviation is 2.6×
the estimate. At small shifts the perturbation's effect on the constrained
quantity is indistinguishable from batch-to-batch variation, so those points
carry no usable signal about `b`. Shift 0.08 is flagged marginal automatically.
This is why the stage-2 bracket was widened to include 0.3 and 0.9: the
low-shift points establish the feasible side, but only the high-shift points
can establish the infeasible one with a bias that is measurably non-zero.

**The threshold, with an averaged estimator** (`2 × probe_batches` batches per
point, local, seed 0):

| logit shift | span `s` | bias `b` | headroom `f·s` | b/headroom | verdict |
|---|---|---|---|---|---|
| 0.01 | 0.007715 | 0.000175 | 0.003858 | 0.05 | usable |
| 0.02 | 0.007778 | 0.000682 | 0.003889 | 0.18 | usable |
| 0.04 | 0.007904 | 0.001710 | 0.003952 | 0.43 | usable |
| 0.08 | 0.008157 | 0.001826 ± 0.001596 | 0.004078 | 0.45 | usable |
| 0.16 | 0.008667 | 0.005421 ± 0.002023 | 0.004334 | **1.25** | infeasible **(marginal)** |

The last row is flagged marginal automatically: the estimate sits within its
own error of the threshold, which is the condition the guard was added for.
Reported as marginal rather than as a crossing.

The single-batch sweep put the crossing between 0.04 and 0.08; the averaged one
puts it between 0.08 and 0.16 — one step apart, consistent given the noise, and
further evidence that the earlier machine-to-machine discrepancy was estimator
variance rather than a property of the models. A like-for-like rerun is queued
to settle it.

**Outcome: the wildfire task cannot support this experiment.** Run either side
of the supposed threshold, the margin exceeded the feasible headroom at every
shift:

| shift | limit `d` | headroom `f·s` | margin required | |
|---|---|---|---|---|
| 0.01 | 0.04422 | 0.003445 | 0.016440 | 4.8× over |
| 0.9 | 0.09446 | 0.006171 | 0.087670 | 14.2× over, 93% of the limit |

Violation stayed at 0.65–0.80 for every arm including ours, because `d_eff`
sits below anything the planner can reach. The guards added earlier fired on
every run and refused to report the arms.

**This is a property of the task, not the method.** The constrained quantity
has a batch-to-batch spread comparable to the entire range the planner can move
it over, so no margin covering that spread fits the headroom — at any
perturbation, with any limit placement. Miscalibration, an under-converged dual
and insufficient crew were each tested and were not the cause.

**What a write-up can claim here:** the precondition, with this task as the
worked example of a system that fails it — more useful to a reader than another
demonstration on a task that passes. The margin's validation rests on the PDE
families (§3.5), where it is measured at adequate power.

**Superseded plan:** the four margin recipes at shifts **0.01, 0.08, 0.3, 0.9**,
3 seeds, queued on Vista. The bracket was widened from the original
0.01/0.04/0.08/0.16 after the averaged estimator moved the threshold: under
either estimate the widened set has at least one feasible and two infeasible
points, where the narrow set risked landing entirely inside the feasible
region and showing only half the two-sided result §8e requires. Success is two-sided and pre-registered in
`paper/REVISION_NOTES.md` §8e: inside the feasible region `residual` and
`episode` should hold δ = 0.1 while `none` violates heavily; outside it **every
arm should fail, ours included**, because the tightened limit is unreachable
there and an arm that appears to succeed is measuring something else.

**What this section can and cannot claim.** Not "the margin works on real fire
data" — that is unavailable on this task, for the structural reason above.
What it can claim is that the margin holds its stated rate where the policy has
room to act and fails where it does not, with the boundary measured rather than
assumed, on dynamics fitted to observed fires.

### 5.6 Flood: a solver-validated intervention route (in progress, 2026-09-25)

> **Scope correction, 2026-09-26.** Everything in this section below the solver
> validation was measured on a **synthetic fluvial testbed**: a channel
> over-topping a berm through narrow gaps, which is the geometry a levee is built
> for. The span, the amplification and the precondition sweep are properties of
> *that* testbed, not of flood control in general. Moving the same study onto the
> Australia 2022 DEM (§5.6a) cut the achievable span from ~100% to a few percent,
> because that event is **pluvial** — rainfall-driven sheet flow — where a levee
> is the wrong instrument. Read every number below as scoped to fluvial
> geometry. Generalising them to "flood" would repeat, one level up, exactly the
> over-reach this project flagged in the draft's retirement of `swe`.



**Why flood, and why now.** Every planning result above carries §6.2: no
observational record contains the counterfactual. §6.2 lists two ways out, and
the second is "a physics-based simulator as a surrogate for reality — which
substitutes one model for another." **The flood work takes that route
explicitly.** The substitution is more defensible here than for fire on three
counts: the shallow-water equations are *conservation laws* rather than empirical
spread rates; the LISFLOOD-FP local-inertial discretisation is the operational
standard used in government flood mapping; and it is independent of our learned
surrogate. It remains a model, not the world, and every claim must say so.

**Dataset.** FloodCastBench (Nature Sci Data 2025; FloodCast, arXiv 2403.12226):
continuous **water depth** from a LISFLOOD-FP-style staggered finite-difference
solution, four events (Pakistan 2022 and Mozambique 2019 at 480 m; Australia
2022 at 30 m; UK 2015 at 60 m), 300 s timestep, TIFF, 20.1 GB on Zenodo
(`records/11431853`), CC-BY-4.0. The cross-event resolution split supplies the
surrogate-error axis our ρ = σ/ε diagnostic needs, from a published design
choice rather than from noise we inject.

**Why `ShallowWaterTransport` could not be reused, which explains defect 12.**
The retired `swe` testbed is *linearised and has no topography*:
`h_t = -H(u_x + v_y) - c_h h + s`, with control adding an equal source to surface
height everywhere. There is no geometry for an intervention to act on, so the ~7%
achievable band was a property of the actuator, not of flood control. Momentum in
the real equations acts on the **free surface** `h + z`; raising `z` adds no water
but changes where water can go, and the interface depth
`h_f = max(h_i+z_i, h_j+z_j) - max(z_i, z_j)` collapses to zero across a crest the
flood has not topped. A levee *gates* flow; the `swe` actuator could only *add* it.

**New solver: `pspe/simulate/flood.py`.** The LISFLOOD-FP local-inertial scheme
with Manning friction, upwind interface depth, adaptive CFL timestep, per-cell
flux limiting, and normal-depth free-outflow boundaries. Verified by seven
properties in `tests/test_flood.py`:

| property | result |
|---|---|
| lake at rest over uneven bed (C-property) | drift **0.000 m**, free-surface std 1.2e-7 m |
| closed-domain mass conservation, 600 steps | **+0.00001%** |
| flux limiter keeps depth ≥ 0 without the clamp | holds every step |
| open edge drains, closed edge does not | 768 → 1.4 vs 768 → 768 |
| **raising the bed displaces water without adding any** | water budget unchanged |
| levee sites are distinguishable (max off-diagonal corr) | < 0.9 |
| exposure weighting normalised | sums to 1 |

**Defect 19 — an unlimited explicit scheme manufactures the constrained
quantity.** The first solver *gained 6.4% volume* on a dam break: when a flux
over-drains a shallow cell, `clamp(min=0)` creates water. Every margin result is
declared on flood depth, so a solver that invents depth would have invalidated
the whole experiment while still producing plausible-looking floods. Caught only
by the explicit conservation test, not by inspection. **Rule 16: when the solver
supplies the quantity a constraint is declared on, test conservation of that
quantity before using it as reality.**

**Scenario, and a discarded first attempt.** My first terrain put the settlement
in an isolated depression. It flooded from **rain falling inside the
depression** — which no levee can prevent — while channel discharge never reached
it (town depth 0.000 m at rain = 0; 0.303 m at rain > 0 *regardless of inflow*).
That would have been as artificial as `swe`. The replacement is a floodplain
separated from its channel by a berm with three low gaps, which is how real levee
systems fail. The town now floods *from the channel*:

| channel inflow (m/s over the patch) | exposure-weighted damage | town max depth |
|---|---|---|
| 1e-3 | 0.0526 | 0.318 m |
| 2e-3 | 0.0930 | 0.461 m |
| 4e-3 | 0.1665 | 0.680 m |
| 8e-3 | 0.3095 | 1.166 m |

Six candidate levee sites along the berm, 3.0 m per-site cap, 6.0 m total budget —
deliberately too small to close every gap, so allocation is a real problem.

**Span attempt 1: 1.63% — worse than `swe`.** Six sites, 3.0 m cap, 6.0 m budget,
greedy in 0.5 m increments over 80 rollouts. Two sites made the objective *worse*
(−1.35%, −0.73%); spreading the budget evenly was worse than doing nothing
(−0.35%); the best plan put 3 m at each of two adjacent sites for +1.63%.

**Diagnosis, before any change.** Two faults, both mine, neither a property of
flood control:

1. **The forcing submerged the structure.** Domain peak depth was **9.2–12.8 m**
   against a **2.2 m** berm. The flood crossed the berm along its entire length,
   so a 3 m levee at one site was irrelevant. The inflow, 4e-3 m/s over 48 cells
   of 480 m, is ≈ **44,000 m³/s sustained** — larger than the Indus at peak 2022
   flood. A levee can only matter when flood depth is comparable to levee height.
2. **The settlement was too far from the structure.** At 17 cells west of the
   berm, the floodplain between drained south before water arrived.

A slope sweep ruled out the obvious third explanation. Flattening the valley from
1.2e-3 to 3e-5 raised damage 0.167 → 0.717 (town mean peak 0.26 → 1.42 m), so the
down-valley slope *was* draining water — but westward flux across the berm line
barely moved (1.48e6 → 2.04e6), so the slope was not what made levees useless.
The submergence was.

Calibrating for partial over-topping confirmed the two regimes did not coexist in
that geometry: at inflows giving 3–4 over-topped rows of 96, town mean peak depth
was only 0.011–0.076 m; at inflows flooding the town, all 96 rows over-topped.

**Attempt 2 moves the settlement adjacent to the levee**, in the threshold regime
where over-topping is the whole question, with the forcing capped so domain peak
stays within about twice the levee height.

**On changing a scenario after seeing a bad number.** This is only legitimate
because both corrections are justified independently of the span: a sustained
44,000 m³/s on a 46 km domain is not a flood anyone plans for, and floodplain
settlements sit *beside* levees — that is what levees are for. The 1.63% is kept
here so the change is visible rather than quietly overwritten. **The gate still
stands as written:** if a properly-scaled scenario cannot separate, flood planning
dies as a second worked example of defect 12.

**Attempt 2 found two further faults, both in my scenario, both caught by
measuring the geometry instead of assuming it.**

*Fault A — the inflow straddled the structure.* The hydrograph was injected at
`cols 40:56`, and the berm sits at **col 39**. So a share of every flood was
delivered *directly onto the west floodplain*, where it ran south to the
settlement without ever crossing the berm. No levee anywhere can stop water that
starts on the protected side. Fixed by injecting only inside the channel
(`cols 46:51`, measured from the meander, not assumed).

*Fault B — the settlement overlapped the structure and the channel.* At
`town_centre=(0.62, 0.37)` with `town_radius=0.09`, the depression spanned cols
27–44 while the berm sat at col 37 and the channel edge at col 44. The
settlement had therefore **carved its own gap through the berm**, and part of
"town flooding" was simply channel water inside the scoring region.

**Rule 17: verify geometry separation numerically before running any physics.**
Both faults are invisible in a plausible-looking flood field and both make an
intervention study meaningless. The check is now an assertion, not an inspection:

```
town cols 13..25   berm col 34..44   chan col 43..52
town strictly WEST of berm: True     <- asserted before any solve
```

**Calibrating to the regime where a levee can matter.** With the geometry
separated, the freeboard measured from the built terrain shows the gaps are
genuinely the weak points:

| row | gap fraction | crest | adjacent floodplain | freeboard |
|---|---|---|---|---|
| 39 | 0.55 | 5.25 m | 4.10 m | 1.15 m |
| **57** | **0.80** | 3.16 m | 2.78 m | **0.37 m** |
| 74 | 0.40 | 3.03 m | 1.56 m | 1.47 m |
| 20 | — | 8.06 m | 5.47 m | 2.59 m |
| 50 | — | 5.83 m | 3.28 m | 2.55 m |
| 88 | — | 3.19 m | 0.55 m | 2.63 m |

A 2.2 m window separates topping the weakest gap from topping the intact berm.
Sweeping the hydrograph through it:

| inflow | equiv. Q | damage | town mean peak | berm rows wet | gap r57 |
|---|---|---|---|---|---|
| 1.0e-2 | 34,560 m³/s | 0.0099 | 0.042 m | 4/96 | 0.37 m |
| **1.4e-2** | **48,384 m³/s** | **0.1016** | **0.282 m** | **9/96** | **0.82 m** |
| 1.8e-2 | 62,208 m³/s | 0.2251 | 0.583 m | 30/96 | 1.19 m |
| 2.4e-2 | 82,944 m³/s | 0.4712 | 1.142 m | 57/96 | 0.65 m |

1.4e-2 is the operating point: the gaps are topped, 87 of 96 berm rows stay dry,
and the settlement floods measurably. **Stated honestly, the equivalent discharge
is roughly 2.5× the 2022 Indus peak**, because this synthetic channel is 2.4 km
wide; the number that matters for the gate is the *regime*, and the real Pakistan
DEM and hydrograph replace this scenario in Phase 3.

**Span attempt 3 — the gate passes decisively.** Do-nothing damage 0.10163,
settlement mean peak depth 0.282 m. Six sites, 3.0 m cap:

| site | row | damage | reduction | town mean peak |
|---|---|---|---|---|
| 0 | 0.42 | 0.10122 | +0.40% | 0.281 m |
| 1 | 0.50 | 0.08221 | +19.10% | 0.233 m |
| **2** | **0.60** | **0.00024** | **+99.77%** | **0.002 m** |
| 3 | 0.68 | 0.07564 | +25.57% | 0.215 m |
| 4 | 0.78 | 0.10680 | **−5.09%** | 0.295 m |
| 5 | 0.86 | 0.10372 | **−2.06%** | 0.287 m |
| all six | — | 0.00000 | +100.00% | 0.000 m |

**Achievable span ≈ 99.8% on one site, 100% unconstrained, against `swe`'s ~7%.**

**This settles defect 12 as a statement about actuators, not physics.** `swe` and
this testbed are the *same PDE family*. What changed is that the control acts on
geometry — raising `z`, which gates where water can go — instead of adding an
equal source to the surface everywhere. The achievable band went from 7% to 99.8%
on the same equations. **Rule 18: when a testbed cannot separate planners, suspect
the actuator's mechanism before concluding the domain is uncontrollable.** The
retired `swe` verdict was correct about `swe` and wrong as a claim about
shallow-water control.

**A safety finding worth more than the gate.** Sites 4 and 5 — *downstream* of the
settlement — make flooding **worse**, by −5.09% and −2.06%. This is the levee
backwater effect: a levee downstream of a protected area impedes drainage and
backs water up into it. It is well known in flood management and it is exactly the
failure mode the paper's thesis is about, now in a domain where the solver can
demonstrate it. A planner that optimises a mis-specified objective, or that treats
every actuator as beneficial, does not merely waste budget here — **it harms the
thing it was deployed to protect.** Two of six available actions are actively
damaging, which makes this a far better constrained-planning testbed than the
wildfire task ever was.

**The task is now too easy at a 6.0 m budget, which is a design note not a
problem.** 0.5 m at site 2 already recovers +69.85%, because that gap has only
0.37 m of freeboard. So the informative regime for Phase 3 is a **tight budget
(~0.5–1.0 m) under hydrograph uncertainty**: the planner must commit levee heights
before seeing the flood magnitude, and the conformal margin bounds the shortfall.
That is the real levee-design problem — freeboard under flood-frequency
uncertainty — and it is where a calibrated margin earns its place rather than
being decoration.

**Reproducible runner:** `eval/run_flood_span.py`, which asserts geometry
separation and inflow placement before solving, so faults A and B cannot recur.



**Phase 3 apparatus, built 2026-09-25.** `eval/run_flood_margin.py`. The flood
decision is *one-shot*, not sequential: levee heights are committed before the
flood magnitude is known. That is the real levee-design problem — freeboard under
flood-frequency uncertainty — and it instantiates §3.5's mismatch exactly. The
planner controls `E_Q[g(a, Q)]`, an expectation over hydrographs; the limit is
declared on a single flood's *realised* depth. So matched-pair scores cancel the
per-instance scatter that actually breaches the limit, which is the regime where
the default under-covers. Both knobs of ρ = σ/ε are explicit: `--q-log-sigma`
sets σ, the training budget sets ε.

**Reality is the accepted solver, and that is now checkable.**
`pspe/simulate/real/floodcast.py` loads the published archive — DEM, Sentinel-2
land cover, GPM-IMERG rainfall, initial condition, and the reference depth
sequence (Pakistan 2022: 480 m, 14 days, 4032 frames at 300 s).
`eval/run_floodcast_validate.py` forces our solver with the event's own rainfall
and compares against the reference on the data paper's own metric, CSI at 0.01 m
and 0.05 m, plus RMSE and bias over wet cells. **Nothing is fitted to the
reference**: roughness comes from land cover via a published Manning table, not
from matching depths, because a solver tuned to reproduce its comparison target
says nothing about independence from it. Frames are sorted by integer, not
lexically — `10.tif` before `9.tif` would silently shuffle a flood's time axis.

Solver extended to a **spatially varying Manning field** (averaged to interfaces),
since roughness is the one calibration knob a hydrodynamic model has. Lake-at-rest
remains exact with a varying field.

**Two bugs found in the new code, both mine, both of a kind this project has seen
before:**

* *SLURM exit codes, again.* `scripts/tacc/vista_flood.slurm` had `wait` inside
  the case **and** a `jobs -p` loop to collect exit codes. `wait` reaps the jobs,
  so `jobs -p` returns nothing and `rc` stays 0 however many runs failed — the
  mirror image of the bug that once marked five COMPLETED jobs FAILED. Now
  collects PIDs at launch.
* *The precondition arithmetic was wrong.* I had computed headroom as
  `limit − mean realised depth`. Proposition 2 compares against `f·s` where **`s`
  is the measured achievable span**, not the distance from an arbitrary limit to
  an arbitrary reference plan. The wrong version reports FAIL trivially whenever
  the reference plan sits above the limit, which says nothing about whether a
  margin fits. It now measures `s` directly (do-nothing vs best-budget) and
  reports what fraction of the span the margin consumes.

**Conformal caveat, stated not hidden.** The margin is calibrated at the
margin-free plan `a0` and deployed at `a1 = plan(q)`. Exchangeability therefore
holds in the hydrograph, not across the plan shift from `a0` to `a1`. Any coverage
gap attributable to that shift is a real effect and is reported.

**First precondition measurement, and it is more interesting than a verdict.**
A local probe at hydrograph log-spread 0.25 measured:

| quantity | value |
|---|---|
| surrogate error ε | 0.1401 m |
| realised-depth spread σ | 0.2416 m |
| **ρ = σ/ε** | **1.72** |
| bias b | −0.0060 m (negligible) |
| required margin b + z_δσ | 0.3036 m |
| do-nothing depth (caps the span s) | 0.2898 m |

**The required margin exceeds the entire achievable span.** Same outcome as the
wildfire task, but for a structurally different and far more informative reason.
On wildfire the bias was 2.0–4.9× the span — a mundane mis-specification. Here
the bias is negligible and the whole margin is `z_δσ`.

**The precondition is dimensionless.** Since a good plan drives depth to ≈ 0, the
span is capped by the do-nothing depth, and both σ and s scale with flood
magnitude, so absolute flood size cancels:

```
b + z_δ·σ < f·s      with b ≈ 0      ⟺      σ/s  <  f/z_δ
```

| f | required σ/s | measured σ/s | shortfall |
|---|---|---|---|
| 0.2 | 0.156 | **0.834** | 5.3× |
| 0.3 | 0.234 | **0.834** | 3.6× |
| 0.5 | 0.390 | **0.834** | 2.1× |

This converts Proposition 2 from an abstract inequality into a statement about
**how much forecast skill a calibrated margin requires** — a quantity a flood
agency could check before adopting one.

**Threshold amplification is why it fails, and it is the finding.** The
hydrograph carried a 25% log-spread, but realised depth spread was 83% of the
mean — a **3.33× amplification**. The cause is that over-topping is a threshold:
right at the freeboard crest, a small discharge error moves depth a great deal.

So there is a real tension, and it is not specific to flood:

> **The regime where an intervention has leverage is the regime where the hazard
> is most sensitive to forecast error, hence where a calibrated margin is hardest
> to fit.** A levee matters precisely at the freeboard threshold; that is also
> where depth uncertainty is amplified most.

This is a statement about safety margins in **threshold-governed** hazards
generally, and it is testable: the precondition should pass once σ/s < f/z_δ. The
queued sweep tests exactly that.

**The amplification mechanism, measured (`eval/run_flood_elasticity.py`).** The
claim above — that threshold behaviour amplifies forecast error — is testable as
an elasticity, since for a small log-spread `σ_lnD ≈ E·σ_lnQ` with
`E = d ln D / d ln Q`. All operating points share one batched solve, so the arms
cannot drift apart on timestep sequence. Prediction recorded before the run: **E
peaks where the flood is just topping the weakest gap**, which is where a levee
has leverage.

| channel inflow | berm rows wet | **elasticity E** |
|---|---|---|
| 4.0e-3 | 0/96 | — (town dry) |
| 6.0e-3 | 0/96 | — (town dry) |
| **8.0e-3** | **2/96** | **40.5×** |
| 1.0e-2 | 4/96 | 12.5× |
| 1.2e-2 | 6/96 | 4.6× |
| 1.4e-2 | 9/96 | 3.2× |
| 1.8e-2 | 32/96 | 2.5× |
| 2.4e-2 | 56/96 | 2.4× |

**Confirmed, and more sharply than predicted.** Amplification is **40× at the
onset of over-topping** and decays monotonically to 2.4× once the berm is broadly
submerged. A 10% discharge error becomes a four-fold depth error at the threshold.
The mechanism is measured rather than asserted, and it is a property of the
threshold, not of the scenario's particular numbers.

This sharpens the tension into something quantitative: the conditions that make an
intervention *worth planning* — water near the crest, where a levee decides
whether the settlement floods — are the conditions under which realised depth is
most sensitive to forecast error, and therefore where a conformal margin needs to
be widest relative to the gain it protects.

*One defect found and fixed in the first run of this experiment.* The span column
used an **even spread of the budget across all six sites**, but two of those sites
worsen flooding (the backwater effect), so the even spread understates the
achievable span and biased `σ/s` upward — the same fault I had just corrected in
`run_flood_margin.py` and then repeated here. Both now scan single-site
allocations and take the best. The elasticity column is unaffected, since it is
computed from do-nothing depths across discharge. The first run's `σ/s` column is
withdrawn; the corrected values follow.

**`σ/s` has an interior minimum — feasibility is squeezed from both sides.**
Corrected run, with the span taken over the best allocation:

| channel inflow | E | span s | best-plan depth | **σ/s at σ_log = 0.25** | verdict (f = 0.3) |
|---|---|---|---|---|---|
| 8.0e-3 | 40.5 | 0.0001 | 0.0000 | 10.187 | FAIL |
| 1.0e-2 | 12.5 | 0.0429 | 0.0000 | 3.132 | FAIL |
| 1.2e-2 | 4.6 | 0.1529 | 0.0000 | 1.154 | FAIL |
| **1.4e-2** | 3.2 | 0.2688 | 0.0027 | **0.803** ← minimum | FAIL |
| 1.8e-2 | 2.5 | 0.3195 | 0.2282 | 1.071 | FAIL |
| 2.4e-2 | 2.4 | 0.3199 | 0.7888 | 2.049 | FAIL |

Two distinct mechanisms bound the feasible region, one at each end:

* **Small floods — threshold amplification.** E reaches 40×, so σ explodes
  relative to a span that is still nearly zero.
* **Large floods — protection saturation.** The levee can no longer keep the
  settlement dry (best-plan depth climbs 0.003 → 0.23 → 0.79 m), so the span
  stops growing while σ keeps rising.

A calibrated margin is therefore most nearly feasible at an *intermediate* flood
magnitude, and fails at both extremes for unrelated reasons. At σ_log = 0.25 it
fails everywhere, by 3.4× at the best operating point.

**Cross-validation.** Two independent routes to σ/s at Q = 1.4e-2 agree to 3.7%:
direct Monte Carlo over 96 solver hydrographs gives **0.834**; the
elasticity-implied value `E·σ_lnQ·D/s` gives **0.803**. The mechanism and the
measurement are consistent, which is what licenses reading the elasticity column
as an explanation rather than a coincidence.

**Figure.** `docs/figures/paper/fig9_flood_elasticity.pdf`, two panels: the
amplification against how much of the berm is topped, and `σ/s` against flood
magnitude with the feasibility rule `f/z_δ` drawn. Generated by
`scripts/make_paper_figures.py` from `runs/flood_elasticity/elasticity.json`.

**The engineering statement this yields.** Inverting the precondition at the best
operating point: discharge log-spread must fall from 0.25 to below **0.073** for a
δ = 0.1 conformal margin to fit inside 30% of the achievable span. That is, **a
calibrated margin is usable on this levee task only with discharge forecasts
accurate to roughly 7%** — a checkable requirement, and the kind of number a flood
agency could hold a forecast product to before adopting a margin at all.

**Sweep design corrected before it ran.** My first σ ladder (0.10–0.45) sat
*entirely* on the failing side. That is the rdf-sweep mistake — a ladder that does
not straddle the boundary can only restate the failure and can never show the
precondition holding, which is the half that makes it a diagnostic rather than a
complaint. Resubmitted as job 1025409 over σ_log ∈ {0.02, 0.05, 0.08, 0.12, 0.20,
0.30}, which should bracket the crossing near 0.07 given the measured 3.33×
amplification.

**ρ is not controllable through the hazard's spread — a design finding.** The σ
sweep measured, at a fixed training budget of 240 samples:

| σ_log | σ (depth) | ε | **ρ = σ/ε** |
|---|---|---|---|
| 0.02 | 0.0152 | 0.0111 | **1.37** |
| 0.05 | 0.0378 | 0.0283 | **1.34** |
| 0.08 | 0.0603 | 0.0431 | **1.40** |
| 0.12 | 0.0917 | 0.0568 | 1.62 |
| 0.20 | 0.1601 | 0.0795 | 2.02 |
| 0.30 | 0.2817 | 0.0461 | 6.12 |

**ρ is nearly constant at ≈1.35 across the first three arms**, because ε scales
with σ: at a fixed training budget a surrogate's error tracks the variability it
must capture, so widening the hydrograph spread moves numerator and denominator
together. Widening the hazard's spread therefore does *not* walk ρ across 1.

Two consequences. First, the `rho` experiment must vary **surrogate quality** at
fixed σ, which is how it is written — the σ ladder would have produced a sweep
that never crossed the boundary, the same failure mode as the original rdf sweep,
for a different underlying reason. Second, wherever ρ is reported from this
testbed, the coupling between ε and σ has to be stated: they are not independent
knobs, and a ρ quoted without its training budget is not reproducible.
**Rule 19: before treating a diagnostic ratio as a control variable, check that
its numerator and denominator can actually be moved independently.**

**A timing defect, caught before it destroyed a run.** The span scan evaluated
each of the eight candidate allocations in its own batched solve over `n_cal`
hydrographs. With six arms sharing a node that needed ~56 minutes against a
50-minute limit, so job 1025437 would have died with nothing written. The span is
an *average* and needs far fewer draws than calibration does, and all candidates
can share one solve. Now batched across candidates at `--n-span 48`, roughly a
quarter of the work. The σ/ε/ρ values above were already recovered from the
cancelled job's stdout, so nothing was lost.

**Precondition verdicts across hydrograph spread (job 1025566, COMPLETED
32:32).** The prediction from the elasticity analysis — PASS below σ_log ≈ 0.08,
FAIL above — is confirmed, and the ladder straddles the boundary, which is what
makes this a diagnostic rather than a complaint:

| σ_log | ε | σ | ρ | span s | required b+z_δσ | headroom f·s | % of span | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.0111 | 0.0152 | 1.37 | 0.2632 | 0.0187 | 0.0789 | 7% | **PASS** |
| 0.05 | 0.0274 | 0.0378 | 1.38 | 0.2567 | 0.0478 | 0.0770 | 19% | **PASS** |
| 0.08 | 0.0431 | 0.0603 | 1.40 | 0.2487 | 0.0727 | 0.0746 | 29% | **PASS** |
| 0.12 | 0.0632 | 0.0917 | 1.45 | 0.2372 | 0.1141 | 0.0712 | 48% | FAIL |
| 0.20 | 0.0356 | 0.1601 | 4.50 | 0.2153 | 0.2444 | 0.0646 | **114%** | FAIL |
| 0.30 | 0.0409 | 0.2817 | 6.89 | 0.1947 | 0.3396 | 0.0584 | **174%** | FAIL |

**Phase 3 is feasible** at σ_log ≤ 0.05, where the margin costs 19% of the span.
The measured span, 0.195–0.263, brackets the elasticity run's 0.2688 at the same
operating point.

Three observations, none of them anticipated:

* **It is a double squeeze.** The span *itself* shrinks as spread widens
  (0.2632 → 0.1947), because averaging over wider hydrographs pulls in extreme
  floods the levee cannot stop. Headroom therefore falls (0.0789 → 0.0584) while
  the requirement rises 18×. Feasibility is attacked from both sides at once, so
  the precondition degrades faster than `z_δσ` alone would suggest.
* **Margin collapse is reached, not just approached.** At σ_log ≥ 0.20 the
  required margin is **114% and 174% of the entire achievable gain**. Tightening
  the limit by it leaves nothing to plan for — the degenerate regime
  `run_flood_margin.py` now detects and labels rather than reporting as coverage.
* **ρ is non-monotonic at the high end** (1.45 → 4.50 → 6.89), because ε *falls*
  at σ_log 0.20 and 0.30 (0.0632 → 0.0356 → 0.0409). At 240 training samples that
  is fitting noise, not signal, and the high-σ ρ values are not read as
  meaningful. It is a further instance of rule 19: ε and σ are not independent,
  and ε is itself noisy at this budget.

**Figure.** `docs/figures/paper/fig10_flood_precondition.pdf`: required margin
against headroom across forecast spread, crossing marked at 8.2%. Both curves are
drawn because the headroom erodes as the requirement rises — the double squeeze.

**Proposition 2 works as an acceptance test.** Across six spreads it separates
the feasible regime from the infeasible one, agreeing with a prediction made
before the run from an independently measured mechanism. The crossing sits
between σ_log 0.08 and 0.12, i.e. **discharge forecasts accurate to roughly
8–12%** — the number a flood agency would hold a forecast product to before
adopting a calibrated margin.

**The archive does not match its own data paper, and this constrains the work.**
The paper describes a "relevant data" folder with DEM, land use, rainfall,
georeferencing and initial conditions. The published archive ships
`Study regions/` containing **only DEMs**. Two consequences, both reported rather
than worked around:

* *No rainfall*, so the event's own forcing cannot be replayed. Validation runs
  instead on a window where the forcing is negligible, and the residual is
  **measured from the reference's own volume budget** rather than assumed.
* *No land cover*, so Manning's n is a single literature value (0.035), not a
  land-cover map and above all not fitted to the reference — a solver tuned to
  reproduce its comparison target says nothing about independence from it.

**Only one event can be placed at all.** Depth TIFFs carry **no georeferencing
tags**. Tested physically, since flooded cells must sit in low ground:

| event | wet cells at elevation percentile |
|---|---|
| Australia 30 m | **0.253** — DEM and depth grids match 1073×1073 exactly |
| Pakistan 480 m | **0.452** across nine candidate 16× block-average offsets — indistinguishable from chance (0.5) |

Pakistan's placement is not recoverable from the archive, so `load_event` now
**raises rather than guessing**: a wrong correspondence would yield a plausible
but meaningless CSI. Validation runs on Australia 2022.

**Solver validation: our implementation reproduces the accepted scheme.**
`eval/run_floodcast_validate.py`, Australia 2022 at 30 m, initialised from the
reference at t = 192 h and free-running with no forcing:

| quantity | value |
|---|---|
| window | t = 192–206 h |
| unmodelled forcing over the window | **0.055% of volume (0.0039%/h)** |
| bed slope, measured from the DEM | 1.53e-3 |
| **mean CSI@0.01** | **0.979** (closed domain; 0.974 with the outflow bug) |
| **mean CSI@0.05** | **0.978** |
| mean RMSE over wet cells | 0.091 m (depths reach 18 m, so ≈0.5%) |
| bias | **−0.0001 m** — essentially zero once the boundary is corrected |

The small dry bias is consistent with the 0.055% of forcing that cannot be
replayed. For scale, flood mapping against SAR observations typically reports CSI
0.6–0.8; **0.96 is agreement between two implementations of the same scheme**,
which is what §5.6's claim requires. This turns "our reality model is the accepted
solver" from an argument from construction into a measurement.

**The decay curve (job 1025760, 15-minute frames).** Mean CSI **0.974** at 0.01 m
and **0.973** at 0.05 m, mean RMSE 0.171 m:

| t (h) | CSI@0.01 | CSI@0.05 | RMSE (m) | bias (m) | wet ref/ours |
|---|---|---|---|---|---|
| 192.00 | 1.000 | 1.000 | 0.000 | +0.000 | 548865 / 548865 |
| 192.25 | 0.982 | 0.981 | 0.166 | −0.014 | 548571 / 548093 |
| 192.50 | 0.977 | 0.975 | 0.178 | −0.018 | 548316 / 546106 |
| 193.00 | 0.970 | 0.969 | 0.192 | −0.024 | 547817 / 542979 |
| 193.50 | 0.966 | 0.965 | 0.202 | −0.028 | 547313 / 540531 |
| 194.00 | 0.963 | 0.962 | 0.210 | −0.032 | 546821 / 538571 |

Decay is slow and **decelerating** — most of the loss falls in the first fifteen
minutes, then ~0.005 per quarter-hour and shrinking, which is the signature of two
solutions settling toward a similar quasi-steady state rather than diverging.

**A systematic dry bias, traced to our own boundary treatment.** Unmodelled
forcing over this window is 0.012% of volume, while the wet-cell deficit grew to
8,250 cells — about **1.5%, roughly 100× larger than the forcing could account
for**. So it was a genuine difference between implementations, and attributing it
to the archive's gaps would have been convenient and wrong. The leading suspect
was our own configuration: all four domain edges opened with a normal-depth
outflow driven by a *single domain-average* bed slope over a 1073² domain.

A closed-domain run (job 1025861) confirms it:

| | open, 4 edges | **closed** |
|---|---|---|
| mean CSI@0.01 | 0.9739 | **0.9785** |
| mean CSI@0.05 | 0.9726 | **0.9783** |
| mean RMSE over wet cells | 0.171 m | **0.091 m** |
| bias at t+2 h | −0.0315 m | **−0.0001 m** |
| wet cells at t+2 h (ref 546,821) | 538,571 | 542,674 |

**The bias was entirely ours.** It falls from −0.032 m to essentially zero and
RMSE nearly halves. The closed domain is the configuration to report: over a
two-hour window the flood does not reach the domain edge, so a crude outflow law
there could only remove water that should have stayed. Corrected agreement is
therefore **CSI 0.979, RMSE 0.091 m, bias ≈ 0** — better than the first number
reported.

A residual wet-cell deficit of 4,147 (0.76%) survives with bias ≈ 0, so the depths
agree on average and the remaining disagreement sits in marginal cells near the
0.01 m threshold. Candidates are the uniform Manning of 0.035 against the
reference's unknown roughness, and genuine differences in the discretisation.
Not chased further: it does not affect any claim made here, and saying so is
cheaper than an explanation that is not measured.

### 5.6a Moving the planning study onto real terrain (2026-09-26)

**The gap this closes.** Everything in §5.6 above — span, elasticity, the
precondition sweep — ran on `floodplain_terrain`, a valley built and tuned until
the settlement flooded from the channel. Only the *solver validation* used the
archive. So "the framework was run on real flood data" was true of a quarter of
the package, and the planning claims rested on synthetic geometry. That is not a
caveat to bury in a limitations paragraph; it is the difference between two
different papers.

`pspe/simulate/real/floodcast_scenario.py` replaces every synthetic ingredient:

| ingredient | now taken from |
|---|---|
| terrain | the Australia 2022 DEM, 4× block-averaged to 120 m |
| initial state | the reference depth field at t = 144 h |
| flood magnitude and timing | the reference's own **mass budget** over t = 144–192 h → **5.12 mm/h** |
| settlement | the compact block flooding deepest in the reference — (47,50), bed 6.6 m, peak 1.32 m |
| levee sites | the **lowest cell of each perimeter sector**, elevations 2.6–12.9 m |

Two choices to state plainly rather than defend later.

*Coarsening to 120 m is what makes this possible at all.* The local-inertial
timestep scales with `dx` while cell count falls as `dx²`, so 4× buys ~64×, and a
study needing hundreds of solves is otherwise out of reach on 1073². FloodCastBench
publishes Australia at 30 m and 60 m, so coarsening is within the dataset's own
conventions — but it is a genuine loss of fidelity.

*Only the forcing's spatial distribution is approximated.* The archive ships no
rainfall field, so rainfall is uniform. Magnitude and timing come from the data;
the spatial pattern does not. This is carried in the JSON output so it travels
with the numbers.

*Levee placement is terrain-derived.* Each candidate sits at the lowest point of
a perimeter sector — where water actually enters. The sites were not chosen to
make the result come out well, and downstream sectors are included, so the
backwater effect can reappear if it is real.

**An anticipated failure mode, recorded before the result.** Uniform rainfall
falls *inside* the settlement as well as outside, and no levee can stop rain
landing within the ring — exactly what killed the first synthetic scenario. At
5.12 mm/h over 48 h that is 0.245 m of direct accumulation against a reference
peak of 1.32 m, so roughly **19% of the settlement's flooding is uncontrollable by
construction**, capping the achievable span near 81% before any routing argument.
If the measured span comes in far below that, direct accumulation rather than
routing is the first hypothesis to test.

**Real-terrain span: 2.4%, against 100% on the synthetic valley.** Do-nothing
exposure-weighted peak depth 1.03803 m; the best single levee buys **+2.43%**,
and greedy reaches only +1.25% at 1.0 m of budget. That is **below the ~7% band
at which `swe` was retired for being unable to rank planners**. Had the synthetic
numbers been written up as "flood", the paper would have carried a ~40×
overstatement of what a levee achieves.

| site | perimeter elevation | reduction |
|---|---|---|
| 0 | 12.9 m | +1.97% |
| **1** | **2.6 m** | **−18.52%** |
| 2 | 7.5 m | +2.43% |
| 3 | 7.4 m | +0.68% |
| 4 | 6.8 m | +0.27% |
| 5 | 6.7 m | +0.39% |

**The mechanism, measured rather than inferred.** A diagnostic decomposing the
settlement's 1.038 m:

| source | depth | share |
|---|---|---|
| already present at t = 144 h | 0.280 m | **27%** |
| direct rain inside any ring | 0.246 m | **24%** |
| routing of existing water | 0.262 m | 25% |
| rainfall-driven total | 0.495 m | 48% |

**51% is uncontrollable by construction** — water in place before planning starts,
plus rain landing inside whatever a levee encloses. And **the lowest perimeter
point is an outlet, not an inlet**: leveeing it moves depth 1.038 → 1.230 m, a
confirmed **−18.5%**. My terrain-derived placement rule assumed the low point was
where water enters; on a settlement sitting in a broad floodplain it is where
water leaves.

**The transferable claim: the instrument must match the mechanism.** A levee earns
its ~100% against **fluvial** flooding arriving through a constriction — the
geometry I built synthetically, in which a berm gap is the only door. Australia
2022 at this site is **pluvial**: rainfall-driven sheet flow, where water arrives
everywhere at once and a quarter of it falls inside the protected area. No levee
placement helps, because there is no door to close. PSPE's planning layer is only
as useful as the actuator it is handed, and selecting that actuator requires
knowing the hazard's mechanism. For a pluvial event the right instruments are
drainage, retention or pumping — actuators this framework does not currently have.

**A design error of mine, corrected — and the pre-registered prediction held.**
Starting at t = 144 h put the planner *mid-flood*, with 27% of the water already
in place before any levee could act. A planner must commit before the flood
arrives. Rerun from t = 48 h (volume 4,950, near-dry) to t = 120 h. Expectation
recorded before the run: the span improves, because the already-present term
vanishes, but stays at or below the ~7% threshold, since neither the direct rain
nor the sheet-flow character changes; above 20% would indicate a bug, not a
result.

**Measured: 2.4% → 6.18%, plateauing, and still below 7%.**

| site | perimeter elev | early start (t=48 h) | mid-flood (t=144 h) |
|---|---|---|---|
| 0 | 3.1 m | −8.79% | +1.97% |
| 1 | 2.4 m | −11.94% | −18.52% |
| 2 | 7.3 m | +0.72% | +2.43% |
| 3 | 7.5 m | +0.71% | +0.68% |
| 4 | 7.5 m | **+5.55%** | +0.27% |
| 5 | 8.0 m | −0.72% | +0.39% |

Greedy trace: +0.97 → +4.59 → +5.26 → +5.80 → +5.90 → +5.95 → **+6.18%** at 3.5 m
of a 4.0 m budget, clearly flattening. **Both real-terrain runs sit below the ~7%
band at which `swe` was retired for being unable to rank planners**, against 100%
on the synthetic fluvial valley.

**~~Conclusion: the real-terrain levee task fails the span gate~~ — WITHDRAWN,
2026-09-26.** That conclusion was drawn from one settlement and it was wrong. See
§5.6b: screening all 933 candidate settlements and repeating the identical levee
test at the best-defended one gives **+15.37% single-site, +17.83% all-sites**,
comfortably above the ~7% band. **The task passes the gate about a kilometre from
where I first measured it.**

**Both jobs hit their wall clock** (40:29 and 50:10), so the greedy figures are
lower bounds. The trace is flattening and the single-site maxima (+2.43%, +5.55%)
bound the achievable, so a longer run would not change the verdict — but the
numbers are reported as what they are, incomplete greedy searches.

**Two reporting errors of mine, recorded.** I twice characterised the early-start
sweep from partial stdout: first calling it "worse" on three of six sites (the
best site had not yet reported), then retracting a prediction that was in fact
correct. The rule is the one already written in Part 7 and broken here anyway:
**do not read a trend from an incomplete sweep.**

### 5.6b The negative result was a site-selection artefact (2026-09-26)

**The confound, and why it had to be tested.** §5.6a's span was measured at the
block that floods *deepest* in the reference. That criterion selects whichever
area holds the most water, which is not the same as the area a levee can defend —
plausibly a sump that fills regardless. A headline negative resting on one
hand-picked target is not a finding, so it was screened.

**The screen is cheap because the depth field does not depend on the settlement,
only the exposure weighting does.** Two solves — one with rainfall, one without —
decompose the water budget at every candidate site at once
(`eval/run_flood_sites.py`, 933 sites, 10 minutes):

| row | col | elev | total | routed | rain-driven | controllable | ctrl % |
|---|---|---|---|---|---|---|---|
| **52** | **52** | 5.2 | 0.279 | 0.001 | 0.277 | **0.193** | **69%** |
| 44 | 52 | 6.3 | 0.273 | 0.001 | 0.272 | 0.187 | 68% |
| 52 | 60 | 9.2 | 0.232 | 0.001 | 0.231 | 0.146 | 63% |
| 76 | 172 | 0.5 | 0.224 | 0.001 | 0.222 | 0.138 | 61% |

**The site I chose was not a bad target.** It sits among the top few of 933 on
controllable depth (~69%), essentially tied with the winner. The controllability
hypothesis was *not* what separated them.

**What separated them was levee geometry.** The identical test at (52,52):

| site | perimeter elev | reduction |
|---|---|---|
| 0 | 13.5 m | −0.27% |
| 1 | 3.6 m | +0.09% |
| 2 | 6.7 m | −0.43% |
| **3** | **6.7 m** | **+15.37%** |
| 4 | 6.2 m | +1.58% |
| 5 | 9.3 m | −1.18% |
| all six | — | **+17.83%** |

**Centres 8 cells (~1 km) apart differ 3× in achievable span**: +5.55% at (44,50)
against +15.37% at (52,52), with near-identical controllable fractions. One
perimeter sector at (52,52) intercepts the inflow; at (44,50) no sector does.

**Rule 20: a feasibility gate measured at a single hand-picked target measures the
target, not the task.** Screen the target space before reporting infeasibility.
This is the second time this project has drawn a strong conclusion from one
configuration — the first was the wildfire task — and the cost here was a wrong
headline held for several hours.

**Corrected standing.** Real terrain gives **17.8%** achievable span against
~100% on the synthetic fluvial valley and the ~7% floor at which `swe` was
retired: real but substantially smaller leverage than a constructed geometry
suggests, and enough to plan against. Phase 3 therefore runs (job 1026669).

*A reporting bug found while writing this up.* The screen's "ranks 1/933" line
matched the default settlement against the sorted candidate list within a
±stride box, so `next` returned the best-ranked site in the window rather than
the nearest one. It now takes the nearest by distance and prints that distance.

### 5.6b-note Defect 20 — the levee's physical size was an accident of the grid

Building the portal's scenario library at 60 m rather than 120 m produced a
six-fold drop in the same site's effectiveness: **+15.37% at 120 m against
+2.44% at 60 m**. That is not a solver resolution effect. `site_width` was
specified in **cells**, so refining the grid halved the structure:

| grid | levee footprint | where used |
|---|---|---|
| 120 m | **360 m** | site screening (§5.6b), precondition (§5.6c) |
| 60 m | **180 m** | portal scenario library |

The two numbers describe different structures, not the same structure measured
twice. **Rule 21: an intervention's physical dimensions must be specified in
physical units. If refining the grid changes the thing being built, the
comparison is meaningless.** Fixed: `site_width_m`, converted to cells at
construction.

**What this does to prior results.** §5.6b and §5.6c are internally consistent —
both used 120 m, so both describe a 360 m levee — and their conclusions stand as
statements about *that* structure. But the levee's size was never a design
choice, and a 360 m levee is a modest local work, not a scheme. Levee length is
properly a *planning variable*, and neither the span nor the precondition has
been measured as a function of it. That is the obvious next experiment and it is
not done.

### 5.6c The margin on real terrain: usable only below ~3.6% forecast spread (2026-09-26)

With the span gate passed at the screened site (§5.6b), the precondition was
measured on the Australia 2022 DEM at three rainfall forecast spreads. Prediction
recorded before the run: fail at 10%, possibly pass at 2%.

| rainfall spread | do-nothing | best | span s | σ | z_δ·σ | headroom f·s | % of span | verdict |
|---|---|---|---|---|---|---|---|---|
| **2%** | 0.27782 | 0.23499 | 0.04283 | 0.00556 | 0.00713 | 0.01285 | **17%** | **PASS** |
| 5% | 0.27559 | 0.23406 | 0.04153 | 0.01379 | 0.01768 | 0.01246 | 43% | FAIL |
| 10% | 0.27265 | 0.23373 | 0.03892 | 0.02808 | 0.03598 | 0.01168 | 92% | FAIL |

**Crossing at ≈3.6% rainfall spread**, against **8.2%** on the synthetic fluvial
testbed (§5.6). **Real terrain is 2.3× more demanding.** The double squeeze
recurs: the span itself contracts as spread widens (0.0428 → 0.0389), so
feasibility is attacked from both ends, not one.

The best allocation is a single site in every case — the one sector of six that
intercepts inflow rather than outflow — which is consistent with §5.6b and means
the result does not depend on a clever multi-measure plan.

**What this means, stated plainly.** Operational quantitative precipitation
forecasting does not achieve 3.6% accuracy at 48-hour lead; errors of tens of
percent are normal. **So on this task, with today's forecast skill, a δ = 0.1
conformal margin does not fit inside the gain a levee buys.** The margin is not
unusable in principle — it passes at 2% — but the forecast accuracy it requires
is well beyond what is available.

That is a harder claim than the synthetic testbed supported, and it is the honest
one: the framework's safety guarantee is available only where the hazard can be
forecast far more precisely than this hazard currently can be. Three ways it
could still hold — a shorter lead time where forecasts are sharper, a hazard with
tighter forcing uncertainty, or an intervention with a larger span — are testable
and none has been tested here.

**Compute status, 2026-09-25.** Vista reachable again (the ControlMaster socket
was for `login2`, so the round-robin `vista` name resolved to a different
ControlPath). **Allocation ATM23014 holds 1761 SUs and expires 2026-09-30** — five
days — so Phases 1 and 3 as scoped cannot complete on it; renewal is the binding
schedule item. FloodCastBench (20.1 GB) is downloading to `$WORK`; it cannot land
on the laptop, which has 21 GiB free. Precondition screen queued as job 1025375
across four hydrograph spreads (σ = 0.10/0.20/0.30/0.45), time limit cut to 55 min
to make it backfillable — the `gh` partition is at 565/576 nodes allocated.

### 5.7 Observed extent from Sentinel-1: the reference is not the world (2026-09-27)

**Why this had to be built.** Every flood number in this project was scored
against a *model*. §5.6 reports CSI **0.979** for our solver against
FloodCastBench's reference — but that reference is itself a shallow-water
solution. Two models agreeing is not evidence that either matches the world.
`pspe/observe/sar.py` supplies the missing term: what a satellite saw.

**Method.** Sentinel-1 RTC from Planetary Computer (terrain-corrected and
radiometrically calibrated already, so no SNAP stage). Flood scene **2 March
2022, relative orbit 74**, against the **median of five same-track baselines**
(1, 13, 25 Jan; 6, 18 Feb). Two rules carry the judgement:

* **Same relative orbit only.** Backscatter depends on incidence angle, so
  differencing across tracks manufactures "flooding" wherever geometry changed.
  Seven of thirteen available scenes were rejected on this ground.
* **Flood is water now that was not water before.** A threshold on one scene
  marks every smooth surface — tarmac, sand, the sea — as flood. Requiring both
  a dark pixel (Otsu, −15.2 dB) and a ≥3 dB drop against the baseline, while
  excluding pixels dark in both, leaves standing water.

Observed: **290 km² flooded**, plus 112 km² of permanent water correctly
separated out.

**Result — the reference matches observation at CSI 0.38.** Scanning every
reference frame for the best match also recovers the timing (day 7.62):

| reference depth threshold | CSI | POD | FAR |
|---|---|---|---|
| 0.02 m | 0.382 | 0.754 | 0.563 |
| 0.05 m | **0.378** | **0.733** | **0.561** |
| 0.10 m | 0.370 | 0.704 | 0.562 |
| 0.25 m | 0.341 | 0.622 | 0.570 |

Stable across thresholds, so not a threshold artefact. The disagreement has a
direction: both agree on 213 km², the satellite sees 78 km² the model misses,
and **the model floods 272 km² the satellite says is dry**.

### 5.7a Defect 21 — 119 km² of the "over-prediction" was the sea (2026-09-27)

That 272 km² is wrong, and the way it was caught matters: I drew the comparison
map for the appendix (`figA1`) and the orange "model only" band ran straight
down the coast and along the river channel. The model marks the ocean and the
permanent channel as water because they *are* water; the change-detection rule
excludes them from the observed flood by construction, because they were water
in the baseline too. Neither is disagreeing with the other. Counting them as
model error measured nothing except that the sea is wet.

Excluding permanent water (ESA WorldCover class 80, 123 km² over this
catchment) and then the radar-blind classes separately:

| comparison set | CSI | POD | FAR | both | model only | satellite only |
|---|---|---|---|---|---|---|
| all comparable cells (as first reported) | 0.378 | 0.733 | 0.561 | 213 km² | **272 km²** | 78 km² |
| excluding permanent water | 0.478 | 0.732 | 0.421 | 211 km² | **153 km²** | 77 km² |
| excluding permanent water and radar-blind cover | **0.535** | 0.738 | **0.339** | 208 km² | 107 km² | 74 km² |

Two separate corrections, each worth about 0.06–0.10 of CSI, and they compose:
the reference's over-prediction against ground a satellite can actually
adjudicate is 107 km² against 208 km² of agreement, not 272 against 213. The
stratified figures in the table above are unchanged — they were already
computed per land-cover class, and permanent water is its own class — but the
*headline* number moves from 0.378 to 0.535.

**Rule 26: before scoring extent against an instrument, remove the water that
was already there.** A change detector answers "what is newly wet"; a hydraulic
model answers "what is wet". Those are different questions over the sea, over
lakes and over the channel, and the difference is charged entirely to the model
unless permanent water is excluded from both sides. Nothing in the pipeline
flagged this — the arithmetic was correct throughout, and the error was only
visible as a shape. **Plot the disagreement before quoting it.**

**Stratified by land cover — the headline number was mostly an artefact.**
ESA WorldCover over the catchment: 34% grassland, **31% tree cover**, 16%
herbaceous wetland, 12% permanent water, 7% cropland, 0.7% built-up. C-band
radar cannot adjudicate standing water under canopy or among buildings, so
scoring there charges the model for error the instrument cannot see.

| stratum | share | CSI | POD | FAR |
|---|---|---|---|---|
| all (the headline) | 100% | 0.378 | 0.733 | 0.561 |
| **where radar can see** | **57%** | **0.532** | 0.738 | **0.344** |
| canopy and built-up | 32% | 0.063 | 0.495 | **0.932** |

| cover class | CSI | FAR |
|---|---|---|
| herbaceous wetland | 0.596 | 0.357 |
| cropland | 0.528 | **0.192** |
| grassland | 0.493 | 0.368 |
| **tree cover** | **0.059** | **0.937** |

**Correction to the claim above.** The 0.378 figure charged the model for 31%
forest where the sensor is blind by physics, and tree cover alone carries
FAR 0.937. Restricted to ground radar can judge, the reference scores **CSI
0.53, POD 0.74, FAR 0.34**: moderate over-prediction, not the failure the
headline implied. Cropland — where radar is most reliable — is where the model
is most accurate (FAR 0.19), which is the pattern one would expect if the
remaining disagreement were largely instrumental rather than modelling error.

**What this does and does not establish.**

It does *not* show our solver is wrong: it reproduces the reference closely, and
that stands. It shows the reference **over-predicts extent moderately where this
can be checked**, and that our CSI 0.979 was agreement with a model of moderate
rather than unknown skill.

**Rule 23: agreement with a reference is not validation unless the reference has
itself been scored against observation.** That remains the lesson. But rule 23
has a companion learnt the same day — **Rule 24: before scoring a model against
an instrument, establish where the instrument can see.** An unstratified CSI
against radar silently penalises a model for every flooded forest, and I
published 0.378 before checking, which overstated the case by a wide margin.

**Replication attempt — one event validated, one not, and the failure is
instructive.** Of the archive's four events only two can be placed at all:
Australia, whose DEM grid matches its depth rasters exactly, and **UK**, whose
offset proved recoverable (a symmetric 5-pixel crop, r=5 c=5, confirmed by the
physics test at elevation percentile 0.298 against 0.365 for the worst
candidate). Pakistan and Mozambique are resampled onto unrelated grids at ~16x
and cannot be placed.

The UK event is Storm Desmond at **Carlisle, Cumbria** (−2.90, 54.91), 5–6
December 2015. The nearest same-track scene is **8 December, orbit 132**, with
two baselines. It returns CSI 0.059, and that number should not be read as a
verdict on the model:

| threshold (dB) | SAR detects | CSI (visible) | POD | FAR |
|---|---|---|---|---|
| −18 | 0.06 km² | 0.005 | 0.771 | 0.995 |
| −16 | 0.44 km² | 0.037 | 0.678 | 0.962 |
| −14 | 1.27 km² | 0.094 | 0.586 | 0.900 |
| −12 | 2.07 km² | **0.127** | 0.514 | 0.856 |
| −7.6 (Otsu) | 2.10 km² | 0.079 | 0.377 | 0.910 |

**The scene contains no water signal.** Flood-scene backscatter runs p1 = −16.4
dB and median −8.3 dB, against p5 = −23.6 dB at Richmond. Otsu, which assumes a
bimodal histogram, split the *land* distribution at −7.6 dB because the water
mode was too small to find. No threshold recovers the model's 10.3 km².

The satellite did not capture this flood: 8 December is two to three days past
the peak in a confined valley where water recedes quickly, over a city that is
19% built-up and where flooded structures scatter bright rather than dark.

**Rule 25: a radar overpass only validates a flood it actually caught.** With a
12-day repeat, most events are seen either side of their peak rather than at it,
and an empty scene is indistinguishable from a dry one unless the backscatter
distribution is inspected. Check for a water mode before scoring anything
against it; the absence of one is a statement about the overpass, not the model.

**Standing position: one validated event.** Richmond 2022 gives CSI 0.53 where
radar can adjudicate. That is a single event on a single date, and the intended
replication did not materialise, so it should be reported as an indication and
not a measurement of the reference's general skill.

**What is needed before the finding is firm:** more events and more dates
within this event. The land-cover mask on that list is now built (§5.7a and the
stratified table above), and the UK replication was attempted and failed for an
instrumental reason rather than a modelling one. What remains missing is
independent events, and Pakistan and Mozambique cannot supply them because
their rasters cannot be placed. One scene, one date, one event is a strong hint
and not a verdict.

### 5.8 Why better perception made worse decisions: the estimator inflates area (2026-09-27)

**Context.** §5.3 and `fig7` report the project's most uncomfortable result: a
learned estimator raises reconstruction quality on occluded cells from AP 0.005
to 0.346 at 60% occlusion — seventyfold, and 0.006 to 0.594 at 15% — and the
burn reduction it supports goes *down*, from 34.5% to 27.3%. The experiment already contained
the diagnostic (`perceive-hard`, the same estimate thresholded, recovers most of
the loss), but the reason was never measured. Building the appendix figure
forced the question, because the spatial panel shows it directly.

**What the panel shows.** `figA3` takes the *median* NDWS patch by hidden-cell
recall among patches where the cloud covers about half the fire front — not the
best case. The estimator recovers 65% of the hidden front, and spreads its
prediction over **3.3× the area that was actually burning**.

**That is not one patch.** Measured over every eval patch with more than five
hidden burning cells, at each occlusion level, as the ratio of predicted to true
hidden fire area:

| occlusion | patches | thresholded area ratio | **soft probability mass ratio** | planner loss vs blind |
|---|---|---|---|---|
| 0.15 | 360 | 1.92 | 2.38 | −1.5 pts |
| 0.35 | 584 | 1.67 | 3.05 | −4.1 pts |
| 0.60 | 763 | 1.20 | **3.74** | **−7.2 pts** |

(medians; "planner loss" is `perceive` against `blind`, averaged over three
seeds. `python scripts/build_fire_panel.py` reprints the two ratio columns and
rebuilds `figA3`'s cached patch.)

**The two columns move in opposite directions, and the decision follows the
second.** As occlusion rises the *thresholded* estimate gets tighter — 1.92 down
to 1.20 — which is why `perceive-hard` costs only 1.5 points at 60% occlusion.
The *probability mass* the planner actually consumes goes the other way, 2.38 up
to 3.74, and the planner's loss tracks it monotonically. The estimator is not
wrong about where the fire is; it is uncertain, it expresses that uncertainty as
diffuse probability, and the downstream surrogate — trained on near-binary
masks — reads diffuse probability as a large weak fire.

**So the failure is representational, not informational.** The information is
there: thresholding recovers it. What destroys the decision is handing a
calibrated-ish probability field to a component that was fitted on indicator
masks. This is the objective-mismatch story of §9.4 appearing inside the
perception interface rather than at the model-learning stage, and it is a
sharper claim than "better perception did not help": **the belief that best
matches the hidden state is the one that plans worst, and it is the softness,
not the content, that does the damage.**

**What this does not license.** Three occlusion levels is three points. The
monotone correspondence is consistent and the mechanism is checkable, but the
honest statement is a mechanism supported at three settings on one dataset with
one architecture, not a law. It does suggest a cheap fix worth testing —
calibrate the belief to the surrogate's training distribution, or train the
surrogate on soft inputs — and neither has been run.

### 5.9 Application figures for the appendix (2026-09-27)

`python scripts/make_appendix_figures.py` writes seven figures showing the
framework as something used rather than something measured. Cached inputs live
in `runs/sar_validation/` and `runs/ndws_partial_local/panel.npz` (rebuilt by
`scripts/build_fire_panel.py`); the portal frames are captured by
`portal/scripts/appendix-shots.mjs` and `appendix-shots2.mjs` against a running
build (`npm run build && npm run start`), into `runs/portal_shots/`.

| figure | what it shows | section |
|---|---|---|
| `figA1_sar_validation` | before / during / observed flood / model vs observation, permanent water excluded | §5.7, §5.7a |
| `figA2_sar_stratified` | CSI by land-cover class, coloured by whether radar can adjudicate | §5.7 |
| `figA3_wildfire_perception` | one NDWS patch: hidden front, persistence fill, and the estimator's 3.3× smear | §5.8 |
| `figA4_wildfire_decision` | reconstruction AP against burn reduction; the two panels disagree | §5.3, §5.8 |
| `figA5_portal_appraisal` | ranked mitigation options, two flagged as deepening flooding | §5.6b |
| `figA6_portal_anywhere` | Cedar Rapids solved on demand from public DEM tiles in 47 s | §5.6a |
| `figA7_validation_chain` | our solver vs the reference, and the reference vs the satellite | §5.6, §5.7 |

**A caveat inside `figA7`.** Per-frame numbers survive only for the *open-edge*
configuration that §5.6 superseded — the closed-domain rerun (job 1025861) left
summary statistics, not frames. The panel therefore plots the open-edge curve,
labels it as superseded, and draws the reported closed-domain mean (0.979) as a
reference line rather than quietly substituting one run's headline onto another
run's curve. The comparison bar in panel (b) uses the reported 0.979.

**What `figA5` is evidence of.** Every finding in §5.6b appears there as a
decision rather than a number: two of six candidate levee sites are labelled
"deepens flooding 3%" in the option list — the backwater effect — and the
option the tool recommends is reported alongside the fact that it keeps no road
open (+0.03 km against no action). The tool does not oversell the measure it
just ranked first. That is the Explain stage doing its job in the place it
matters.

**Defect 22 — the header named the wrong place.** The page chrome carried a
constant "Richmond Valley · NSW" from the layout, so analysing Cedar Rapids
produced a screen that said Richmond Valley over a map of Iowa. Caught by
reading the first version of `figA6`. Fixed with a small region context
(`portal/src/components/RegionContext.tsx`) that the planner sets and the header
reads. Minor as a bug and worth recording as a habit: **a figure of an interface
is a review of that interface**, and this one had been on screen for a day
without anyone noticing.

### 5.10 Exchangeability of the calibration scores: tested, and it holds (2026-09-27)

**The hole.** §9.3 states it plainly: split conformal guarantees its rate under
exchangeability of the calibration scores, our margin calibrates on residuals
collected *during training*, and a learning policy is a distribution shift.
Every coverage result in the project (`run_margin_coverage.py`) draws its scores
i.i.d. by construction, which tests the estimator and never the assumption. A
conformal-prediction reviewer goes straight here, and until now the honest
answer was "assumed, not tested".

**Method.** `eval/run_exchangeability.py`, `scripts/tacc/vista_exchangeability.slurm`
(job 1029518, three seeds, one per node, ~5 min each). The trainer now keeps the
iteration each calibration score came from, so the sequence survives the run.
dar, grid 64, 400 iterations, a probe every 10, 8 episodes per probe: **320
scores per seed**, split 160 calibration / 160 test, 480 pooled test points.
That sizing is deliberate — at 160 calibration points the quantile sits at rank
145, an order statistic rather than the sample maximum, and the test half
resolves δ = 0.1 to 1/160. The script refuses to report quietly below either
threshold (§7, defect 18).

**Three tests of the assumption, all non-rejections:**

| test | what would break the guarantee | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| Spearman ρ of \|score\| on iteration | the spread widening as the policy moves | p 0.51 | p 0.47 | p 0.74 |
| KS, first third vs last third | the distribution shifting over the run | p 0.98 | p 0.93 | p 0.40 |
| lag-1 autocorrelation vs permutation null | serial dependence beyond drift | p 0.48 | p 0.47 | p 0.60 |

The permutation null is the exchangeability null stated directly: under
exchangeability every ordering is equally likely, so the observed statistic
should sit inside the permutation distribution. It does, in all three seeds.

**And the number a reviewer actually wants.** Calibrating on the prefix and
scoring the tail is exactly how the margin is used inside a run, so the realised
exceedance rate there is the honest coverage:

| calibration sequence | realised rate | 95% CI | δ |
|---|---|---|---|
| **real order** | **0.0771** (37/480) | [0.0549, 0.1047] | 0.1 |
| shuffled (exchangeable by construction) | 0.0854 (41/480) | [0.0620, 0.1141] | 0.1 |
| adaptive CP, Gibbs & Candès, γ = 0.02 | 0.1000 (48/480) | [0.0747, 0.1304] | 0.1 |

**The ordering carries no information: real against shuffled is Fisher exact
p = 0.723.** That is the result. Destroying the temporal structure — which is
what non-exchangeability would exploit — changes the realised rate by four test
points out of 480. All three intervals contain δ.

**Reading it correctly.** The fixed quantile sitting *below* δ is conformal
behaving as advertised: the guarantee is on violation at most δ, so 0.077 is
mild conservatism, not undercoverage. Adaptive CP lands on 0.1000, i.e. it
recovers that conservatism and buys a little return, but it does not fix
anything, because nothing was broken. **We do not need adaptive conformal here,
and can now say why rather than hope.**

**What this does not establish.** Three non-rejections are not a proof of
exchangeability; they bound how large a violation could have hidden in 320
scores per seed, and no more. It is one testbed (dar), one policy
parameterisation, one probe schedule. The claim that belongs in the paper is
narrow and sufficient: *in the setting where the headline conformal result is
measured, the exchangeability assumption is not detectably violated, and the
realised rate matches the shuffled control.* If the probe schedule or the policy
optimiser changes, this needs re-running — which is now one `sbatch`.

**Rule 27: an assumption a method depends on is a claim, and a claim gets
measured.** The margin recipe has been the project's strongest contribution
since §3.2, and its load-bearing assumption went untested for as long as it was
convenient. The test cost five minutes of GPU time once the sequence was
persisted; the reason it went unrun was that nothing in the pipeline recorded
the order, and what is not recorded does not get questioned.

### 5.11 The portal runs the loop, and building it found three more defects (2026-09-27)

**Why this section exists.** The portal was demonstrating one stage of four. An
audit against the framework found: Simulate genuinely running (the service
imports `FloodSolver` and solves on Copernicus DEM on request), Plan replaced by
`helpful.sort()` over precomputed options, Perceive absent, Explain absent, and
the displayed uncertainty band **not** the conformal margin — a
normal-approximation interval whose docstring called it "the product-facing form
of the framework's safety margin". A showcase that omits the paper's strongest
contribution is not a showcase.

| stage | before | now |
|---|---|---|
| Perceive | absent | `POST /observe` — most recent Sentinel-1 pass, same-orbit differencing, permanent water held separately |
| Simulate | real | unchanged |
| Plan | `sort()` over 13 precomputed options | `POST /api/plan` — greedy allocation under a real budget, in milliseconds |
| Explain | absent | leave-one-out counterfactual attribution per measure |
| margin | Gaussian, 6 runs | split-conformal, 30 runs, via `pspe.plan.margins` |

**The loop, end to end, in one click.** Budget A$20M → search → *site 3 at 3.0 m
and site 5 at 1.0 m, A$18.2M* → **estimated 52%, at least 39% nine times in
ten** → attribution: site 3 adds 43 points here against 47 alone, site 5 adds 6
against 9, the 4-point gap in each being overlap → "adopting this should be
conditional on a full hydrodynamic run of this exact height vector." Plan in the
surrogate, bound the error against reality, verify what you commit to.

**Defect 27 — the tool displayed a confidence it had no data to support.** Split
conformal needs `n >= 1/delta - 1` calibration points. The scenario library had
**six** multi-measure runs; 90% needs nine. `conformal_quantile` would have
returned `None` and applied no margin, which is the correct behaviour — but the
portal was not calling it. It was showing `1.645 * RMSE * sqrt(1 + 1/n)`, which
assumes Gaussian residuals nothing established and, crucially, **always returns
a number**. Rebuilt at 30 calibration points, the honest margin is **±20.2
points against the ±10.5 displayed: the tool was overstating its confidence by
about 2×.** Two causes compound — a fit tuned on six points reported RMSE 5.9
where thirty give 10.21, and a normal would place the 90% point at 1.64×RMSE
≈ 16.7 where the conformal quantile is 20.2, so the residuals are heavier-tailed
than assumed.

**Rule 31: a displayed uncertainty states which guarantee it carries, and
refuses when the calibration set cannot support it.** A method that always
returns a number cannot tell you it has run out of evidence. This is §3.2's
result recurring inside the product: a margin fitted under a convenient
assumption is not a margin.

**Defect 28 — the surrogate was discontinuous, and only a planner could find
it.** The interaction model was `S*tanh(linear/S)` for two or more measures and
exact alphas for one. Those do not meet: crossing from one measure to two
applies the saturation to the whole sum, so **adding a levee that helps lowers
the prediction** — measured at −1.3 points for site 5 (alpha +28.4) placed
beside site 3. A catalogue of thirteen fixed options never meets that edge. A
planner asking "what if I add one more?" meets it on its first step, and had
been silently refusing to spend the second half of every budget.

Reformulated as `lead + S*tanh(rest/S)`: the largest single contribution passes
through untouched and only what is stacked on top saturates, so single measures
stay exact and the model is continuous as any added height goes to zero. It also
simply fits better — **RMSE 10.21 → 7.67, margin ±20.2 → ±13.4** — and the
planner now finds a **52%** two-site plan where the catalogue's best single
option was 47%.

**Rule 32: a model that a search will optimise over must be continuous across
the whole search space.** Fitting a correction that applies only above some
count creates a seam, and a fixed menu of options will never walk over it. The
defect was latent for as long as nothing searched.

**Defect 29 — an interface figure is an interface review.** `figA6` showed the
header reading "Richmond Valley · NSW" over a map of Iowa (§5.9). Recorded again
here because the pattern repeated: each of defects 21, 22 and 28 was found by
*rendering the thing and looking at it*, not by a test.

**What is honest about the Explain stage now.** §4.4 withdrew this project's
first explanation module: briefs scored well against the model and carried zero
state-specific information, which a permutation control exposed. The
replacement is not narration. A leave-one-out marginal — "remove this measure
from *this* plan and you lose 43 points, though it would give 47 alone" —
depends on what else is in the plan, so it cannot be reused for a different
plan, and it is checkable against a solver run. That is the property §4.4 says
to require.

### 5.11a The framework anywhere, not just where a library was precomputed (2026-09-27)

**The limitation, stated plainly.** Everything in §5.11 keyed off a scenario
library solved offline for one valley. Enter any other place and the tool fell
back to what it had always done: one solve, and a levee you position by hand.
Plan, margin and attribution were Richmond-only, which makes "works anywhere" a
claim about the *solver* and not about the framework.

**Why it was fixable.** The solver is batched over scenarios — one timestep
loop, one bed per plan — so the library does not have to be precomputed. It can
be **built on demand**, and twenty candidate plans cost roughly what one does.

`POST /plan` builds the domain from public DEM tiles, enumerates do-nothing plus
each candidate alone plus a calibration set of combinations at varied heights,
solves them in **one batched run**, reads the single-measure effects off exactly,
fits the interaction and takes a split-conformal margin, then allocates greedily
under a budget and returns leave-one-out attribution.

The fit and margin call `pspe/plan/surrogate.py` — the *same module* the district
path now uses, extracted for that purpose. Rule 13 is the reason: two numbers
produced by different code are not comparable, and a guarantee that means one
thing in Richmond and another in Iowa is worse than no guarantee.

**Measured on Cedar Rapids, where nothing is precomputed.** Five candidate sites,
90 m grid, severe-storm forcing:

| | |
|---|---|
| scenarios solved in one batch | **29**, in 137 s, 4,732 timesteps |
| per-site effect alone | −3.7, **+3.5**, +0.7, +0.1, −0.1 % |
| plan under A$20M | site 1 at 3.0 m, site 2 at 2.0 m (A$18.2M) |
| reduction | 4.0%, **guaranteed 3.8% at 90%** |
| margin | split-conformal over 23 held-out combinations, fit RMSE 0.07 |
| rejected | site 0 at **−3.7%**, site 4 at −0.1% |

**It found a harmful site on terrain it had never seen.** Site 0 deepens flooding
at the protected area by 3.7%, and the planner rejected it because increments are
chosen by measured effect rather than by assuming a levee helps. The backwater
result of §5.6b was not a property of the Richmond valley; it reproduces wherever
the geometry does.

**Two things this measurement does not say.** The 4% is small because the five
sites were placed by drawing a line on a map, not by any hydraulic reasoning —
the framework correctly reports that arbitrary levees do little, which is the
honest answer rather than a failure. And the number of candidate sites is a hard
constraint on what can be promised: with three sites the combinations give seven
calibration points where δ = 0.1 needs nine, so the service refuses to state a
90% margin and says why. **Four sites is where a 90% guarantee becomes
attainable**, and the interface says so before the user spends two minutes.

**Defect 30 — a fitted parameter reported from the edge of its own search.** The
saturation constant came back as 498 on a grid that stopped at 500, which does
not mean "S = 498"; it means the search ran out of room and no saturation was
needed, because effects this small do not overlap. Reported as a fitted value it
would look like a measurement. The grid now extends to 20,000 and the fit returns
a `saturating` flag, so "no interaction correction was required" is stated rather
than disguised as a number. **Rule 33: a fitted parameter that lands on the
boundary of its search is not a fit, and must be reported as the boundary it
is.**

**What is still not demonstrated.** The planner is greedy over six sites in
half-metre increments, not the paper's gradient planner; over that space greedy
with full re-evaluation is near-exhaustive and auditable, which is the right
trade for a tool someone spends public money on, but it is not the same
algorithm. The verification step is *stated* rather than wired — adopting a plan
should trigger a solver run and currently tells the user to do so. And Perceive
reports the observation beside the model rather than assimilating it: the twin
loop of §5.2a is not closed in the product.

### 5.12 What drives the flood, and the trench underneath every ad-hoc result (2026-09-28)

**The symptom.** A coastal city returned every mitigation option at "no
change", with one pair marginally *worse*. Correct, for what was being solved,
and the reason is structural: the ad-hoc solver applied rainfall uniformly over
the whole box with every edge open. **A levee redirects water that arrives from
somewhere; it has nothing to block when the rain lands on both sides of it.**
The pair that came out worse is the same mechanism — two walls trapping the rain
falling inside them.

That is why Richmond shows 47% and an arbitrary flat city shows nothing. Rain
on a river valley still concentrates into a channel and a levee blocks the
route; flat coastal ground has no route.

**Defect 31 — nodata was filled with sea level, and inland that is a trench.**
`build_domain` did `nan_to_num(nan=0.0)`, reasoning that Copernicus reads ocean
as nodata. True, and right for a coastal box. Reprojecting WGS84 tiles into UTM
leaves wedge-shaped gaps along the edges, so **Cedar Rapids, a city at 247 m,
was solved with a 247 m trench ringing the domain**: 3.6% of cells at exactly
0 m, covering 84–100% of every boundary, an artificial sink for the entire
catchment to drain into.

| | before | after |
|---|---|---|
| elevation range | 0 – 296 m | **208 – 296 m** |
| relief | 296 m | 88 m |
| cells below 10 m | 3.6% | 0.0% |

Nearest-neighbour fill instead: it extends real terrain outward, so inland
stays inland, and a true coastline — where the nearest valid cells are already
near zero — still fills to about sea level. **Every ad-hoc flood extent
produced before this was drawn over that trench**, including the 29 m depths
that prompted the question in the first place.

**Rule 34: nodata is unknown ground, not a value.** Substituting a plausible
number for "no data" is a modelling decision disguised as a cleanup, and one
that is right in the case you were thinking of is usually wrong in the case you
were not.

**Three drivers, inferred from the terrain.** `pspe/simulate/real/forcing.py`
reads the DEM for what can force it, with nothing for the user to draw:

| | Cedar Rapids | Richmond NSW |
|---|---|---|
| river | west edge, 240 m wide, drains east, 8.4 m fall | none |
| coastal | none | east edge, 27.7 km |
| min elevation | 208 m | −0.9 m |

Two corrections were needed to get that right, and both were caught by a number
looking wrong rather than by a test. Measuring "channel" as cells below the edge
**median** selected 78 cells — 4.7 km of floodplain, and spreading a river's
discharge across that is rainfall again; measured from the edge **minimum** it
finds the four cells that are the channel. And taking the highest crossing as
upstream picked a 238 m road cutting, where the real channel runs 217 m in to
208 m out — the two lowest crossings are the river, and upstream is the higher
of those.

**Defect 32 — an inflow poured in as a volume rather than held as a stage.**
Injecting 6,036 m³/s as a source term into the few cells of a channel makes
water arrive faster than it can spread: the first run reached **112 m deep**, a
number with no physical meaning. A boundary sets a *state*. Manning's normal
depth for the discharge over the measured width and bed fall does it with no
free parameters:

    h = (Q n / (W sqrt(S)))^(3/5)   ->   10.2 m stage, 11.7 m peak

**Rule 35: a boundary condition sets a state, not a volume.** Adding mass at a
boundary makes the cell the bottleneck and the depth an artefact of the cell
size. The coastal surge uses the same mechanism — hold a level at named cells,
raise-only so the boundary never drains the land — so the two drivers differ
only in where the level comes from.

**What it changes.** Cedar Rapids, four candidate sites, A$20M:

| forcing | best site alone | best plan | verdict |
|---|---|---|---|
| rainfall | 0.4% | — | "no measure worth building" |
| river, as a volume | 7.6% | 8.5% | physically void (112 m deep) |
| **river, as a stage** | **16.2%** | **21.3%, guaranteed 20.8%** | 17 held-out combinations |

**And the honesty requirement that came with it.** A driver the terrain cannot
support falls back to rainfall, and both `/analyse` and `/plan` now report what
was *actually* applied rather than what was asked — "6,036 m³/s on the west
edge, 2× the forecast peak of 2,414, held as a 10.2 m stage". The baseline takes
the driver too, so "doing nothing" and the plan cannot disagree about what they
are comparing.

**What this does not fix.** The discharge is Open-Meteo's forecast peak scaled
by the chosen storm, not a fitted design flood; the 2× at Cedar Rapids lands at
6,036 m³/s against the 2008 event's ~5,400, which is the right order and not a
return period. Roughness is a uniform literature value. And the surge height is
a user input, not a joint-probability analysis. These are the ordinary
limitations of a planning tool and are stated in the interface.

## Part 6 — The climate digital-twin framework: where we actually stand

A digital twin makes four claims. Assessed honestly:

| property | status | evidence |
|---|---|---|
| **1. A model synchronised to a real system from observation** | **Demonstrated** | §5.2 — 8 fires, state sync 8.4×, including days with no overpass |
| **2. What-if rollouts through that model** | **Demonstrated** | §4.5 — surrogate exceeds the published benchmark on the test split |
| **3. Decisions taken on that basis, under constraints** | **Demonstrated** | §5.1 — +12.0 pts over operational practice, t = 11.15 |
| **4. Reality feeding back to correct model and decision** | **Half.** State and model correction: yes (§5.2). Correction from *intervention outcomes*: **no** | — |

### 6.1 What is genuinely built

- **Observation → state**, under realistic occlusion, with a decision-relevant
  evaluation (§5.3) — including the negative result that imputation does not
  help on a sparse field.
- **State → forecast**, at published-benchmark skill on real fire data (§4.5).
- **Forecast → constrained plan**, beating operational practice with the
  mechanism measured (§5.1).
- **Plan → safety**, with a conformal bound on violation (§3.2).
- **Observation → model correction**, over real multi-day sequences (§5.2).
- **Plan → brief → certificate**, with honest coverage (§4.4).

### 6.2 The one gap that will not close

**No observational dataset contains counterfactual interventions.** NDWS and
FIRMS record what fires did; neither records what a fire *would have done* had a
firebreak been cut where our planner asked. So:

```
validated against observation:        𝒪 → û,     G_θ(û) → u',     θ-correction
model-based, not validated:           effect of a on u'
```

**Switching domain does not fix this.** Flood, heat, rainfall and air-quality
records have the same structure: outcomes under whatever interventions actually
occurred, never under ours. Closing it requires either (a) a controlled burn or
fuel-treatment programme with recorded treatment locations and matched
untreated controls, or (b) a physics-based fire simulator (FARSITE/WRF-SFIRE)
as a surrogate for reality — which substitutes one model for another.

**Update, 2026-09-25: the flood work takes route (b), deliberately.** §5.6 plans
interventions against a learned surrogate and evaluates them with the
LISFLOOD-FP local-inertial solver. This does not close the gap — it is still one
model standing in for the world — but it is the strongest version of route (b)
available to the project, because the shallow-water equations are conservation
laws rather than empirical spread rates, the discretisation is the operational
standard in government flood mapping, and the reference model is independent of
the surrogate being tested. The sentence above stays true as written: no
*observational* record, in any domain, contains our counterfactual.

### 6.3 Honest scope of the twin claim

> **The sync half is demonstrated on real data; the act half is not.**
>
> PSPE is a decision system whose *perception, dynamics and correction* are
> validated against real observations, and whose *intervention effects* rest on
> a stated action model. That is more than a forecaster and less than a
> validated controller, and the boundary should be drawn explicitly wherever
> the work is presented.

### 6.4 What would make it a deployable climate twin

| step | effort | what it buys |
|---|---|---|
| Ingest live FIRMS + gridded meteorology for an active incident | days | the loop running forward in real time rather than on archives |
| A second hazard with sequential observation (flood extent from Sentinel-1) | 2–3 weeks | shows the framework is not fire-specific; a **dense** field, which would test §5.3's rare-event caveat |
| Treatment records with matched controls (fuel breaks, prescribed burns) | months, institutional | the only path to validating intervention effects |
| Operator study on real briefs | weeks | the Explain module's missing human evaluation |

---

## Part 7 — Measurement defects found, and the rules that follow

This genre is established: [Henderson et al., 2018] on evaluation protocol in
deep RL, [Agarwal et al., 2021] on point estimates and statistical uncertainty
with only a handful of runs, and — in our own application domain and
concurrent with this work — *WildfireSpreadBench* (arXiv 2609.22191), which finds that in wildfire
spread prediction the metric outranks the architecture. What follows is a
domain-specific instance for constrained model-based planning, not a new kind
of argument.

| # | defect | effect | how it was caught |
|---|---|---|---|
| 1 | Constraint read from one final snapshot | Identical runs scored 0% or 100% violation on timing luck | multi-seed reruns disagreeing |
| 2 | Evaluation perturbed training via global RNG | baseline return shifted −2.51 → −2.30 | forking the RNG changed the baseline |
| 3 | Optional-dependency tests skipped silently | a "tested" loader failed on first real use | `importorskip` audit |
| 4 | Truth solver unstable above 64² | NaN at every finer resolution | resolution sweep |
| 5 | Faithfulness REINFORCE had exactly zero gradient | samples never parsed; all scores equalled the fallback | parse-rate diagnostic |
| 6 | Cost limits uncalibrated | dual never activated; five unconstrained learners tying | measuring do-nothing and greedy cost |
| 7 | `F = exp(−ΣKL)` with summed KL | every arm scored exactly 0.0 at 64 actions | running Explain on a large action space |
| 8 | Faithfulness never measured state-specificity | a constant brief scores as well as a correct one | **permutation control** |
| 9 | dar's policy is state-independent | the Explain testbed had nothing to explain | action diversity across states |
| 10 | Conformal margin bounded model error, not policy spread | violations rose to 18.2%, worse than no margin | running it where the surrogate is accurate but the policy is variable |
| 11 | Baselines compared against their own initialisation | four methods "structurally failed" | solving for the action the parameterisation implies at init |
| 12 | Testbed calibrated for a binding constraint, not a controllable objective | swe ranked methods inside a 7% band for the whole project | **return** separation — *but see §3.4a: rdf's band is 4.4% and it ranks planners fine, so the band width was never the discriminator* |

| 13 | Calibration scores collected per fire while violation was declared on a batch mean | Every margin inflated by √32; margins consumed 100% of the limit, `d_eff` → 0, all arms stuck at 40–80% | Printing the margin next to the limit |
| 14 | A constraint whose limit left no room for a margin | Infeasible read as "the method fails"; cost four runs | Measuring span, bias and headroom before fitting anything (`--calibrate-only`) |
| 15 | A fidelity sweep whose range could not reach the regime it was meant to test | Would have confirmed the failure five times and never shown the default working | Computing ρ at the sweep's endpoint before submitting |
| 16 | Two "different" fidelity levels clamped to the same 8 trajectories by a `max(8, ·)` floor | 3 of 5 levels identical **at grid 32**, where it was diagnosed. At grid 64, where the sweep runs, 0.02 and 0.05 map to 8 and 12 and would not have collapsed: the defect was real, the reported cost was measured at the wrong scale | The log printing the same rel L2 twice; the overclaim by re-checking at the grid actually used |
| 17 | A feasibility verdict estimated from one batch of 32 fires | Same configuration read 8.6× apart on two machines; the verdict could flip on noise | Running the same probe in two places |
| 18 | Violation rate measured over 11 evaluations while testing a 10% target | Every rate is a multiple of 1/11; δ falls between 1/11 and 2/11, so the headline metric could not resolve the claim either way | The observed rates being 0.091, 0.182, 0.273 — all `k/11` |

### Standing rules

**Added 2026-09-25 from the flood testbed (§5.6):**

| # | rule | what forced it |
|---|---|---|
| 16 | when the solver supplies the quantity a constraint is declared on, test conservation of that quantity before using it as reality | an unlimited explicit scheme gained 6.4% water volume |
| 17 | verify geometry separation numerically before running any physics | inflow straddled the berm; the settlement carved its own gap |
| 18 | when a testbed cannot separate planners, suspect the actuator's mechanism before concluding the domain is uncontrollable | same equations, 7% → 99.8% span on changing the actuator |


1. A baseline starts from the same initial condition as the method, or the
   comparison reports an initialisation.
2. Constrained comparisons are scored at matched constraint, always.
3. Any result depending on a hand-specified component is re-run with that
   component removed.
4. A margin or bound names the distribution it bounds, and that distribution is
   the one that produces the failure.
5. Any metric comparing a generated artefact to a target ships with a
   permutation control.
6. A testbed is calibrated for the variation a metric needs before the metric
   runs on it.
7. A testbed is admitted for method comparison only if the span between doing
   nothing and the best achievable return is wide relative to the differences
   being measured — and that span is reported.
8. A belief, estimate or reconstruction is evaluated on the **decision**, not on
   its own reconstruction metric.
9. A calibration sample is exchangeable with the quantity that will be compared
   against the limit: **same estimator, same unit, same aggregation**.
10. Before fitting a correction, measure whether the system has room to apply
    it. Infeasible and miscalibrated are indistinguishable in an outcome metric.
11. A sweep reports the value of its own axis at both endpoints, so a range that
    cannot reach the regime under test is visible before the jobs run.
12. A threshold verdict carries its own uncertainty, and is flagged when the
    estimate sits within that uncertainty of the threshold.
13. Two numbers are compared only after checking they were produced by the same
    code. A version skew between runs is indistinguishable, in the output, from
    a finding about the systems being compared.
14. A metric's resolution is checked against the effect size before the run:
    a rate targeting δ needs enough evaluations that δ is not between two
    adjacent attainable values.
15. A defect found in a cheap configuration is re-checked at the configuration
    that will actually run before its cost is stated. Dataset size, grid and
    batch all change whether a boundary case is reached, and a diagnosis that
    is right about the mechanism can still be wrong about the consequence.

### Defects 19–32, from the flood and wildfire datasets and the portal

The first eighteen are failures of *measurement protocol*. These eight are
mostly failures of *belief about data* — what a number in a file means — and
they are harder, because the code is correct and the output looks right.

| # | defect | effect | how it was caught | § |
|---|---|---|---|---|
| 19 | Unlimited explicit scheme manufactured the constrained quantity | solver gained 6.4% volume from `clamp(min=0)` creating water | closed-domain mass conservation test | 5.6 |
| 20 | Levee size specified in cells, not metres | the same site looked 6× less effective at a different grid | re-deriving the physical width at two resolutions | 5.6b |
| 21 | Permanent water counted as model over-prediction | 119 km² of "error" was the sea and the river channel; headline CSI 0.378 vs 0.535 | **drawing the map and looking at it** | 5.7a |
| 22 | Page header named a constant region | the tool said "Richmond Valley" over a map of Iowa | reading the first version of a figure of the interface | 5.9 |
| 23 | "No detection" read as "not observed" | would have deleted 37.4% of days — exactly those where fires go out | comparing finite fractions across bands | 5.2b |
| 24 | Acquisition time read as radiative power | `frp` held clipped HHMM clock values | the values were 742–2142 with minutes always a multiple of 6 | 5.2b |
| 25 | Downsample cropped the raster margin | lost up to **59%** of a fire's detections, smoothly and only for some fires | conservation check: detections before vs after | 5.2b |
| 26 | Year fold keyed on the first date | a one-fire "2017" fold from a fire starting 30 December | the fold listing printing a fold of size 1 | 5.2b |
| 27 | A displayed band that could not carry its stated level | the portal claimed 90% from 6 calibration runs, understating the honest margin ~2x | calling the conformal function, which refuses below n = 1/delta - 1 | 5.11 |
| 28 | Surrogate discontinuous between one measure and two | adding a levee that helps LOWERED the prediction; the planner refused to spend half of every budget | building a planner that asks "what if I add one more?" | 5.11 |
| 29 | Plan, margin and attribution keyed to a precomputed library | the framework worked in one valley; everywhere else was a bare solve | entering another location and finding only a levee slider | 5.11a |
| 30 | Fitted parameter reported from the edge of its search grid | S = 498 on a grid ending at 500 read as a measurement, not as "no saturation needed" | the value sitting exactly at the boundary | 5.11a |
| 31 | DEM nodata filled with sea level | a 247 m trench ringing every inland domain; 3.6% of cells at 0 m across 84-100% of each boundary, draining the catchment into it | terrain analysis reporting a coast 1,500 km inland | 5.12 |
| 32 | River inflow poured in as a volume, not held as a stage | 112 m of water: the cells could not spread it as fast as it arrived | a depth with no physical meaning | 5.12 |
| 33 | Best-of-N action search drew fresh initial conditions per rollout | 29.4% span on swe where the published sweep found none — selection on lucky initial states | a testbed admitted on a manufactured span | 3.4a |
| 34 | Distillation cloned raw actions with an MSE, on a sparse target | the policy learned the MSE-optimal constant, −1 everywhere: treated 6e−05%/day for a 0.0% reduction | a null that would have *confirmed* our own amortisation claim | 5.1a |
| 35 | SMBPO's truncated prefix cost rescaled linearly, on a cost that accrues non-linearly | −4.7% bias, 4× the pessimism term: the arm was net optimistic and violated more than baseline | a baseline failure that matched our predicted direction | 9.2a |

**Defect 25 is the one to remember.** It degrades data continuously rather than
breaking it, affects only some fires, and leaves every downstream number
plausible. Nothing short of a conservation check finds it.

### Rules 16–35

16. A scheme that cannot violate the constrained quantity cannot be used to
    study violating it; verify conservation before trusting a solver.
17–21. *(flood testbed; see §5.6–5.6b)*
22. An intervention's physical dimensions are specified in physical units.
23. Agreement with a reference is not validation unless the reference has itself
    been scored against observation.
24. Before scoring a model against an instrument, establish where the instrument
    can see.
25. A satellite overpass only validates an event it actually caught; check for
    the signal before scoring against its absence.
26. Before scoring extent against an instrument, remove the thing that was
    already there. A change detector and a state model answer different
    questions over permanent water.
27. An assumption a method depends on is a claim, and a claim gets measured.
28. Past a few hundred paired samples, report an effect size and a confidence
    interval next to every t — the t alone stops discriminating between
    "certain" and "important".
29. When a channel encodes absence as NaN, it cannot also tell you whether
    anyone looked. Find a channel that is present when the instrument worked.
30. When adopting an external dataset, verify each channel's semantics against
    raw values before running anything, and check that any resampling conserves
    the quantity it claims to conserve.
31. A displayed uncertainty states which guarantee it carries, and refuses when
    the calibration set cannot support it. A method that always returns a number
    cannot tell you it has run out of evidence.
32. A model that a search will optimise over must be continuous across the whole
    search space. A fixed menu of options never walks the seams a planner does,
    so the defect stays latent until something searches.
33. A fitted parameter that lands on the boundary of its own search grid is not
    a fit. Report it as the boundary it is, or widen the grid until it is not.
34. Nodata is unknown ground, not a value. Substituting a plausible number for
    "no data" is a modelling decision disguised as a cleanup, and one that is
    right in the case you were thinking of is usually wrong in the case you
    were not.
35. A boundary condition sets a state, not a volume. Adding mass at a boundary
    makes the cell the bottleneck and the resulting depth an artefact of the
    grid rather than a property of the flood.

---

## Part 8 — Status, and what to do next

> **Read [`RELATED_WORK.md`](RELATED_WORK.md) alongside this part.** It audits
> every contribution below against the current literature and downgrades
> several of them: the silent-failure diagnosis is already named in the
> model-based safe-RL literature, the joint-training negative restates
> objective mismatch (Lambert et al. 2020), state sync against open loop is the
> textbook data-assimilation comparison, and the amortisation claim is
> contradicted by the planner-amortisation literature. Four of twelve areas
> survive as contributions. The paper it recommends is narrower than §0.3's.


### 8.1 Contributions as they stand, after the literature audit

Strength is graded **against the field** (Part 9), not against our own baselines.

| # | contribution | strength | nearest prior work |
|---|---|---|---|
| 1 | **The conformal margin**, with the threshold derived (`ε ≥ σ`) and confirmed at ρ = 1.00 in a controlled study | **Strongest, and stronger than it was.** A falsifiable critique of the field's default, the fix, and the rule that says when each applies (§3.5) | §9.3 — Lindemann 2023; arXiv 2503.17678, 2603.02196, 2606.15366 |
| 1b | **Two preconditions for a margin**: same estimator/unit/aggregation, and `b < f·s` | **Novel and checkable.** Both were violated by our own experiments before being stated; each costs two rollouts to verify (§3.5, §5.5) | none found |
| 2 | **The horizon mechanism** on observed fire data | **Strong.** Null where theory requires, +14.3 at 5 days. No firebreak paper reports it | §9.6 — arXiv 2404.08523; Appl. Soft Comput. 2025 |
| 3 | **Better reconstruction, worse decision** | **Novel direction, modest evidence.** Contradicts the assimilation stack's working assumption | §9.10 — Sci. Adv. 2026; arXiv 2609.22191 |
| 4 | **The measurement catalogue** | **Careful replication in a new domain.** Worth publishing, not a headline | §9.11 — Henderson 2018; Agarwal 2021; arXiv 2609.22191 |
| 5 | Safety by measurement (probe + margin decomposition) | **Narrow.** The diagnosis is already named; neither-half-alone is ours | §9.2 — Ma 2022; Jayant & Bhatnagar 2022; Huang 2024 |
| 6 | Planning on observed data, +12.0 pts | **Real, but a trade-off not a win.** Prior work can evaluate the counterfactual; we cannot | §9.6 |
| 6b | Silent violation on real fire dynamics, 40–75% | **Confirmed, larger than on the PDE testbeds.** Whether the margin fixes it there is pending (§5.5) | §9.2 |
| 7 | Sample efficiency, 9× / 5× | **Exposed.** All our baselines are model-free, so the claim is near-tautological | §9.2 — SMBPO, SafeDreamer, CAP |
| 8 | Joint training negative | **Known phenomenon, inverted data point** | §9.4 — Lambert 2020; arXiv 2310.06253 |
| 9 | The twin loop on real sequences | **Demote to a control.** Gain-1 nudging against free-running is what DA is | §9.8 — Evensen 2003; Sci. Adv. 2026 |
| 10 | Certified explanations | **Halved.** Certificate holds; faithfulness withdrawn by a standard control | §9.12 — Adebayo 2018 |

### 8.2 Weaknesses to state in the paper, not hide

- Intervention effects are model-based (§6.2). Permanent, and prior work on the
  same task does not share this limitation (§9.6).
- Every safe-RL baseline is model-free, so the sample-efficiency claim is not
  yet earned (§9.2).
- The twin loop runs on 8 fires when 607 were publicly available (§9.9).
- Forecast skill is below current state of the art — 0.3162 against 0.3790
  (§4.5, §9.5).
- Exchangeability of episode-cost deviations is assumed, not tested (§9.3).
- We report mean ± std over 3–5 seeds, which [Agarwal et al., 2021] argue
  against; stratified bootstrap intervals would cost nothing (§9.11).
- The amortisation claim is contradicted by the planner-amortisation literature
  (§9.7).
- rdf is parity, not a win. swe is retired with a measured reason.
- The partial-observation finding may be specific to sparse fields (§9.10).

### 8.3 Venue read

- **ICML / NeurIPS main track:** viable, but as the **narrow** paper of §9.13 —
  one claim (what a conformal margin should bound), one negative result against
  a named alternative, two validations. The four-contribution framing of §0.3
  does not survive Part 9 and would be read as replication.
- **Climate-ML venue (CCAI workshop, Environmental Data Science, Tackling
  Climate Change with ML):** strong now, and the natural home for the wildfire
  planning result with its trade-off stated honestly against §9.6.
- **A safe-RL venue (L4DC, CDC):** the conformal margin result would land well,
  and that audience will demand the model-based baselines of §9.2.
- The twin framing is **motivation only**. After §9.8, claiming an unqualified
  digital twin invites a reviewer to check four properties; property 4 fails
  outright and properties 1–2 sit well below the DA state of the art.

### 8.4 Remaining experiments, by value

Re-ranked after Part 9. Items 1–3 remove the claims most exposed to a
knowledgeable reviewer and are cheap.

| # | experiment | cost | what it buys |
|---|---|---|---|
| 1 | ~~Port the twin loop to **WildfireSpreadTS**~~ | **done** | 607 fires, year-wise CV, job 1029659. Corrected two §5.2 claims downward: adaptation +6.3% not +21% at day 2, and 62-76% of fires not 8/8 (§5.2a) |
| 2 | ~~Add a **model-based** safe-RL baseline~~ | **done; second one written 2026-09-28** | CAP transplanted onto our planner at a matched probe budget: 0.1199 violating against our 0.0780, ours vs CAP z = +2.08, CAP vs no margin z = −0.43 (§3.2). `eval/run_smbpo_baseline.py` transplants SMBPO the same way. **Result: 12.20% against CAP's 11.99% (p = 0.83, indistinguishable) and ours at 5.00% (+7.20 pts, p = 0.00135).** The pre-registered prediction that it would lose by *more* than CAP failed, and the failure generalises the claim: how you summarise disagreement does not matter, computing on the wrong distribution does (§9.2a). SafeDreamer remains unrun |
| 3 | ~~Add a **DA baseline** (EnKF or learned gain) to the twin loop~~ | **done (2026-09-28)** | 32-member EnKF, σ_o swept over {0.05, 0.15, 0.30}, 606 fires, job 1031659. **No setting beats gain-1 nudging**; skill tracks the implied Kalman gain (0.979/0.837/0.562 → 0.329/0.127/0.032), so gain 1 is approximately the filter's own optimum, not a naive stand-in (§5.2c) |
| 4 | ~~Head-to-head on an inaccurate surrogate~~ | **done** | the diagnostic is derived and confirmed at ρ = 1.00 (§3.5); the PDE ladder is queued |
| 5 | **Flood extent from Sentinel-1** — a dense field, sequential, free on GCS | 2–3 weeks | decides whether §5.3 is a sparse-field quirk or a property of budgeted intervention (§9.10). *Partly done*: `pspe/observe/sar.py` and one validated event (§5.7); the dense-field intervention study is not started |
| 6 | Run the planner against **Cell2Fire** | ~1 week | one setting where the intervention counterfactual is evaluable (§9.6) |
| 7 | ~~Test **distillation** from planner solutions into the amortised policy~~ | **done (2026-09-28)** | 3 seeds, job 1031950. Distillation recovers **46.9%** of the planner: 18.32% against the planner's 39.04% and the from-scratch arm's 10.4%. So §9.7's warning was half right — the recipe helps by +7.9 points — but 20.7 points survive it, and the claim gets *stronger* for having been tested (§5.1a) |
| 8 | ~~Test exchangeability of episode-cost deviations; consider adaptive CP~~ | **done** | Not violated: real-order coverage 0.0771 against a shuffled control's 0.0854, Fisher p = 0.723; drift, KS and serial-dependence tests all non-rejections (§5.10). Adaptive CP unnecessary |
| 9 | Re-report all results with stratified bootstrap CIs and IQM | **partly done (2026-09-28)** | meets the evaluation standard we ourselves cite (§9.11). `eval/metrics.py` provides `iqm`, `bootstrap_ci`, `seed_report`; `eval/run_bootstrap_report.py` applies them. Planning gap: IQM 11.50, 95% CI [9.97, 14.63]. Safety margin on rdf, 5 seeds: Δ −6.1 pts, CI [−21.2, +6.1], straddles zero (§3.2a) — **re-run at 20 seeds: Δ −9.1 pts, CI [−11.8, −5.5], p = 0.0002, and no return cost (§3.2b)**. Horizon mechanism survives at 5 seeds (§5.1). **Still to do:** every remaining 5-seed comparison in the document needs the same treatment |
| 10 | **Live-incident loop** on current FIRMS plus meteorology | ~1 week | the one role FIRMS keeps that WSTS cannot fill (§9.9) |
| 11 | **Explain redesign** — structured head over (patch, amplitude) | weeks | probably a separate paper (§9.12) |
| 12 | **swe revived by matching target to actuator authority** | **objective solved (2026-09-28)** | third PDE testbed recovered (§3.4a). At **horizon 96, target 0.12**: span **47.9%** and 15/40 improving directions against rdf's 7.1% and 23/60 — better than the reference family on both criteria. The missing quantity was horizon, not amplitude or objective. Caps placed: `u_max` 0.14, `budget` 0.60, `cost_limit` ≈ 0.083 by the 35% convention every shipped limit follows. Remaining: re-derive the limit against a *trained* reward-greedy anchor rather than best-of-40-random, which makes 0.083 a lower bound |
| 13 | **Operator study** on real briefs | weeks | the Explain module's missing human evaluation |

---

### 8.5 The ICML revision plan (2026-09-28, submission in ~3 months)

An external read of the draft as an ICML main-track submission scored it
**6.5–7/10, weak accept / borderline**, with novelty 7.5, experiments 7.5,
theory 6.5, reproducibility 6.5. The assessment is well calibrated and the
criticisms are worth recording verbatim, because two of them are already
addressed and one is sharper than the reviewer made it.

**What the review got right, and what has moved since.**

| criticism | status |
|---|---|
| "exchangeability check is only on DAR — not RDF, despite RDF carrying much of the headline margin evidence" | **Confirmed.** `runs/exchangeability/` is dar only. Tier 0 below. |
| "CAP is the only model-based safe-RL baseline … does not compare against SMBPO or SafeDreamer" | **Half closed.** SMBPO run 2026-09-28 (§9.2a): 12.20% against CAP's 11.99%, p = 0.83. SafeDreamer outstanding. |
| quoted numbers 0.078 / 0.120 / 0.132 | **Superseded.** At 20 seeds: 5.00 / 11.99 / 14.09, run-level p = 0.0003, and **no objective-value cost** (§3.2b). |
| "the real-world experiment does not validate the proposed safety method" | **Correct, and the sharpest point in the review.** See Tier 3. |
| abstract says "the default is structurally wrong" while the body is more precise | **Correct.** Fix the abstract. |

**The one to resist.** *"Proposition 1 is essentially standard conformal coverage
after redefining the residual."* That is true, and the response is **not** to make
Proposition 1 harder. It is to (a) show empirically that the residual choice is
the load-bearing decision, which §9.2a now does — two opposite summaries of model
disagreement land 0.2 points apart and neither beats doing nothing — and (b) add
theory where the paper has an *unexplained* empirical phenomenon, which is the
ρ transition, not the coverage statement.

---

#### Tier 0 — cheap, and a reviewer can check both in five minutes

Submitted as Vista job **1033187**, `scripts/tacc/vista_tier0.slurm`.

| # | experiment | why |
|---|---|---|
| T0.1 | **Exchangeability battery on rdf**, 3 seeds, same sizing as the dar run | the assumption is currently untested on the family that carries every margin number in §3.2 |
| T0.2 | **20 seeds** for `constraint_fix_sat_rdf` (§3.3) and `alpha_rule` (§4.2) | §3.2c found both are comparative claims whose CIs span zero at five seeds; §3.2b showed what happens when that is fixed properly |

#### Tier 1 — the theory that answers "Proposition 1 is trivial"

| # | experiment | why |
|---|---|---|
| **T1.1** | **A separation result.** Prove that any margin measurable w.r.t. the model-error distribution alone has realised coverage independent of σ_ep, and so cannot hold δ once ρ = σ_ep/ε > 1 | **Highest-leverage item in the plan.** It converts §9.2a's empirical finding into a theorem: CAP and SMBPO land identically *because they are functions of the same insufficient statistic*. It also answers the "standard conformal" objection with new theory rather than with decoration. |
| T1.2 | **Closed-form coverage as a function of ρ** for the model-error margin, overlaid on the measured transition | turns the reviewer's favourite figure from an observed transition into a predicted curve |

Sequencing note: draft T1.1 **first**. If the separation result does not work, that
must surface in week 3, not week 10.

#### Tier 2 — the deepest objection, nonstationarity

| # | experiment | why |
|---|---|---|
| T2.1 | **Beyond-exchangeability bound** [Barber et al., 2023] instantiated for policy drift, with the drift term **measured** on both families | converts "we assume exchangeability" into "here is the coverage penalty and here is its size" — a qualitatively different answer to a theory reviewer |
| T2.2 | **Adaptive conformal as a first-class arm** on both families, not a diagnostic | `run_exchangeability.py` already computes adaptive coverage; promote it to a comparison arm |
| T2.3 | Stratify calibration by **stage of learning** (early / mid / late windows) | shows where split conformal holds and where it degrades, rather than asserting either |

#### Tier 3 — closes the structural hole

| # | experiment | why |
|---|---|---|
| **T3.1** | **Flood as the margin's real-data validation**: the δ-sweep with a hydrodynamic solver as reality, plus an explicit Proposition 2 feasibility check | The wildfire CMDP is **infeasible** under Prop 2, so the strongest real-data experiment currently validates the *planner*, not the *margin*. Scoping found the apparatus already built (`eval/run_flood_margin.py`, solver as reality, Prop 2 working as an acceptance test) and the gap narrower than budgeted: `--delta` was a single value, so **the δ-sweep — the paper's most persuasive figure — had never been run on a real hazard.** Now a `--deltas` flag sharing one surrogate and calibration set. Job **1033263**. |

**T3.1's pre-registered prediction (2026-09-28, before the result).** The run at
`q-log-sigma 0.04` reports **σ = 0.03231 m, ε = 0.02270 m, ρ = 1.424** — measured
from the solver, not chosen. §3.6's Corollary 3.2 then *predicts the whole sweep*
for the `model_error` arm, with no free parameters:

| requested δ | predicted realised `1 − Φ(z_δ/ρ)` | over-run |
|---|---|---|
| 0.02 | **0.075** | 3.7× |
| 0.05 | **0.124** | 2.5× |
| 0.10 | **0.184** | 1.8× |
| 0.15 | **0.233** | 1.6× |
| 0.20 | **0.277** | 1.4× |
| 0.30 | **0.356** | 1.2× |

with the `residual` arm tracking δ at every level. Two things ride on this. It
would be the closed form validated **on a hydrodynamic solver rather than a
synthetic testbed**, and it would show the characteristic signature — the
default's error growing as δ *tightens*, from 1.2× at δ = 0.30 to 3.7× at
δ = 0.02 — which is why requesting a stricter guarantee from the wrong
distribution makes matters worse, not better. Recorded before the run because
§9.2a's prediction was written after the mechanism and got it wrong.

#### Tier 4 — optics and presentation

| # | item | why |
|---|---|---|
| T4.1 | **SafeDreamer** transplant, with a pre-registered prediction that it lands near 12% | low scientific information after §9.2a, high review value; converts baseline-adding into hypothesis-testing |
| T4.2 | **Narrow the narrative**: ICML = the structural result; wildfire becomes external validation | the draft reads as two papers joined. `docs/WORKSHOP_PAPER.md` and `docs/POSTER.md` now give the application material its own home |
| T4.3 | **Fix the abstract's precision** — "appropriate for pointwise controllers, inappropriate under an expectation constraint" rather than "structurally wrong" | more accurate *and* more interesting; removes an easy overreach flag |

#### Schedule

| month | work |
|---|---|
| 1 | Tier 0; draft T1.1 early and kill it fast if it fails; T1.2 |
| 2 | T2.1–T2.3; T3.1 (the long pole) |
| 3 | T4.1; rewrite; **freeze experiments at week 10** |

**The two items that move this from weak accept to accept are T1.1 and T3.1.**
T1.1 answers the theory reviewer's actual objection; T3.1 removes "your real-data
experiment does not test your method." Everything else is insurance.

## Part 9 — Related work, and an honest positioning

Literature checked September 2026. The long form, with the argument for each
verdict, is [`RELATED_WORK.md`](RELATED_WORK.md); this part is the summary that
belongs with the results. Full entries are in the References below.

### 9.1 Verdicts

| § | area | verdict | one line |
|---|---|---|---|
| 9.3 | Conformal margin for constraint violation | **Novel, strongest result** | The field's default conformalises model error; we show it fails and give the version that holds |
| 9.6 | Horizon mechanism on observed fire data | **Novel as evidence** | Null at 1 day where theory requires, +14.3 at 5 days; no firebreak paper reports this ablation |
| 9.10 | Better reconstruction, worse decision | **Novel direction** | Contradicts the working assumption of the assimilation and twin literature |
| 9.11 | Measurement-defect catalogue | **Converging with an established line** | Henderson, Agarwal, and now WildfireSpreadBench argue the same thing |
| 9.2 | Constrained planning fails silently | **Known** | Stated outright in model-based safe RL |
| 9.4 | Joint training does not help | **Known, inverted data point** | Objective mismatch, 2020; our twist is the model improved 5–9× and the decision did not move |
| 9.8 | State sync beats open loop | **Baseline run 2026-09-28; reading changed** | A 32-member EnKF loses to gain-1 nudging at every swept σ_o, and skill tracks the implied gain (0.979/0.837/0.562). Gain 1 is approximately the filter's optimum here, not the crudest form of it — the observations are far more reliable than the forecast, which drives K → 1 (§5.2c) |
| 9.7 | Decision-time beats amortised policy | **Tested 2026-09-28; claim survives, narrowed** | Distillation recovers 46.9% of the planner (18.32% vs 39.04% vs 10.4% from scratch): the recipe helps by +7.9 pts, and 20.7 pts survive it (§5.1a) |
| 9.12 | Permutation control for faithfulness | **Standard practice we omitted** | Credit for applying it, none for inventing it |
| 9.5 | Wildfire next-day forecast skill | **Behind** | 0.3162 against 0.3673 single / 0.3790 ensemble |
| 9.9 | Sequential wildfire data | **Closed** | Ported to WildfireSpreadTS, 607 fires, year-wise CV; the eight-fire estimate was ~3x too generous (§5.2a) |
| 9.5 | PDE surrogate architecture | **Behind, and not our contribution** | FNO is a 2021 baseline; Poseidon, DPOT and BCAT are the frontier |

Four of twelve survive as contributions. The broad thesis of §0.2 is real but is
being reached independently by several groups, so §0.3's four-contribution
framing is weaker than it looked. §9.13 states the paper that survives.

### 9.2 Constrained and safe reinforcement learning

Our baselines — CPO [Achiam et al., 2017], PPO-Lagrangian and PID-Lagrangian
[Stooke et al., 2020], Sauté RL [Sootla et al., 2022], primal-dual NPG [Ding et
al., 2020] — are the standard set, benchmarked in the Safety Gym lineage [Ray et
al., 2019; Ji et al., 2023]. The closer neighbours are **model-based** safe RL:
SMBPO [Thomas et al., 2021], CAP [Ma et al., 2022], constrained PPO inside a
model [Jayant & Bhatnagar, 2022], SafeDreamer [Huang et al., 2024], and
conservative safety critics [Bharadhwaj et al., 2021].

That literature already names our §3.1 failure. It is not a discovery. Two
consequences for the write-up:

1. Present §3.1 as *characterisation*, and the probe-plus-margin decomposition
   — with the measurement that neither half works alone — as the contribution.
2. **The sample-efficiency claim is exposed.** Every baseline we run is
   model-free, so "9× fewer real transitions" is close to tautological for a
   model-based method. Against SMBPO or SafeDreamer it may not survive. This is
   the largest gap in the baseline set and the highest-value missing experiment
   after §9.9.

### 9.3 Conformal prediction for safety

Foundations: [Vovk et al., 2005], split conformal [Lei et al., 2018], the
practitioner's account [Angelopoulos & Bates, 2023], and adaptive conformal
inference under distribution shift [Gibbs & Candès, 2021].

Applied to control and RL this is an active 2023–2026 line: conformal prediction
regions inside an MPC [Lindemann et al., 2023], adaptive CP with control barrier
functions (arXiv 2503.17678), conformal policy control (arXiv 2603.02196),
robust conformal CBF/CLF under iterative policy updates (arXiv 2606.15366), and
proactive context-forecasted constraints giving a cumulative violation bound
linear in the horizon (arXiv 2609.08080).

**The pattern: the dominant instantiation conformalises the error between the
learned dynamics model and the true system.** One strand does not:
[Prinster et al., 2026] calibrate permissible *behavioural change* against a
safe reference policy and identify no dynamics model at all. That is the
closest prior work to ours and shares the instinct that the calibration target
should be the controlled object; we differ in conformalising realised cost
against the quantity a Lagrangian dual holds, which yields a CMDP budget rather
than a behavioural-change budget. Any write-up must not lump it in with the
model-error line — an author-reviewer would catch that immediately. That is exactly the recipe §3.2 tried
first, and on the reaction-diffusion front it raised violations to 18.2% against
a 14.5% no-margin baseline. The reason is structural — where the surrogate is
accurate and the policy is variable, a quantile over model error never sees what
breaches the limit. Conformalising realised episode costs holds the rate on both
families.

This is the clearest contribution in the project: a falsifiable, empirically
supported critique of the field's default instantiation, plus the fix. Two
caveats belong in any write-up:

- Their setting is CBF-constrained continuous control with GP dynamics; ours is
  a CMDP with an episode-cost budget. The failure mode should generalise; we
  have not shown it in *their* setting.
- ~~**Exchangeability of episode-cost deviations across a learning run is
  assumed, not tested.**~~ **Tested (§5.10).** Drift, early-vs-late KS and a
  permutation test on serial dependence are all non-rejections across three
  seeds, and — the number that matters — the realised rate on a held-out tail
  is 0.0771 against 0.0854 for a shuffled control, Fisher exact p = 0.723. The
  ordering carries no information, so the guarantee is not being propped up by
  an assumption that fails. Adaptive CP [Gibbs & Candès, 2021] recovers the
  mild conservatism (0.1000 against δ = 0.1) but is not needed. This is one
  testbed and one probe schedule; it is not a proof.

### 9.4 Objective mismatch and decision-aware model learning

That a more accurate, task-agnostic model need not yield a better policy is
**objective mismatch** [Lambert et al., 2020]; the remedies are sorted into
distribution correction, control-as-inference, value equivalence and
differentiable planning by *A Unified View* (arXiv 2310.06253). The
value-equivalence line is [Grimm et al., 2020] and value-aware model learning
[Farahmand et al., 2017]; the optimisation analogue is decision-focused learning
[Donti et al., 2017; Elmachtoub & Grigas, 2022], with OptNet [Amos & Kolter,
2017] as the differentiable-layer machinery. Model-based RL context: PETS [Chua
et al., 2018], MBPO [Janner et al., 2019].

Our anchored joint training sits in the differentiable-planning family, so §4.3
confirms a known phenomenon. **The data point is the mirror image of the usual
one**: decision-aware training made the surrogate 5–9× better on held-out
trajectories and still never moved the decision (t = +0.66, +1.00, identical).
Report as a nuance, not a discovery.

### 9.5 Neural operators and PDE surrogates

FNO [Li et al., 2021], DeepONet [Lu et al., 2021], GNOT [Hao et al., 2023],
benchmarked on PDEBench [Takamoto et al., 2022] — the set we use — are
2021–2023 baselines. The current frontier is transformer PDE foundation models:
Poseidon [Herde et al., 2024], which needs an order of magnitude fewer samples
than FNO on 13 of 15 tasks, DPOT [Hao et al., 2024], multiple-physics
pretraining [McCabe et al., 2023], and BCAT (arXiv 2501.18972), which reports
state of the art on PDEBench shallow water and Navier–Stokes.

**Not a contribution of this work, and nothing here depends on it.** The
surrogate is an input to the planning result. Say so and move on.

### 9.6 Wildfire intervention planning

Firebreak placement has direct prior work, all of it deep RL over the
**Cell2Fire** cellular-automaton simulator [Pais et al., 2021]: *Advancing
Forest Fire Prevention* (arXiv 2404.08523) and *Deep RL for optimal firebreak
placement* (Appl. Soft Comput. 2025). Also combinatorial: *A graph-based
optimization framework for firebreak planning* (2025), motivated exactly by
insufficient resources to treat every threatened location. Suppression-resource
allocation: *Spatiotemporal Wildfire Prediction and RL for Helitack Suppression*
(arXiv 2601.14238). Operational physics baselines are FARSITE [Finney, 1998] and
WRF-SFIRE [Mandel et al., 2011]. Domain review: [Jain et al., 2020].

| | prior work | PSPE |
|---|---|---|
| dynamics | Cell2Fire simulator | U-Net fitted to observed NDWS fire days |
| intervention effect | simulated, **counterfactually evaluable** | stated action model, **not validated** |
| spread realism | simulator calibration gap | fitted to what real fires did |

**Neither dominates**; claiming superiority would be unsupportable. What is ours:
decision-time optimisation beating learned policies on this problem (a
substantive critique, since the prior work is all DRL — but see §9.7), and the
**horizon ablation**, which is our best empirical evidence and which none of
these papers report.

### 9.7 Decision-time planning versus amortised policies

Established MPC-versus-RL territory, and the recent evidence runs against our
reading: a model-based planner **can** be amortised into a compact policy
without performance loss, via MPO plus behaviour cloning on planner-generated
data [Byravan et al., 2022]; TD-MPC and TD-MPC2 [Hansen et al., 2022, 2024] are
state of the art across diverse suites.

We trained a CNN from scratch under a budget dual for 200 iterations and did not
try distillation from planner solutions — the recipe the literature identifies
as the one that works. The defensible claim is about *our* amortised arm, not
about learned policies in general. The four constrained-RL baselines survive
better, though three of the four exceed the budget (§5.1), so that comparison is
not matched-constraint in the way the phrase implies.

> **Tested 2026-09-28 (§5.1a).** Distillation was run: same policy class as the
> amortised arm, demonstrations from the planner's own trajectories, matched
> budget. It recovers **46.9%** of the planner's advantage — 18.32% against the
> planner's 39.04% and the from-scratch arm's 10.4%. So this section's warning
> was half right and the original framing overstated the case by crediting
> amortisation with what was partly our recipe; that concession stands. But
> "amortised without performance loss" does not hold here either: **20.7 points
> survive the recommended recipe.** The claim to make is the one that survived
> the objection — decision-time planning retains a large advantage on observed
> fire data even after distillation. Remaining caveat: one recipe, not the
> family (no MPO step, no DAgger correction, 256 of 1,024 fires as demonstrations).

### 9.2a A second model-based baseline, and a prediction made before running it

§9.2 reports CAP transplanted onto our planner — 11.99% violating against our
7.80%, z = +2.08 — and then concedes "SMBPO and SafeDreamer unrun". §3.2 explains
CAP's loss *structurally*: ensemble disagreement is not the distribution that
breaches the limit, for the same reason model error is not. A structural
explanation is worth more than a win only if it predicts the next method, so the
prediction is recorded here **before** the run rather than after.

The three mechanisms, on the one axis this paper is about:

| method | what the correction is derived from | statistic | adapts? |
|---|---|---|---|
| **ours** | the realised cost distribution | conformal quantile | δ is stated, not tuned |
| CAP | disagreement among surrogates | mean + k·σ | yes, from probe violations |
| SMBPO | disagreement among surrogates | **max** over members, truncated horizon | **no** |

**Prediction: SMBPO loses to the conformal margin, and by more than CAP does.**
Two reasons, both structural rather than empirical. It draws on the same
disagreement distribution §3.2 argues is the wrong one; and `max` is a cruder
statistic than `mean + k·σ` with no adaptation to walk it back when disagreement
turns out to be uninformative — CAP's adapted `k` can at least shrink toward the
uncorrected estimate, and SMBPO's pessimism cannot.

The transplant follows `run_cap_baseline.py` exactly: same ensemble
construction, same member count, same planner, dual, testbed, limit, probe
budget and evaluation protocol, with only `_dual_input` differing. SMBPO's
truncated imagination horizon is implemented rather than assumed away
(`--imagine-frac`, the prefix cost scaled back to a full episode), because
trusting the model only over the near future is the mechanism, not a detail.

If the prediction fails — if SMBPO beats the margin — then §3.2's structural
argument is wrong and the paper's central safety claim rests on CAP being a weak
comparison. That is the outcome worth knowing, which is why it is written down
first.

**The first attempt to test it was invalid, and the invalidity pointed our way.**
`runs/smbpo_truncbias_void/`, job 1031901, killed at 3 of 5 seeds. It reported
rdf violating 0.2439, 0.1707, 0.122 — mean 17.9% against a 14.1% no-margin
baseline, so worse than applying no correction at all, which is the predicted
direction and by a larger gap than CAP's. It was wrong.

The tell was in a column beside the result: **`mean pessimism` was 0.020–0.031
against a cost limit of 3.26**, under 1%. A correction that inert cannot make an
arm violate 4 points more than baseline, so something else was moving the dual.

| quantity | rdf | dar |
|---|---|---|
| per-step cost, first → last | 0.193 → 0.229 | 0.180 → 0.182 |
| episode cost | 2.5322 | 2.1598 |
| half-horizon prefix × 2 | 2.4142 | 2.1541 |
| **bias from linear extrapolation** | **−0.118 (−4.7%)** | −0.006 (−0.3%) |

The first implementation returned `max(prefix) / imagine_frac` as *the cost
estimate*. rdf's cost accrues faster late in an episode, so that understates the
episode cost by 4.7% — and **the −0.118 truncation bias is four times the +0.03
pessimism gain.** Net, the arm handed the dual a cost *below* the baseline's own
estimate: it was optimistic, not pessimistic, and it under-constrained. The
violations were my arithmetic. dar's cost is nearly linear (−0.3%), which is why
its 0% arm gave no hint.

**The conceptual error, which is the part worth remembering.** I conflated
*truncating the horizon* with *truncating the cost estimate*. SMBPO penalises
states from which a violation is reachable inside a short horizon; it does not
rescale the episode cost. The corrected arm keeps the full-horizon estimate the
baseline uses and **adds** short-horizon disagreement:

    c_SMBPO  =  c_surrogate(full horizon)  +  [max - mean](prefix) / imagine_frac

Two properties follow, both absent before. The penalty is **≥ 0 by
construction**, so this arm can only ever be more conservative than the
baseline — the failure mode above is now unreachable. And scaling a *difference*
to the full horizon is sound where scaling the *level* was not, because both
ensemble members share the accrual profile and it cancels in the difference.

**Rule 39: when an arm's headline number moves in the predicted direction,
check that the mechanism you attributed it to is large enough to have caused
it.** One column of the output falsified the result. The same check applies to
any penalty-based arm: compare the penalty's magnitude against the effect being
claimed for it, and if the penalty is a rounding error, the effect came from
somewhere else.

**The result (job 1032361, 5 seeds each on rdf and dar, 3 h 12 m).** Run-level
comparison, resampling over runs:

| arm | runs | violating % | vs ours | 95% CI | p |
|---|---|---|---|---|---|
| no margin | 20 | 14.09 | — | — | — |
| CAP | 12 | 11.99 | +6.99 | [+3.96, +9.98] | **0.00029** |
| **SMBPO** | 5 | **12.20** | **+7.20** | **[+4.01, +10.05]** | **0.00135** |
| **ours** | 20 | **5.00** | — | — | — |

dar: 0% for SMBPO on all five seeds, the regression check passing.

**The prediction was half right, and the half that failed matters more than the
half that held.**

*Held:* SMBPO loses to the conformal margin, by 7.2 points with p = 0.00135.

*Failed:* it was predicted to lose by **more** than CAP, on the reasoning that
`max` is a cruder statistic than an adapted `mean + k·σ` and has no way to walk
itself back. **CAP − SMBPO = −0.20 points, 95% CI [−2.84, +2.52], p = 0.83.**
They are indistinguishable. That reasoning was wrong.

**Why the failure strengthens the claim.** The two methods summarise model
disagreement in about as different a way as one can: CAP takes a mean plus a
multiple of the standard deviation and *adapts* the multiple from observed
violations; SMBPO takes the hard maximum over the ensemble with no adaptation at
all. They land 0.2 points apart. And **neither is distinguishable from applying
no margin whatsoever** — SMBPO against no margin is −1.89 points, p = 0.75, which
is the same verdict §3.2 reached for CAP at −0.43σ.

So the finding is not "our margin beats these two baselines." It is:

> **How you summarise model disagreement does not matter. Computing on the wrong
> distribution does.** An adapted second moment and an unadapted maximum give the
> same answer, because both describe *how wrong the model is*, and under an
> expectation constraint that is not the quantity that carries a trajectory over
> the limit. Conformalising the realised cost is a different distribution, and it
> is worth 7 points against either of them.

That is a stronger and more general claim than the one predicted, and it is
falsifiable in the same way: a third disagreement-based correction should also
land near 12%, and a method that bounds realised outcomes by some other route
should land near 5%.

**Rule 41: when a prediction fails, check whether the failure refutes the
mechanism or generalises it.** This one generalised it — the mechanism said
*disagreement is the wrong distribution*, and my prediction added an untested
claim about which summary of disagreement would be worse. The mechanism survived;
the embellishment did not, and dropping it makes the argument cleaner.

### 9.8 Data assimilation and digital twins

4D-Var and the Ensemble Kalman Filter [Evensen, 2003; Houtekamer & Zhang, 2016]
are the classical machinery; the ML era has latent data assimilation with
reduced-order surrogates [Cheng et al., 2023], latent-space DA at global scale
(*Physically consistent global atmospheric data assimilation with machine
learning in latent space*, Sci. Adv. 2026), deep latent particle filters with UQ
(arXiv 2406.02204), and the learned forecast models that motivate it:
FourCastNet [Pathak et al., 2022], Pangu-Weather [Bi et al., 2023], GraphCast
[Lam et al., 2023], NeuralGCM [Kochkov et al., 2024], Aurora [Bodnar et al.,
2025]. Programme-scale twins: Destination Earth, NVIDIA Earth-2.

**Our "state sync" is nudging with gain 1** — no covariance, no observation-error
model, no ensemble, no variational step. Comparing it to a free-running forecast
is what assimilation is *for*. The 8.4× is a wiring check, not a finding, and
presenting it otherwise to a DA audience would land badly. Either add a real DA
baseline over the same sequences, or demote §5.2 to a control.

### 9.9 Wildfire datasets: the FIRMS pipeline was avoidable

NDWS [Huot et al., 2022] cannot be chained — no date, no fire id, no location —
which is why `pspe/simulate/real/firms.py` exists. But **WildfireSpreadTS**
[Gerard et al., 2023] was built for exactly that reason and has been public
since NeurIPS 2023: **607 fire events, 13,607 daily images, 23 multi-modal
channels, 2018–2021**. Extensions and successors: WSTS+ (WACV 2026), FireSentry,
BCWildfire, and the evaluation study *WildfireSpreadBench* (arXiv 2609.22191).

~~Our twin loop runs on **8 fires** where 607 were available.~~ **Ported
(§5.2a).** It now runs on all 607, held out by year rather than by fire, and the
port cost a day rather than the "days of work" estimated here. It was worth it
for the reason this section gives and for one it does not: at n = 607 the
adaptation benefit turned out to be about a third of the eight-fire estimate and
the "8/8 fires" unanimity became 62–76% (§5.2a). The underpowering was not only
a presentational weakness, it was producing numbers that were too good.

Still only 1 of the 23 channels is used — the active-fire detection. Fuel,
terrain and weather are all sitting in the same rasters and the spread model
ignores them, which is the obvious next experiment on this dataset. FIRMS keeps
one legitimate role: a **live incident**, which a static archive cannot
support.

### 9.10 Beliefs evaluated on decisions

The assimilation stack in §9.8 optimises **analysis quality** on the assumption
that a better analysis yields a better outcome; the Science Advances latent-DA
result is stated in those terms — better analysis *and* better forecast skill.
Neither is the decision.

§5.3 is a counterexample in the regime that matters for intervention: under a
hard budget on a sparse field, an estimator ~100× better by average precision
produced a worse decision in **0 of 9** configurations. *WildfireSpreadBench*
finds the same shape on the forecast side of the same domain — the best-AP model
flags 4–5× the area that actually burned while ranking fifth on F1/IoU.

Two independent instances of "component metric improves, operational quantity
worsens" in one domain is a real pattern, and the closest thing to a general
claim this project has. **The flood experiment (§8.4) decides whether it is a
sparse-field quirk or a property of budgeted intervention.**

### 9.11 Evaluation rigour

[Henderson et al., 2018] on evaluation protocol and variance in deep RL;
[Agarwal et al., 2021] on point estimates, few-run uncertainty, stratified
bootstrap intervals and the `rliable` recommendations; and, concurrent and in
our own domain, *WildfireSpreadBench* (arXiv 2609.22191): "metrics outrank
architecture, with AP and F1 picking different winners."

Part 7 is a domain-specific instance of this genre for constrained model-based
planning, not a new kind of argument. Worth publishing, not a headline. One
concrete debt: we report mean ± std over 3–5 seeds, which is precisely what
[Agarwal et al., 2021] argue against; stratified bootstrap CIs and interquartile
means would cost nothing and are what a careful reviewer now expects.

### 9.12 Explanation faithfulness

The canonical precedent for §4.4's permutation control is [Adebayo et al.,
2018]: a method "independent both of the model and of the data generating
process" fails a randomisation test and is inadequate whatever it scores. Our
briefs fail that test in its original form. The chain-of-thought faithfulness
literature runs the same family of controls [Turpin et al., 2023; Lanham et al.,
2023], and interpretability more broadly has debated exactly this
[Jacovi & Goldberg, 2020; Rudin, 2019].

**Credit for applying the control and withdrawing the claim; none for inventing
the method.** The certificate machinery is the part that stands.

### 9.13 What the paper should be

§0.3 proposed a four-contribution paper led by the failure catalogue. After this
audit the individual failures each have precedent (§9.1), so a knowledgeable
reviewer reads the catalogue as careful replication in a new domain. The
narrower paper survives intact:

> **What should a conformal margin bound in constrained model-based planning?**
>
> Constrained model-based planners violate their budget because the dual is fed
> cost measured in the model while violation is realised under true dynamics.
> The standard conformal remedy bounds the *model's* prediction error. We show
> this fails whenever the model is accurate and the policy is variable — on a
> reaction-diffusion front it raises violations from 14.5% to 18.2%, worse than
> no margin at all — and that conformalising *realised episode costs* holds the
> stated rate on two PDE families. We validate the resulting planner on 1,500
> real wildfire records, where planning through a learned spread model beats the
> operational forecast-then-treat heuristic by 12 points at matched crew budget,
> with the advantage appearing only as the decision becomes sequential
> (t = 1.40 at one day, +14.3 points at five).

One claim, one mechanism, one negative result against a named alternative, two
independent validations. The digital-twin framing becomes motivation only: after
§9.8 it cannot carry weight, and §6.2's property 4 fails outright.

---

### 9.14 Comparison with related work — qualitative

What each line of work calibrates, synchronises or optimises, against what this
project does, and what was actually measured about the difference. "Verdict" is
the honest standing of our claim against that line, not a score.

| § | line of work | representative | what they do | what we do | measured difference | verdict |
|---|---|---|---|---|---|---|
| 9.3 | Conformal safety in control | Lindemann 2023; CBF+ACP (2503.17678); conformal policy control (2603.02196) | conformalise the **error of the learned dynamics model**, then plan inside the inflated tube | conformalise **realised episode cost against the quantity the dual controls** | the default raises rdf violations to 14.15% against 13.17% for no margin; ours reaches 7.80% (§3.2) | **Novel and falsifying.** We do not merely propose an alternative; we show the field's default is worse than nothing in a regime it is used in |
| 9.3 | Behavioural-change conformal | Prinster et al. 2026 | calibrate permissible deviation from a safe reference policy; no dynamics model | calibrate realised cost against a Lagrangian's controlled quantity | not run head-to-head; different constraint object (behavioural budget vs CMDP cost budget) | **Closest prior art.** Must be cited as such, not lumped with the model-error line |
| 9.3 | Adaptive conformal under shift | Gibbs & Candès 2021 | update δ online because a learning policy shifts the score distribution | fixed split-conformal quantile over the run's residuals | exchangeability tested and **not violated**: real-order coverage 0.0771 vs shuffled 0.0854, Fisher p = 0.723; adaptive gives 0.1000 (§5.10) | **Not needed here, and now demonstrated rather than assumed** |
| 9.2 | Model-based safe RL | CAP (Ma 2022); SMBPO (Thomas 2021); SafeDreamer | inflate the cost estimate by model uncertainty (ensemble disagreement) | inflate by a conformal quantile of realised-cost residuals | At matched sample budget, run-level: **CAP 11.99%, SMBPO 12.20%, ours 5.00%**. Both lose by ~7 pts (p = 0.0003 / 0.0014); CAP vs SMBPO p = 0.83; neither separates from no margin at all (§9.2a) | **Ours wins, and the reason generalises further than expected**: an adapted mean+kσ and an unadapted max land 0.2 pts apart, so the *summary* of disagreement is irrelevant — the distribution is what is wrong. SafeDreamer unrun |
| 9.2 | Model-free constrained RL | CPO; PPO-Lagrangian; Sauté; primal-dual NPG | constrain in the true environment, many samples | plan in a surrogate, probe the truth periodically | dar: better return (t = +3.84 to +17.95) at **12.5× fewer real transitions**, but 7.3% violating where they violate 0% (§3.1) | **Mixed and stated as such.** Part of the sample-efficiency advantage was bought with violations |
| 9.4 | Objective mismatch / decision-focused learning | Lambert 2020; decision-focused learning | better model ≠ better control; train the model for the decision | tested joint Simulate+Plan training | model improved **5–9×**, decision did not move (§4.3) | **Known, inverted data point.** Our contribution is the magnitude: the improvement was large and the decision was flat |
| 9.10 | Belief quality → decision quality | latent DA (Sci. Adv. 2026); deep latent particle filters | optimise analysis quality, assume better analysis ⇒ better outcome | score the belief on the decision | reconstruction AP 0.005 → 0.346 (70×) and burn reduction **falls** 34.5% → 27.3%; mechanism is 3.74× soft-mass inflation (§5.8) | **Novel direction, now with a mechanism.** The softness, not the content, does the damage |
| 9.8 | Data assimilation | 4D-Var; EnKF (Evensen 2003) | optimal gain, covariance, observation error model | gain-1 nudging, now with a 32-member EnKF beside it | 606 fires, 3 seeds: EnKF loses to gain 1 at every σ_o (−0.0497 at σ_o = 0.05, t = −20.95); skill tracks the implied gain 0.979/0.837/0.562 → 0.329/0.127/0.032 (§5.2c) | **Gap closed, and the answer favours the simple method for a stated reason.** K → 1 because the observations beat the forecast. Caveat: perturbed-observation EnKF has a sampling-noise floor, so σ_o = 0.05 does not exactly recover gain 1 — a square-root filter is the follow-up |
| 9.6 | Wildfire intervention planning | firebreak / fuel-treatment optimisation | optimise placement, usually in simulation | horizon ablation on observed fire data | null at 1 day where theory requires, +14.3 at 5 days | **Novel as evidence.** No firebreak paper reports this ablation |
| 9.9 | Sequential wildfire benchmarks | WildfireSpreadTS (Gerard 2023); WSTS+; WildfireSpreadBench | 607 fires, 23 channels, year-wise CV recommended | ported to it; 1 of 23 channels used | eight-fire estimate was ~3× too generous; "8/8" became 62–76% (§5.2a) | **Closed, and it cost us a claim.** Channel use is the remaining gap |
| 9.5 | PDE surrogates | FNO; Poseidon; DPOT; BCAT | frontier operator architectures | FNO, a 2021 baseline | not benchmarked against the frontier | **Behind, and not our contribution.** The claim is about what the margin bounds, not the operator |
| 9.11 | Evaluation rigour | Henderson 2018; Agarwal 2021; WildfireSpreadBench | the metric outranks the architecture | 26 defects, 30 rules, each with the measurement that caught it | — | **Converging with an established line.** Our instance is constrained model-based planning |
| 9.12 | Explanation faithfulness | faithfulness metrics for post-hoc explanation | score explanation against model behaviour | permutation control against another state | F = 0.47 vs post-hoc 0.18, t = +21, yet **zero** state-specific information | **Withdrawn claim.** The control killed our own result |

### 9.15 Comparison with related work — quantitative

Two kinds of number appear below and they must not be read together.

**A. Measured head-to-head in our setting.** Same testbed, same limit, same
evaluation protocol, same real-sample budget, only the named mechanism differs.
These are the comparisons the paper can defend.

*Constraint violation on `rdf`, full-fidelity surrogate, matched budget of 6,400
training transitions plus 1,920 probe (§3.2). Lower is better; δ = 0.1.*

| method | what it conformalises / penalises | violating | seeds | z vs ours |
|---|---|---|---|---|
| no margin | — | 13.17% | 5 | −3.68 |
| model-error conformal (**field default**) | \|c − g\| , matched pairs | **14.15%** | 5 | not computed on this run† |
| CAP, ensemble penalty | mean + k·std over 5 surrogates | 11.99% | 12 | −2.08 |
| **ours, residual conformal** | c − ĝ, the quantity the dual holds | **7.80%** | 10 | — |

† The model-error contrast was established separately and stratified by
ρ = σ/ε, which is where the mechanism lives (§3.2): `model_error` vs `residual`
gives z = +0.86 at ρ < 8 and **z = +2.78 at ρ ≥ 8**. That stratification is the
claim — the default is fine where per-instance model error dominates and fails
where policy spread does — so a single pooled z against it would blur the
result rather than sharpen it. CAP vs no margin is **z = −0.43**.

The two rows that matter: the field's default scores **worse than applying no
margin at all** (14.15% against 13.17%), and CAP — the closest model-based
safe-RL method, with its adapted k demonstrably active between 0.50 and 1.46 —
is statistically indistinguishable from no margin.

*Sample efficiency and constraint respect on `dar`, d = 0.936, 5 seeds (§3.1).*

| method | return | violating evals | real transitions |
|---|---|---|---|
| **PSPE (adaptive α)** | **−2.448 ± 0.029** | 7.3% | **3,072** |
| PPO-Lagrangian | −2.558 ± 0.039 | 0% | 38,400 |
| CPO | −2.541 ± 0.035 | 0% | 38,400 |
| Sauté RL | −2.537 ± 0.024 | 0% | 38,400 |
| primal-dual NPG | −2.544 ± 0.026 | 0% | 38,400 |

Best return at 12.5× fewer real transitions, and the only method that violates.
Both halves belong in the abstract.

*Wildfire intervention, frozen U-Net shared by every arm, 3% daily budget (§5.1).*

| method | burn reduction | treated/day | fires over budget |
|---|---|---|---|
| greedy heuristic | 24.1 ± 3.8% | 3.00% | 0 |
| PPO-Lagrangian | 4.6 ± 0.4% | 4.32% | 58% |
| CPO | 4.3 ± 0.1% | 2.84% | 6% |
| Sauté RL | 4.7 ± 0.5% | 7.37% | 79% |
| primal-dual NPG | 4.6 ± 0.1% | 3.26% | 100% |
| **PSPE per-instance** | **36.5 ± 1.0%** | 3.00% | 0 |

**B. Not comparable, and listed so the gap is explicit.** Published numbers from
these works were obtained on other benchmarks, metrics and budgets. Quoting them
beside ours would be the exact error §5.7 is about — agreement or disagreement
with a number is meaningless until the two were measured against the same thing.

| line of work | why no head-to-head | what would make one possible |
|---|---|---|
| SMBPO, SafeDreamer | not implemented here | the same transplant treatment CAP got: their correction, our dual, testbed and probe budget |
| Prinster et al. 2026 | different constraint object (behavioural-change budget, not CMDP cost) | a setting where both budgets are definable; not obviously worth it |
| EnKF / 4D-Var | no DA baseline in the twin loop | §8.4 item 3, days of work, and it decides whether §5.2a is a finding or a control |
| Poseidon, DPOT, BCAT | we use FNO and do not claim operator quality | swap the surrogate; the margin result should be architecture-independent, which is itself testable |
| WildfireSpreadBench | an evaluation study, not a method | its protocol could be adopted wholesale for §5.2a |

**What the two tables say together.** Where we have run the comparison properly,
the result is consistent and it is not that our method is uniformly better —
it is that **the mechanisms the literature reaches for first do not act on the
quantity that breaks the constraint.** Model error, ensemble disagreement and
reconstruction accuracy are all defensible things to measure and improve, and
all three were measured here to be ineffective or actively harmful for the
decision. That is the paper.

## References

Conventions: where I could not verify authorship, the entry is given by title
and identifier rather than invented. Everything below was checked in September
2026 or is a canonical work cited from established knowledge.

### Constrained and safe reinforcement learning

- Achiam, Held, Tamar, Abbeel. *Constrained Policy Optimization.* ICML 2017.
- Stooke, Achiam, Abbeel. *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.* ICML 2020.
- Sootla et al. *Sauté RL: Almost Surely Safe RL Using State Augmentation.* ICML 2022.
- Ding, Zhang, Başar, Jovanović. *Natural Policy Gradient Primal-Dual Method for Constrained MDPs.* NeurIPS 2020.
- Ray, Achiam, Amodei. *Benchmarking Safe Exploration in Deep Reinforcement Learning.* 2019. (Safety Gym)
- Ji et al. *Safety Gymnasium / OmniSafe: an infrastructure for accelerating safe RL research.* 2023. [arXiv 2305.09304](https://arxiv.org/pdf/2305.09304)
- Thomas, Luo, Ma. *Safe Reinforcement Learning by Imagining the Near Future.* [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/73b277c11266681122132d024f53a75b-Abstract.html). (SMBPO)
- Ma, Shen, Bastani, Jayaraman. *Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning.* [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/download/20478/20237).
- Jayant, Bhatnagar. *Model-based Safe Deep RL via a Constrained Proximal Policy Optimization Algorithm.* [NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/9a8eb202c060b7d81f5889631cbcd47e-Paper-Conference.pdf).
- Huang et al. *SafeDreamer: Safe Reinforcement Learning with World Models.* [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/ece182f93af26c64187ba3f7dfd4309a-Paper-Conference.pdf).
- Bharadhwaj et al. *Conservative Safety Critics for Exploration.* ICLR 2021.

### Conformal prediction

- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA, 2018.
- Angelopoulos, Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in ML, 2023.
- Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.
- Lindemann, Cleaveland, Shim, Pappas. *Safe Planning in Dynamic Environments using Conformal Prediction.* [arXiv 2210.10254](https://arxiv.org/abs/2210.10254).
- Zhou, Zhang, Luo. *Computationally and Sample Efficient Safe Reinforcement Learning Using Adaptive Conformal Prediction.* [arXiv 2503.17678](https://arxiv.org/abs/2503.17678).
- Prinster, Fannjiang, Park, Cho, Liu, Saria, Stanton. *Conformal Policy Control.* [arXiv 2603.02196](https://arxiv.org/html/2603.02196v2). (Calibrates against a reference **policy**, not a dynamics model.)
- Mirzaeedodangeh, Shekhtman, Matni, Lindemann. *Robust Conformal CBF and CLF Controllers via Iterative Policy Updates.* [arXiv 2606.15366](https://arxiv.org/html/2606.15366).
- Tomashevskiy. *Proactive Context-Forecasted Safety Constraints for Nonstationary Reinforcement Learning.* CoLLAs 2026, [arXiv 2609.08080](https://arxiv.org/html/2609.08080).

### Model-based RL, objective mismatch, decision-focused learning

- Lambert, Amos, Yadan, Calandra. *Objective Mismatch in Model-based Reinforcement Learning.* [L4DC 2020, arXiv 2002.04523](https://arxiv.org/abs/2002.04523).
- *A Unified View on Solving Objective Mismatch in Model-Based Reinforcement Learning.* [arXiv 2310.06253](https://arxiv.org/pdf/2310.06253).
- Grimm, Barreto, Singh, Silver. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* NeurIPS 2020.
- Farahmand, Barreto, Nikovski. *Value-Aware Loss Function for Model-based Reinforcement Learning.* AISTATS 2017.
- Chua, Calandra, McAllister, Levine. *Deep RL in a Handful of Trials using Probabilistic Dynamics Models.* NeurIPS 2018. (PETS)
- Janner, Fu, Zhang, Levine. *When to Trust Your Model: Model-Based Policy Optimization.* NeurIPS 2019. (MBPO)
- Donti, Amos, Kolter. *Task-based End-to-end Model Learning in Stochastic Optimization.* NeurIPS 2017.
- Elmachtoub, Grigas. *Smart "Predict, then Optimize".* Management Science, 2022.
- Amos, Kolter. *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML 2017.

### Planning, amortisation, model predictive control

- Byravan et al. *Evaluating model-based planning and planner amortization for continuous control.* 2022.
- Hansen, Wang, Su. *Temporal Difference Learning for Model Predictive Control.* [ICML 2022, arXiv 2203.04955](https://arxiv.org/pdf/2203.04955). (TD-MPC)
- Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024.
- *Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination.* [arXiv 2605.04568](https://arxiv.org/pdf/2605.04568).

### Neural operators and PDE surrogates

- Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- Lu, Jin, Pang, Zhang, Karniadakis. *Learning nonlinear operators via DeepONet.* Nature Machine Intelligence, 2021.
- Hao et al. *GNOT: A General Neural Operator Transformer for Operator Learning.* ICML 2023.
- Takamoto et al. *PDEBench: An Extensive Benchmark for Scientific Machine Learning.* NeurIPS 2022 Datasets & Benchmarks.
- Herde et al. *Poseidon: Efficient Foundation Models for PDEs.* [NeurIPS 2024, arXiv 2405.19101](https://proceedings.neurips.cc/paper_files/paper/2024/file/84e1b1ec17bb11c57234e96433022a9a-Paper-Conference.pdf).
- Hao et al. *DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training.* ICML 2024, arXiv 2403.03542.
- McCabe et al. *Multiple Physics Pretraining for Physical Surrogate Models.* 2023.
- *BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics.* [arXiv 2501.18972](https://ww3.math.ucla.edu/wp-content/uploads/2025/02/2501.18972v1.pdf).
- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.

### Wildfire data, forecasting and intervention

- Huot, Hu, Goyal, Sankar, Ihme, Chen. *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data.* [IEEE TGRS 2022, arXiv 2112.02447](https://arxiv.org/pdf/2112.02447).
- Gerard, Zhao, Sullivan. *WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction.* [NeurIPS 2023 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf).
- *Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark.* [WACV 2026, arXiv 2502.12003](https://arxiv.org/pdf/2502.12003).
- *Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction.* [arXiv 2609.17763](https://arxiv.org/abs/2609.17763). (0.3673 single, 0.3790 ensemble)
- *WildfireSpreadBench: The Metric Decides the Model in Wildfire Spread Prediction.* [arXiv 2609.22191](https://arxiv.org/html/2609.22191).
- *Advancing Forest Fire Prevention: Deep Reinforcement Learning for Effective Firebreak Placement.* [arXiv 2404.08523](https://arxiv.org/pdf/2404.08523).
- *Deep reinforcement learning for optimal firebreak placement in forest fire prevention.* [Applied Soft Computing, 2025](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003540).
- *A graph-based optimization framework for firebreak planning in wildfire-prone landscapes.* [2025](https://www.sciencedirect.com/science/article/pii/S1574954125003486).
- *Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression.* [arXiv 2601.14238](https://arxiv.org/pdf/2601.14238).
- Pais, Carrasco, Martell, Weintraub, Woodruff. *Cell2Fire: A Cellular Automata Based Forest Fire Growth Model.* Frontiers in Forests and Global Change, 2021.
- Finney. *FARSITE: Fire Area Simulator — Model Development and Evaluation.* USDA Forest Service, 1998.
- Mandel, Beezley, Kochanski. *Coupled atmosphere-wildland fire modeling with WRF-Fire.* Geoscientific Model Development, 2011.
- Jain et al. *A review of machine learning applications in wildfire science and management.* [arXiv 2003.00646](https://arxiv.org/pdf/2003.00646).
- NASA FIRMS. *Fire Information for Resource Management System, VIIRS active fire detections.*

### Data assimilation and climate digital twins

- Evensen. *The Ensemble Kalman Filter: theoretical formulation and practical implementation.* Ocean Dynamics, 2003.
- Houtekamer, Zhang. *Review of the Ensemble Kalman Filter for Atmospheric Data Assimilation.* Monthly Weather Review, 2016.
- Cheng et al. *Reduced order digital twin and latent data assimilation.* [2022/2023](https://egusphere.copernicus.org/preprints/2022/egusphere-2022-1167/egusphere-2022-1167.pdf).
- *Physically consistent global atmospheric data assimilation with machine learning in latent space.* [Science Advances, 2026](https://www.science.org/doi/10.1126/sciadv.aea4248).
- *The Deep Latent Space Particle Filter for Real-Time Data Assimilation with Uncertainty Quantification.* [arXiv 2406.02204](https://arxiv.org/pdf/2406.02204).
- Pathak et al. *FourCastNet: A Global Data-driven High-resolution Weather Model.* 2022.
- Bi et al. *Accurate medium-range global weather forecasting with 3D neural networks.* Nature, 2023. (Pangu-Weather)
- Lam et al. *Learning skillful medium-range global weather forecasting.* Science, 2023. (GraphCast)
- Kochkov et al. *Neural general circulation models for weather and climate.* Nature, 2024. (NeuralGCM)
- Bodnar et al. *A foundation model for the Earth system.* Nature, 2025. (Aurora)
- ECMWF. *Destination Earth (DestinE) climate change adaptation digital twin.*

### Explanation, faithfulness and evaluation rigour

- Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim. *Sanity Checks for Saliency Maps.* [NeurIPS 2018, arXiv 1810.03292](https://arxiv.org/abs/1810.03292).
- Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS 2023.
- Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* 2023.
- Jacovi, Goldberg. *Towards Faithfully Interpretable NLP Systems.* ACL 2020.
- Rudin. *Stop explaining black box machine learning models for high stakes decisions.* Nature Machine Intelligence, 2019.
- Henderson, Islam, Bachman, Pineau, Precup, Meger. *Deep Reinforcement Learning that Matters.* AAAI 2018.
- Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* [NeurIPS 2021, arXiv 2108.13264](https://arxiv.org/pdf/2108.13264).

### Components used

- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Zhai, Mustafa, Kolesnikov, Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. (SigLIP)
- Qwen Team. *Qwen2.5 Technical Report.* 2024.

---

## Appendix — Index of analyses and runs

| file | contents |
|---|---|
| `docs/results/PHASE1_ANALYSIS.md` | Joint training, Eq. 8, Lipschitz/Prop 1, weight sweep, conformal on dar |
| `docs/results/PHASE3_ANALYSIS.md` | Constraint fix and joint training on swe and rdf |
| `docs/results/RDF_PLANNER_FIX.md` | Three-defect diagnosis, dar regression, rdf safe-RL baselines |
| `docs/results/NDWS_PLANNING.md` | Firebreak planning on observed fire data |
| `docs/results/NDWS_PARTIAL.md` | Planning under partial observation |
| `docs/results/NDWS_EXPLAIN.md` | Briefs and certificate on real fire plans |
| `docs/results/EXPLAIN_PERMUTATION.md` | The permutation control and three failed fixes |
| `docs/results/FIRMS_TWIN.md` | The twin loop on real satellite sequences |
| `docs/results/ROBUSTNESS_NOTES.md` | Seven checks, the results they overturned, the standing rules |
| `docs/RELATED_WORK.md` | Positioning audit against the state of the art |
| `docs/MASTER_REPORT.md` | Narrative report (md / html / pdf) |
| `docs/figures/` | Diagrams (below) |
| `docs/figures/paper/` | Publication figures, PDF and 400 dpi PNG |
| `scripts/make_thesis_figures.py` | Regenerates the diagrams in this document |
| `scripts/make_paper_figures.py` | Publication figures, ICML column widths (below) |
| `scripts/make_margin_figures.py` | The margin comparison, for this document |
| `pspe/plan/margins.py` | The three recipes; every experiment calls this one module |
| `eval/run_margin_synthetic.py` | Controlled coverage study, the ρ crossover |
| `eval/run_margin_coverage.py` | Realised rate against the stated level δ |
| `eval/run_margin_choice.py` | Fidelity sweep on a real planner |
| `eval/run_ndws_margin.py` | The margin on real fire dynamics; `--calibrate-only` for feasibility |
| `tests/test_margins.py` | The distinguishing properties, as executable claims |

### Figures

All are written as svg, pdf and png by `python scripts/make_thesis_figures.py`.
Boxes size themselves from their contents, so editing the text cannot push a
line outside its frame.

| figure | contents | section |
|---|---|---|
| `pspe_silent_failure` | The thesis: seven component metrics against the decision | §0.2 |
| `pspe_system` | Combined architecture, validated and model-based paths | Part 1 |
| `pspe_safety` | Reality probe, conformal margin, measured violation rates | §3.2 |
| `pspe_mod_perceive` | Perceive: mechanism, what held, what did not | §4.6, §5.3 |
| `pspe_mod_simulate` | Simulate: operator, benchmarks, the vacuous bound | §4.1, §4.5 |
| `pspe_mod_plan` | Plan: hybrid gradient, dual, decision-time optimisation | Part 3, §5.1 |
| `pspe_mod_explain` | Explain: certificate held, faithfulness withdrawn | §4.4 |
| `pspe_wildfire` | Budget Pareto and horizon ablation, measured | §5.1 |
| `pspe_twin_loop` | Open loop against state sync against state and model | §5.2 |

### Publication figures

`python scripts/make_paper_figures.py` writes `docs/figures/paper/`, sized for
an ICML two-column layout (3.25 in and 6.75 in), serif type, Okabe–Ito palette
with distinct markers so they survive greyscale.

| figure | claim | paper |
|---|---|---|
| `fig0_calibration` | realised rate against stated δ; the default barely responds to δ | margin |
| `fig1_mechanism` | (a) coverage fails past ρ = 1; (b) why — the margin vanishes | margin |
| `fig2_rdf_violations` | the default is worse than no margin at all | margin |
| `fig3_precondition` | `b < f·s`; a second trained model never reaches it | margin |
| `fig5_tradeoff` | −4.8 points of violation for −1.8% of return | margin |
| `fig6_ablation` | neither probe nor margin works alone | margin |
| `fig4_wildfire` | budget Pareto and the horizon ablation | PSPE |
| `fig7_partial_observation` | reconstruction 100× better, decision worse | PSPE |
| `fig8_twin_loop` | syncing to observation on real fire sequences | PSPE |

Appendix figures, from `python scripts/make_appendix_figures.py` (§5.9):

| figure | claim | paper |
|---|---|---|
| `figA1_sar_validation` | what the satellite saw, against the reference | appendix |
| `figA2_sar_stratified` | CSI by land cover; radar is blind under canopy | appendix |
| `figA3_wildfire_perception` | the estimator recovers the front and smears it 3.3× | appendix |
| `figA4_wildfire_decision` | reconstruction improves, the decision does not | appendix |
| `figA5_portal_appraisal` | the planner as an operator meets it; harm flagged | appendix |
| `figA6_portal_anywhere` | any town, solved from public DEM in 47 s | appendix |
| `figA7_validation_chain` | 0.979 against a model, 0.535 against the world | appendix |

Superseded by the above, kept because slide decks reference them:
`pspe_architecture` (draws joint training as a headline path, refuted in §4.3),
`pspe_perceive` / `pspe_simulate` / `pspe_plan` / `pspe_explain` (pre-date the
permutation control and the conformal margin), and the six application figures
`pspe_digital_twin`, `pspe_twin_timeline`, `pspe_applications`,
`pspe_app_wildfire`, `pspe_app_flood`, `pspe_app_heat`.
