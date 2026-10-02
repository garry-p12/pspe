# Constrained Planning in Learned Dynamics: Which Distribution the Margin Must Bound

*Target: ICML. The structural result and its validation. Applications live in
`WORKSHOP_PAPER.md`; this paper cites them and does not lead with them.*

---

## Abstract

A planner that optimises inside a learned surrogate inherits that surrogate's
errors, and inherits them invisibly: a dual controller enforcing a budget is fed
cost measured *in the model*, while violation is realised under *true* dynamics.
The dual converges, reports the constraint satisfied, and the real system
violates it anyway.

The natural correction is to inflate the cost estimate by how wrong the model
is. We prove this cannot work. **No margin that is a measurable function of the
model-error distribution alone attains a stated violation level uniformly in the
spread of realised cost** (Proposition 1). The obstruction is not a tightness
issue: realised violation is strictly increasing in that spread and tends to ½
for symmetric error, so every such margin fails beyond a computable threshold.
For Gaussian error the failure has the closed form `R(ρ) = 1 − Φ(z_δ/ρ)` with
`ρ = σ/ε`, which predicts measured violation to within **0.003** across two
orders of magnitude in ρ.

The admissible correction follows: calibrate on the realised-cost distribution.
A split-conformal quantile of `c − ĝ`, with `ĝ` the quantity the dual actually
controls, cuts violations on a reaction–diffusion hazard from **14.1% to 5.0%**
(20 seeds, paired −9.1 points, 95% CI [−11.8, −5.5], p = 0.0002) **at no cost in
objective value**, and refuses to return a number when its calibration set
cannot support the requested level.

Three published disagreement-based corrections — conformal ensemble
disagreement, SMBPO's pessimism, and SafeDreamer's cost critic — land at 11.99%,
12.20% and 13.17%, none better than applying no margin at all. That is the
theorem's prediction rather than three baselines that happened to lose; we
pre-registered 12% for SafeDreamer before running it. We additionally test the
exchangeability assumption the guarantee rests on, find it **violated** on one
of two testbeds through serial dependence, and find coverage attained anyway.

We report five original claims of our own that did not survive, three retracted
results, and a defect catalogue of 39 measurement errors with the check that
caught each.

---

## 1. The failure

Constrained planning in a learned model is usually a Lagrangian: maximise
reward, penalise constraint cost, let a dual multiplier find the trade-off. It
works when the cost estimate is unbiased. Inside a surrogate it is not, and the
failure mode is specific and quiet.

> The dual converges. Its own diagnostics report the constraint satisfied. The
> realised system violates it.

Measured on a reaction–diffusion hazard (§5.1): the dual settles, surrogate
episode cost sits under the limit, and **7.3% of evaluations violate the true
limit** — found only by periodically probing the true environment, which is not
something a self-consistent training loop does on its own.

**Why the obvious fix is the wrong fix.** Inflate the cost estimate by the
model's error, and violations get *worse*: 14.5% → 18.2% on the same task. This
is not a tuning failure. It is the subject of §3.

**Figure 1** `docs/figures/paper/fig1_mechanism.png` — the dual's view against
the realised distribution, same run.

---

## 2. Setup

An episode has realised cost `c` under true dynamics and surrogate estimate `g`.
The constraint is on an **expectation**: `E[c] ≤ L`. Write

- `F` for the distribution of realised cost, with spread `σ`;
- `G` for the distribution of model error `c − g`, with bias `b` and scale `ε`;
- `R(q; σ)` for the realised violation rate when margin `q` is applied;
- `δ` for the violation level the user *states*, not tunes;
- **`ρ = σ / ε`**, the one dimensionless quantity that matters.

A margin rule is **model-error measurable** if `q = Q(b, G)` — a function of how
wrong the model is, with no dependence on `F` or `σ`. Every correction of the
form "inflate by model uncertainty" is of this type, however the uncertainty is
summarised: an error bound, an ensemble spread, a learned variance, a
pessimism bonus.

---

## 3. Theory: the choice of distribution is a feasibility constraint

**Proposition 1 (separation).** Let `Q` be model-error measurable. If `F` has
full support then

1. `R(q; σ)` is strictly increasing in `σ`;
2. `R(q; σ) → 1 − F(0) = ½` as `σ → ∞`, for symmetric `F`;
3. coverage fails for every `σ > (Q(b, G) − b) / z_δ`.

Consequently **no model-error-measurable margin attains level δ uniformly in σ.**

*Proof.* `(q − b)/σ` is decreasing in `σ` and `F` is increasing, so `R`
increases; the limit is `1 − F(0)`; and `R > δ` exactly when `(q − b)/σ < z_δ`. ∎

The proof is three lines and that is the point: the missing object was the
statement, not a harder argument. It says the choice of distribution to
conformalise is not a modelling preference but a feasibility question.

**Corollary 1 (where ρ = 1 comes from).** The matched-pair recipe
`q = Q_{1−δ}(c − g)` equals `b + z^G_δ ε`, so coverage fails iff
`σ > (z^G_δ / z^F_δ) · ε`, which for `F`, `G` of shared shape is exactly **ρ > 1**.

**Corollary 2 (closed form).** For Gaussian `F`, a model-error margin's realised
violation rate is

```
R(ρ) = 1 − Φ( z_δ / ρ )
```

**Validated against simulation**, δ = 0.1, 600 calibration draws per point:

| ρ = σ/ε | measured | predicted | error |
|---|---|---|---|
| 0.10 | 0.000 | 0.000 | 0.0000 |
| 0.50 | 0.004 | 0.005 | 0.0012 |
| 0.80 | 0.054 | 0.055 | 0.0006 |
| **1.00** | **0.099** | **0.100** | 0.0010 |
| 1.25 | 0.153 | 0.153 | 0.0004 |
| 2.00 | 0.258 | 0.261 | 0.0028 |
| 10.0 | 0.449 | 0.449 | 0.0000 |

Maximum error 0.003 over two orders of magnitude in ρ. The corollary also
predicts something counter-intuitive and testable: **requesting a stricter
guarantee from the wrong distribution makes matters worse**, the default's error
growing from 1.2× at δ = 0.30 to 3.7× at δ = 0.02. Measured, and it holds.

**Corollary 3 (what a valid margin must use).** Any margin attaining δ uniformly
must depend on `F` and `σ`. The recipe below is measurable with respect to the
realised-cost distribution and is therefore admissible.

**What this buys.** The contribution is not "we picked a better conformal
score". It is a **feasibility statement**: an entire family of corrections
cannot attain a coverage level under an expectation constraint once policy
spread exceeds model error, and ρ says exactly when. The baselines in §5.2 are
then not methods that happened to lose — they are instances of a class the
theorem excludes, and their near-identical results are the prediction.

**Figure 2** `docs/figures/paper/fig0_calibration.png` — measured against
`R(ρ)`, with the δ-sweep inset.

---

## 4. The margin

Let `ĝ` be the quantity the dual controls. Calibrate on matched pairs
`s_i = c_i − ĝ` drawn from a periodic probe of the true environment, and take
the split-conformal quantile. Adding it to the limit gives, under
exchangeability of the scores,

```
P[ realised cost > limit ] ≤ δ
```

with δ **chosen and stated**. Two operational properties matter.

**It refuses.** `ceil((n+1)(1−δ)) ≤ n` requires `n ≥ 1/δ − 1`; below that no
order statistic carries the guarantee and the function returns `None` rather
than extrapolating. This caught a real bug in our own interface, which displayed
90% confidence from six calibration runs where nine are needed (§7).

**It needs no error model.** The score is an outcome, so nothing is assumed
about the surrogate's error structure — which is the practical content of
Corollary 3.

---

## 5. Experiments

### 5.1 The main result

Reaction–diffusion hazard (`rdf`), grid 64, 20 seeds, δ = 0.1:

| arm | violating evaluations | objective |
|---|---|---|
| no margin | 14.1% | — |
| model-error margin | 18.2% | — |
| **probe + conformal margin** | **5.0%** | **no loss** |

Paired difference **−9.1 points, 95% CI [−11.8, −5.5], p = 0.0002**. The
objective is statistically indistinguishable: the margin is not buying safety
with performance, it is removing a bias.

An earlier draft reported this on 5 seeds with a z-value that did not survive
recomputation at the correct unit of analysis (evaluations, not runs). The 20-seed
rerun is the number above; §7 records the correction.

**Figure 3** `docs/figures/paper/fig2_rdf_violations.png`.

### 5.2 Three published corrections, and the theorem's prediction

All three are model-error measurable. All are implemented inside *our* planner so
the only difference is how model error is corrected for.

| method | violating | beats no margin? |
|---|---|---|
| no margin | 14.1% | — |
| conformal ensemble disagreement (CAP-style) | 11.99% | no |
| SMBPO (full-horizon cost + truncated pessimism) | 12.20% | no |
| SafeDreamer (cost critic) | 13.17% | no |
| **ours** | **5.00%** | — |

They land within 1.2 points of each other, and none separates from no margin.
**We pre-registered 12% for SafeDreamer before running it**; it returned 13.17%.
A prediction made from a theorem and confirmed on an unseen method is stronger
evidence than a baseline table.

Two of our own implementations were wrong first and are reported as such: SMBPO's
truncated prefix cost rescaled linearly on a cost that accrues non-linearly,
giving a −4.7% bias four times the pessimism term and making the arm net
*optimistic*; and SafeDreamer's cost critic shadowed the trainer's own `critic`
attribute, scoring the wrong head. Both failures initially pointed the way that
favoured us.

### 5.3 A second family, as a regression check

Diffusion–advection (`dar`): 0% → 0%. Nobody violates, so the margin cannot
help and must not hurt — it does not. This is a regression check, **not** a
second validation, and we say so rather than counting it.

### 5.4 A third family, and what "not rejected" means

Shallow water (`swe`) needed its cost limit calibrated against a *trained*
reward-greedy anchor rather than assumed: greedy 3.2647, do-nothing 0.0017,
limit **1.1437**. At that limit the constraint binds hard — realised cost reaches
**10× the limit** at worst — and over 20 seeds:

| arm | violating |
|---|---|
| baseline | 47.7% |
| margin alone (no probe) | 47.7% |
| probe | 33.2% |
| **probe + margin** | **16.8%** |

Paired **−16.4 points, 95% CI [−24.7, −8.0]**.

The margin-alone arm matching baseline to four decimal places is the design
working: with no probe there are no matched pairs, nothing to calibrate, and the
margin is correctly inert. It is the cheapest available evidence that the margin
reads its calibration set rather than a constant.

**The aggregate 16.8% is not a coverage measurement.** `effective_limit` — the
limit minus the margin — is exactly zero on 5 of the 20 seeds, where the margin
consumes the entire budget: the planner is instructed to spend nothing, and a
violation under that instruction measures the task's feasibility. Pooling
evaluations at 11 per seed:

| pool | seeds | violations | rate | 95% CI |
|---|---|---|---|---|
| all | 20 | 37/220 | 16.82% | [11.88, 21.76] — excludes δ |
| **feasible** | 15 | 20/165 | **12.12%** | **[7.14, 17.10] — contains δ = 0.1** |
| infeasible | 5 | 17/55 | 30.91% | [18.70, 43.12] |

So: `swe` is **infeasible at δ = 0.1 on a quarter of its seeds**, and where a
plan exists the realised rate is consistent with the level. Failing to reject δ
is weaker than attaining it, and 165 evaluations cannot resolve 10% tightly; a
60-seed extension is running to settle it. The honest summary across families:

> **The mechanism replicates on three PDE families. The stated rate is attained
> on one, trivially satisfied on a second, and not rejected on the third over the
> seeds where a plan exists.**

"Validated on three families" is not sayable and we do not say it.

### 5.5 The exchangeability assumption, tested where it fails

The guarantee rests on exchangeability of the calibration scores, and our scores
are collected *during a learning run*, which is a distribution shift by
construction. Every coverage result drawn i.i.d. tests the estimator and never
the assumption. Three seeds per testbed, 320 scores each, 160/160 split:

| testbed | drift (Spearman) | shift (KS) | **lag-1 dependence** | coverage | shuffled | adaptive |
|---|---|---|---|---|---|---|
| dar | p 0.47–0.74 | p 0.40–0.98 | p 0.47–0.60 | 0.0771 | 0.0875 | 0.1000 |
| rdf | p 0.16–0.90 | p 0.51–0.93 | **p = 0, all seeds** | 0.0854 | 0.0729 | 0.0959 |

**On rdf exchangeability is violated** — significant serial dependence, no drift,
no distribution shift: dependent but not drifting. **And coverage holds anyway**,
0.0854 against δ = 0.1. Dependence of this kind inflates the variance of the
empirical quantile without biasing it, which is why the beyond-exchangeability
bound (coverage ≥ 1 − δ − Δ) is the right frame and why the measured Δ does not
push miscoverage past δ. Adaptive conformal inference is therefore unnecessary
here — demonstrated, not assumed.

### 5.6 Where the margin does not help

On an evacuation-routing decision built on the same framework (§`WORKSHOP_PAPER`
§5), the route-level conformal margin is **exactly 0.000 m on all 96 decisions**.
The route the central forecast chooses is dry under every ensemble member, so the
residual of the reported scalar is identically zero: what differs between members
is not *how deep the chosen road gets* but **which roads exist**. The decision is
discrete and the margin is continuous, so there is nothing for it to bind on.

We report this because a feasibility theorem cuts both ways. Corollary 3 says
which distribution a margin must use; it does not say a margin exists for every
decision, and for discrete route choice under forecast uncertainty it does not.

---

## 6. Related work

Silent constraint violation under learned dynamics is named in safe RL; our
contribution is the *feasibility* statement about which distribution admits a
guarantee, and the measurement that exposes the failure. Conformal prediction
under distribution shift is addressed by adaptive conformal inference
(Gibbs & Candès) and by beyond-exchangeability bounds (Barber et al.); we use
the latter as the frame for §5.5 and show the former is not needed on these
tasks. SMBPO, CAP and SafeDreamer are the three concrete instances of the
excluded class we test. `RELATED_WORK.md` audits each claim against precedent,
including the places where our "novel failure" turned out to be documented.

---

## 7. What did not work, and what we retracted

Five original claims, none surviving in its original form: joint end-to-end
training (refuted on three families, decision never changed); the Eq. 8 mixing
rule (degenerate — measured `B² ≈ 0.0013`, selects the same α); Corollary 1's
asymptotic violation → 0 (false as stated on PDE testbeds); faithful natural
language briefs (withdrawn — a permutation control shows zero state-specific
information); cross-family transfer (direction only).

Three results retracted during development, **all three initially favouring us**:
a distillation number that was an artefact of a loss collapsing to "treat
nothing"; the SMBPO comparison invalidated by an extrapolation bias 4× the effect
it tested; a statistical claim whose interval, recomputed over the correct unit,
spanned its own effect size.

39 measurement defects are catalogued in `THESIS_AND_RESULTS.md` §7 with the
check that caught each, and 45 rules derived from them. The three most
instructive: a downsample that lost up to 59% of a fire's detections smoothly and
only for some fires; an interface displaying 90% confidence from six calibration
runs where nine are required; and a coverage rate averaged over configurations
in which no admissible plan existed, which moved a headline by four points.

---

## Figures

| # | file | shows |
|---|---|---|
| 1 | `docs/figures/paper/fig1_mechanism.png` | the dual's view vs the realised distribution |
| 2 | `docs/figures/paper/fig0_calibration.png` | measured violation against `R(ρ)`, δ-sweep inset |
| 3 | `docs/figures/paper/fig2_rdf_violations.png` | the 20-seed rdf result |
| 4 | `docs/figures/paper/fig3_precondition.png` | ρ across the sweep, crossover at 1 |
| 5 | `docs/figures/paper/fig5_tradeoff.png` | objective vs violation: the margin is not a trade |
| 6 | `docs/figures/paper/fig6_ablation.png` | recipe ablation across the four margin modes |

## Reproducibility

`eval/run_constraint_fix.py` (main result and all arms), `run_margin_coverage.py`
(closed form), `run_cap_baseline.py`, `run_smbpo_baseline.py`,
`run_safedreamer_baseline.py` (the three excluded-class methods),
`run_exchangeability.py` (§5.5). Cluster scripts in `scripts/tacc/`. Seeds,
cost limits and δ are command-line arguments in every case; no result in this
paper comes from a default we did not state.
