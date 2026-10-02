# What Uncertainty Must Be Calibrated When Planning Under Expectation Constraints?

*ICML main submission. This manuscript owns the **methodological** contribution
and nothing else. Full derivations are `THESIS_AND_RESULTS.md` Part 10, which
remains the source of record; the digital-twin systems contribution is a separate
paper (`WORKSHOP_PAPER.md`) and is referenced here, not reproduced.*

---

## Abstract

A planner optimising inside a learned model inherits that model's errors, and
under an **expectation** constraint it inherits them invisibly: the dual
controller is fed cost measured in the model while violation is realised under
true dynamics. The dual converges, reports the constraint satisfied, and the real
system violates it.

The natural correction — inflate the cost estimate by how wrong the model is —
cannot work, and we prove it. **No margin that is a measurable function of the
model-error distribution alone controls violation probability uniformly in the
spread of realised cost.** Realised violation is strictly increasing in that
spread and tends to ½ for symmetric error, so every such margin fails beyond a
computable threshold. For Gaussian error the failure is closed-form,
`R(ρ) = 1 − Φ(z_δ/ρ)` with `ρ = σ/ε`, predicting measured violation to within
**0.003** across two orders of magnitude in `ρ` and locating the transition at
`ρ = 1`.

The admissible correction follows from the same statement: calibrate the error
around **the quantity the dual actually controls**. A split-conformal quantile of
`c − ĝ` from a periodic probe of the real environment cuts violations on a
reaction–diffusion hazard from **14.1% to 5.0%** (20 seeds, paired −9.1 points,
95% CI [−11.8, −5.5], p = 0.0002) **at no cost in objective value**, and refuses
to return a bound when its calibration set cannot support the requested level.

Three published disagreement-based corrections — conformal ensemble disagreement,
SMBPO's pessimism, SafeDreamer's cost critic — land at 11.99%, 12.20% and 13.17%,
none better than applying no margin. That is the theorem's prediction rather than
three baselines that lost; we pre-registered 12% for SafeDreamer before
implementing it. We test the exchangeability assumption the guarantee rests on,
find it **violated** on one of two testbeds through serial dependence, and find
coverage attained regardless.

---

## 1. Introduction: a calibration mismatch

Constrained model-based planning is usually a Lagrangian: maximise reward,
penalise constraint cost, let a dual multiplier find the trade-off. It is correct
when the cost estimate is unbiased. Inside a learned surrogate it is not, and the
resulting failure is quiet rather than loud.

Measured on a reaction–diffusion hazard: the dual settles, surrogate episode cost
sits under the limit, every internal diagnostic reads correct, and **7.3% of
evaluations violate the true limit.** Nothing in a self-consistent training loop
exposes this, because the quantity the loop measures is the quantity the model
believes.

The obvious repair makes it worse. Inflating the cost estimate by the surrogate's
own error raises violations from **14.5% to 18.2%**. This paper explains why that
is structural, says exactly when it happens, and derives what to calibrate
instead.

**Contributions.**

1. A **separation result** (§3): no margin measurable from model-error
   uncertainty alone controls violation probability uniformly in realised-cost
   spread. The choice of distribution is a feasibility constraint, not a
   modelling preference.
2. A **closed form** for the failure, `R(ρ) = 1 − Φ(z_δ/ρ)`, validated to 0.003
   and locating the transition at `ρ = 1` (§3, §4).
3. The **admissible recipe**: conformalise the error around the dual-controlled
   quantity. 14.1% → 5.0% at no objective cost (§5).
4. **Three published methods shown to be instances of the excluded class**, with
   a pre-registered prediction confirmed on one we had not implemented (§5).
5. **The assumption tested where it fails** (§6), and a negative result: our own
   margin contributes nothing to a discrete decision (§8).

---

## 2. Problem formulation: two distributions, one of which matters

An episode under policy `π_φ` has realised cost `c` and surrogate estimate `g`;
the constraint is `E[c] ≤ d`. Decompose

```
c_i = μ + b + σ ξ_i ,   ξ_i ~ F , E ξ = 0, scale 1            (realised cost)
c_i − g_i = b − e_i ,   e_i ~ G , E e = 0, scale ε            (model error)
```

with `μ` the quantity the dual holds, `b` the surrogate's bias, `σ` the
**episode-to-episode spread of realised cost**, and `ε` the **scale of model
error**. The single dimensionless quantity is

```
ρ = σ / ε .
```

A margin `q` lowers the effective limit to `d − q`, so violation is
`σ ξ_i > q − b` and

```
R(q; σ) = 1 − F((q − b)/σ) ,     coverage at δ  ⟺  q ≥ b + z^F_δ σ .
```

The second line is the entire problem: **the margin must scale with `σ`.** Call a
rule **model-error measurable** if `q = Q(b, G)`, with no dependence on `F` or
`σ`. Every "inflate by model uncertainty" correction is of this type, however the
uncertainty is summarised — error bound, ensemble spread, learned variance,
pessimism bonus, cost critic.

**That `ε` and `σ` are different quantities is the paper's premise, and it is
measurable.** On `rdf`, `ρ` ranges **1.20 to 96.24** across 15 surrogate-quality
settings, exceeding 1 in every one — including a surrogate at relative `L2` 0.82,
worse than predicting zero. Surrogate accuracy and realised-cost spread are not
the same axis, and improving the first does not shrink the second.

---

## 3. Theory

**Proposition 1 (separation).** Let `Q` be model-error measurable and let `F` have
full support with positive density. Then `σ ↦ R(q; σ)` is strictly increasing;
`R → 1 − F(0)` as `σ → ∞`, which is ½ for symmetric `F`; and `R > δ` for every
`σ > (q − b)/z^F_δ`. Consequently, for any `δ < 1 − F(0)`,

```
sup_{σ>0} R(Q(b,G); σ) = 1 − F(0) > δ
```

so no model-error-measurable margin attains `δ` uniformly in `σ`.

*Proof.* `t(σ) = (q−b)/σ` is strictly decreasing and `F` strictly increasing, so
`R = 1 − F(t)` increases; `t → 0⁺` gives the limit; and `R > δ ⟺ t < z^F_δ`. ∎

Three lines, and that is the point: the missing object was the statement.
`THESIS_AND_RESULTS.md` §10.3 records which hypotheses do work — full support
gives strictness, symmetry only evaluates the limit, and `δ < 1 − F(0)` is not a
removable technicality.

**Corollary 1 (the matched-pair recipe, and where `ρ = 1` comes from).**
`q_mp = Q_{1−δ}(c − g) = b + z^G_δ ε`, which is model-error measurable, so
coverage fails iff `σ > (z^G_δ/z^F_δ) ε` — exactly `ρ > 1` when `F` and `G` share
a shape.

**Corollary 2 (closed form).** For Gaussian `F`, `G`:

```
R(ρ) = 1 − Φ( z_δ / ρ ) .
```

`R(1) = δ` identically for every `δ`: the recipe is **exactly** calibrated at the
crossover, and (Cor. 2) describes the whole departure from it. A consequence that
is counter-intuitive and testable: the *relative* error `R/δ` **rises as `δ`
tightens**, because margin and requirement scale by the same factor while the
tolerance shrinks. At `ρ = 1.4`, `R/δ` runs 1.18× at `δ = 0.30` to 3.56× at
`δ = 0.02`.

**Corollary 3 (necessity and sufficiency).** Any margin attaining `δ` uniformly
must depend on `F` and `σ`. The score `s_i = c_i − μ̂`, with `μ̂` the
dual-controlled quantity, has `Q_{1−δ}(s) = b + z^F_δ σ` — the coverage threshold
met with equality.

**The distinction that matters in implementation.** Subtracting the *per-episode*
surrogate estimate `g_i` gives `s_i = b − e_i` and recovers the inadmissible
recipe; subtracting the *controlled scalar* `μ̂` gives the admissible one. It is
one line of code and the whole difference.

**Finite sample.** With `n` exchangeable scores and rank
`k = ⌈(n+1)(1−δ)⌉`, coverage is `≥ 1−δ`, and `k ≤ n` requires

```
n ≥ 1/δ − 1 .
```

Below that no order statistic carries the guarantee and the estimator returns
**nothing** rather than extrapolating. `δ = 0.1` needs 9; with `n = 30` the
highest honest confidence is 96.8%.

**Beyond exchangeability.** Under a total-variation budget (Barber et al., 2023)
coverage is `≥ 1 − δ − Δ`. §6 measures the realised rate directly; we do not
claim a value for `Δ`, because miscoverage below `δ` is consistent with any
`Δ ≥ 0`.

---

## 4. Controlled mechanism experiment

**Figure 1 — the hero result.** Measured violation against `R(ρ)`,
`eval/run_margin_coverage.py`, δ = 0.1, 600 calibration draws per point:

| `ρ = σ/ε` | measured | predicted | error |
|---|---|---|---|
| 0.10 | 0.000 | 0.000 | 0.0000 |
| 0.50 | 0.004 | 0.005 | 0.0012 |
| 0.80 | 0.054 | 0.055 | 0.0006 |
| **1.00** | **0.099** | **0.100** | 0.0010 |
| 1.25 | 0.153 | 0.153 | 0.0004 |
| 2.00 | 0.258 | 0.261 | 0.0028 |
| 10.0 | 0.449 | 0.449 | 0.0000 |

Maximum error **0.003** over two decades in `ρ`, with the transition where the
theory puts it. The figure is the paper's centre: a predicted phase boundary,
measured.

`docs/figures/paper/fig0_calibration.png`, with the δ-sweep inset showing
`R/δ` rising as `δ` tightens.

---

## 5. Safe model-based planning

**The main result.** Reaction–diffusion hazard, grid 64, 20 seeds, δ = 0.1:

| arm | violating evaluations | objective |
|---|---|---|
| no margin | 14.1% | — |
| naive model-error margin | 18.2% | — |
| **probe + conformal (ours)** | **5.0%** | **no loss** |

Paired **−9.1 points, 95% CI [−11.8, −5.5], p = 0.0002**; objective
statistically indistinguishable. The margin removes a bias rather than trading
performance for safety. `docs/figures/paper/fig2_rdf_violations.png`.

An earlier draft reported this on 5 seeds with a z-value that did not survive
recomputation over the correct unit of analysis. The 20-seed rerun is the number
above.

**Three published corrections, all in the excluded class.** Each reimplemented
inside *our* planner, so the only difference is how model error is corrected for:

| method | violating | beats no margin? |
|---|---|---|
| no margin | 14.1% | — |
| conformal ensemble disagreement | 11.99% | no |
| SMBPO | 12.20% | no |
| SafeDreamer | 13.17% | no |
| **ours** | **5.00%** | — |

Within 1.2 points of one another, none separating from no margin. **We
pre-registered ≈12% for SafeDreamer before implementing it**; it returned 13.17%.
A prediction derived from Proposition 1 and confirmed on an unseen method is
stronger evidence than a baseline table, which is why that arm exists.

Two of our implementations were wrong first and both errors favoured us: SMBPO's
truncated prefix cost rescaled linearly on a non-linearly accruing cost, a −4.7%
bias four times the pessimism term that made the arm net *optimistic*; and
SafeDreamer's cost critic shadowing the trainer's own `critic`, scoring the wrong
head.

---

## 6. Cross-dynamics robustness, and the assumption

**Three families.**

| family | result |
|---|---|
| reaction–diffusion | 14.1% → 5.0%, level attained |
| diffusion–advection | 0% → 0%: nobody violates, so the margin cannot help and must not hurt — a **regression check**, not a validation |
| shallow water | limit calibrated against a *trained* reward-greedy anchor (1.1437); 37.4% → 21.5% over **80 seeds**, paired −15.9 points, 95% CI [−19.8, −12.1]; on the 52 of 80 seeds where a plan exists at δ = 0.1, **12.4%, 95% CI [9.4, 15.5]** |

On shallow water the margin consumes the entire limit on **28 of 80 seeds** — no
admissible action exists, and a violation there measures the task's feasibility
rather than the method's calibration. Pooled over all seeds the rate is 21.48%,
CI [18.76, 24.19], excluding `δ`; pooled over feasible seeds it is 12.41%,
cluster-robust CI [9.35, 15.47] with the seed as the unit of independence,
containing it. **A coverage rate may only be pooled over configurations in which
a plan exists**, and conflating the two moves this headline by nine points.

**The estimate is stable, and that is what the extra seeds bought.** At 20 seeds
the feasible pool read 12.12%, CI [7.14, 17.10] — an interval wide enough to
contain `δ` whatever the truth was. At 80 seeds it reads 12.41% with an interval
2.2× tighter. The point estimate moved 0.3 points across a 3.5× increase in
sample, so the containment is now informative rather than a statement about
power.

It is also no longer comfortable. The estimate sits 2.4 points above nominal and
a seed-level bootstrap puts `P(rate > δ) = 0.945`, just under the conventional
threshold. **We read the excess as the exchangeability gap rather than as noise**:
Barber et al. give coverage `≥ 1 − δ − Δ` beyond exchangeability, scores here are
collected during a learning run, and the measured `Δ̂ = 0.024`, 95% CI
`[−0.006, 0.055]`, is an estimate of precisely that quantity. The violations are
diffuse — 43 of 52 feasible seeds violate at least once, the five worst carrying
31% of the total — which is the signature of a distribution gap rather than of a
few pathological seeds.

> **The honest summary: the mechanism replicates on three families; the stated
> rate is attained on one, trivially satisfied on a second, and on the third is
> consistent with the level while trending above it by the amount the
> beyond-exchangeability bound predicts.** Failing to reject `δ` is weaker than
> attaining it. We do not write "validated on three families".

**Exchangeability, tested where it fails.** Scores are collected *during* a
learning run, which is a distribution shift by construction. Three seeds per
testbed, 320 scores, 160/160 split:

| testbed | drift (Spearman) | shift (KS) | lag-1 dependence | miscoverage | shuffled | adaptive |
|---|---|---|---|---|---|---|
| dar | p 0.47–0.74 | p 0.40–0.98 | p 0.47–0.60 | 0.0771 | 0.0875 | 0.1000 |
| rdf | p 0.16–0.90 | p 0.51–0.93 | **p ≈ 0, all seeds** | 0.0854 | 0.0729 | 0.0959 |

On `rdf` exchangeability is **violated** — significant serial dependence with
drift and distribution-shift tests both non-rejections: *dependent, not
drifting*. Coverage is attained anyway, 0.0854 against δ = 0.1. Dependence of
this kind inflates the quantile's variance without biasing it, which is the
mechanism, and is why adaptive conformal inference is unnecessary on both
testbeds — demonstrated rather than assumed.

---

## 7. One application: planning through a learned hazard model

Included to show the framework matters beyond synthetic PDE control, and
deliberately compressed; the systems contribution is a separate paper.

WildfireSpreadTS, **607 observed fires**, learned spread dynamics, decision-time
planning of crew allocation under a daily budget.

**The falsification experiment is the result worth the space.** At a one-day
horizon a greedy allocation is optimal and a planner should buy nothing.
Burned-area reduction, planner against operational heuristic:

| horizon | heuristic | planner | gap | t |
|---|---|---|---|---|
| 1 day | 24.8 ± 1.1 | 25.4 ± 1.7 | **+0.6** | **1.40 — n.s.** |
| 2 days | 24.2 ± 3.6 | 32.6 ± 3.4 | +8.5 | 10.74 |
| 3 days | 22.0 ± 4.4 | 34.1 ± 4.4 | +12.1 | 11.08 |
| 5 days | 17.7 ± 4.3 | 32.0 ± 5.8 | **+14.3** | 8.99 |

The advantage **vanishes exactly where theory requires it to** and grows with the
horizon over which sequential reasoning can act. A result that survives its own
falsification test is worth more than one never exposed to it.

An earlier version of this claim was measured on 8 fires at t = 3.0. At 607 fires
the benefit is **a third the size** and holds on 76% of them.

`docs/figures/paper/fig4_wildfire.png`.

---

## 8. Limitations, and where our own margin does not help

- **The guarantee is marginal, not per-step.** It bounds the probability that a
  trajectory's total cost exceeds the limit, not cost at every timestep.
- **Exchangeability fails on one of two testbeds**, with coverage attained
  regardless (§6). We report the failure rather than the easier sentence.
- **`ρ > 1` throughout the reachable regime.** The region where the matched-pair
  recipe is correct is not reachable on these testbeds, so the contribution is
  better stated as a correction than as a diagnostic for choosing recipes.
- **The intervention arm of §7 is unvalidated in reality.** No fire in the
  archive had a firebreak cut on our instruction, and no retrospective dataset
  can fix that.
- **Our margin contributes nothing to a discrete decision.** On an evacuation
  routing problem built on the same framework, the route-level conformal margin
  is **exactly 0.000 m on all 96 decisions**: the uncertainty lives in *which
  roads exist* rather than in the continuous scalar being reported, and a
  margin on a continuous scalar cannot represent variation in a discrete
  feasible set. A feasibility theorem cuts both ways — Corollary 3 says which
  distribution a margin must use, not that a margin exists for every decision.
  (Developed in the systems paper.)
- **Five original claims of our own did not survive**, three results were
  retracted during development — all three initially favouring us — and 41
  measurement defects are catalogued with the check that caught each.

---

## Reproducibility

`eval/run_constraint_fix.py` (main result, all arms, all three families),
`run_margin_coverage.py` (§4), `run_cap_baseline.py`, `run_smbpo_baseline.py`,
`run_safedreamer_baseline.py` (§5), `run_exchangeability.py` (§6),
`run_ndws_plan.py` (§7), `agg_margin_seeds.py` (pooling with the feasibility
split). Cluster scripts in `scripts/tacc/`. Seeds, cost limits and `δ` are
command-line arguments in every case.

Full derivations with the hypotheses each step needs, and a table mapping every
number here to the script that produced it: `THESIS_AND_RESULTS.md` Part 10.
