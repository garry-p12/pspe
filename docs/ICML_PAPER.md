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
Against four published model-free safe-RL algorithms **at a matched real-sample
budget** it returns more than **all four** (Figure 11), seed-paired with every
interval excluding zero; given 4.6× more data than us they close most of that gap and one
of them draws level — so the sample-efficiency claim is stated as a quantity,
not a ratio. It **violates more often than any of them**, which we report as the
trade rather than a caveat: three reach 0% by operating at a quarter of the
limit, below a do-nothing policy's own cost. Every arm is anchored to that zero
policy, because a 0% violation rate is what inaction scores.

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
theory puts it — in a controlled CMDP where `σ` and `ε` are set by hand.

**And on a learned planner?** We now measure both quantities on the real
planner, at every probe, beside an arm running the excluded recipe. 20 seeds,
reaction–diffusion:

| quantity | measured |
|---|---|
| `σ_ep` — spread of realised episode cost | **0.454** |
| `ε` — spread of the matched-pair model error | **0.0265** |
| `ρ = σ/ε` | **median 17.9**, range [7.4, 91.1], **> 1 on 20/20 seeds** |

**The separation result's precondition holds by an order of magnitude**, and the
mechanism is visible: the matched pair `c − ĝ` is taken on common random numbers,
so the policy's own spread cancels and `ε` collapses. The excluded recipe's
margin is consequently **0.059 of budget where ours applies 0.583 — 9.9× too
small** (paired difference +0.5245, CI [+0.4766, +0.5768]), and it recovers only
about an eighth of what ours does: 10.45% violating against probe-only's 10.91%,
and **exactly identical to probe-only on 11 of 20 individual seeds**.

**But the closed form does not transfer.** `R(ρ)` predicts **0.470** at the
measured `ρ`; the arm violates **0.105** — a **4.5× over-prediction**. `R(ρ)` is
derived for a dual that drives `ĝ` exactly to `d_eff` and tends to ½ as `ρ → ∞`;
a real planner's rate is bounded above by its own no-margin rate (14.09%),
because a negligible margin degenerates the arm toward the unmargined planner
rather than toward the asymptote.

> **The theorem's conclusion and its closed form transfer independently.** The
> qualitative statement — *this statistic is insufficient, so a margin built from
> it must fail* — survives contact with a learned planner and is confirmed by a
> margin 10× too small. The curve, exact to 0.003 where its assumptions hold, is
> an **upper bound the real system does not approach**. We report it as such.

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

**Anchored to a do-nothing policy.** "No loss in objective" compares the treated
arm to the untreated one, which says nothing about whether either controls the
system. We therefore evaluate the **zero policy** under the identical protocol —
same grid, same 8 evaluation episodes, same initial conditions per seed — and
pair every arm against it on its own seed:

| arm | real transitions | return | over do-nothing | 95% CI | violating |
|---|---|---|---|---|---|
| do-nothing | **0** | −7.400 | — | — | **0.0%** |
| no margin | 6,400 | −6.822 | +0.578 | [+0.514, +0.651] | 14.1% |
| **probe + conformal** | **8,320** | **−6.900** | **+0.501** | [+0.451, +0.556] | **5.0%** |

The zero policy never violates — a 0% violation rate is what *inaction* scores,
so it certifies nothing on its own, and every constraint-side number in this
paper should be read against that floor.

**Against model-free safe RL, at their budget and ours.** Four published
algorithms, each given **38,400** real transitions against our 8,320, 20 seeds,
paired against the same null:

| method | real transitions | return | over null | violating | worst cost / limit |
|---|---|---|---|---|---|
| Sauté RL | 38,400 | −7.253 | +0.147 | 0.0% | 0.25× |
| CPO | 38,400 | −7.198 | +0.202 | 0.0% | 0.30× |
| primal-dual NPG | 38,400 | −7.199 | +0.201 | 0.0% | 0.30× |
| PPO-Lagrangian | 38,400 | −6.970 | +0.430 | 2.3% | 1.22× |
| **ours** | **8,320** | **−6.900** | **+0.501** | **5.0%** | **1.54×** |

Seed-paired on return: **+0.354 over Sauté RL [+0.286, +0.427]**, **+0.300 over
primal-dual NPG [+0.237, +0.361]**, **+0.299 over CPO [+0.236, +0.361]** — all
excluding zero, at **4.6× fewer real transitions**. Against PPO-Lagrangian the
difference is **+0.070, CI [−0.039, +0.167]: not resolved.**

**At a matched real-sample budget we beat all four.** A sample-efficiency claim
is a comparison, not a ratio, so we also cap each baseline at the planner's own
real-transition count (~8,300) and re-run, 20 seeds, same pairing:

| method | return | over null | seed-paired vs ours | violating |
|---|---|---|---|---|
| **ours** | **−6.899** | **+0.501** | — | 5.0% |
| Sauté RL | −7.293 | +0.107 | **+0.394** [+0.318, +0.472] | 0.0% |
| PPO-Lagrangian | −7.316 | +0.084 | **+0.417** [+0.358, +0.476] | 0.0% |
| CPO | −7.360 | +0.041 | **+0.460** [+0.407, +0.513] | 0.0% |
| primal-dual NPG | −7.360 | +0.041 | **+0.460** [+0.407, +0.513] | 0.0% |

![Matched budget](docs/figures/paper/fig11_matched_budget.png)

*Figure 11. Both families at a matched real-sample budget, 20 seeds,
seed-paired, 20,000-resample bootstrap. The dashed line is the zero policy. The
percentage beside each interval is that arm's violation rate — on (b) three of
four baselines cannot be separated from inaction, and all four reach 0%
violation by staying far from the limit.*

**Every interval excludes zero**, and the one baseline we could not separate from
at full budget we separate from at matched budget: PPO-Lagrangian goes from
+0.070 [−0.039, +0.167] to **+0.417 [+0.358, +0.476]**.

> **Stated as a quantity rather than a ratio: PPO-Lagrangian needs 4.6× our real
> transitions to draw level with us.**

**On diffusion–advection at matched budget, no baseline clears the null floor at
all** — CPO +0.0052 [−0.0009, +0.0111], Sauté +0.0066 [−0.0021, +0.0158],
primal-dual NPG +0.0052 [−0.0039, +0.0226], PPO-Lagrangian **+0.0001**
[−0.0091, +0.0088] — while ours returns +0.0866 on 20/20 seeds, **13× the best
baseline effect.** We registered the prediction "safe but not learning" before
that run; all four came back at 0% violation and at the null.

**We are the least safe arm in that table, and we state it as the result rather
than a footnote.** Three of the four reach 0% by operating far from the
constraint — Sauté's mean episode cost is 0.533 against a limit of 3.26, *below
the do-nothing policy's 0.565*, with a worst case at a quarter of the limit.
That conservatism is what costs them 0.30–0.35 of return, and it is the trade
this paper's margin makes in the opposite direction. PPO-Lagrangian is the one
baseline that uses its budget (mean cost 1.703) and it is the one we do not
separate from.

**The margin reduces how often the limit is exceeded, not how badly.** Our worst
excursion over the run is 1.54× the limit against the untreated arm's 1.55×,
while the violation *rate* falls from 14.1% to 5.0%. That is exactly what a
guarantee on the expected episode cost promises and no more, and it means the
method is not a substitute for a per-episode safety mechanism. The margin
already consumes 17.9% of the budget at δ = 0.1 (effective limit 2.677 against
3.260), and **a smaller δ does not buy a lower rate**. Swept over 20 seeds:

| δ | violating | return over null | margin takes |
|---|---|---|---|
| 0.10 | 5.00% | **+0.501** | 18% of budget |
| **0.05** | **4.09%** | +0.474 | 24% |
| 0.02 | 5.00% | +0.420 | 33% |
| 0.01 | 5.91% | +0.335 | **50%** |

The rate is **non-monotone in δ**, with an interior minimum at δ = 0.05 that
**dominates δ = 0.02 and δ = 0.01 on both axes**. Over a tenfold range it spans
4.09–5.91% and never approaches 0%, while return falls monotonically and the
worst excursion stays pinned at 1.54–1.55×. **δ sets the margin's width, not how
often the limit is crossed**, so this method should not be offered as a safety
dial. The comparison above does not depend on it being one: +0.335 at δ = 0.01,
with half the budget surrendered, still exceeds every baseline's gain at 4.6×
the data.

**A testbed caveat that changes which comparison is meaningful.** On our
diffusion–advection testbed, three of these four algorithms **cannot be
distinguished from the do-nothing policy** over 20 seeds at full budget — CPO
+0.012 [−0.001, +0.024], PPO-Lagrangian +0.014 [−0.005, +0.034], primal-dual NPG
+0.010 [−0.004, +0.023], with only Sauté clearing it (+0.028 [+0.022, +0.034]).
The zero policy is near-optimal there, so a comparison on that family measures
very little; the table above is on reaction–diffusion, where all four clear the
floor by 0.15–0.43 and are genuine opponents. We report the weaker testbed
rather than omitting it.

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

**Exchangeability, tested where it fails — 20 seeds per testbed.** Scores are
collected *during* a learning run, which is a distribution shift by
construction. 320 scores per seed, 160/160 split.

| testbed | drift (Spearman) | shift (KS) | **lag-1 dependence** | coverage | shuffled | adaptive |
|---|---|---|---|---|---|---|
| diffusion–advection | rejects 0/20 | rejects 0/20 | rejects **0/20** | 0.1125 | 0.1134 | 0.1013 |
| reaction–diffusion | rejects 5/20 | rejects 1/20 | **rejects 20/20**, median p = 0.00175 | **0.0997** | 0.0944 | 0.0972 |

**The violation is specific, and naming it matters.** On reaction–diffusion the
lag-1 permutation test rejects on **every seed** while the drift test rejects on
5 and the distribution-shift test on 1: the scores are **dependent, not
drifting**. Three tests disagreeing by an order of magnitude in rejection rate
means "exchangeability fails" would have been true and nearly useless.

**Coverage is attained anyway — 0.0997 against δ = 0.100 — on the family where
the assumption is rejected on 20 of 20 seeds.**

**And there is a mechanism rather than a coincidence.** The lag-1
autocorrelation is **negative on 20 of 20 seeds**, mean **−0.181**, range
[−0.246, −0.119]. Negatively correlated scores *alternate* rather than trend, so
a sample spans its range faster than an i.i.d. one and the empirical order
statistic is, if anything, conservative. **Exchangeability is violated in the
direction that does not hurt.**

That is falsifiable, and we say how: a family whose scores are *positively*
serially correlated should degrade coverage. Gibbs–Candès adaptive conformal does
not improve the mean here, but it does tighten the spread across seeds
([0.0875, 0.1187] against the fixed arm's [0.0563, 0.1812]) — so it is the right
instrument where the sign of the dependence is unknown, which is the honest
recommendation rather than "unnecessary".

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

**The 3-day comparison at 20 seeds**, re-run because 5 seeds (t = 11.15, df 4)
was the thinnest evidence behind any headline here. Same runner and horizon; the
1,500 held-out fires are a fixed archive, so the seed varies initialisation over
a fixed test set rather than resampling evaluation conditions.

| policy | burn reduction | 95% CI | over budget |
|---|---|---|---|
| none | 0.00% | — | 0 |
| random placement | 4.32% | [4.11, 4.52] | 0 |
| greedy — forecast, treat riskiest, repeat | 22.51% | [21.19, 23.75] | 0 |
| **planner, per-instance** | **32.50%** | [31.30, 33.64] | **0** |
| planner, amortised policy | 16.92% | [12.41, 21.48] | **0.371** |
| unconstrained (reference) | 88.85% | [87.35, 90.22] | 1.000 |

Seed-paired against the heuristic: **+9.99 points, CI [+9.15, +10.81],
t = +23.0, 20/20 seeds improved.**

**The effect is smaller than the 5-seed figure and far better established** —
+12.0 → +9.99 with t rising 11.15 → 23.0. `random` (4.32 against 4.5) and
`greedy` (22.51 against 22.0) both reproduce, so the drift is in the treated arm:
five seeds sampled a favourable tail. We report the 20-seed number.

**And the amortised policy loses to the heuristic it was meant to replace** —
16.92% against 22.51%, while exceeding the budget on 37% of evaluations. We
include it because it is the fourth independent case in this work of a learned
component losing to its non-learned alternative, and the pattern belongs in the
paper rather than in a reader's inference: **the contribution is the calibration,
not the learning.**

An earlier version of this claim was measured on 8 fires at t = 3.0. At 607 fires
the benefit is **a third the size** and holds on 76% of them.

`docs/figures/paper/fig4_wildfire.png`.

---

## 7a. A second silent failure: a dual that is working and cannot bind

The failure this paper corrects is a dual fed the wrong *measurement*. A
distinct failure, found while porting the planner to a flood-control task, is a
dual fed the right measurement and attached to a policy that can no longer
respond. It is worth a section because the two look identical from the dual's
own logs, and because the second is invisible to every diagnostic the first one
motivates.

**Setup.** A polder gate and pump station on real terrain, budget 2.0
pump-hours over a twelve-interval episode (§5.14y of the record). The dual is
the same PID controller used throughout this paper.

**What happens.** Over 150 iterations the policy's pre-squashing mean grows
without bound, reaching `|μ| ≈ 1.4 × 10³`. Past `|μ| ≈ 3` the `tanh` is flat to
machine precision, so `∂a/∂μ ≈ 0` and *every* gradient through the policy
vanishes — including the constraint gradient the dual multiplies. Episode cost
pins at the action ceiling of 6.00 pump-hours from iteration 50 and never moves
again. The multiplier grows **exactly linearly** thereafter, 0.2 per iteration,
because the constraint error is frozen at a constant 4.0: an integral term
winding up against a quantity that cannot change.

![Saturation](docs/figures/paper/fig13_saturation.png)

*Figure 13. The same planner with a saturation penalty disabled (the package
default) and enabled. (a) the policy leaves the region where its own squashing
function has slope; (b) cost pins at the ceiling, far outside the budget;
(c) the multiplier winds up linearly against a quantity that can no longer
respond. With the penalty enabled the policy settles at `|μ| ≈ 1.9`, cost
enters the budget at 1.62, and the multiplier rises and then releases.*

**Why it matters for the method.** Our adaptive mixing weight `α` is chosen to
minimise estimator variance. In the saturated regime the pathwise branch has a
gradient of zero and therefore a *variance* of zero, which the rule reads as
perfect precision: `α → 0.979`. The estimator confidently selects a branch that
carries no signal. We measured this and it is not a tuning accident — it is
what a variance-minimising rule must do when handed a dead branch, and any
method that selects between estimators by variance alone inherits it.

**The honest account of how we found it.** Two configurations — truth dynamics
with `α` pinned to 0, and surrogate dynamics with `α` adaptive — returned
`−32.009300231933594` and `−32.009300231933594`. Identical to thirteen
significant figures is not a coincidence; both policies had reached the same
action-space corner, where the evaluation is the same trajectory regardless of
the gradient that produced it. The remedy, a quadratic penalty on `relu(|μ| −
1.5)`, was already implemented in our own codebase, documented with this exact
failure mode, and disabled by default because the first testbed never needed
it. **Every flood-domain number we had produced ran with it off.**

**What it does not buy.** With the penalty enabled the planner is feasible and
still loses to a level-deadband controller already in operational use (−60.62
against −41.50 at a third of the energy). The mechanism is repaired; the
comparison is not won. We report both.

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
- **A working dual does not imply a binding constraint.** §7a: a saturated
  policy makes every gradient vanish, so the multiplier can grow without limit
  while the cost it controls does not move. Diagnosing this needs the policy's
  pre-squashing magnitude, which no constraint-satisfaction metric reports.
- **The repaired planner still loses on that task.** On the flood polder a
  level deadband beats every learned method we ran, ours included. We state it
  as a result about the task rather than a defect of the method: sweeping a
  one-threshold controller over its whole range yields **three** reachable
  behaviours, with a cost band of 2.3× over which the outcome does not change
  at all, so the achievable gap for any learner is small and discontinuous.
- **Our margin contributes nothing to a discrete decision.** On an evacuation
  routing problem built on the same framework, the route-level conformal margin
  is **exactly 0.000 m on all 96 decisions**: the uncertainty lives in *which
  roads exist* rather than in the continuous scalar being reported, and a
  margin on a continuous scalar cannot represent variation in a discrete
  feasible set. A feasibility theorem cuts both ways — Corollary 3 says which
  distribution a margin must use, not that a margin exists for every decision.
  (Developed in the systems paper.)
- **The margin bounds the rate, not the severity.** Violations fall from 14.1%
  to 5.0% while the worst excursion over the run is unchanged at 1.54× the
  limit against 1.55%. Anything needing a bounded worst case needs a different
  mechanism.
- **Every baseline we compare against satisfies the constraint more often than
  we do** (§5): 0.0% for three of four, 2.3% for the fourth, against our 5.0%.
  Our advantage is return per real transition, not safety, and a deployment that
  requires zero violations should not use this method at δ = 0.1.
- **A do-nothing policy is a competitive baseline on one of our testbeds.** On
  diffusion–advection three of four published safe-RL algorithms cannot be
  distinguished from inaction over 20 seeds at full budget. We report it because
  it means a comparison on that family measures almost nothing, including ours.
- **The explanation module's own mechanism is net harmful.** With the brief
  length cap lifted from 96 to 160 tokens — the reference briefs average 136,
  so every arm had been truncated before its footer — a supervised generator
  reaches the parser's ceiling (F = 0.901 against a reference of 0.901) while
  adding our faithfulness objective *drops* it to 0.505, on all three seeds and
  at 2.3× the training cost. Prompting an off-the-shelf 0.5B instruction model
  with no training scores 0.353, statistically the same as our generator trained
  on a different policy's briefs (0.350). No arm produces state-specific briefs:
  the aligned-minus-shuffled permutation gap is within ±0.006 everywhere. We
  report the faithfulness term as a measured negative and do not include it in
  any headline configuration.
- **One safe-RL baseline's own correctness canary fails on 1 seed in 3.** Sauté
  RL collapses to the action boundary on a closed-form CMDP for seed 0,
  returning −3.24 where −0.16 was available; the cost assertion passes there
  because the collapsed policy spends nothing. It is not collapsed on either PDE
  family. We report it as validated on 2 of 3 canary seeds.
- **Dispersion from one sweep is not seed variance.** An ablation driver passed
  its `--seed` only to the global RNG, so five "seeds" shared one evaluation
  initial-condition set; intervals from that sweep measure initialisation
  variance and are five to six times too narrow. The contrasts stand, the ±
  does not, and every interval quoted in this paper comes from drivers that pass
  the seed.
- **Five original claims of our own did not survive**, three results were
  retracted during development — all three initially favouring us — and **54
  measurement defects** are catalogued with the check that caught each.

---

## The learned model, named

The paper's claims are about what a dual is fed, not about an architecture, but
"planning through a learned surrogate" should say which one. A **Fourier Neural
Operator** throughout, selected against two alternatives on the planning testbed
at 20 seeds:

| operator | `rel L2` (final) | sd | 95% CI |
|---|---|---|---|
| **FNO** | **0.2168** | 0.1324 | **[0.1625, 0.2760]** |
| DeepONet | 0.3090 | 0.0042 | [0.3072, 0.3108] |
| GNOT | 0.4325 | 0.0824 | [0.4005, 0.4697] |

FNO's upper bound sits below DeepONet's lower bound, so the ordering is
established rather than assumed. **FNO is also far more seed-sensitive** — sd
0.132 against DeepONet's 0.004 — which a single-seed comparison would have
hidden in either direction, and which matters here because `ρ` is a property of
the surrogate as much as of the policy.

On the published **PDEBench 2-D shallow water** benchmark the same model reaches
`rel L2` 0.0018 one-step and 0.0098 on rollout, against a published FNO nRMSE of
0.0044 — an anchor rather than a like-for-like comparison, since the metrics
differ, and single-seed.

**The physics-informed loss is reported as buying convergence speed only** — no
asymptotic effect, −0.0127 at 20 seeds with CI [−0.0886, +0.0650]. We include
the null because the term is cheap to add and widely assumed to help.

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
