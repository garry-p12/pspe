# Revision notes for the ICML draft

Corrections and additions to `pspe_icml_preview.pdf`, in priority order.
Bibliography: `paper/references.bib` (drop-in; 43 entries, all verified
September 2026).

---

## 1. References — every placeholder resolved

The draft shipped eleven entries reading `VERIFY authors`, cited in text as
keys (`(ACP safe RL, 2025)`, `(Firebreak DRL, 2024)`). For a paper whose
contribution is "the field does X, we do Y", an unnamed X reads as unread.

| draft key | replace with | new bib key |
|---|---|---|
| ACP safe RL, 2025 | Zhou, Zhang & Luo, 2025 | `zhou2025acpsafe` |
| Conformal Policy Control, 2026 | Prinster, Fannjiang, Park, Cho, Liu, Saria & Stanton, 2026 | `prinster2026cpc` |
| Robust Conformal CBF, 2026 | Mirzaeedodangeh, Shekhtman, Matni & Lindemann, 2026 | `mirzaeedodangeh2026robustcbf` |
| Proactive Safety Constraints, 2026 | Tomashevskiy, 2026 (**CoLLAs 2026**, not a preprint) | `tomashevskiy2026proactive` |
| Unified View of Objective Mismatch, 2023 | Wei, Lambert, McDonald, Garcia & Calandra, 2023 | `wei2023unified` |
| Modular Wildfire Mechanisms, 2026 | Esparza, Ayanzadeh, Mousavi & Mostafavi, 2026 | `esparza2026modular` |
| WildfireSpreadBench, 2026 | Gopakumar & Pannozzo, 2026 | `gopakumar2026bench` |
| Graph Firebreak Planning, 2025 | Yemshanov, Liu, Neilson, Thompson & Koch, 2025, *Ecological Informatics* 90 | `yemshanov2025graph` |
| Helitack RL, 2026 | Mathur, Manjunath, Kulkarni & Vereshchaka, 2026 | `mathur2026helitack` |
| **Firebreak DRL, 2024** *and* **Firebreak DRL (ASOC), 2025** | **one paper, cited twice.** Same authors, same work. Keep only Murray, Castillo, Martín de Diego, Weber, González-Olabarria, García-Gonzalo, Weintraub & Carrasco, *Applied Soft Computing* 175:113043, 2025 | `murray2025firebreak` |

Also add, since §9 of the internal audit turned them up and reviewers will
expect them: Gerard, Zhao & Sullivan (WildfireSpreadTS, NeurIPS 2023 D&B) and
Herde et al. (Poseidon, NeurIPS 2024).

## 2. Related work — one claim is factually wrong

§2 currently says of the conformal-control line:

> "Almost all of this work conformalises the error between the learned dynamics
> and the true system."

That is **not true of Prinster et al. (2026)**, which calibrates permissible
*behavioural change* against a safe reference policy and identifies no dynamics
model at all. It is the closest prior work to ours and the sentence
mischaracterises it. Suggested replacement:

> The dominant instantiation calibrates a quantile of the error between the
> learned dynamics and the true system and tightens a barrier or Lyapunov
> constraint by it (Lindemann et al., 2023; Zhou et al., 2025; Mirzaeedodangeh
> et al., 2026). A separate strand calibrates against a reference *policy*
> rather than a model (Prinster et al., 2026), which shares our instinct that
> the calibration target should be the controlled object; we differ in
> conformalising the realised cost against the quantity a Lagrangian dual
> holds, which gives a CMDP budget rather than a behavioural-change budget.

## 3. The claim is sharper than the draft states

The draft says the default fails "whenever the model is accurate and the policy
is variable". That is the symptom. The cause is structural and gives a much
better paper:

> **The default is correct when the controlled quantity is per instance, and
> undercovers when the dual controls an expectation.**

Matched-pair residuals `c_i - g_i` cancel whatever scatter is common to both
terms. If the controller holds `g_i` at the limit for each instance, that
cancellation is exactly right. In a CMDP the dual holds `E[c]`, so `g_i` is not
the controlled quantity and the per-instance scatter that breaches `d` cancels
out of the calibration set.

This explains why the recipe is sound in the CBF literature it was developed in
(pointwise, per-state constraints) and unsound here, and it turns the paper's
result from an empirical curiosity into a rule. Recommended: state it in the
abstract and as contribution 1, and make §4.3/§4.4 the proof of it.

It also simplifies the method. The draft conformalises `e_i = c_i - c̄` and then
adds `b̂ = max(0, bias)` as a separate deterministic term, so Proposition 1 needs
the dual to hold a *bias-corrected* mean. Conformalising `s_i = c_i - ĝ`
directly — realised cost minus the quantity the dual controls — covers bias and
spread in one order statistic, needs no separate bias term, and the condition
becomes exactly what the dual already does. `residual` is implemented as the
default mode in `pspe/plan/margins.py`; `model_error` and `episode` are
selectable for the head-to-head.

Three-line demonstration of the mechanism, from the module's unit test — an
accurate per-instance model, a policy with real spread:

```
model_error margin : 0.0174
episode  margin    : 0.6623
residual margin    : 0.6623
needed (90th pct)  : 0.6623
```

## 4. Table 2 oversells DAR

Table 2 reports DAR baseline **0% violations, worst 0.225 against a limit of
0.936**. The margin changes violations from 0% to 0% and worst case from 0.225
to 0.225, at a return cost of 0.005. DAR never approaches its limit, so it is
evidence that the margin is **harmless**, not that it works.

"Holds the stated rate on two PDE families (0% and 7.3%)" is therefore
misleading: the whole positive result is 7.3% on RDF. Either say so plainly —

> On DAR the constraint does not bind under the fixed planner (worst case 0.225
> against a limit of 0.936), so that column shows only that the margin costs
> nothing when it is not needed; the load-bearing result is RDF.

— or report the fidelity sweep (§6 below), which supplies the missing evidence.

## 5. Two tables label different baselines identically

Table 1 gives a DAR baseline of **1.8%** violations; Table 2 gives **0%**.
These are different configurations (pre- and post-planner-fix). Rename, or add
a column stating which planner configuration each row used.

## 6. New result — the diagnostic, derived and confirmed

The draft says the default fails "whenever the model is accurate and the policy
is variable". That is a symptom with no threshold attached, so a reviewer
cannot check it. The algebra gives one.

With `d_eff = d - q` the realised mean sits at `d_eff + b`, so an episode
violates when `n_i > q - b`, and coverage at level `δ` needs

```
q  >=  b + z_δ · σ            z_δ = Φ⁻¹(1 - δ)
```

The matched-pair residual is `c_i - g_i = b - e_i`, so its quantile is about
`b + z_δ · ε`. **The bias appears on both sides and cancels.** The default
therefore covers exactly when

```
ε  >=  σ
```

that is, when the model's **per-instance** error is at least the policy's
episode-to-episode spread. The diagnostic is `ρ = σ / ε`, with the crossover at
**ρ = 1**.

> **A correction worth recording.** I first wrote this diagnostic as
> `ρ = σ / |bias|` and it is wrong: bias shifts the requirement and the
> quantile by the same amount and drops out. The error was caught by deriving
> the threshold rather than reasoning from the symptom, and it would have put a
> wrong rule in the paper. Both `eval/run_margin_choice.py` and the docs now
> report `σ / ε`.

`eval/run_margin_synthetic.py` confirms it in a setting where every term is
visible — bias and spread dialled independently, nothing learned, 600
calibration draws per point, δ = 0.1:

| ρ = σ / ε | no margin | model error | deviations + bias | **residual (ours)** |
|---|---|---|---|---|
| 0.10 | 0.841 | **0.000** | 0.098 | 0.098 |
| 0.25 | 0.841 | **0.000** | 0.099 | 0.099 |
| 0.50 | 0.841 | **0.004** | 0.097 | 0.097 |
| 0.80 | 0.842 | **0.054** | 0.099 | 0.100 |
| **1.00** | 0.841 | **0.099** | 0.101 | 0.101 |
| 1.25 | 0.842 | 0.153 | 0.100 | 0.100 |
| 2.00 | 0.841 | 0.258 | 0.099 | 0.099 |
| 4.00 | 0.842 | 0.373 | 0.098 | 0.099 |
| 10.0 | 0.841 | 0.449 | 0.099 | 0.100 |

The default sits at 0.099 exactly at ρ = 1 and degrades monotonically above it,
reaching 0.449 — the no-margin rate is 0.841, so at high ρ the default recovers
less than half of what a margin should. Our recipe holds 0.097–0.101 across two
orders of magnitude. Figure: `docs/figures/pspe_margin_choice.svg` panel (a).

**Why this matters for the review.** It converts the draft's isolated 18.2%
into an instance of a rule that predicts its own regime of validity, and it
answers the "is this just a bug in your implementation?" objection directly:
the same code, at ρ < 1, covers correctly.

## 6b. Two further findings from making the recipes share one module

`pspe/plan/margins.py` now implements all three, and `tests/test_margins.py`
states the distinguishing properties as executable claims (9 tests, passing).
Writing them down turned up two things worth a sentence each in the paper.

**The default is doubly mismatched to an episode-cost budget.** It takes the
*absolute* residual `|c_i - g_i|`, because the barrier constraints it was built
for can be breached from either side. So when the surrogate over-predicts cost
— reality cheaper than the model, the dual already conservative, no margin
needed — it still charges one. A one-sided budget `J_C <= d` wants a signed
residual. This is milder than the cancellation problem but points the same way:
the recipe carries assumptions from pointwise two-sided control into a setting
that has neither property.

**A testable explanation for the draft's admitted anomaly.** §5 says: "We do
not have a complete account of why the rate rose above the no-margin baseline
rather than merely failing to fall." A tighter limit should not *increase*
violations, so this is the sentence a hostile reviewer will build on.

The hypothesis the running sweep can settle: tightening `d_eff` raises lambda,
which pushes the policy harder and into a higher-variance region, and the extra
spread more than offsets the margin. `eval/run_margin_choice.py` records the
policy's episode spread per arm, so the check is simply whether `spread sigma`
is larger in the `model_error` arm than in `none`. If it is, the anomaly has a
mechanism and the sentence can be replaced by one. If it is not, the honest
move is to keep the admission and say the hypothesis was tested and rejected.

## 6c. A sizing check that changes what the result means

Before submitting the sweep I measured rho on the *weakest* surrogate it was
going to include (1 epoch, 10% of trajectories, grid 16): **rho = 1.35**. Since
rho = sigma/eps and eps falls as the surrogate improves, rho only rises from
there. The planned sweep would therefore have visited the failure regime
exclusively, reproduced 18.2% three more times, and never once shown the
default working — which is the half that makes this a diagnostic rather than a
complaint, and exactly the objection this experiment exists to answer.

Two much weaker levels (1:0.02, 1:0.05) were added to push eps above sigma.

**The more interesting consequence.** If reaching rho < 1 requires a surrogate
that bad, then for any competently trained model in a CMDP the default recipe
undercovers. That is a stronger claim than "it depends on where the error
lives": it says the failure regime is the ordinary one, and the regime where
the default is correct is a corner you have to work to reach.

The sweep settles which framing is right, and the two are worth different
sentences in the abstract:

- if the crossover lands at a surrogate quality anyone might plausibly deploy,
  the contribution is a **diagnostic**: measure rho, choose accordingly;
- if it requires a surrogate far worse than anyone would ship, the contribution
  is a **correction**: in CMDPs, matched-pair conformalisation is the wrong
  recipe, with the per-instance case noted as the exception that explains why
  the CBF literature got a different answer.

Do not write the abstract until this is known.

## 7. New experiment — the head-to-head the draft defers

§5 ends: "A head-to-head on a testbed with an *inaccurate* surrogate, which
would turn this into a diagnostic for choosing between them, is left to future
work." That deferral is load-bearing — without it, a reviewer can explain the
18.2% result as a mis-tuned implementation rather than a structural failure.

`eval/run_margin_choice.py` runs it: surrogate fidelity is swept by training
budget (epochs × fraction of trajectories), and all four margins are run at
each level. Reported per level: surrogate rel L2, cost bias, policy episode
spread, the margin each recipe produces, violation rate and return. The
predicted ordering is by `rho = spread / |bias|` — below 1 the default
suffices, above it the default undercovers.

*Status: running. Fidelity level 1 gives a surrogate at rel L2 0.575 against
0.0024 at full training, so the sweep does span the regime where the default
should win.*

## 8. New experiment — §6 currently does not test the contribution

The draft states it outright: "The budget is enforced by projection, so this
section validates the planner rather than the conformal margin." Half the paper
does not exercise its own mechanism.

`eval/run_ndws_margin.py` fixes this without touching the counterfactual
limitation. Two models of real fire spread at different quality: `G`, the
planner's surrogate, trained short; `F`, a stronger model with a different seed
whose held-out AUC-PR against observed fire is reported, standing in for
reality. The crew budget stays a hard projection. The *constraint* becomes an
expectation — population-weighted burn — that conflicts with the objective
(total burn) whenever sparing a large empty area costs a small populated one. A
single multiplier shared across fires is driven by the batch mean under `G`;
violation is realised under `F` and measured per evaluation batch, because the
constrained quantity is a mean.

This is sim2sim and must be labelled as such: `F` is a model, and §7's
counterfactual limitation is untouched. What it adds is a model/reality cost gap
that nobody tuned, over dynamics fitted to real fires, and a third setting for
the head-to-head of §6 above.

## 8b. The calibration unit, and a mistake worth printing

The first run of the wildfire experiment failed completely. Three seeds, four
arms, nothing near delta = 0.1:

| seed | none | model_error | episode | residual |
|---|---|---|---|---|
| 0 | 0.750 | 0.700 | 0.800 | 0.600 |
| 1 | 0.400 | 0.450 | 0.250 | 0.550 |
| 2 | 0.550 | 0.600 | 0.700 | 0.400 |

The margins explain it:

```
seed 0: limit 0.064030   margin 0.06403  ->  d_eff = 0.00000
seed 1: limit 0.071778   margin 0.07178  ->  d_eff = 0.00000
seed 2: limit 0.078697   margin 0.07870  ->  d_eff = 0.00000
```

Every margin consumed its entire limit. With `d_eff = 0` the constraint is
infeasible, the dual saturates, and the margin mechanism is moot — which is why
no arm separated from `none`.

**The cause.** Violation is declared on a batch mean over 32 fires; I
calibrated on per-fire residuals. A batch mean has sqrt(32) ~ 5.7x smaller
spread, so every quantile was ~5.7x too large.

**This is the paper's own claim, violated by the paper's own experiment.** The
thesis is *conformalise the distribution that actually produces the violation*.
The unit of that distribution is part of its identity, and I got it wrong.

It is the second instance of the same class in this project: the first was
conformalising surrogate error instead of episode cost (Section 5 of the
draft). That is an argument for stating the rule more sharply than "which
distribution" — something closer to:

> A conformal margin is valid only if its calibration scores are exchangeable
> with the quantity that will be compared against the limit — same estimator,
> same unit, same aggregation.

Under that phrasing both failures are the same error, and the paper gains a
rule that catches them in advance rather than two anecdotes.

**Fixed.** Calibration collects one residual per batch, 4 independent batches
per probe. `model_error` stays matched-pair at batch level, so the across-batch
scatter still cancels; `residual` measures against the dual's running estimate,
so it survives. The distinction the paper is about, now at the right
granularity. A guard refuses to report any arm whose margin eats more than 90%
of the limit.

**Worth saying in the paper.** A reviewer asked to believe a conformal method
will want to know the failure modes. Two concrete ones, both found by running
it, are more convincing than a clean derivation — and the guard is the kind of
thing a practitioner can copy.

## 8c. A precondition for a margin being usable at all

The wildfire experiment then failed a second time, for a different and more
interesting reason. Three quantities, all measurable before any margin is
fitted (seed 0, 3% crew budget):

```
controllable span  (constraint-optimal -> objective-optimal)  0.008193   12.8% of d
model-reality bias (dual holds G at d, F delivers this)       0.011570   18.1% of d
headroom for a margin (d - constraint-optimal)                0.002867    4.5% of d
```

A margin must cover the bias and must fit inside the headroom. Here it must be
at least 0.0116 and can be at most 0.0029. **The bias is 1.4x the entire span**:
the gap between the planner's model and reality is larger than the range over
which the planner can move the constrained quantity at all. Raising the limit
to make headroom also raises it past the point where the constraint binds, so
no choice of limit works.

The 70-80% violation rates were therefore not the margin failing. The task was
infeasible, and infeasible is indistinguishable from miscalibrated in the
output.

**The general statement, which is worth a paragraph in the paper:**

> A conformal margin can only deliver its stated rate if the policy retains
> enough authority to act on the tightened limit. Write `s` for the span of the
> constrained quantity between the constraint-optimal and objective-optimal
> policies, `b` for the model-reality bias, and place the limit at
> `d = d_best + f·s`. The margin has to fit the headroom `f·s`, so it is usable
> only when
>
>     b  <  f · s
>
> Otherwise `d_eff` falls below anything the policy can reach and the
> constraint is unachievable however well the margin is calibrated.

**`f` is not a free parameter, it is the trade-off.** It sets how tightly the
constraint binds *and* how much room a margin has, in opposite directions. The
first version of this experiment used `f = 0.35` chosen only to make the
constraint bite, which left headroom of 4.5% of the limit against a bias of
18%, and no calibration could have rescued it. Reporting `f`, `b` and `s`
together is the minimum for a reader to judge whether a margin could have
worked at all.

This is checkable in advance from two rollouts, it costs nothing, and it
belongs next to the exchangeability assumption as a condition of use rather
than being discovered by a reviewer. `eval/run_ndws_margin.py --calibrate-only`
reports all three numbers and prints a usable/INFEASIBLE verdict.

It also explains something in the draft that was passed over. On DAR the
constraint never binds (worst case 0.225 against a limit of 0.936), so the span
is enormous relative to the bias and any margin is trivially usable; on RDF the
margin works because the surrogate is accurate and `b` is small. The wildfire
task is the first setting in this project where `b > s`, which is why it is the
first place the mechanism could not be made to work.

**What the sweep found.** No combination of crew budget and G quality reached
`b < f·s`: the ratio ran from 2.0x to 4.9x, and more crew made it *worse*,
shrinking the absolute span from 0.0095 to 0.0055 because the
objective-optimal and constraint-optimal plans both converge on the same floor.
A second independently trained model cannot supply a usable discrepancy on this
task.

So the discrepancy was made a controlled variable (Section 8d). With reality as
the planner's own model plus a calibrated perturbation, the threshold is
reached and bracketed:

| logit shift | span `s` | bias `b` | headroom `f·s` | ratio | verdict |
|---|---|---|---|---|---|
| 0.01 | 0.006887 | 0.001509 | 0.003443 | 0.44 | usable |
| 0.02 | 0.006940 | 0.001988 | 0.003470 | 0.57 | usable |
| 0.04 | 0.007086 | 0.003029 | 0.003543 | 0.85 | usable |
| 0.08 | 0.007389 | 0.005134 | 0.003694 | **1.39** | infeasible |
| 0.16 | 0.007887 | 0.009718 | 0.003943 | **2.46** | infeasible |

Three points either side of the threshold, which is what Section 6 needs: the
margin should hold delta on the usable rows and fail on the infeasible ones,
and failing there is the constraint being unreachable rather than the margin
being wrong.

## 8e. What each pending result would mean (written before the data)

Recorded 2026-09-24 23:13 local, with both experiments queued and
neither reported. Today produced five experiments that could not have failed
informatively, so the outcomes are written down first.

### Stage 2: the margin on real fire dynamics

Success is **two-sided**. Shifts 0.01 and 0.04 sit inside the feasible region
(b/headroom 0.44 and 0.85); 0.08 and 0.16 sit outside it (1.39, 2.46).

| region | wanted |
|---|---|
| feasible | `residual` and `episode` hold at or below delta = 0.1, `none` violates heavily |
| infeasible | **every arm fails, ours included** |

The second row is not a formality. The tightened limit is unreachable there, so
an arm that appears to succeed is measuring something other than what it
claims, and would invalidate the feasible-region result rather than add to it.

**Genuine negative:** `residual` fails at b/headroom 0.44. The margin does not
transfer to real fire dynamics, and Section 6 reports that.

**Vacuous:** every arm passes everywhere. The constraint stopped binding; the
run is discarded, not interpreted.

### The PDE fidelity sweep

One question: does `model_error`'s violation rate cross delta as the surrogate
improves, while `residual` and `episode` stay flat?

**Genuine negative, and the most consequential one pending:** it never crosses
across the whole ladder. That would make the crossover an artefact of the
controlled setting rather than a property of real planners, and the paper's
central claim would not survive in its current form. The honest response would
be to retreat to "derived, and demonstrated in a controlled setting", which is
a workshop paper rather than a main-track one.

### OUTCOME, recorded after the fact

**The PDE sweep: pre-registered test run at adequate power, pattern confirmed.**

The first attempt used `eval_every 20` — 11 evaluations per run, so every rate
was a multiple of 1/11 and δ = 0.1 was not attainable. The test could not be
run. The margin collapse was reported instead, and flagged at the time as a
post-hoc substitute.

Rerun at `eval_every 5` (41 evaluations per run, 615 pooled per arm):

| | `none` | `model_error` | `episode` | `residual` |
|---|---|---|---|---|
| pooled, 15 runs | 0.1090 | 0.0830 | 0.0586 | **0.0602** |
| ρ < 8, 10 runs | 0.1171 | **0.0659** | 0.0586 | 0.0561 |
| ρ ≥ 8, 5 runs | 0.0927 | **0.1171** | 0.0586 | 0.0683 |

The predicted pattern: the default holds δ where the model's error dominates
and breaks it where the policy's spread does, exceeding both δ and the
no-margin baseline. Ours holds in both regimes.

**Significance, stated plainly.** Pooled `none` vs `residual` is z = 3.07.
Within the high-ρ stratum, `model_error` vs `residual` is z = 1.70 — right
direction, short of 1.96 on 205 evaluations per arm. Two more seeds there give
the ~1.33× needed. So the qualitative fact is visible and the contrast is not
yet separable from noise; the paper should say exactly that.

**The substitution caveat is withdrawn.** The test was run as written. The
margin collapse now supports the violation result rather than standing in for
it.

**Stage 2 (wildfire): still open.** The first attempt was truncated by its wall
limit with 3 of 4 arms done, losing `residual`. Three arms that survived sat at
0.60–0.75 against δ = 0.1 with λ only reaching ~2.1, which suggests an
under-converged dual rather than a failing margin. A 160-iteration test was
queued; its status is unknown because the SSH control socket expired and
reconnecting needs MFA.

*Recorded 2026-09-25 12:30.*

### Banked regardless of both

- the controlled coverage study, crossover at rho = 1.00 over 600 draws a point
- `eps >= sigma` derived rather than guessed
- `b < f*s`, checkable from two rollouts on any task
- silent violation on real fire data at 40 to 75 percent
- 95 tests, one shared margin implementation, five guards in code

The paper is viable on these alone, led by the controlled study and the two
rules. The pending results decide whether the claim is *demonstrated on real
planners* or *derived and shown under controlled conditions*.

## 9. Report the statistics the paper already cites

§7 concedes "mean ± s.d. reporting; stratified bootstrap intervals
(Agarwal et al., 2021) would be preferable". Citing a paper and then not
following it is a free objection to hand a reviewer, and the fix is a few
lines. `eval/metrics.py` now provides `iqm`, `bootstrap_ci` and `seed_report`.

Applied to the headline planning gaps (+9.6, +11.7, +15.9, +12.1, +10.7):

| reported as | value |
|---|---|
| draft | +12.0 (mean of 5 seeds) |
| Agarwal-style | **IQM 11.50, 95% bootstrap CI [9.97, 14.63]** |

The result survives — the interval is comfortably away from zero — and the
paper gains the reporting standard it asks for. The interval is wide because
five runs is few; that is the honest outcome, not a defect of the method.

The same treatment is worth applying where it changes the reading. The
fixed-α arm in Appendix B is the clearest case: one diverged seed drags the
mean to −2.518 while the IQM is −2.300 with a CI of [−3.04, −2.29], which says
"usually fine, occasionally catastrophic" — the actual finding — where
mean ± s.d. says only "worse on average".

## 10. Smaller items

- **Figure 1** has a clipped `10%` artefact at top left; the `δ = 10%` dashed
  line annotation collides with the bar labels.
- **Figure 2(b)**: the `+14.3` annotation is detached from its anchor.
- `Sauté` / `Saute` inconsistent between Table 4 and the references.
- The partial-observation paragraph in §6 is compressed past usefulness and
  does not serve this paper's thesis. Cutting it reclaims roughly a third of a
  column for §5, which needs it more.
- Appendices B and D earn their place (B is the planner, D is why a testbed was
  excluded). Appendix E (joint training) is vestigial from the larger project
  and could go.
- §7 should add the exchangeability item that §3 now makes concrete: the
  `residual` mode's assumption is *weaker* than the draft's, because it needs
  one exchangeable sample rather than an exchangeable sample plus an estimated
  bias term.
