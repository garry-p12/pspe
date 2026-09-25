# PSPE against the state of the art

**A positioning audit.** For each area PSPE touches: what the field's current
best is, what we actually do, and an honest verdict on whether our result is
ahead, at parity, behind, or novel in a different direction.

Companion to [`THESIS_AND_RESULTS.md`](THESIS_AND_RESULTS.md), which states what
we measured. This document states what it is worth **relative to other people's
work**, which is a different and less comfortable question.

Literature checked September 2026. Where a comparison is not like-for-like
(different split, different metric, different environment), that is said
explicitly rather than glossed.

---

## 0. Summary of verdicts

| area | our position | one line |
|---|---|---|
| Conformal margin for constraint violation | **Novel, and it is the strongest thing we have** | The field's default conformalises model error; we show that fails and give the version that works |
| Horizon mechanism on observed fire data | **Novel as evidence, not as method** | Insignificant at 1 day, +14.3 at 5 days: a mechanism measurement, not a leaderboard number |
| Reconstruction helps, decision worsens | **Novel and under-discussed** | Contradicts the working assumption of the data-assimilation and twin literature |
| Measurement-defect catalogue | **Converging with an established line** | Henderson, Agarwal, and now WildfireSpreadBench are making the same argument |
| Constrained planning fails silently | **Known** | Stated outright in the model-based safe-RL literature |
| Joint training does not help | **Known phenomenon, inverted data point** | Objective mismatch, 2020; our twist is that the model got 5–9× better and the decision did not move |
| State sync beats open loop | **Textbook** | This is what data assimilation is. Our version is the crudest possible form of it |
| Decision-time beats amortised policy | **Known, and the literature is against our reading** | Planner amortisation works when done properly; we did not do it properly |
| Permutation control for faithfulness | **Standard practice we initially omitted** | Credit for applying it, none for inventing it |
| Wildfire next-day forecast skill | **Behind** | We report 0.3162; current best is 0.3673 single model, 0.3790 ensemble |
| Sequential wildfire data | **Behind, and avoidably so** | WildfireSpreadTS has 607 fire time series; we built a pipeline for 8 |
| PDE surrogate architecture | **Behind, and not our contribution** | FNO is a 2021 baseline; Poseidon and BCAT are the current frontier |

**Net.** Four of twelve hold up as contributions. The broad thesis ("model-based
decision systems fail silently") is real but is being arrived at independently
by several groups; the specific, defensible paper is narrower than the one
`THESIS_AND_RESULTS.md` §0.3 proposes. See §10.

---

## 1. Constrained safe RL: the failure we diagnose is already named

**The field.** CPO (Achiam et al. 2017), PPO-Lagrangian, PID-Lagrangian
(Stooke et al. 2020), Sauté RL (Sootla et al. 2022), primal-dual NPG — all
baselines we already run. Model-based safe RL is the closer neighbour:
SafeDreamer (ICLR 2024) puts Lagrangian methods inside a world model; CAP
(Conservative and Adaptive Penalty, AAAI) and constrained-PPO-in-a-model
(NeurIPS 2022) address exactly the case where the cost estimate comes from a
learned model.

**What they already say.** The premise of our contribution 1 — that a dual fed
surrogate cost is enforcing a constraint the environment does not respect — is
stated plainly in this literature: cost estimates are error-prone when computed
under an environment model, and Lagrangian methods degrade under low cost
thresholds because of critic/model estimation error.

**Verdict: the diagnosis is known; the mechanism is a narrow contribution.**
We should not present "constrained model-based planning violates silently" as a
discovery. What is ours:

- the **probe + margin decomposition**, with the measurement that *neither half
  works alone* (margin-only is bit-identical to baseline, because with no probe
  there is no measured error to derive a margin from). This is a clean, small,
  transferable result.
- the finding that **the same diagnosis repaired PPO-Lagrangian** (1.8% → 0% on
  rdf via `K_i`), which says the defect belongs to the PID controller, not to us.

**What a reviewer will ask.** "How does this compare to SafeDreamer and to CAP?"
We have no answer — they are not in our baseline set. Our baselines are all
*model-free* safe RL, which makes the sample-efficiency claim (9×/5×) close to
tautological: of course a model-based method uses fewer real transitions than
model-free methods. **Against a model-based safe-RL baseline the sample
efficiency claim may vanish entirely.** This is the single most damaging gap in
the current baseline set.

---

## 2. Conformal prediction for safety: our negative result is the contribution

**The field.** An active 2025–2026 area. The dominant recipe is:

> conformalise the **error between the learned dynamics model and the true
> system**, then push that bound into a Control Barrier Function or Control
> Lyapunov Function constraint.

Representative: Zhou, Zhang & Luo (arXiv 2503.17678) — GP dynamics with
Quadrature Fourier Features, adaptive conformal prediction on the model's
online prediction error, fed to a CBF. Also Mirzaeedodangeh, Shekhtman, Matni
& Lindemann (2606.15366) and Tomashevskiy (CoLLAs 2026, 2609.08080), which
gives a cumulative violation bound growing at most linearly in the horizon.

**One paper is not in that pattern, and it is the closest to ours.** Prinster,
Fannjiang, Park, Cho, Liu, Saria & Stanton, *Conformal Policy Control*
(2603.02196), calibrates permissible **behavioural change** against a safe
reference policy and identifies no dynamics model at all. It shares our
instinct that the calibration target should be the controlled object; we differ
in conformalising realised cost against the quantity a Lagrangian dual holds,
which gives a CMDP budget rather than a behavioural-change budget. The ICML
draft lumped it in with the model-error line, which is wrong and which an
author-reviewer would catch on sight; corrected in
`paper/REVISION_NOTES.md` §2.

**What we did.** We tried that recipe first. On the reaction-diffusion front it
made things **worse**: 18.2% violations against a 14.5% no-margin baseline,
because the margin it produced was 0.25 where 0.69 was needed. The reason is
structural and general:

> Where the surrogate is **accurate** (bias −0.08) but the policy is
> **variable**, a quantile over model error never sees the quantity that
> actually breaches the limit, which is the policy's own episode-to-episode
> spread.

**The cause, stated properly.** That wording names the symptom. Matched-pair
residuals `c_i − g_i` cancel whatever scatter is common to both terms, so the
recipe is *correct* whenever the controller holds a per-instance quantity at
the limit — which is exactly what a CBF does, pointwise, in the setting the
recipe was developed for. A CMDP dual holds an **expectation**, so `g_i` is not
the controlled quantity and the per-instance scatter that breaches `d` cancels
out of the calibration set. Sound where it came from; unsound here; and the
distinction can be written down in advance rather than discovered empirically.

Conformalising the **realised cost against the quantity the dual controls**
covers bias and spread in a single order statistic, needs no separate bias
term, and reduces to the matched-pair recipe when control *is* per-instance.
It holds the stated rate on both families (dar 0%, rdf 7.3%, at δ = 0.1).

**Honest note on "both families":** dar's baseline already violates 0% with a
worst case of 0.225 against a limit of 0.936, so that column shows the margin
is harmless, not that it works. The load-bearing evidence is rdf, which is one
number — hence the two experiments below.

**Verdict: this is the strongest thing in the project.** It is a direct,
falsifiable, empirically supported critique of the field's default
instantiation, plus the fix. Nobody in the papers above reports what happens
when model error is small and policy variance is not.

**Caveats to state, not bury.**
- Their setting is CBF-constrained continuous control with GP dynamics; ours is
  a CMDP with an episode-cost budget. The failure mode should generalise, but we
  have not demonstrated it in *their* setting, and we should say so.
- Our bound is weaker than theirs in one respect: exchangeability of episode-cost
  deviations across a training run is an assumption we state but do not test.
  Adaptive conformal prediction (Gibbs & Candès) exists precisely because that
  assumption fails under distribution shift, and a learning policy is a
  distribution shift. **This is a real hole and a reviewer will find it.**
- We give a marginal rate, not the per-step cumulative bound of 2609.08080.

**Action — now implemented, running.**
- `eval/run_margin_choice.py` sweeps surrogate fidelity by training budget and
  runs all four margins at each level, so the crossover is measured. Fidelity
  level 1 yields rel L2 0.575 against 0.0024 at full training, so the sweep
  does span the regime where the default should win.
- `eval/run_ndws_margin.py` carries the same recipes onto real fire dynamics
  under an **expectation** constraint, which is the regime the sharpened claim
  predicts the default fails in — and which §6 of the ICML draft did not test
  at all, since it enforced its budget by projection.
- Both call one module, `pspe/plan/margins.py`, so the comparison cannot differ
  by implementation. Its unit test is the mechanism in three numbers: with an
  accurate per-instance model and a policy that scatters, the model-error
  margin is 0.017 where 0.662 is needed, and the residual margin returns 0.662
  exactly.

---

## 3. Wildfire forecasting: we are behind, and the doc overstates us

**The claim in `THESIS_AND_RESULTS.md` §4.5.** U-Net at 0.3162 AUC-PR on the
NDWS test split, "111% of the published benchmark" (Huot et al.'s 0.284).

**The problem.** 0.284 is the *2022 originating paper's* number, not the state
of the art. Current results on the same benchmark:

| model | AUC-PR | source |
|---|---|---|
| mixed ensemble, two augmented plus one non-augmented | **0.3790** | arXiv 2609.17763 (3-seed mean) |
| SwinUNETR with all modular augmentations | **0.3673** | same |
| **PSPE U-Net** | **0.3162** | ours, 3 seeds |
| published originating baseline | 0.284 | Huot et al. 2022 |

**Verdict: behind by roughly 16–20% relative.** "111% of the published
benchmark" is technically true and rhetorically misleading, and a wildfire
reviewer will know the current numbers. It must be restated as: *above the
originating baseline, below current best, on a surrogate we did not set out to
optimise.*

That restatement costs us nothing, because forecast skill is **not our
contribution** — the surrogate is an input to the planning result. But leaving
the sentence as it stands invites the reviewer to distrust the rest.

**Caveat needing resolution.** I could not confirm from the fetched paper that
0.3790 is on the identical test split and protocol as our 0.3162. Splits and
augmentation protocols differ across NDWS papers. Verify before writing a
comparison table into a paper.

**A finding that helps us.** *WildfireSpreadBench: The Metric Decides the Model*
(arXiv 2609.22191) argues that in wildfire spread prediction, AP "outranks
architecture" as a determinant of who wins, that the best-AP model flagged
**4–5× the area that actually burned** while ranking fifth on F1/IoU, and that
models with operationally sensible predictions score 24–37% lower on AP. They
recommend pairing AP with F1, IoU and threshold-fixed precision/recall.

This is independent corroboration of our central thesis, from the wildfire
community, published concurrently. It is an ally to cite — and also evidence
that **the thesis is in the air**, not ours alone.

---

## 4. Sequential wildfire data: the FIRMS pipeline was avoidable work

**What we built.** `pspe/simulate/real/firms.py` — VIIRS active-fire detections
for 8 named fires, 14–20 consecutive days, gridded to 64², with `observed`
flags preserving days with no overpass. Used for the twin-loop result (§5.2),
8 fires, leave-one-fire-out, paired t = 15.9.

**What already exists.**

| dataset | scale | modality |
|---|---|---|
| **WildfireSpreadTS** (NeurIPS 2023 D&B) | **607 fire events, 13,607 daily images, 2018–2021** | 23 multi-modal channels: active fire, fuel, topography, weather |
| WSTS+ (WACV 2026) | extension, time-series input models | best overall accuracy comes from multi-temporal input |
| FireSentry | fine-grained spatio-temporal spread benchmark | multi-modal |
| BCWildfire | long-term multi-factor boreal benchmark | risk prediction |

WildfireSpreadTS exists *specifically* because NDWS cannot be chained — the same
motivation written into our module docstring. It has been public since 2023.

**Verdict: behind, and this one is on us.** Our twin loop runs on 8 fires when
607 were available, pre-processed, benchmarked, with 23 channels instead of our
2. The paired test across 8 fires is honest but badly underpowered, and "8 of 8
fires" is a weaker statement than it sounds at n = 8.

**This is also the single highest-value fix available.** Porting the twin loop
(`eval/run_firms_twin.py`) to WildfireSpreadTS would:
- raise n from 8 to ~600, turning a suggestive paired test into a solid one;
- let the model adaptation result (+21% day 2, +30% day 3) be measured per fire
  across a real distribution of fire behaviours;
- put the result on a benchmark reviewers already know, removing "why your own
  dataset?" from the review entirely;
- cost days, not weeks — the loop code is dataset-agnostic apart from the loader.

FIRMS is not wasted: it remains the right tool for a **live incident**, which
WSTS (a static archive) cannot support. That is the honest role for it.

---

## 5. Firebreak planning: direct prior work exists

**The field.** Our §5.1 headline is not the first attempt at this problem.

- *Advancing Forest Fire Prevention: Deep RL for Effective Firebreak Placement*
  (arXiv 2404.08523) — DQN / Double DQN / Dueling DDQN over the **Cell2Fire**
  simulator.
- *Deep reinforcement learning for optimal firebreak placement in forest fire
  prevention* (Applied Soft Computing, 2025) — same simulator family.
- *A graph-based optimization framework for firebreak planning in
  wildfire-prone landscapes* (2025) — combinatorial optimisation, explicitly
  motivated by insufficient resources to treat all threatened locations, which
  is our budget constraint.
- *Spatiotemporal Wildfire Prediction and RL for Helitack Suppression* (2026) —
  suppression resource allocation.

**The real difference, and it cuts both ways.**

| | prior work | PSPE |
|---|---|---|
| dynamics | Cell2Fire, a physics-based simulator | U-Net fitted to observed NDWS fire days |
| intervention effect | **simulated, but internally consistent and counterfactually evaluable** | stated action model, **not validated** |
| realism of spread | simulator, with its own calibration gap | fitted to what real fires did |
| evaluation | in-simulator, so the counterfactual is exact | in-model, so the counterfactual is unverifiable |

**Verdict: a different trade-off, not a strict improvement.** They can actually
evaluate a firebreak's effect because their environment defines it; they pay for
it with a simulator gap. We have real spread dynamics and cannot evaluate the
intervention at all (§6.2 of the thesis doc). Claiming superiority over them
would be unsupportable.

**What is genuinely ours in this area:**
1. **Decision-time optimisation beats learned policies on this problem**
   (36.5% vs 4.3–4.7% for four constrained-RL methods, and 10.4% for our own
   amortised arm). Since the prior work is *all DRL*, this is a substantive
   critique of the dominant approach in the subfield. See §6 for why it is
   weaker than it looks.
2. **The horizon mechanism** (§7).

**Action.** Cite these four papers, state the trade-off table above, and run our
planner against Cell2Fire so there is at least one setting where the
intervention counterfactual is evaluable. That converts "we cannot validate
intervention effects" from a permanent limitation into a bounded one.

---

## 6. Decision-time planning versus amortised policies: the literature is against us

**Our claim.** Per-instance projected gradient (34.0%) beats an amortised CNN
policy (10.4%) and four constrained-RL methods (4.3–4.7%); the reason is
amortisation, not the constraint; "on a per-instance decision problem this
diverse, decision-time optimisation is the right tool and a learned policy is
not."

**The field.** This is well-trodden MPC-versus-RL ground, and the recent
evidence points the *other* way:

- *Evaluating model-based planning and planner amortization for continuous
  control* — a model-based planner **can be amortised into a compact policy
  without performance loss**, via MPO plus behaviour cloning on MPC-generated
  data, and this outperforms strong model-free baselines in multi-task/multi-goal
  settings.
- TD-MPC / TD-MPC2, Dream-MPC — hybrid decision-time planning with learned
  value/policy amortisation, state of the art across diverse task suites.

**Verdict: our conclusion overreaches.** We showed that *a CNN policy trained
with a budget dual for 200 iterations over 1,024 fires* fails. The literature's
recipe for amortisation — distil the planner's own solutions into the policy —
is exactly what we did not try. The honest statement is:

> An amortised policy trained from scratch under a budget dual does not reach
> the planner's performance in our compute budget. We did not test distillation
> from planner solutions, which the amortisation literature identifies as the
> method that works.

The constrained-RL comparison survives better than the amortisation claim,
because those four methods are the standard baselines and were given a fair
initialisation. But note our own table: three of the four **exceed** the 3%
budget (Sauté by 2.5×, over budget on 79% of fires). They lose while spending
more, which is a fair thing to report, but it means the comparison is not
matched-constraint in the way the phrase implies.

---

## 7. The horizon result: our best empirical evidence

**The measurement.** On 1,500 held-out real fires, 5 seeds:

| horizon | gap over the operational heuristic | paired t |
|---|---|---|
| 1 day | +0.6 | **1.40, not significant** |
| 2 days | +8.5 | 10.74 |
| 3 days | +12.1 | 11.08 |
| 5 days | **+14.3** | 8.99 |

**Why this is the strongest empirical thing we have.** The one-day objective is
separable over cells, so treating the highest-risk cells within budget is its
*exact* optimum — theory says the gap must be zero there, and it is. The
statistic then jumps an order of magnitude the moment the decision becomes
sequential, and the heuristic **degrades** (24.8 → 17.7) because it keeps
optimising for today.

A result that is null exactly where theory requires and significant exactly
where theory predicts is much harder to attribute to tuning or luck than a
single large number. None of the firebreak-DRL papers in §5 report a horizon
ablation of this form.

**Verdict: novel as evidence, on a real dataset, with the mechanism isolated.**
This should be Figure 1 of the paper, not the conformal margin's supporting act.

**Caveat.** The intervention effect is still model-based, so what we have shown
rigorously is *the planner exploits the surrogate's multi-day structure*, not
*firebreaks placed this way reduce real burned area*. Say it in those words.

---

## 8. Data assimilation: our twin loop is the trivial baseline

**Our result (§5.2).** State sync 0.608 vs open loop 0.073 at one day, 8.4×,
better on 8 of 8 fires, t = 15.9. Model adaptation adds +21% at day 2, +30% at
day 3.

**The field.** Data assimilation is a mature discipline: 4D-Var, Ensemble Kalman
Filter, and now a large ML-DA literature — latent data assimilation (ROM + ML
surrogate + DA), *Physically consistent global atmospheric data assimilation
with machine learning in latent space* (Science Advances, 2026) showing latent
DA improves both analysis quality and forecast skill over model-space DA, deep
latent particle filters with UQ, EnKF combined with ML for turbulent state
estimation, and the DestinE / Earth-2 digital-twin programmes.

**What "state sync" actually is.** Direct replacement of the belief with the
observation where observed, persistence of the model estimate elsewhere. In DA
terms this is **nudging with gain 1**, the crudest possible analysis step. No
covariance, no observation-error model, no ensemble, no variational step.

**Verdict: the result is real, the method is textbook, and "8.4× over open loop"
is what assimilation is for.** Comparing an assimilating system to a free-running
forecast is not a contribution; it is a sanity check that the loop is wired
correctly. Presented to a DA audience as a finding, it would land badly.

**How to salvage it.** Two honest framings, both smaller than the current one:
1. **As a control, not a result:** "the loop closes, verified on real
   sequences" — one paragraph, not a section.
2. **As the substrate for the interesting question** in §9, which *is* ours.

**Action.** Either add a real DA baseline (an EnKF or a learned gain over the
same FIRMS/WSTS sequences) so "sync" has something above it to be compared
against, or demote the section. Reporting gain-1 nudging against free-running
forecast as an 8.4× improvement, with no DA baseline, is the kind of comparison
this project's own standing rules (§7 of the thesis doc) exist to prevent.

---

## 9. Better reconstruction, worse decision: genuinely under-discussed

**Our result (§5.3).** Under correlated occlusion, an estimator that recovers
hidden cells ~100× better by average precision (0.594 vs 0.006) produces a
**worse decision in 0 of 9 configurations** (−4.28 points, t = −4.33). The
`perceive-hard` control splits it: 3.4 points are a representation artefact (the
surrogate was trained on near-binary masks and reacts badly to a smeared
probability field), 0.86 points are a real cost of spending a hard budget on
cells that merely *might* be burning.

**Why this matters against the field.** The entire DA and digital-twin stack in
§8 optimises **analysis quality** — how close the reconstructed state is to
truth — on the assumption that a better analysis yields a better outcome. The
Science Advances latent-DA result is stated in exactly those terms: better
analysis *and* better forecast skill. Neither is the decision.

Our result is a counterexample in the regime that matters for intervention:
under a **hard budget** on a **sparse field**, a better analysis produced worse
decisions, every time.

**Verdict: novel direction, modest evidence, and it lines up with the outside
literature.** WildfireSpreadBench's finding — the best-AP model flags 4–5× the
area that burned and is operationally unusable — is the same phenomenon measured
on the forecast rather than the analysis. Two independent instances of
"component metric improves, operational quantity worsens" in the same domain is
a real pattern.

**The caveat that bounds it, restated.** Fire covers 1.3% of cells, so "assume
nothing burns where I cannot see" is close to the base rate and `blind` gets a
strong prior nearly free. On a dense field — flood depth, surface temperature —
this could invert. **This is why the Sentinel-1 flood experiment is the highest
scientific-value item remaining**, and the literature check strengthens that:
it is the difference between a quirk of sparse fire masks and a general claim
about beliefs under budgeted intervention.

---

## 10. What the paper should actually be

`THESIS_AND_RESULTS.md` §0.3 proposes "Constrained planning with learned
dynamics: silent failure and measurement-based correction," with the failure
catalogue as the differentiator. **After this audit, that framing is weaker than
it looked**, because the individual failures have precedent:

- silent violation under model-based constraints — named in the safe-RL literature
- objective mismatch — Lambert et al. 2020, with a 2023 unified taxonomy
- metric-versus-operational-outcome divergence — WildfireSpreadBench, 2026
- evaluation-protocol defects — Henderson et al. 2018, Agarwal et al. 2021

A reviewer who knows these will read the catalogue as a careful replication
across a new domain, which is worth publishing but is not a headline.

**The narrower paper that does survive:**

> **What should a conformal margin bound in constrained model-based planning?**
>
> Constrained model-based planners violate their budget because the dual is fed
> cost measured in the model while violation is realised under true dynamics.
> The standard conformal remedy bounds the *model's* prediction error. We show
> this fails whenever the model is accurate and the policy is variable — on a
> reaction-diffusion front it raises violations from 14.5% to 18.2%, worse than
> no margin — and that conformalising *realised episode costs* holds the stated
> rate on two PDE families. We validate the resulting planner on 1,500 real
> wildfire records, where planning through a learned spread model beats the
> operational forecast-then-treat heuristic by 12 points at matched crew budget,
> with the advantage appearing only as the decision becomes sequential
> (t = 1.40 at one day, +14.3 points at five).

That is one claim, one mechanism, one negative result against a named
alternative, and two independent validations. It is a better ICML paper than
the four-contribution version, and every sentence survives §1–§9.

**The digital-twin framing should be motivation only.** After §8, the twin
sections cannot carry weight: property 4 fails (§6.2 of the thesis doc), and
properties 1–2 are executed well below the DA state of the art.

---

## 11. Ranked actions

| # | action | cost | what it buys |
|---|---|---|---|
| 1 | Restate the NDWS forecast claim against current SOTA, not the 2022 baseline | hours | removes the one sentence most likely to lose reviewer trust |
| 2 | Add a **model-based** safe-RL baseline (SafeDreamer, CAP) | ~1 week | the sample-efficiency claim is currently near-tautological without it |
| 3 | Port the twin loop from FIRMS (8 fires) to **WildfireSpreadTS** (607) | days | turns an underpowered paired test into a solid one on a known benchmark |
| 4 | Head-to-head: conformalise model error vs episode cost, on a testbed with an **inaccurate** surrogate | ~1 week | converts "ours is right" into "here is the diagnostic for which to use" |
| 5 | Sentinel-1 flood extent, dense field | 2–3 weeks | decides whether §9 is a sparse-field quirk or a general claim |
| 6 | Either add a DA baseline to the twin loop, or demote it to a control | days | stops us reporting gain-1 nudging against free-running as a finding |
| 7 | Run the planner against **Cell2Fire** | ~1 week | one setting where the intervention counterfactual is evaluable |
| 8 | Test **distillation** from planner solutions into the amortised policy | days | our amortisation claim is currently contradicted by the literature |
| 9 | Test exchangeability of episode-cost deviations; consider adaptive CP | days | closes the hole a conformal-prediction reviewer will go straight to |

Items 1, 3 and 6 are cheap and remove the three claims most exposed to a
knowledgeable reviewer. Items 2 and 4 are what turn the surviving contribution
into a defensible paper.

---

## Sources

Wildfire forecasting and benchmarks
- [Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction](https://arxiv.org/abs/2609.17763)
- [WildfireSpreadBench: The Metric Decides the Model in Wildfire Spread Prediction](https://arxiv.org/html/2609.22191)
- [WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction (NeurIPS 2023 D&B)](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf)
- [Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ (WACV 2026)](https://arxiv.org/pdf/2502.12003)
- [Next Day Wildfire Spread (Huot et al.)](https://arxiv.org/pdf/2112.02447)

Firebreak and suppression planning
- [Advancing Forest Fire Prevention: Deep Reinforcement Learning for Effective Firebreak Placement](https://arxiv.org/pdf/2404.08523)
- [Deep reinforcement learning for optimal firebreak placement in forest fire prevention](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003540)
- [A graph-based optimization framework for firebreak planning in wildfire-prone landscapes](https://www.sciencedirect.com/science/article/pii/S1574954125003486)
- [Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression](https://arxiv.org/pdf/2601.14238)

Safe and constrained RL
- [SafeDreamer: Safe Reinforcement Learning with World Model (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/ece182f93af26c64187ba3f7dfd4309a-Paper-Conference.pdf)
- [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/download/20478/20237)
- [Model-based Safe Deep RL via a Constrained Proximal Policy Optimization (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/file/9a8eb202c060b7d81f5889631cbcd47e-Paper-Conference.pdf)

Conformal prediction for safety
- [Computationally and Sample Efficient Safe RL Using Adaptive Conformal Prediction](https://arxiv.org/abs/2503.17678)
- [Conformal Policy Control](https://arxiv.org/html/2603.02196v2)
- [Robust Conformal CBF and CLF Controllers via Iterative Policy Updates](https://arxiv.org/html/2606.15366)
- [Proactive Context-Forecasted Safety Constraints for Nonstationary RL](https://arxiv.org/html/2609.08080)

Objective mismatch and decision-aware model learning
- [Objective Mismatch in Model-based Reinforcement Learning (Lambert et al. 2020)](https://arxiv.org/abs/2002.04523)
- [A Unified View on Solving Objective Mismatch in Model-Based RL](https://arxiv.org/pdf/2310.06253)

Planning, amortisation and PDE surrogates
- [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/pdf/2203.04955)
- [Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination](https://arxiv.org/pdf/2605.04568)
- [Poseidon: Efficient Foundation Models for PDEs](https://www.alphaxiv.org/abs/2405.19101)
- [BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://ww3.math.ucla.edu/wp-content/uploads/2025/02/2501.18972v1.pdf)

Data assimilation and digital twins
- [Physically consistent global atmospheric data assimilation with machine learning in latent space (Science Advances)](https://www.science.org/doi/10.1126/sciadv.aea4248)
- [Reduced order digital twin and latent data assimilation](https://egusphere.copernicus.org/preprints/2022/egusphere-2022-1167/egusphere-2022-1167.pdf)
- [The Deep Latent Space Particle Filter for Real-Time Data Assimilation with UQ](https://arxiv.org/pdf/2406.02204)

Evaluation rigour
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice (Agarwal et al. 2021)](https://arxiv.org/pdf/2108.13264)
