# Paper ↔ implementation coverage, and the plan for what's left

Audit of the ICLR paper (`pspe_iclr.pdf`) against the current repo, then a
prioritized plan to close the gaps — with real open-source datasets as the
first-class target, since that is the largest hole.

The paper is a **proposal** ("we lay out an experimental program", "we intend
to demonstrate"). So the honest framing is: the *framework* is largely built
and unit-validated; the *experimental program* (Section 7) is mostly not yet
run, and nothing has touched a real dataset or a real backbone.

---

## 1. Coverage map

Legend: ✅ built + run · 🟡 built, not run at scale / stub only · ❌ missing

### Section 4 — the four modules (architecture)

| Paper element | Status | Evidence / gap |
|---|---|---|
| 4.1 Perceive: VLM + LoRA, regression + contrastive loss (Eq. 6) | 🟡 | `pspe/perceive/` complete; **only the `tiny` stub backbone has ever run**. Qwen2-VL/moondream never loaded; contrastive term runs on synthetic captions |
| 4.2 Simulate: FNO + physics residual + rollout consistency (Eq. 7) | ✅ | `pspe/simulate/`; validated against Burgers (canary, rel L2 0.008). DeepONet + GNOT also implemented |
| 4.3 Plan: hybrid pathwise/LR gradient, adaptive α, PID-Lagrangian (Eq. 8–10) | ✅ | `pspe/plan/`; adaptive α reaches 0.99 at a 43,000× variance ratio; dual validated on closed-form CMDP + `dar` |
| 4.4 Explain: LM + LoRA, frozen parser, faithfulness KL (Eq. 11–12) | 🟡 | `pspe/explain/`; F(b) logged in training. **Stub LM only; generated briefs are degenerate at the smoke budget** |

### Section 5 — theory

| Paper element | Status | Evidence / gap |
|---|---|---|
| Prop 1: hybrid bias-variance, α* from calibration error | 🟡 | Mechanism implemented and empirically consistent (variance ratio drives α↑). No formal proof reproduced; α is driven by fold-wise gradient variance, a proxy for the paper's "calibration error" |
| Cor 1: PID-Lagrangian violation → neighborhood of 0 | ✅ | Time-averaged violation →0 on toy CMDP and `dar` |

### Section 7 — the experimental program

| Paper element | Status | Evidence / gap |
|---|---|---|
| 7.1 three PDE families | ✅ | `dar`, `swe`, `rdf` |
| 7.1 **multiple spatial resolutions** | ✅ | `eval/run_resolution.py`; FNO trained at 32² gives rel L2 0.089/0.089/0.088 at 32/48/64² — flat (discretization-invariant) |
| 7.1 **real-world proxy datasets (imagery + sensors)** | 🟡 | EuroSAT (Sentinel-2 → NDVI) loader built + tested (`pspe/perceive/eurosat.py`); PDEBench loader built + tested (`pspe/simulate/pdebench.py`). Real downloads are Colab steps (6.6 GB PDEBench; HF EuroSAT) — code runs, data not yet pulled here |
| 7.1 "tens of thousands of grid cells" | 🟡 | default 64² = 4,096; 128² path exists, not run |
| 7.2 surrogate baselines FNO/DeepONet/GNOT (± physics) | ✅ | `baselines/run_operator_baselines.py`; physics on/off run |
| 7.2 constrained-planning baselines (CPO/PID-Lag/Sauté/PD-NPG) | ✅ | validated + real comparison on `dar`, calibrated limits |
| 7.2 **perception baselines: zero-shot VLM, CNN/ViT regression** | 🟡 | CNN-from-scratch + frozen linear probe implemented and run over 5 seeds (`eval/run_perception_baselines.py`): **both beat the method on field rel L2**; the method wins retrieval acc 4x. Zero-shot VLM still missing (needs a real instruction-tuned backbone) |
| 7.2 **explanation baseline: post-hoc TalkToAgent-style control** | ✅ | `ExplainTrainConfig(posthoc=True)` + `eval/run_explain_baselines.py`, 5 seeds: trained-in > post-hoc (t = +5.53). But the faithfulness *term* adds nothing (t = -1.40 vs no-faithfulness) |
| 7.3 ablation: physics on/off | ✅ | run |
| 7.3 ablation: fixed vs adaptive α | ✅ | run |
| 7.3 ablation: freeze vs fine-tune perception | ✅ | 5 seeds, full budget: frozen wins (t = −3.97) — opposite sign to the paper's expectation, on the stub backbone |
| 7.x **multi-seed error bars on every table** | ✅ | `eval/run_seeds.py` + `aggregate_seeds` + `tests/test_seeds.py`; Phase 2 (`runs/seeds_full/`), ablations (`runs/seeds_rest/`) and the transfer matrix (`runs/transfer_seeds/`) all rerun over 5 seeds on Vista GH200. Only the resolution sweep and the real-backbone runs remain single-seed |
| Cor 1 revisited: violation → 0 **on the PDE testbed** | ❌ | run-level: planner exceeds the limit on 7.3% of evaluations (fixed α 12.7%), every baseline on 0%. Holds only on the toy CMDP |
| 7.3 ablation: faithfulness on/off | 🟡 | 5 seeds, full budget: no separation (t = +0.67, gain driven by 1 of 5 seeds). Needs a real LM |
| 7.3 **ablation: cross-PDE-family transfer** | ✅ | `eval/run_transfer.py` runs the full 3×3 matrix with padded surrogates |
| 7.4 surrogate fidelity vs horizon | ✅ | `surrogate_fidelity` |
| 7.4 **resolution-generalization error** | ✅ | `eval/run_resolution.py` / `resolution_generalization` |
| 7.4 planning reward / sample-eff / violation | ✅ | reported, real-vs-surrogate sample accounting split |
| 7.4 perception field recon error | 🟡 | on synthetic only |
| 7.4 **perception agreement with human descriptions** | ❌ | not implemented |
| 7.4 faithfulness F(b) | ✅ | logged in training |
| 7.4 **human expert study (Likert)** | 🟡 | `eval/human_rating.py` harness exists; no participants, no briefs worth rating yet |
| 7.4 **cross-domain transfer gap (all metrics)** | 🟡 | `planning_transfer_gap` implemented + smoke-tested (`eval/run_transfer.py --planning`); full-budget run outstanding |
| 7.5 wall-clock + peak memory reported | ✅ | in every `summary.json` |
| 7.5 mixed precision + gradient checkpointing "uniformly" | 🟡 | gradient checkpointing added to the FNO (`use_checkpoint`, identical grads verified), opt-in; AMP stays CUDA-only |

### Section 3 — formulation

| Paper element | Status | Evidence / gap |
|---|---|---|
| C-POMDP, budget + pointwise safety constraints | ✅ | `TaskSpec`: exposure (u_max) + budget cost terms |
| **distributional equity constraint** (Sec 3 + Sec 8 "first-class") | ✅ | `TaskSpec.equity_cost` (variance of per-sub-region harm) as a real g_2 with its own PID dual in the planner; `equity.enabled=true`. Off by default so Phase 2 stays a controlled single-constraint comparison |

### Headline reality check

- **Zero real-world datasets.** Every number in the repo is synthetic PDE +
  synthetically-rendered imagery.
- **Zero real backbones.** Every perception/explanation number is from the
  offline `tiny` stub (watermarked `backbone_is_stub: true`).
- **One real baseline comparison** (constrained planning on `dar`), single seed.
- Cross-domain transfer, the paper's contribution #5, has **never been run**.

---

## 2. Plan

Ordered by credibility-per-effort. The two things a reviewer checks first are
"does the surrogate beat published numbers on a real benchmark" and "does the
cross-domain claim hold up" — so those lead.

### STATUS (this pass)

Done: **E** (equity constraint, resolution-generalization eval, gradient
checkpointing — all local, tested), **B** (cross-family transfer matrix — run,
surrogate fidelity only), **A** (PDEBench + EuroSAT loaders, downloaders, tests,
full-path proof on synthetic HDF5), **F** *machinery* (`eval/run_seeds.py`,
`make seeds`: the ablation sweep across N seeds, aggregated to mean ± sample std
by `aggregate_seeds` in `eval/metrics.py`, resumable per seed).

Remaining: the real multi-GB PDEBench download and the real VLM runs are Colab
steps (notebook + loaders ready); **C/D** (real backbones) outstanding; **F**
now needs compute, not code — every table in the README is still single-seed
until its sweep is rerun through the runner.

### Phase A — Real PDE benchmark for the surrogate (highest value)

**Dataset: PDEBench** (Takamoto et al., NeurIPS 2022; open, ~GB scale). It ships
exactly our three families — diffusion-reaction, shallow-water, advection/Burgers
— with *published* FNO/U-Net baseline errors. This turns Phase 1 from
"consistent with our own solver" into "competitive on a community benchmark",
and it is the single highest-credibility real-data integration.

Steps:
1. `pspe/simulate/pdebench.py`: loader for PDEBench HDF5 → the existing
   `(states, controls)` tensor schema (controls zero where the benchmark has no
   forcing). Download script + checksum, gitignored data.
2. Wire `ensure_dataset(source="pdebench", ...)` so `train_simulate.py` runs
   unchanged on it.
3. Train FNO/DeepONet/GNOT on the 2D shallow-water and diffusion-reaction sets;
   report relative L2 next to PDEBench's published numbers in `eval/`.
4. Acceptance: our FNO within a stated factor of PDEBench's reported FNO error.

Runs on the existing CPU/MPS path for small resolutions; a Colab GPU for the
full sets. No backbone needed.

### Phase B — Execute cross-domain transfer (contribution #5)

Code exists (`multifamily.py`, `transfer_gap`); it has never been run.
1. Train one padded surrogate per source family (`train.padded=true`).
2. Run the full transfer matrix (dar→{swe,rdf}, etc.) zero-shot and few-shot,
   reporting the transfer gap on **surrogate fidelity and planning reward**, not
   just fidelity (extend `transfer_gap` to roll a planner in the target).
3. Acceptance: a populated 3×3 transfer table with the diagonal (in-family) as
   the reference, in `runs/transfer/`.

Pure compute on existing code. Do right after A.

### Phase C — Real backbones on real imagery (Perceive)

Needs a GPU (Colab/Kaggle free tier); cannot run meaningfully on this CPU box.
1. Perception dataset with paired imagery → continuous field. Best open fit:
   **Sentinel-2 → a continuous index** (NDVI/NDWI/land-surface-temperature),
   the field the encoder regresses. Alternatively **WeatherBench/ERA5** patches
   (reanalysis field with a satellite-like channel) for a cleaner image→field
   pair. Land on one; wire it through the existing `source="real"` loader
   (already present in `perceive/dataset.py`).
2. Run `train_perceive.py model.backbone=Qwen/Qwen2-VL-2B-Instruct data.source=real`
   on Colab. First non-stub perception number (`backbone_is_stub: false`).
3. Add the two missing perception baselines (Section 7.2):
   - CNN/ViT regression, regression-term-only (contrastive off) — mostly a
     config flag plus a small ViT head;
   - zero-shot VLM (prompt the frozen VLM for the field statistic, no adapter).
4. Acceptance: field-reconstruction error for {zero-shot, CNN/ViT, PSPE-LoRA}
   on held-out real imagery, plus agreement with held-out captions.

### Phase D — Explanation: real LM, post-hoc baseline, human study

1. `train_explain.py model.backbone=Qwen/Qwen2.5-1.5B-Instruct` on Colab, on a
   *trained* planner (not the smoke policy) so briefs are non-degenerate.
2. Post-hoc baseline (Section 7.2): generate briefs from the same frozen LM
   *after* the policy is fixed, no faithfulness loss — the trained-in vs
   post-hoc control. Report F(b) for both.
3. Run the human study: `eval/human_rating.py` on the real briefs with
   3–5 volunteers; report Likert correctness/completeness/actionability and its
   correlation with F(b).
4. Acceptance: F(b) trained-in > post-hoc, with a human-score correlation.

### Phase E — Formulation + metric gaps

1. **Equity constraint** (Section 3, flagged "first-class" in Section 8): add a
   distributional cost functional to `TaskSpec` — e.g. variance of intervention
   benefit across a partition of the domain into sub-regions — and a
   corresponding dual. Small, self-contained, and closes a stated contribution.
2. **Resolution-generalization eval** (Section 7.4): evaluate a surrogate
   trained at 64² on 96²/128² inputs (FNO is discretization-invariant, so this
   is a real test) and report the error curve.
3. **Compute-budget honesty** (Section 7.5): enable gradient checkpointing; keep
   AMP as-is but state clearly it is CUDA-only.
4. Bump the headline resolution to 128² (16,384 cells) to match the paper's
   "tens of thousands", keeping 64² as the free-tier default.

### Phase F — Statistical rigor (before any table is reportable)

Runner built: `eval/run_seeds.py` (`make seeds`) reruns `eval/run_ablations.py`
once per seed in a fresh process — so the seed is set before any module touches
the RNG — and collapses the per-seed `results.json` files into one mean ± sample
std table (`runs/seeds/results_seeds.md`, with per-seed values kept in
`results_seeds.json`). Interrupted sweeps resume; `--aggregate-only` re-tables
what already ran; `--only` selects the ablation groups.

Scope note: `ensure_dataset` caches by (testbed, grid), so all seeds share the
PDE dataset. The reported spread is training stochasticity — initialisation,
minibatch order, policy sampling, evaluation rollouts — not dataset
resampling. Dataset variance is a separate, larger study and should be labelled
as such if ever run.

**Phase 2 sweep: run** (5 seeds, Vista GH200 array, `runs/seeds_full/`). It
changed two conclusions:

* the return advantage over all four safe-RL baselines is confirmed and large
  (paired t = +9.7 … +23.4, df = 4, every seed);
* **"all five respect the constraint" was a single-seed artifact.** The planner
  violates the limit on 1/5 seeds with adaptive α and 2/5 with fixed α (worst:
  episode cost 8.105 against a 0.936 limit, λ = 19.1), while every baseline
  satisfies it on every seed. Corollary 1 does not hold uniformly at 200
  iterations, and that is now the most important open item in the repo — it is
  a safety claim, not a performance one.
* adaptive vs fixed α is a *variance* effect, not a mean one (t = +1.47, n.s.);
  adaptive's contribution is suppressing the fixed-α blow-up. The ablation as
  written in the paper ("adaptive improves return") is not supported.

**Remaining sweeps: run** (`runs/seeds_rest/`, `runs/transfer_seeds/`, both
5-seed GH200 arrays at full budget). Three more single-seed claims did not
survive:

* **physics-informed loss: no effect** on final accuracy (t = +0.52, df = 4).
  The 2.8× advantage in the smoke table is a small-budget convergence-speed
  effect, not an accuracy one. Section 7.3 must say which it is claiming.
* **frozen perception beats fine-tuned** (t = −3.97), the opposite of the
  paper's expectation — though measured on the stub backbone, so it is a claim
  about the stub until Phase C runs.
* **faithfulness loss: no separation** (t = +0.67); the apparent gain is one
  seed out of five. Eq. 11 is unsupported until a real LM runs.
* **transfer magnitudes into `swe` are not measurements** — σ ≈ mean (dar→swe
  27.4 ± 26). The direction survives, the numbers do not. `swe → dar` is
  consistent with no penalty at all (0.087 ± 0.14).

What survives multi-seed scrutiny: the planner's return advantage over all four
baselines, and FNO ≫ DeepONet/GNOT (0.057 vs 0.256/0.268, tight).

Left: the violation finding needs a diagnosis (dual gains? step size? PID windup
at this budget?) before the Phase 2 table can be presented as a safety result.
That is now the single most important open item — it is a safety claim, and one
seed reproduces it.

---

## 3. What can run here vs needs a GPU

| Phase | Here (CPU/MPS) | Needs GPU (Colab/Kaggle) |
|---|---|---|
| A PDEBench surrogate | small resolutions | full sets |
| B cross-domain transfer | ✅ (existing code) | faster |
| C real Perceive | ❌ | ✅ real VLM |
| D real Explain + study | ❌ | ✅ real LM (study is manual) |
| E equity / resolution / grad-ckpt | ✅ | — |
| F multi-seed | ✅ (slow) | ✅ |

**Suggested order: A → B → E → F locally, then C → D on Colab.** A and B are the
highest-credibility, fully-local, and unblock the "beats baselines on a real
benchmark" and "cross-domain transfer works" claims — the two that carry the
paper.
