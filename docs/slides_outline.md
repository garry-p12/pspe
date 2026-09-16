# PSPE — slide outline

Four slides. Every number traceable to a run under `runs/`. Full detail in
`docs/technical_report.md`.

---

## Slide 1 — The problem and the system

**Title:** Constrained intervention design for PDE-governed hazards

- Forecasting models (FourCastNet, GraphCast, ClimaX) answer *what will happen*.
  PSPE targets the next step: **what to do about it, under a hard budget**.
- Four modules, one loop:

| Perceive | Simulate | Plan | Explain |
|---|---|---|---|
| imagery → physical field | differentiable surrogate | constrained policy | natural-language brief |
| frozen VLM + LoRA | FNO / U-Net | hybrid gradient + PID-Lagrangian dual | LM + faithfulness objective |

- Constraints are first-class: intervention **budget**, pointwise **safety**
  ceiling, and a **distributional equity** term (harm must not concentrate in
  one sub-region).
- Testbeds map to real hazards: reaction-diffusion → wildfire,
  shallow water → flood, advection-diffusion → air quality.

*Visual: the four-module loop, with the constraint set attached to Plan.*

---

## Slide 2 — Method and evaluation infrastructure

**Title:** Built to be falsifiable

- **Hybrid gradient:** pathwise (low variance, biased by surrogate error) mixed
  with likelihood-ratio (unbiased, high variance); mixing coefficient α adapts
  from the measured variance ratio.
- **PID-Lagrangian dual** enforces the cost constraint; a second dual handles
  equity.
- **Faithfulness** `F(b) = exp(−KL(π ‖ π̂_b))` — the brief is optimised to imply
  the policy that produced it, not written after the fact.
- **Evaluation discipline** (the part that changed the conclusions):
  - 5 seeds by default, paired *t*-tests, mean ± std — one command
  - GH200 job arrays: a 5-seed sweep in **9.5 min** vs **~9 h** on CPU
  - 150+ tests, including regression tests for measurement bugs

**Three measurement bugs found, all invisible at single-seed scale:**

| bug | effect |
|---|---|
| constraint read from one final evaluation | identical runs scored "0% violation" or "100%" on timing luck |
| evaluation perturbed training (global RNG) | baseline returns shifted −2.51 → −2.30 |
| optional-dep tests skipped silently | "tested" loader failed on first real use |

*Visual: before/after of the violation metric — spiky curve vs single snapshot.*

---

## Slide 3 — Results: what survived multi-seed scrutiny

**Title:** Claims tested against their own reruns

**Survived**

| claim | evidence |
|---|---|
| Beats 4 safe-RL baselines on return | paired t = **+3.84 to +17.95** (df=4) vs CPO, PPO-Lag, Sauté, PD-NPG |
| **12.5× fewer real environment samples** | 3,072 vs 38,400 transitions — model-based planning |
| Competitive on a real public benchmark | PDEBench 2D shallow-water: FNO **0.0098** rel L2 rollout (published FNO 0.0044 nRMSE) |
| Trained-in explanations beat post-hoc | F(b) 0.168 vs 0.101, **t = +5.53** |

**Did not survive — reported, not buried**

| claim | finding |
|---|---|
| Constraint satisfaction | violates on **7.3%** of evaluations; all baselines **0%** |
| Physics-informed loss improves accuracy | t = +0.52 — a small-budget *convergence-speed* effect only |
| Adaptive α improves return | t = +1.66 (n.s.) — it improves the **tail** (std 0.029 vs 0.365) |
| Faithfulness *term* drives faithfulness | t = −1.40 — training on briefs does, the objective adds nothing |

**Safety failure — root cause and fix.** The dual was controlled on **surrogate**
cost while violation is realised under **true** dynamics. Fix: probe the real
environment every 20 iterations + plan against a measured error margin.

| | violating evals | worst cost | return | real samples |
|---|---|---|---|---|
| before | 1.8–7.3% | up to 1.73 | −2.30 | 3,200 |
| **after (probe + margin)** | **0% on all 5 seeds** | 0.29 | −2.30 (unchanged) | 4,160 (still 9× fewer) |

Neither half works alone — the margin needs a measurement to be conservative about.

*Visual: two-column survived / did-not-survive ledger.*

---

## Slide 4 — Real climate data, and where operators stop working

**Title:** From simulated testbeds to observed hazards

- **PDEBench** (6.2 GB, real 2D shallow-water): FNO ≫ DeepONet ≫ GNOT —
  0.0098 / 0.0137 / 0.0543 rel L2. Ranking reproduces the synthetic result.
- **Next Day Wildfire Spread** (2.3 GB, ~18k real US fire patches, 64×64 km):
  first dynamics result scored against **observed** rather than simulated truth.
  Pure-numpy TFRecord reader — no TensorFlow dependency.

| model on NDWS (3 seeds) | AUC-PR | vs published 0.284 |
|---|---|---|
| U-Net | **0.277 ± 0.002** | **97.5%** |
| Hybrid (gated FNO + U-Net) | 0.235 ± 0.005 | 83% |
| FNO alone | 0.200 | 70% |
| all-zeros floor | 0.011 | — |

**The finding worth presenting:** the Fourier operator, which is excellent on
smooth PDE fields, **loses to a plain U-Net on sharp fire fronts** (paired
t = +12.5) — a spectral model truncates the high frequencies that *are* the
signal. Given a learned gate over the spectral path, the model settles it at
**0.12 on every seed**: offered the operator, it declines.

Also confirmed with real backbones: SigLIP perception (frozen probe best at
reconstruction, LoRA + contrastive wins alignment 5×), Qwen explanations
(trained-in 0.47 vs post-hoc 0.18, t = +21), and FNO resolution invariance flat
from 64² to **128²** (16,384 cells).

→ PSPE is the four-module loop, not one architecture. **Operators where dynamics
are smooth; convolutions where the front is sharp.**

**Next:** flood testbed (Sen1Floods11 + real DEM), constraint-fix validation,
human expert study on real briefs.

*Visual: side-by-side — smooth SWE field (FNO wins) vs binary fire mask (U-Net wins).*

---

### Sources for citation slide
Li et al. ICLR 2021 (FNO) · Lu et al. NMI 2021 (DeepONet) · Takamoto et al.
NeurIPS 2022 (PDEBench) · Huot et al. TGRS 2022 (NDWS) · Achiam et al. ICML 2017
(CPO) · Stooke et al. ICML 2020 (PID-Lagrangian) · Sootla et al. ICML 2022 (Sauté)
