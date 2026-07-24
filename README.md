# PSPE — Perceive · Simulate · Plan · Explain

A research stack for physics-grounded, constraint-aware, self-explaining
planning. Four modules, three PDE testbeds, one end-to-end loop.

**No paid APIs anywhere.** No OpenAI/Anthropic/Google keys, no hosted
simulation service, no paid tracker. Every model is open-weight and runs
locally; every solver is computed in-process; experiment tracking is
TensorBoard on disk.

---

## The four modules

| Module | What it does | Implementation |
|---|---|---|
| **Perceive** | imagery → physical field estimate | frozen open-weight VLM + LoRA, regression + contrastive-to-weak-text loss |
| **Simulate** | differentiable reduced-order surrogate | in-tree FNO (spectral truncation = the reduced order), physics-informed residual + multi-step rollout consistency |
| **Plan** | constrained intervention under a cost budget | hybrid pathwise/likelihood-ratio policy gradient with variance-optimal adaptive mixing, PID-Lagrangian dual |
| **Explain** | operator brief, scored for faithfulness | frozen small LM + LoRA, frozen brief parser, `F(b) = exp(-KL(π ‖ parse(b)))` in the training loss |

### Testbeds

| id | PDE | channels |
|---|---|---|
| `dar` | diffusion–advection–reaction | 1 (`u`) |
| `swe` | linearised shallow water | 3 (`h, u, v`) |
| `rdf` | FitzHugh–Nagumo front propagation | 2 (`u, v`) |

All on a periodic 64×64 grid with an RK4 integrator, in PyTorch — so the
ground-truth solver is itself differentiable and its `rhs` doubles as the
physics residual for the surrogate. One definition, no drift.

---

## Quick start

```bash
make setup            # conda env "pspe" (python 3.11) + deps + editable install
make smoke            # environment report + one pass through all four modules
make test             # fast suite, ~4 min on CPU
make test-all         # + slow safe-RL convergence checks (~15 min)
make canary           # FNO vs published Burgers numbers; run once, not per-commit

make data             # generate all three PDE testbeds
make train-simulate   # Phase 1
make train-plan       # Phase 2
make train-perceive   # Phase 3
make train-explain    # Phase 4
make train-e2e        # Phase 5: the whole loop
make ablations        # the full results table
make tb               # tensorboard on runs/
```

On **Colab or Kaggle**, where conda is not first-class, skip `make setup`:

```bash
pip install -q -r requirements-notebook.txt && pip install -q -e .
```

Every script is Hydra-driven, so any config key overrides on the command line:

```bash
python scripts/train_simulate.py testbed=rdf surrogate=gnot train.epochs=30
python scripts/train_plan.py planner.adaptive_alpha=false
python scripts/train_perceive.py model.backbone=Qwen/Qwen2-VL-2B-Instruct
python eval/run_ablations.py --testbed dar --full
```

---

## Open-weight backbones

Phases 3 and 4 default to `backbone=tiny`: small in-tree transformers that run
on CPU with no downloads. They exercise the *entire* training path — frozen
backbone, LoRA adapters, parameter-count assertion, faithfulness objective — so
the pipeline and its acceptance checks are verifiable offline. They are
stand-ins for the language models, not claims about them.

Every run records `backbone` and `backbone_is_stub` in its `summary.json`, and
the results table surfaces a `stub backbone` column. **No number with
`backbone_is_stub: true` belongs in a paper draft.**

Swap in the real backbones with one flag:

```bash
pip install -e ".[llm]"
python scripts/train_perceive.py model.backbone=Qwen/Qwen2-VL-2B-Instruct   # or vikhyatk/moondream2
python scripts/train_explain.py  model.backbone=Qwen/Qwen2.5-1.5B-Instruct  # or microsoft/Phi-3.5-mini-instruct
```

`quant=4bit` is honoured on CUDA hosts. On macOS/arm64 it is ignored, because
`bitsandbytes` ships no wheels for that platform — the model loads unquantised
rather than failing.

---

## Deviations from the plan, and why

Substitutions, each because the named dependency does not currently install or
run in this environment. All are one flag away from being swapped back.

**`omnisafe` → `baselines/safe_rl.py`.** OmniSafe does not install cleanly
against Python 3.11+/gymnasium≥0.29. CPO, PID-Lagrangian PPO, Sauté RL and
primal-dual NPG are implemented in-tree instead — all four sharing this repo's
env, reward, cost and policy code, which makes the comparison *tighter* than a
cross-library one would be. The env exposes cost via `info["cost"]`, the
Safety-Gymnasium convention, so an OmniSafe install drops in unchanged.

Reimplemented baselines are the highest-risk thing in this repo: these
algorithms run happily while being subtly wrong. They are therefore validated
against a CMDP with a **closed-form optimum** rather than only against each
other — see [Baseline validation](#baseline-validation) below, which describes
the two real bugs that check found.

**Faithfulness formula.** The proposal writes `F(b) = 1 - KL`; the code uses
`F(b) = exp(-KL)`. Deliberate: `1 - KL` is unbounded below, so it has no fixed
scale as a metric and lets one catastrophic sample dominate the REINFORCE
advantage. The exponential form is bounded in (0, 1], monotone in KL, and
agrees to first order at small KL. **The proposal text needs updating to
match** — tracked in [docs/proposal_deltas.md](docs/proposal_deltas.md).

**`neuraloperator` → in-tree FNO.** Removes a build dependency from Phase 1.
`surrogate=fno_neuralop` uses the reference implementation when
`pip install -e ".[extras]"` is present.

**`py-pde` → in-tree solvers.** The solvers had to be differentiable PyTorch to
serve as both ground truth and physics residual; `py-pde` is NumPy-side. It
remains an optional extra.

**`bitsandbytes`** is not pinned at all: CUDA-only, no macOS/arm64 wheels.

---

## What is checked, and where

| Phase | Acceptance check | Enforced by |
|---|---|---|
| 0 | all four modules import and run | `tests/test_phase0_imports.py`, `make smoke` |
| 1 | relative L₂ rollout error vs numerical truth, all three testbeds | `surrogate_fidelity`, `tests/test_simulate.py` |
| 2 | return, violation rate, sample count vs four baselines | `HybridPlannerTrainer.evaluate`, `baselines/run_safe_rl_baselines.py` |
| 3 | held-out field error **and** backbone provably frozen | `assert_lora_only` — raises, not just logs |
| 4 | `F(b)` logged during training, not post hoc | `tests/test_explain.py` reads it back out of `metrics.jsonl` |
| 5 | one command reproduces the results table | `eval/run_ablations.py` |

Two checks sit outside the phase table because they validate the
*implementations* rather than the results: the safe-RL baselines against a
closed-form CMDP, and the FNO against published Burgers numbers. Both are
described below, and both found real bugs.

Solver stability, the analytic Laplacian, brief round-tripping through the
parser, and the exact value of the hybrid gradient mixture are all asserted
rather than assumed.

### Baseline validation

Comparing four reimplemented algorithms only against each other cannot detect a
wrong update rule — they would be wrong together and the table would look
self-consistent. So `baselines/toy_cmdp.py` fixes a CMDP whose constrained
optimum is known analytically (`a* = min(p, d)`, `λ* = 2(p − d)`), and
`tests/test_safe_rl_correctness.py` checks two levels:

* **analytic** — conjugate gradient against an explicit matrix inverse, the KL
  Hessian-vector product against the analytic Fisher of a Gaussian, the NPG step
  against the trust-region radius it claims to respect, the PID terms against
  hand arithmetic, and CPO's post-update KL against its own `target_kl`;
* **behavioural** — all four must land on `a* = 0.4` with `λ → 0.8`.

Only `primal_dual_npg` was correct as first written. The canary found **four
bugs across the other three**, every one of which would have corrupted the
Phase 2 comparison while producing plausible-looking curves:

| bug | symptom | fix |
|---|---|---|
| **Sauté RL** zeroed reward in the unsafe absorbing state | every PSPE reward is ≤ 0, so violating the budget was the *highest-reward* outcome available; it converged to the unconstrained optimum while looking healthy | shift rewards by a running floor before applying the absorbing rule |
| **PPO-Lagrangian** normalised the *combined* advantage | unit-variance rescaling throws away λ's magnitude, so the dual chased a target it could not move; the policy oscillated to the action bounds | normalise reward/cost advantages separately, blend as `(A_r − λA_c)/(1+λ)`; raise the default dual gains |
| **CPO** scaled the cost advantage to unit variance | `b` then lives in different units from the constraint surplus `c`, biasing the dual solution's trade-off point — a persistent 0.11 action error that did *not* shrink from 300 to 900 iterations | centre the cost advantage without scaling; express `c` in per-step units |
| **CPO** had no slack branch | when the constraint was comfortably satisfied, `2δ − c²/s` went negative, λ exploded and the step collapsed — CPO froze exactly when it was safe and should have been improving reward | reduce to the TRPO step when the constraint is slack and that step keeps it satisfied; clamp the degenerate denominator and the Cauchy–Schwarz numerator instead of taking `sqrt` of a negative |

After the fixes, measured against the closed form (`a* = 0.400`, `λ* = 0.800`):

| algorithm | converged a | error | λ |
|---|---|---|---|
| **cpo** | 0.402 | 0.002 | — |
| **ppo_lagrangian** | 0.410 | 0.010 | 1.007 |
| **primal_dual_npg** | 0.439 | 0.039 | 0.878 |
| **saute** | 0.491 | 0.091 | — |

CPO reaches 0.0001 error at 600 iterations. Sauté's residual 0.09 is expected:
it enforces the budget through state augmentation rather than a dual, so it
converges to a *conservative* interior point (cost 0.309 against a 0.400 limit)
rather than sitting on the boundary.

**The PDE testbed constraints do not currently bind.** On `dar`, every planner
and baseline finishes at episode cost ≈0.16 against a `cost_limit` of 2.0, with
violation rate 0 — the dual stays at zero and all five are effectively running
unconstrained, which is why the Phase 2 table shows no separation. Tighten the
per-testbed `cost_limit` in `pspe/envs/task.py` before generating any
reportable Phase 2 result; see
[docs/proposal_deltas.md](docs/proposal_deltas.md) item 6.

**Still missing:** the toy CMDP is a correctness canary, not a scale test.
`baselines/validate_safety_gym.py` runs all four on SafetyPointGoal1-v0 against
published trends, but **it has not been run** — `safety-gymnasium` fails to
build here (pygame wheel, macOS/arm64). Run it on Linux or Colab before
reporting any Phase 2 comparison.

### FNO correctness canary

Phase 1 scores the surrogate against *this repo's own* solver, which is an
internal-consistency check: a subtly broken spectral convolution would report a
slightly worse relative L2 rather than announcing itself.
`tests/test_fno_canary.py` trains the production `SpectralConv2d` on 1D Burgers
(ν=0.1) — the standard FNO sanity task, where Li et al. report ~1e-3 — and
asserts a much looser bound that separates "works" from "broken". Marked
`canary`, excluded from the regular suite: `make canary`.

---

## Ablation switches

All from the proposal, all one flag:

| Ablation | Flag |
|---|---|
| physics-informed loss on/off | `train.use_physics=false` |
| fixed vs adaptive mixing coefficient | `planner.adaptive_alpha=false` |
| frozen vs fine-tuned perception encoder | `train.freeze_encoder=false` |
| faithfulness loss on/off | `train.use_faithfulness=false` |
| cross-domain transfer | `train.padded=true`, then `pspe.pipeline.transfer_gap` |

### Transfer: two distinct protocols

The three testbeds have different state arities (dar 1, swe 3, rdf 2), so a
bespoke surrogate cannot be rolled out across families —
which would have made the proposal's cross-PDE-family transfer claim
unmeasurable as written. `make_surrogate(..., padded=True)` (config:
`train.padded=true`) fixes that: the core operator always sees `MAX_CHANNELS=3`
channels plus a validity mask, so one head trains on `dar` and rolls out on
`swe` with no weight changes.

* **cross-family** (`dar → swe`) — the proposal's novelty claim. Needs a padded
  surrogate; `transfer_gap` refuses with a reason for a single-family model
  rather than inventing a number.
* **parameter shift** (`dar → dar` at different diffusivity/advection/reaction)
  — a weaker within-family generalisation check, and the default.

These are not interchangeable and the cross-family number will be much worse;
say which one any reported figure is.

---

## Compute

Every run reports wall-clock and peak memory in its `summary.json`. Defaults
are sized for free-tier compute: 64×64 grids, short rollout horizons, LoRA-only
adaptation, mixed precision and gradient clipping on by default. Local GPU is
used when present (CUDA, or Apple MPS — the spectral block falls back to CPU
there, since MPS has no complex FFT kernel), otherwise CPU.

**Grid resolution.** 64×64 is 4,096 cells; the proposal's compute section says
"tens of thousands". Run headline numbers at `data.grid=128 train.grid=128`
(16,384 cells) and keep 64² as the free-tier default — datasets are grid-scoped
(`dar_64.npz`, `dar_128.npz`), so both coexist. The proposal's compute section
should name both figures; tracked in
[docs/proposal_deltas.md](docs/proposal_deltas.md).

Rough CPU costs on an M-series laptop: data generation ~1 min/testbed, Phase 1
~4 min/epoch at 64², the full test suite ~4 min. Phases 2–4 at default settings
are minutes, not hours.

---

## Licenses

The stack is free to use for research, but "free" is per-artifact — check each
model card before any commercial or large-scale deployment, since these terms
do change.

| Artifact | License | Note |
|---|---|---|
| This repository | MIT | |
| PyTorch, NumPy, SciPy, Gymnasium, Hydra, TensorBoard | BSD-3 / Apache-2.0 / MIT | permissive throughout |
| HuggingFace `transformers`, `peft`, `accelerate` | Apache-2.0 | |
| `Qwen2-VL-2B-Instruct` | Apache-2.0 | |
| `Qwen2.5-1.5B-Instruct` | Apache-2.0 | larger Qwen2.5 sizes use the Tongyi Qianwen license — check if you scale up |
| `Phi-3.5-mini-instruct` | MIT | |
| `moondream2` | Apache-2.0 | |
| Sentinel-2 (Copernicus) | free, attribution | |
| Landsat / NAIP (USGS) | public domain | |

Reimplemented algorithms (CPO, PID-Lagrangian, Sauté RL, primal-dual NPG) are
this repo's own code under MIT, written from the published papers; cite Achiam
et al. 2017, Stooke et al. 2020, and Sootla et al. 2022 for the methods.

---

## Layout

```
configs/     Hydra configs, one per phase
data/        generated PDE testbeds; proxy imagery recipes (free sources only)
pspe/
  simulate/  solvers, FNO/DeepONet/GNOT, losses, rollout, trainer
  plan/      policy, hybrid gradient estimator, PID-Lagrangian, trainer
  perceive/  VLM+LoRA encoder, field decoder, weak-text supervision, trainer
  explain/   brief templates, frozen parser, faithfulness objective, LM+LoRA
  envs/      actuator basis, task functionals, Gymnasium + batched envs
  pipeline.py  the end-to-end loop and the transfer-gap protocol
baselines/   safe-RL four; operator comparison runner
eval/        metrics, ablation runner, results table, human-rating harness
docs/        proposal ↔ implementation deltas (outstanding paper edits)
scripts/     one entry point per phase
tests/       79 tests: 73 fast, 5 slow (safe-RL convergence), 1 canary (FNO/Burgers)
```
