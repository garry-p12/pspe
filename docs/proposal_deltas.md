# Proposal ↔ implementation deltas

Places where this repository and the written proposal disagree, and which one
should change. Kept as a checklist so the paper draft and the code cannot drift
apart silently — a reviewer comparing an equation in the paper against the
metric in the code will catch any of these immediately.

Status: **the proposal text has not yet been edited.** Every row below is an
outstanding edit to the proposal, the code, or both.

---

## 1. Faithfulness formula — edit the proposal

| | |
|---|---|
| Proposal says | `F(b_t) = 1 - D_KL(π_φ ‖ π̂_{b_t})` |
| Code does | `F(b_t) = exp(-D_KL(π_φ ‖ π̂_{b_t}))` |
| Resolution | **Change the proposal to the exponential form** |

Why the code is right: `1 - KL` is unbounded below, so a badly wrong brief
scores arbitrarily negative. That breaks the metric's comparability across runs
and, more seriously, breaks it as a REINFORCE reward — a single catastrophic
sample dominates the batch advantage. `exp(-KL)` is bounded in (0, 1], monotone
in KL exactly as the linear form is, and agrees with it to first order at small
KL, which is where a converged model lives.

Edit needed: the faithfulness equation in the detailed proposal, and any prose
describing F's range (it is (0, 1], not "near 1").

Implemented in [`pspe/explain/faithfulness.py`](../pspe/explain/faithfulness.py).

---

## 2. Cross-PDE-family transfer — code fixed, proposal now matches

| | |
|---|---|
| Proposal says | cross-PDE-family transfer (novelty #5, Phase 5 deliverable) |
| Code did | within-family parameter shift only; cross-family refused, because dar/swe/rdf have 1/3/2 state channels and a bespoke surrogate cannot be rolled out across arities |
| Resolution | **Code fixed.** `make_surrogate(..., padded=True)` builds a channel-padded shared head |

[`pspe/simulate/multifamily.py`](../pspe/simulate/multifamily.py) pads every
state to `MAX_CHANNELS = 3` and passes a validity mask, so one head trains on
`dar` and rolls out on `swe` or `rdf` with no weight changes. `transfer_gap`
now scores cross-family transfer whenever the surrogate supports it, and still
refuses (with a reason) for a single-family model rather than inventing a
number.

Both protocols are reported, and they measure different things:

* **cross-family** — `dar → swe`: different PDE, different arity. The
  proposal's novelty claim.
* **parameter shift** — `dar → dar` with different diffusivity/advection/
  reaction rate. A weaker, within-family generalisation check.

Remaining proposal edit: state explicitly which of the two the headline
transfer number refers to. They are not interchangeable, and the cross-family
number will be much worse.

---

## 3. Grid resolution — decide, then align

| | |
|---|---|
| Proposal says | testbeds with "tens of thousands of grid cells" |
| Code does | 64×64 = **4,096** cells by default |
| Resolution | either bump the default to 128×128 (16,384 cells) or soften the proposal |

128×128 satisfies the proposal's wording and is a one-flag change
(`train.grid=128 data.grid=128`); datasets are grid-scoped, so both resolutions
can coexist on disk. The cost is roughly 4× per training step, which pushes
Phase 1 outside a comfortable free-tier session on CPU but is fine on any GPU.

Recommendation: keep 64×64 as the free-tier default, run the headline numbers
at 128×128, and change the proposal's compute section to state both — the
"reduced resolution keeps this inside free-tier compute" argument is already
made there, so it costs nothing to be specific.

---

## 4. Safe-RL baselines are reimplementations — disclose in the paper

The proposal names OmniSafe as the source of CPO, PID-Lagrangian, Sauté RL and
primal-dual NPG. OmniSafe does not install against Python 3.11+/gymnasium≥0.29,
so all four are implemented in [`baselines/safe_rl.py`](../baselines/safe_rl.py).

This must be stated in the paper, not just the repo README: reimplemented
baselines are a legitimate methodological choice but an undisclosed one is not.
Cite the original papers (Achiam et al. 2017; Stooke et al. 2020; Sootla et al.
2022), state that the implementations are the authors' own and share the
environment, reward, cost and policy code with the proposed method, and point
at the validation described in item 5.

---

## 5. Baseline validation — partially done, one gap remaining

**Four bugs** were found by validating against a CMDP with a closed-form
optimum ([`baselines/toy_cmdp.py`](../baselines/toy_cmdp.py)). Three of the four
algorithms were wrong; only primal-dual NPG was correct as first written. Every
one of these would have corrupted the Phase 2 comparison silently:

* **Sauté RL** zeroes reward in the absorbing unsafe state, which only
  penalises the agent when rewards are non-negative. All PSPE rewards are ≤ 0,
  so violating the budget was the *highest-reward* outcome available. It
  converged to the unconstrained optimum while appearing to run correctly.
  Fixed by shifting rewards by a running floor.
* **PPO-Lagrangian** normalised the *combined* advantage `adv_r - λ·adv_c` to
  unit variance, which rescales away the multiplier's magnitude — the dual
  controller then chases a target it cannot move. Fixed by normalising the two
  advantages separately and blending as `(adv_r - λ·adv_c)/(1+λ)`. The default
  dual gains were also too weak to reach the optimal multiplier within any
  realistic iteration budget.
* **CPO** scaled the cost advantage to unit variance, putting the cost gradient
  in different units from the constraint surplus `c` that the dual solution
  mixes it with. This produced a persistent 0.11 action bias that did not shrink
  between 300 and 900 iterations — the signature of bias rather than slow
  convergence.
* **CPO** had no slack branch: with the constraint comfortably satisfied,
  `2δ - c²/s` turns negative, λ explodes and the step collapses to zero, so CPO
  froze precisely when it was safe and should have been improving reward. The
  same expression also took `sqrt` of a negative number in two places
  (degenerate denominator; a Cauchy–Schwarz numerator made negative by inexact
  CG). Fixed by reducing to the TRPO step when slack, and clamping both.

After the fixes, CPO converges to within 0.0002 of the closed-form optimum.
Before them it sat at 0.519 against a 0.400 optimum with a 0.519 episode cost
against a 0.400 limit — a 26% steady-state constraint violation that would have
read as "CPO is a weak baseline" rather than "CPO is misimplemented".

Remaining gap: none of this is a scale test.
[`baselines/validate_safety_gym.py`](../baselines/validate_safety_gym.py) runs
all four on SafetyPointGoal1-v0 against published reward/violation trends, but
**it has not been run** — `safety-gymnasium` does not install in this
environment (its `pygame` dependency fails to build a wheel on macOS/arm64).
Run it on Linux or Colab before reporting any Phase 2 comparison.

---

## 6. The testbed constraints do not bind — fix before reporting Phase 2

Measured on `dar` at the default settings, every planner and every baseline
finishes with an episode cost around **0.16** against a `cost_limit` of **2.0**,
and a violation rate of exactly 0:

| run | episode cost | limit | violation rate |
|---|---|---|---|
| plan (adaptive α) | 0.201 | 2.0 | 0.00 |
| ppo_lagrangian | 0.159 | 2.0 | 0.00 |
| cpo | 0.158 | 2.0 | 0.00 |
| saute | 0.163 | 2.0 | 0.00 |
| primal_dual_npg | 0.162 | 2.0 | 0.00 |

The constraint is slack by more than 10×, so the dual variable stays at zero
and every algorithm is effectively running unconstrained. A Phase 2 table built
on this compares five *unconstrained* learners and would show no separation on
the metric the proposal cares about — which is exactly what the current numbers
show.

Fix before generating any reportable Phase 2 result: tighten `cost_limit` in
[`pspe/envs/task.py`](../pspe/envs/task.py) per testbed until an unconstrained
policy actually violates it (calibrate by running the planner with the dual
disabled and setting the limit below its achieved cost), so the constrained
comparison has something to compare. The `TaskSpec` thresholds `u_max` and
`budget` are the other two knobs.

---

## 7. Stand-in backbones — never cite these numbers

Phases 3 and 4 default to `backbone=tiny`, small in-tree transformers that run
offline. They exercise the full training path but are **not** Qwen2-VL /
Qwen2.5 / Phi-3.5 results. Every run records `backbone` and
`backbone_is_stub` in its `summary.json`; any number destined for the paper
must come from a run with `backbone_is_stub: false`.
