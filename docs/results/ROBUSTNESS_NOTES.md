# Robustness checks, and what they changed

Running log of the checks applied to each claim, including the ones that
overturned a result I was about to report. Kept because two of them caught
errors that would have survived into a submission.

## 1. Baselines must start where the method starts

**The near-miss.** The first constrained-RL comparison on the wildfire task
had PPO-Lagrangian, CPO, Sauté and primal-dual NPG all treating 22–38% of the
sector against a 3% budget, on every held-out fire, after 200 iterations with
the dual fix our own planner got. Four algorithmically distinct methods failing
identically looked like a structural finding: constrained RL cannot satisfy a
budget that binds on the sum of 64 continuous action components, while
decision-time optimisation with an explicit projection satisfies it by
construction.

**Why it was wrong.** Under the squared intensity map, a randomly initialised
policy (pre-tanh mean ~ 0) emits intensity ((0+1)/2)^2 = 0.25 — i.e. **25%
treated**, almost exactly the observed numbers. The agents had barely moved off
their initialisation on the cost axis. Meanwhile PSPE's per-instance optimiser
*starts* at the uniform budget allocation and projects onto the budget after
every step, so it never travels that distance at all. The comparison was
measuring an initialisation, not an algorithm.

**The check.** `--fair-init` sets the baselines' policy bias so their initial
action already meets the budget — the same starting position PSPE gets. With
it, PPO-Lagrangian holds 2.99% against the 3% limit with zero violations after
**three** iterations.

**What this changes.** The structural claim is withdrawn before it was ever
made. The comparison reverts to what it should have been: decision quality at
matched budget, with the baselines given the same starting advantage. The
initialisation sensitivity is still worth one sentence in the paper — it is a
real property of this action parameterisation — but it is not a result about
constrained RL.

## 2. Every comparison is scored at matched budget

A method that ignores the budget reduces more burn for free, so a raw table of
burn reduction cannot be read as a comparison. Every run now also reports a
**matched-budget pass**: each plan is rescaled onto the budget before scoring,
so the reduction column is apples-to-apples whatever the method did during
training. Both columns are reported; the matched one is the comparison.

## 3. The result must survive removing the hand-specified physics

The fire action model has a learned half (fuel removed from the NDVI channel,
the surrogate decides the effect) and an imposed half (spread suppressed in
treated cells). Setting `--block 0` removes the imposed half. The planner's
advantage over the heuristic **widens** from +12.4 to +13.6 points, because the
heuristic loses 76% of its effect without the imposed physics and the planner
loses 47%. The claim rests on the learned model, which is what it asserts.

## 4. A margin must bound the distribution that actually violates

The first conformal margin took a quantile over the **surrogate's cost error**
and made rdf worse (18.2% violating against a 14.5% baseline). On rdf the
surrogate is accurate; what violates the limit is the policy's own
episode-to-episode spread, which that quantile never sees. Conformalising the
deviations of individual real episode costs from their probe mean gives a
margin that holds at delta = 0.1 on both families (0% and 7.3%).

## 5. A faithfulness metric must be checked against a permutation

Scoring each brief against a *different* state's action leaves the score
unchanged to four decimals on both testbeds and every arm. The briefs carry no
state-specific information, and the published comparison was measuring format
learning. See `EXPLAIN_PERMUTATION.md`.

## 6. A testbed must be able to express the effect being measured

The trained dar policy varies 1.2% across states and produces 8 distinct
actions in 64 states on the brief quantisation grid. A constant brief is
therefore near-optimal there, and the permutation gap is zero whatever the
generator does. Any testbed used for an explanation metric needs its action
diversity measured first; `plan_diversity` in `eval/run_ndws_explain.py` is
that check.

## Standing rules adopted from these

1. A baseline starts from the same initial condition as the method, or the
   comparison reports an initialisation.
2. Constrained comparisons are scored at matched constraint, always.
3. Any result that depends on a hand-specified component is re-run with that
   component removed.
4. A margin or bound names the distribution it bounds, and that distribution is
   the one that produces the failure.
5. Any metric comparing a generated artefact to a target ships with a
   permutation control.
6. A testbed is calibrated for the variation a metric needs before the metric
   is run on it.

## 7. A testbed needs a controllable objective, not just a binding constraint

**What the calibration checked.** `scripts/calibrate_constraints.py` measured
the cost of doing nothing and the cost a reward-greedy policy incurs, and put
the limit between them. That guarantees the constraint *binds*.

**What it never checked.** Whether the *return* separates — whether actuation
can move the objective at all. A testbed can pass the first test and fail the
second, and then every planning result on it is noise. `return_separation` is
now reported alongside the cost spread.

**swe, measured.** Three probes:

| probe | result |
|---|---|
| 400 random constant action vectors | not one beats doing nothing |
| hand-written state-feedback damper | +0.0098 (5.5% of the objective) |
| trained reward-greedy planner | +0.0122 (7% of the objective) |

So the task *is* controllable, and the planner *does* find control — it beats
both doing nothing and the hand damper. The margin is simply ~7%, and
differences between planning methods live inside it. That is why swe's
joint / disaggregated / unanchored arms all returned -0.1733 to four decimals:
not because the policy never moved, but because the whole span of achievable
outcomes is narrower than the noise between arms.

**A correction.** Earlier notes in this project said "the swe policy never
leaves its initial behaviour". That was wrong. The policy moves and improves;
the task's controllable band is just too narrow to rank methods within.

**The fix that does not work.** Raising actuator amplitude grows the margin
available to a *deterministic* controller (5.5% at amplitude 1.0 to 32.6% at
4.0) but does nothing for a stochastic policy, because amplitude scales the
useful signal and the exploration noise equally: planner separation is +0.0122
at amplitude 1.0 and +0.0081 at 4.0. Amplitude was therefore left at 1.0.

**What would work.** A different objective, not a louder actuator — one whose
optimum is far from zero action, e.g. tracking a non-trivial target profile
that requires sustained forcing, rather than driving height to zero in a system
where forcing only injects energy.

**Standing rule 7.** A testbed is admitted for method comparison only if the
span between doing nothing and the best achievable return is wide relative to
the differences being measured. Report that span.
