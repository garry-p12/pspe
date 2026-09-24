# Experiment 5: certified briefs for firebreak plans on real fire data

Vista, 2026-09-22. `eval/run_ndws_explain.py`, Qwen2.5-0.5B-Instruct with LoRA,
3 seeds. Jobs 1015909 (trained-in + certificate, 28 min) and 1015910 (controls,
18 min); raw output under `runs/ndws_explain_ti/` and `runs/ndws_explain_ctrl/`.

## Setup

The plans are the per-instance firebreak plans of Experiment 4: a real NDWS
fire patch in, 8 x 8 break intensities out, 3% daily budget, three days
ahead. The brief names the treated patches with their intensities, the day-1
treated fraction against the budget, and the expected population-weighted
burn. A frozen parser reads the intensities back and
F(b) = exp(-KL(plan || parsed)) scores the agreement.

- 1,024 training fires planned and explained; 608 held-out fires, never seen
  by the explainer, split into 200 calibration and 400 test.
- Arms as on the synthetic testbeds: `trained-in` (supervised on the plans'
  briefs plus the faithfulness term), `no-faithful` (supervised only),
  `post-hoc` (generator never trained on these plans).
- Briefs name at most 12 patches; the plans treat 6.7 on average.

## The metric had to be fixed first

The first run (`runs/ndws_explain_summed_kl/`) scored **every arm at exactly
0.0** while the reference brief scored 0.985. Cause: F = exp(-KL) with KL
summed over action dimensions. At 9 actuators (dar, rdf) that is fine; at 64
patches a near-miss brief carries KL 15 to 22 and exp(-KL) underflows, so the
metric has no resolution left to rank generators by, and the certificate
certifies the vacuous floor F >= 0.

Fix: report KL per action dimension (`per_dim_kl`, off by default so the
dar/rdf numbers stay comparable). Same quantity in nats per actuator, one
scale whatever the action space. Regression test in `tests/test_explain.py`.
This is a defect in the metric as written in the proposal, not in the
implementation; `docs/proposal_deltas.md` needs the note.

## Results, 3 seeds, held-out fires

| arm | F(b) held-out | KL per dimension |
|---|---|---|
| trained-in | 0.814 ± 0.039 | 0.218 |
| no-faithful (supervised on the plans only) | 0.815 ± 0.016 | 0.208 |
| post-hoc control | 0.741 ± 0.005 | 0.299 |
| reference brief (the ceiling) | 0.9996 | — |

Paired t (df 2): trained-in vs post-hoc **+3.61**; no-faithful vs post-hoc
**+6.12**; trained-in vs no-faithful **-0.01**.

Conformal certificate for the trained-in explainer, 200 calibration / 400
test held-out fires:

| delta | certified floor | empirical coverage | nominal | holds |
|---|---|---|---|---|
| 0.2 | F >= 0.747 ± 0.053 | 0.795 | 0.80 | 2 of 3 seeds |
| 0.1 | F >= 0.706 ± 0.063 | 0.900 | 0.90 | 3 of 3 |
| 0.05 | F >= 0.680 ± 0.071 | 0.956 | 0.95 | 3 of 3 |

## The caveat that decides how this is reported

The generated briefs are nearly identical across different fires. Two
held-out examples from seed 0 (`runs/ndws_explain_ti/seed_0/briefs_heldout.jsonl`)
name the same two patches with the same amplitudes; only the cost and reward
lines track the plan:

    generated: step 4 . actuator 43 at 0.44 0.44 set to +0.60 ... actuator 52
    at 0.56 0.81 set to +0.60 ... predicted cost 3.0 is within the limit 3.0 .
    expected reward -0.9 . confidence likely .

    reference (fire A): actuator 16 ... +0.35 , actuator 30 ... +0.15 , ...
    reference (fire B): actuator 27 ... +0.95 , actuator 35 ... +0.55 , ...

The model learned the brief's format and a generic central plan. That still
scores 0.81 because the plans concentrate near the centre of the patch and the
budget line is always "3.0 within 3.0", so a generic brief is most of the way
right by construction.

**The permutation control confirms it** (job 1015981, 3 seeds). Scoring each
brief against a *different* fire's plan changes nothing:

| seed | F aligned | F shuffled |
|---|---|---|
| 0 | 0.7439 | 0.7436 |
| 1 | 0.7673 | 0.7674 |
| 2 | 0.8630 | 0.8639 |

Equal to three decimals. The briefs carry no fire-specific information: F is
measuring the shared structure of every plan (treatment near the centre, the
budget line, the confidence word), not whether this brief describes this
fire.

## Status of the claim

What holds:

- The certificate machinery works on real data. Coverage holds at 90% and 95%
  on 3 of 3 seeds with a non-vacuous floor, which is the property a
  deployment needs: whatever the generator's quality, the certificate reports
  it honestly rather than overstating it.
- Trained-in beats a genuine post-hoc control (t = +3.61), replicating the
  synthetic-testbed result on real plans.
- The faithfulness term adds nothing (t = -0.01). Third independent
  replication.

What does not:

- **Fire-specific faithfulness.** Shuffled equals aligned; the generator has
  learned the brief format and a generic plan. The trained-in advantage is
  format and calibration learning, and the certified floor of 0.71 is the
  floor of a near-constant brief.
- Any claim that these briefs explain *this* fire's plan to an operator.

What it would take to fix, in order of likelihood:

1. **Conditioning bandwidth.** The condition vector is the plan's 64
   intensities plus five scalars, projected to a prefix; a 64-patch plan may
   not survive that bottleneck. Widening the prefix, or conditioning on the
   fire state directly (the Perceive embedding), is the first thing to try.
2. **Training signal.** 200 iterations x batch 8 = 1,600 briefs over 1,024
   distinct fires: about 1.5 briefs per fire. The synthetic testbeds had one
   9-dimensional action family; this is 1,024 distinct 64-dimensional plans.
3. **Metric sensitivity.** Even per-dimension KL is dominated by the 57
   untreated patches both distributions agree on. A score restricted to the
   union of treated patches (aligned and parsed) would put the signal where
   the decision is. That changes what F means, so it belongs in the paper as
   a definition, not a patch.

Until at least (1) and (3) are done, report Experiment 5 as: the conformal
certificate transfers to real data and reports an honest floor; the generator
does not yet individuate briefs, shown by a permutation control.
