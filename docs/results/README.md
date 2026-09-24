# Result write-ups

Analyses of experiment batches, copied here from `runs/` (which is gitignored)
so they travel with the repository. Each names the run directories its
numbers come from.

- `PHASE1_ANALYSIS.md`: joint vs disaggregated training, Eq. 8 mixing rule,
  Lipschitz assumption and Proposition 1, faithfulness weight sweep and
  conformal certificate on Qwen (Vista, 2026-09-16, fetched 2026-09-21).
- `PHASE3_ANALYSIS.md`: constraint fix and joint training on swe and rdf
  (2026-09-21). swe's policy never moves; rdf's dual failed as shipped.
- `RDF_PLANNER_FIX.md`: the three-step diagnosis and fix of the planner on
  rdf, the dar regression check, and the safe-RL baselines on rdf
  (Experiments 1 and 2, 2026-09-21).
- `NDWS_PLANNING.md`: constrained firebreak planning on observed wildfire
  data (Experiment 4, 2026-09-22); `ndws_planning_seed_*.md` are the per-seed
  tables.

`docs/technical_report.md` sections 7.9 to 7.11 summarise these;
`docs/slides_experiments.md` is the slide-by-slide version.
