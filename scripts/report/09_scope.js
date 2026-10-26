module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("10 · What we do not claim"),

H2("10.1 The gap that will not close"),

QUOTE("**No observational dataset contains counterfactual interventions.** NDWS and FIRMS record what fires did; neither records what a fire *would have done* had a firebreak been cut where our planner asked."),

...CODE([
  "validated against observation:   O → û,   G(û) → u',   θ-correction",
  "model-based, not validated:      effect of an action a on u'",
]),

P("**Switching domain does not fix this.** Flood, heat, rainfall and air-quality records have the same structure: outcomes under whatever interventions actually occurred, never under ours. Closing it requires either a treatment programme with recorded locations and matched untreated controls, or a physics simulator standing in for reality — which substitutes one model for another."),

P("The flood work takes the second route **deliberately**, and the choice is defensible only because of three specific properties: the shallow-water equations are conservation laws rather than empirical spread rates, the discretisation is the operational standard in government flood mapping, and the reference model is independent of the surrogate being tested. It does not close the gap. It is the strongest available version of not closing it."),

H2("10.2 The scope of the twin claim"),

QUOTE("**The sync half is demonstrated on real data; the act half is not.** PSPE is a decision system whose perception, dynamics and correction are validated against real observations, and whose intervention effects rest on a stated action model. That is more than a forecaster and less than a validated controller, and the boundary should be drawn explicitly wherever the work is presented."),

...figure("docs/figures/paper/figA7_validation_chain.png",
  "The validation chain. Each link is marked by what it was validated against; the final link has no observational validation available in any domain.",
  { maxHeight: 340 }),

H2("10.3 What is genuinely built"),

table(
  ["Link in the chain", "Status"],
  [["Observation → state, under realistic occlusion, evaluated on the decision", "Built, including the negative result that imputation does not help on a sparse field"],
   ["State → forecast, at published-benchmark skill on real fire data", "Built"],
   ["Forecast → constrained plan, beating operational practice with the mechanism measured", "Built"],
   ["Plan → safety, with a conformal bound on violation", "Built; level attained on one family, not rejected on a second, trivial on a third"],
   ["Observation → model correction, over real multi-day sequences", "Built"],
   ["Plan → brief → certificate, with honest coverage", "Certificate built; the faithfulness claim withdrawn"],
   ["Action → outcome, validated", "**Not built, and not buildable from observational data**"]],
  [52, 48]),

H2("10.4 What would make it a deployable climate twin"),

table(
  ["Step", "Effort", "What it buys"],
  [["Ingest live FIRMS and gridded meteorology for an active incident", "Days", "The loop running forward in real time rather than on archives"],
   ["A second hazard with sequential observation (flood extent from Sentinel-1)", "2–3 weeks", "Shows the framework is not fire-specific, and tests the rare-event caveat on a **dense** field"],
   ["Treatment records with matched controls", "Months, institutional", "The only path to validating intervention effects"],
   ["Operator study on real briefs", "Weeks", "The Explain module's missing human evaluation"]],
  [40, 18, 42]),

H2("10.5 Weaknesses to state rather than hide"),

BULLET("**The guarantee is marginal, not per-step.** It bounds trajectory cost exceeding the limit, not cost at every timestep."),
BULLET("**The margin bounds rate, not severity**, and δ does not change that."),
BULLET("**Every baseline satisfies the constraint more often than we do.** They are largely cautious by being inert."),
BULLET("**ρ > 1 throughout the reachable regime** — the region where the default recipe is correct is not reachable on these testbeds."),
BULLET("**The intervention arm is unvalidated in reality**, and no retrospective dataset can fix that."),
BULLET("**The road layer understates disruption in two independent ways** (§5.5.1)."),
BULLET("**After a literature audit, four of twelve claimed contribution areas survive as contributions.** Silent violation under model-based constraints is named in the safe-RL literature; the joint-training negative restates objective mismatch; state sync against open loop is the textbook data-assimilation comparison. A reviewer who knows that literature reads the catalogue as careful replication rather than as a headline."),

H1("11 · Reproducing everything"),

H2("11.1 Commands"),

...CODE([
  "# Re-measure all 71 gym ladder rungs against the live environments.",
  "# Exits nonzero if any catalogued number has drifted.",
  "python3 scripts/score_gym.py --ladder --check --split eval",
  "",
  "# The closed-form validation of chapter 2.",
  "python3 eval/run_margin_coverage.py --rhos 0.1 0.25 0.5 0.8 1.0 1.25 2.0 4.0 10.0",
  "",
  "# The service and the portal.",
  "python3 -m uvicorn service.app:app --port 8000",
  "cd portal && npx next dev        # FLOOD_SERVICE=... if the port differs",
  "",
  "# This report.",
  "node scripts/make_full_report.js",
]),

H2("11.2 Where things live"),

table(
  ["Path", "What it holds"],
  [["`docs/THESIS_AND_RESULTS.md`", "**The master record.** Every result, dated, with the defect log and the standing rules. Where this report and that file disagree, that file is correct."],
   ["`pspe/envs/catalog.py`", "The single source of truth for all fourteen environments: budgets, ladders, plain-language fields, and the shared `harm_pct` scale"],
   ["`pspe/envs/splits.py`", "The train/eval partition: `SPLIT_SEED = 20261018`, 25% held out"],
   ["`pspe/simulate/flood.py`", "The local-inertial shallow-water solver; state is `(h, qx, qy)`"],
   ["`service/app.py`", "FastAPI service: the flood twin's solves, roads, buildings, plan kit"],
   ["`service/gym_routes.py`", "The gym's six routes. Named `gym_routes` so it cannot shadow Gymnasium"],
   ["`portal/src/components/GymExplain.tsx`", "The three explanation panels of §8.2"],
   ["`scripts/score_gym.py`", "The checker that keeps the catalogue honest"],
   ["`eval/run_margin_coverage.py`", "The controlled CMDP behind chapter 2's table"],
   ["`runs/margin_synthetic/results.json`", "The data for the 0.003 validation"]],
  [34, 66]),

H2("11.3 Index of the numbers in this report"),

table(
  ["Quantity", "Value", "Where"],
  [["Components passing their own test while the decision worsened", "6; in 3, improving the component hurt", "§1.1"],
   ["Violation: none / margin / probe / both", "14.1 / 14.1 / 10.9 / 5.0 %", "§4.2"],
   ["Closed form validated to", "0.003 max error, two decades in ρ", "§2.5"],
   ["Same curve on the real planner", "over-predicts 4.5×", "§2.6"],
   ["Real-sample efficiency", "4.6× less data", "§4.4"],
   ["Conformal refusal threshold", "δ=0.1 below 9 runs; δ=0.05 below 19", "§3.5.3"],
   ["FEMA agreement", "CSI 0.39 — 50% found, 37% of ours outside", "§5.4.1"],
   ["Surface vs bare-earth elevation", "28.84 vs 18.71 km², +54%", "§5.1"],
   ["Roads cut at the car threshold", "29.7 of 898 km, all road classes", "§5.5"],
   ["Dixie Drive", "31 of 78 gauged, 60 of 78 calibrated, worst 4.85 m", "§5.6"],
   ["City-survey vs DEM ground disagreement", "2.05 m RMSE — 12× the floor offset", "§5.6.4"],
   ["Gauge warning time", "2 largest floods: no record; one captured gave 20 min vs 120 needed", "§5.3.1"],
   ["Fire detection", "99 of 406 found, 94 false; base rate ~1%", "§6.3"],
   ["Crews, fair comparison", "−33.1% vs even cover", "§6.4.1"],
   ["Against the operational rule", "+12 points at 3 days, paired t = 11.15", "§6.4.2"],
   ["Planning horizon", "1 day +0.6 n.s.; 3 days +12.0; 5 days +14.3", "§6.4.3"],
   ["Environments / ladder rungs", "14 / 71, all measured on held-out splits", "§7.1"],
   ["Toy optimum and multiplier", "0.400 / 0.800", "§7.3.8"],
   ["Polder", "nothing −57.81, gate-only −28.44; timing-aware 24% but over budget 10/12", "§7.3.1"],
   ["Planner saturation", "pre-activation ≈ 1,436", "§7.3.2"],
   ["Crossing", "greedy in budget 4/12; budgeted 12/12", "§7.3.3"],
   ["Held-out split", "Defend +0.21→+0.59; Firebreak +0.79→+0.90; Buyout +7.57→+0.05", "§7.4"],
   ["Margin precondition on real terrain", "fits only below ~3.6% forecast spread", "§5.9.1"],
   ["Defects found and corrected", "45", "§9"]],
  [38, 44, 18]),

RULE(),

P("*End of report. The master record at `docs/THESIS_AND_RESULTS.md` is the authority for every number above and carries the full derivation, the job IDs, and the dated audit trail for each.*"),

];
};
