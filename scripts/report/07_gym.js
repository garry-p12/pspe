const fs = require("fs");
const CAT = JSON.parse(fs.readFileSync(
  "/private/tmp/claude-501/-Users-guruprasadparasnis-pspe/38a1c11f-e34e-4a54-8c7a-26eb95373499/scratchpad/catalog_full.json", "utf8"));
const by = (id) => CAT.find((c) => c.id === id);

module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;

/** Every rung of one environment, affordable first, as the portal shows it. */
function ladder(id) {
  const e = by(id);
  const rows = e.ladder.map((r) => ({
    name: r.plain, pct: r.pct, ok: r.ok,
    note: r.cost <= 0 ? "spends nothing"
      : r.ok ? (r.cost / e.cost_limit > 0.95 ? "uses all of it"
               : r.cost / e.cost_limit > 0.6 ? "uses most of it"
               : r.cost / e.cost_limit > 0.25 ? "uses some of it" : "uses very little")
      : r.feasible > 0 ? `over in ${r.of - r.feasible} of ${r.of}`
      : `${(r.cost / e.cost_limit).toFixed(1)}x the allowance`,
  }));
  const ok = rows.filter((r) => r.ok).sort((a, b) => a.pct - b.pct);
  const no = rows.filter((r) => !r.ok).sort((a, b) => a.pct - b.pct);
  const fmt = (r, within) => ({
    cells: [r.name, `${r.pct}%`, r.note, within ? "yes" : "NO"],
    opts: within ? undefined : { color: "9B2C1F" },
  });
  return table(["Policy", "Harm", "Spend", "Within allowance"],
    [...ok.map((r) => fmt(r, true)), ...no.map((r) => fmt(r, false))],
    [46, 12, 28, 14], { numericFrom: 1 });
}

return [

H1("7 · The simulation gym: fourteen environments"),

P("A twin tells you what will happen. It does not tell you what to do. The gym wraps both twins, plus three synthetic control testbeds and one analytic oracle, as constrained decision problems behind one Gymnasium API."),

...figure("docs/deck_bundle/images/slide-15-the-decision-gym.png",
  "The gym: each environment is a constrained decision with a reward, a separate cost, and a budget it may not exceed.",
  { maxHeight: 330 }),

H2("7.1 What makes it a benchmark rather than a demo"),

P("Three properties, and all three were added because their absence had already produced a wrong result somewhere in this project."),

NUM("**Every step returns a cost as well as a reward**, and the budget is the *episode sum* (the Safety-Gymnasium and OmniSafe convention). A return without its spend is not a result, and on three of the fourteen a table of returns alone ranks a random policy first."),
NUM("**Every environment carries a ladder** — every policy ever run on it, its return, its spend, and how many episodes broke the budget, including the ones that embarrass us. `scripts/score_gym.py --ladder --check` re-measures all 71 rungs against the live environments and exits nonzero if any has drifted."),
NUM("**Scenario pools are split train and eval under a fixed seed** (`SPLIT_SEED = 20261018`, 25% held out), and every figure quoted comes from the held-out half."),

H3("7.1.1 Two API decisions worth stating"),

P("`terminated` versus `truncated` is not cosmetic: a clock running out is truncation, and only the one-step bandit terminates. Four tests initially encoded a rule that one case legitimately breaks, and had to be relaxed rather than the environment changed."),

P("Reproducing a random policy needs **two** seeds — `reset(seed=)` and `action_space.seed()` drive different generators. Seeding only the first moved published figures by up to 8%, and two comments in the codebase asserted the opposite before a test caught it."),

H2("7.2 The fourteen, by when the decision is taken"),

P("The most useful way to hold the set in mind is not by hazard but by **when you decide relative to the hazard**, because that is what changes the structure of the problem."),

table(
  ["Family", "When you decide", "Environments"],
  [["Oracle", "Once; the answer is known on paper", "`CMDPToy-v0`"],
   ["Pure control", "During, on a synthetic fluid", "`PDEControl-dar/swe/rdf-v0`"],
   ["Operations", "Repeatedly, while the hazard runs", "`FloodOperate-v0`, `FloodOperateFine-v0`, `FloodEvacuate-v0`, `FloodCrossing-v0`"],
   ["Investment", "Once per block, before it starts", "`FloodDefend-v0`, `FloodBuyout-v0`, `FloodGreenInfra-v0`, `HeatRetrofit-v0`"],
   ["Fire", "During, on learned dynamics", "`Firebreak-v0`, `FireDispatch-v0`"]],
  [14, 30, 56]),

P("Dynamics are recorded three ways, not two, and the distinction is a claim about how much to trust a row: **physics** (a conservation law is solved), **learned** (a network predicts the next state and its error compounds), and **observed** (measured data replayed, so nothing is predicted at all)."),

WHY("Why three and not two", "`dynamics` was a two-valued field until `HeatRetrofit-v0` arrived, whose temperature field is measured AORC data. A two-way split rendered it on the portal as \"guessed by a model\" — the opposite of true for the most directly measured data in the gym. A category with two values will meet a third case, and the rendering will be wrong before the assertion is."),

H2("7.3 Reading a ladder"),

P("Every percentage in the gym is a **share of doing nothing**, so 100% means no better than not acting and a policy can legitimately score above 100 by being worse than inaction."),

WHY("A denominator that had to be fixed", "This used to be a share of the **worst** rung, on a docstring's claim that the worst rung is doing nothing on all five environments. With fourteen that is false: on `PDEControl-dar/swe/rdf-v0`, `CMDPToy-v0` and `FloodDefend-v0` the worst rung is a *random* policy, and on four of those five it also breaks the budget. So \"100%\" meant \"the worst anyone happened to try\" — a denominator that moves when somebody adds a row, and that answers no question a reader has. Both the service and the portal now call one shared function, `EnvSpec.harm_pct`."),

H3("7.3.1 FloodOperate-v0 — run the pumps"),

P(by("FloodOperate-v0").plain.decision + " Harm is " + by("FloodOperate-v0").plain.score + "; the allowance is " + by("FloodOperate-v0").plain.budget + "."),

ladder("FloodOperate-v0"),
P(""),

P("**The gate is free.** Opening it is worth 30.9 of return at zero cost, so roughly half the available performance is gravity. Any learned number here has to say how much of it was."),

P("The bar is a **constant**: hold the pumps at the one fraction that spends exactly the two-hour budget. Two parameters, obtained by dividing a budget by a horizon. Four published safe-RL algorithms and our own hybrid planner are all worse than it inside budget."),

QUOTE("The ladder answers *why* a constant is hard to beat, and the answer is not \"timing does not matter\". Pumping only when the river blocks the gate reaches **24%** against the constant's 35%. It is simply **over budget in 10 of 12 episodes**. The structure is there and exploiting it costs more fuel than you are allowed; among affordable policies, spreading evenly is what remains."),

WHY("What this is and is not evidence of", "Not that learning cannot do this. That this environment's headroom is small and arithmetic found most of it — and that a paper reporting a learned win here without the constant in the table would be reporting the budget, not the policy. The terrain is a bank of 64 generated polders, not Onion Creek; the solver is real, the landscape is invented."),

H3("7.3.2 The planner's failure, which is diagnostic"),

...figure("docs/deck_bundle/images/slide-21-why-our-planner-failed.png",
  "Our own planner on the polder. Left: how hard the policy pushes, with the guard off, rising three orders of magnitude past the line where it stops responding. Right: the safety penalty climbing linearly against an error the policy can no longer answer.",
  { maxHeight: 330 }),

P("The pre-activation reaches about **1,436**. Through a tanh that output is 1 to machine precision and the derivative `1 − tanh²` is numerically zero, so **the policy has become a constant function of state** — by arithmetic, not by reluctance. Meanwhile the dual is an integrator, accumulating a constraint error whose actuator is pinned."),

QUOTE("That combination has a name in control engineering — **integral windup against a saturated actuator** — and the standard remedy is to stop integrating while the actuator saturates. It is not a learning pathology and would not be fixed by more data or a better simulator."),

P("None of it is visible in a reward curve. It is visible only if the pre-activation and the multiplier are logged, which is an argument for instrumenting the environment rather than only the agent. The guard that prevents it existed, written months earlier, and was off by default."),

H3("7.3.3 FloodCrossing-v0 — when to close the road"),

P(by("FloodCrossing-v0").plain.decision),

ladder("FloodCrossing-v0"),
P(""),

P("Austin instruments 22 barricade arms at 11 low-water crossings; **five** fall inside the solved domain and those are the environment, split by site, 4 train and 1 held out. Depth is solved at the city's own published coordinates."),

QUOTE("Closing whenever the road is wet scores a **perfect zero** — nobody ever drives through water. It also stays inside the closure budget in **4 episodes of 12**, and closes the road 29 times against a flood lasting about 30. It has not solved the problem; it has closed the road."),

P("The bar is the same rule plus two words — *and stop*. Close while it is wet, and stop when the allowance runs out. The entire difference between the bar and the obvious rule is a clause about spending."),

WHY("A defect that shaped this environment", "Eight of the eleven crossings came back bone dry, because depth was read at the published coordinate. A crossing is the **dip in the road**, not the point the city geocodes. Fixed by taking the local minimum within 120 m, which is the difference between an environment that works and one that silently has no signal."),

...figure("docs/figures/paper/fig12_trigger_frontier.png",
  "Sweeping the evacuation trigger end to end. The rule collapses into three behaviours rather than tracing a smooth trade-off, so a single reported number for the heuristic would pick one of three arbitrarily.",
  { maxHeight: 340 }),

H3("7.3.4 FloodEvacuate-v0 — call the evacuation"),

P(by("FloodEvacuate-v0").plain.decision),

ladder("FloodEvacuate-v0"),
P(""),

P("**Nothing known is both affordable and better than doing nothing.** Every affordable policy sits at 100%; the one that gets people out scores 0% harm and costs 1.5× the allowance. Driving times come from the real Austin street map."),

P("This is the weakest of the fourteen and is reported as such. Six forecast members, three of them inert, and the trigger has no stable behaviour across partitions."),

WHY("A mean that described no episode", "The trigger's cost is bimodal — either 0.00 or 39–48 — so its mean of 28.22 against a limit of 32 describes **no episode that ever ran**. A docstring claimed 0.55 was the best feasible policy on the strength of that mean. A mean cost over episodes is not a feasibility claim; `feasible` counts episodes whose *summed* cost cleared the limit, per episode, never averaged across them."),

H3("7.3.5 FloodDefend-v0 — spend the defence budget"),

P(by("FloodDefend-v0").plain.decision),

ladder("FloodDefend-v0"),
P(""),

P("Real Onion Creek terrain, 35 windows. The budget is **derived from the ground, not invented**: US$202,149 of USACE earth fill at $15.2/m³, which buys a 1.6 m embankment along the whole bank if spread evenly."),

WHY("Why the budget had to be derived", "The first version used a round $3,000,000, which could not be spent — the maximum any plan could absorb was about $527k. A budget that cannot bind measures nothing, exactly as in §2.1. Deriving it from the geometry is what made the environment a decision problem."),

P("Note that `lowest_first` scores **103%** — building where the ground is lowest is *worse* than building nothing. That is a real finding and the old denominator hid it."),

P("Two further defects shaped this one. Column strips on a diagonal creek produced segment lengths of `[0,0,0,0,960,60,0,0]`; fixed by following each window's principal axis via SVD. And `torch.chunk` returns fewer than K pieces when the split is uneven, so `tensor_split` is used instead."),

H3("7.3.6 FloodBuyout-v0 — which houses to buy out"),

P(by("FloodBuyout-v0").plain.decision + " 43,366 real building footprints and FEMA floodway polygons."),

ladder("FloodBuyout-v0"),
P(""),

P("The first version of this environment had **zero FEMA floodway polygons in its box** — 0 of 433 — so every policy scored exactly 0.00. Moved to Onion Creek and re-fetched. This is the environment whose headline claim did not survive the held-out split (§7.4)."),

H3("7.3.7 FloodGreenInfra-v0 and HeatRetrofit-v0 — the two newest"),

P("`FloodGreenInfra-v0`: " + by("FloodGreenInfra-v0").plain.decision + " Real NLCD 2021 impervious cover (classes 21–24 mapped to 0.10/0.35/0.65/0.90 impervious fraction) and **seven real storms** pulled hourly at ~1 km from NOAA AORC."),

P("`HeatRetrofit-v0`: " + by("HeatRetrofit-v0").plain.decision + " This is the only environment whose dynamics are **observed** — six real Austin heat waves from AORC's measured 2 m temperature field. Harm is person-hours above 35 °C, counted where the people are."),

WHY("Why AORC and not ERA5, and why not the downscaling model", "ERA5 understated Halloween 2013's peak **five-fold** — 12.2 against 59.8 mm/h — so it cannot express a convective storm at all. The project's own temperature downscaling model was evaluated and not used, for three reasons: its 10 km target is one to two cells over the study box, a daily mean cannot express \"hours above 35 °C\", and AORC is that model's own training target. Its proper use here is as a **model-error source** inside the environment, which is future work."),

P("`HeatRetrofit-v0` carries an explicit honesty flag: the cooling each block receives is taken from published figures rather than simulated, so the **order** of the rows should be trusted more than the magnitudes."),

H3("7.3.8 The oracle and the control testbeds"),

P("`CMDPToy-v0` is one number, one step, and the constrained optimum is **0.400** with multiplier **0.800**, both from algebra. Its value is that failure is loud: an implementation with a sign error in the dual looks, on every other environment, like one that merely underperforms. It caught two real bugs — one converging to the *unconstrained* optimum because cost was averaged over the batch before the multiplier saw it, and one with the dual step on the wrong side of the update."),

P("The three `PDEControl` presets are nine actuators on a 32×32 periodic grid for eight steps. The best known policy on all three is proportional control, and **the gain was set by sweep, not taste**: on `dar` at gain 1.0 the same controller scores better and violates the budget in every episode, so shipping it unscaled would have left doing nothing as the best affordable policy."),

H2("7.4 The held-out split, and the two claims it killed"),

P("A quarter of each scenario pool is held out under a fixed seed and never touched while tuning. Then every headline comparison is re-run on scenarios the rules have not seen. Five were tested."),

table(
  ["Headline comparison", "Full pool", "Held out", "Survives"],
  [["FloodDefend: uniform − lowest-first", "+0.21, t = 2.00", "+0.59, t = 4.58", "yes"],
   ["Firebreak: greedy − uniform", "+0.79, t = 5.78", "+0.90, t = 4.73", "yes"],
   ["GreenInfra: retrofit − nothing", "+0.87, t = 5.78", "+0.55, t = 8.14", "yes"],
   [{ cells: ["FloodBuyout: even − lowest-first", "+7.57, t = 2.69", "+0.05, t = 0.68", "**NO**"], opts: { fill: "F7E9E7" } }][0],
   [{ cells: ["GreenInfra: even − uphill (siting)", "+0.20, t = 2.41", "+0.03, t = 0.64", "**NO**"], opts: { fill: "F7E9E7" } }][0]],
  [40, 21, 21, 18]),

QUOTE("Two of five failed, and **both were claims already written into our own catalogue as results** before the split existed."),

H3("7.4.1 Why the buyout effect vanished"),

P("The pool is heavy-tailed: a handful of windows carry nearly all of the paired difference, and the split happened to put them on the training side. The pool estimate was not bad arithmetic — it was **a claim about windows we had already looked at**, which is a different and weaker thing than a claim about windows we had not."),

H3("7.4.2 A design detail that had to be fixed for the split to be honest"),

P("Budgets were originally derived **after** the split, which gave train and eval different limits — so the two halves were not the same problem. They are now derived from the full pool before splitting, and there is a test that fails if they diverge."),

WHY("What the split is worth", "It is the only mechanism in this project that has cost us published results. A methodology that never costs its proposer anything is not evidence of discipline. Both deleted claims were things we believed."),

H2("7.5 What the gym does not yet have"),

BULLET("**Eight of the fourteen have had no learned policy run on them at all.** Only `FloodOperate`, `FloodEvacuate`, `CMDPToy` and the three `PDEControl` presets have."),
BULLET("**Two are nearly inert.** On `FireDispatch-v0` the entire spread from doing nothing to the best rule found is 0.72 — under 3% of the return. `Firebreak-v0` is similar. The ordering of the heuristics is real; the magnitudes are a rounding error. Either loosen the budget or do not train there."),
BULLET("**One has nothing affordable that beats doing nothing** (`FloodEvacuate-v0`)."),
BULLET("**Where to train, if asked:** `FloodDefend-v0` and `FloodCrossing-v0`. Real terrain, bars that got *stronger* when held out rather than evaporating, and genuine headroom above the bar."),

];
};
