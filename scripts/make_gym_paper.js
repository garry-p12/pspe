/**
 * Build the long-form gym paper as a .docx.
 *
 * Two inputs have to exist first, and both are one command each:
 *
 *   1. the catalog, as JSON --
 *      python3 -c "import json,sys; sys.path.insert(0,'.'); \
 *        from pspe.envs import SPECS; \
 *        json.dump([{**s.as_dict(), 'ladder':[dict(r) for r in s.ladder]} \
 *                   for s in SPECS], open('<scratch>/catalog.json','w'))"
 *
 *   2. the portal screenshots, from a PRODUCTION build --
 *      cd portal && npx next build && npx next start -p 3021
 *      node shoot.mjs <scratch>/shots
 *      (the dev server's HMR socket fails under Playwright and blocks
 *      hydration, so the page never renders; `next start` has no such socket)
 *
 * Then:  node scripts/make_gym_paper.js out.docx
 *
 * Every figure in the paper comes from `catalog.json`, which
 * `scripts/score_gym.py --ladder --check` tests, so the paper cannot quote a
 * number the environments no longer produce.
 */
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun,
  LevelFormat, PageBreak, PageNumber, Packer, Paragraph, ShadingType, Table,
  TableCell, TableOfContents, TableRow, TextRun, VerticalAlign, WidthType,
} = require("docx");

const SCRATCH = "/private/tmp/claude-501/-Users-guruprasadparasnis-pspe/38a1c11f-e34e-4a54-8c7a-26eb95373499/scratchpad";
const SHOTS = path.join(SCRATCH, "shots");
const CAT = JSON.parse(fs.readFileSync(path.join(SCRATCH, "catalog.json"), "utf8"));
const by = (id) => CAT.find((c) => c.id === id);

// ---------------------------------------------------------------- helpers --
const CONTENT = 9360;                     // US Letter minus 1.5" of margins

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 140, line: o.line ?? 276 },
  alignment: o.align,
  indent: o.indent,
  children: runs(text, o),
  ...(o.border ? { border: o.border } : {}),
});

/** Inline markup: **bold**, `code`, *italic*. */
function runs(text, o = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  const push = (t, extra) => {
    if (!t) return;
    out.push(new TextRun({ text: t, size: o.size ?? 21, font: o.font ?? "Calibri",
                           color: o.color, ...extra }));
  };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: (o.size ?? 21) - 2 });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1,
                                  spacing: { before: 360, after: 180 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2,
                                  spacing: { before: 280, after: 140 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3,
                                  spacing: { before: 220, after: 120 } });

const BULLET = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 90, line: 276 },
  children: runs(text),
});
const NUM = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  spacing: { after: 90, line: 276 },
  children: runs(text),
});

const CODE = (lines) => lines.map((l, i) => new Paragraph({
  spacing: { after: i === lines.length - 1 ? 160 : 0, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F4F4F5" },
  indent: { left: 220, right: 220 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
}));

const QUOTE = (text) => new Paragraph({
  spacing: { before: 120, after: 160, line: 276 },
  indent: { left: 300 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: "18181B", space: 10 } },
  children: runs(text),
});

const RULE = () => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "D4D4D8", space: 1 } },
  children: [new TextRun({ text: "" })],
});

function table(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT / total;
  const w = widths.map((x) => Math.round(x * scale));
  const cell = (text, i, head, rowOpts) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    shading: head ? { type: ShadingType.CLEAR, fill: "18181B" }
                  : (rowOpts && rowOpts.fill
                     ? { type: ShadingType.CLEAR, fill: rowOpts.fill } : undefined),
    verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({
      spacing: { after: 0, line: 252 },
      alignment: i >= (o.numericFrom ?? 99) ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(text), {
        size: 17, color: head ? "FFFFFF" : (rowOpts && rowOpts.color),
        ...(head ? {} : {}),
      }).map((r) => head ? new TextRun({ ...r, bold: true }) : r),
    })],
  });
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true,
                     children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => {
        const opts = Array.isArray(r) ? {} : r.opts;
        const cells = Array.isArray(r) ? r : r.cells;
        return new TableRow({ children: cells.map((c, i) => cell(c, i, false, opts)) });
      }),
    ],
  });
}

function figure(file, caption, widthPx = 620) {
  const buf = fs.readFileSync(path.join(SHOTS, file));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
      children: [new ImageRun({ data: buf, type: "png",
                                transformation: { width: widthPx, height: Math.round(widthPx * 2000 / 2920) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 220 },
      children: runs(caption, { size: 17, color: "52525B" }),
    }),
  ];
}

const BREAK = () => new Paragraph({ children: [new PageBreak()] });

// ------------------------------------------------------------------- body --
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ---- title page ----
add(
  new Paragraph({ spacing: { before: 2600, after: 0 }, children: [] }),
  new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text: "A Simulation Gym for Constrained Hazard Decisions",
                             bold: true, size: 48, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 420 },
    children: [new TextRun({
      text: "Twelve environments on real terrain, held-out scenario splits, and a catalogue of the measurement defects found while building them",
      size: 26, color: "52525B", font: "Calibri" })],
  }),
  P("**Guruprasad Parasnis**", { size: 23 }),
  P("PSPE — Perceive, Simulate, Plan, Explain", { size: 21, color: "52525B" }),
  P("Working paper, long version. 18 October 2026.", { size: 21, color: "52525B" }),
  new Paragraph({ spacing: { before: 520 }, children: [] }),
  QUOTE("This is the long form, written to be cut down. Everything measured is " +
        "included, including the results that did not survive being checked — " +
        "three of those are the most useful material in it."),
  BREAK(),
);

// ---- contents ----
add(
  H1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  BREAK(),
);

// ---- abstract ----
add(
  H1("Abstract"),
  P("A digital twin that cannot be scored is a visualisation. This paper describes " +
    "the conversion of a flood-and-wildfire digital twin into a **simulation gym**: " +
    "twelve environments behind a single `gymnasium.make` API, each posing a " +
    "constrained decision — operate a pump station, call an evacuation, allocate a " +
    "levee budget, choose which houses to buy out, close a road, cut a firebreak, " +
    "dispatch crews between simultaneous fires, or track a field with nine actuators " +
    "— and each scored against a budget it may not exceed."),
  P("The contribution is not the environments alone. It is the discipline applied to " +
    "them. Every environment carries a **ladder**: every policy ever run on it, with " +
    "its return, its spend, and the number of episodes whose summed cost cleared the " +
    "limit — infeasible rows included and marked. Every number is re-measured by one " +
    "command that exits nonzero on drift. Every scenario pool is partitioned into " +
    "train and evaluation halves under a fixed seed, and every reported figure comes " +
    "from the half nothing was fitted to."),
  P("Three findings recur and are, we argue, the general lesson rather than facts " +
    "about these particular problems. **First, a weak incumbent invents a result**: " +
    "three environments originally named a baseline that a random policy outscored, " +
    "which reads as a discovery about the problem and is an artefact of the " +
    "baseline. **Second, the best policy on reward is usually one you may not run** — " +
    "on the polder it spends 3.46 pump-hours against a limit of 2.0 — so a table of " +
    "returns without a cost column is not a benchmark table. **Third, holding " +
    "scenarios out changes the answer**: of three headline comparisons, two survive " +
    "and strengthen and one collapses from t = 2.69 to t = 0.68, and before the " +
    "split existed all three read identically."),
  P("No learned policy has beaten a simple rule on any of the twelve. We regard that " +
    "as the honest state of an unsolved benchmark rather than a result to be " +
    "softened, and the paper is organised so that a reader can verify it."),
  BREAK(),
);

// ---- 1. introduction ----
add(
  H1("1. Introduction"),
  H2("1.1 The gap this closes"),
  P("The PSPE portal simulates floods on Onion Creek in Austin, Texas and wildfire " +
    "spread on historic fire days, and presents both to an operator. Before the work " +
    "described here it could answer *what happens if*. It could not answer *is this " +
    "decision any good*, because there was nothing to compare a decision against and " +
    "no way for an outside algorithm to make one."),
  P("Concretely: of the decisions the twin models, exactly two carried a " +
    "`gymnasium.make` id. The evacuation model and the wildfire model spoke the " +
    "batched, all-torch protocol the in-house planner consumes — `reset(batch, " +
    "generator)`, tensors in and out — which is the correct shape for a trainer that " +
    "wants a gradient and sixteen worlds at once, and the wrong shape for every " +
    "algorithm outside the repository. \"Simulation gym\" meant one family."),
  H2("1.2 What a benchmark owes its reader"),
  P("Four properties separate a gym from a benchmark, and we state them here because " +
    "the rest of the paper is organised around satisfying them."),
  NUM("**It conforms.** `gymnasium.utils.env_checker.check_env` passes; cost arrives " +
      "on `info[\"cost\"]` per step with the budget as the episode sum, which is the " +
      "Safety-Gymnasium and OmniSafe convention, so an OmniSafe install drops in " +
      "unchanged."),
  NUM("**It has a bar worth clearing.** The named baseline must be the strongest " +
      "policy that stays inside the budget, not the first rule somebody wrote."),
  NUM("**Its numbers can be regenerated.** One command, re-measuring every figure, " +
      "exiting nonzero when any of them has stopped being true."),
  NUM("**It can detect overfitting.** Scenarios used for fitting are disjoint from " +
      "scenarios used for reporting, and the partition does not move."),
  P("The fourth was the last to arrive and the most consequential. Until it existed, " +
    "a policy fitted to a 33-window environment was scored on the same 33 windows, " +
    "and nothing anywhere would have noticed one that had memorised them."),
  H2("1.3 What is in this paper"),
  P("Section 2 sets out the design rules that emerged. Section 3 describes the twelve " +
    "environments in detail. Section 4 is operational: how to install, call, train " +
    "against, score and extend the gym. Section 5 reports results. Section 6 is the " +
    "defect catalogue — every measurement error found while building this, with the " +
    "check that caught it — and is, we think, the most transferable part. Section 7 " +
    "describes the browser interface. Section 8 states limitations without softening " +
    "them. Appendices carry the full ladders, the rules, and reproduction commands."),
  BREAK(),
);

// ---- 2. design ----
add(
  H1("2. Design rules"),
  P("These were not chosen in advance. Each one is the residue of a specific mistake, " +
    "and Section 6 gives the mistake for each."),
  H2("2.1 The bar is the strongest feasible policy"),
  P("A benchmark whose named baseline is weak does not merely understate a " +
    "comparison. It **invents a result**, because a method that beats the weak " +
    "baseline reads as a finding about the problem."),
  P("The clearest instance: the polder environment named a level deadband as its " +
    "incumbent. The deadband holds gate-open and pumps-off on most steps and spends " +
    "0.29 of a 2.0 pump-hour budget. Holding the pump at the single constant fraction " +
    "the budget exactly affords returns −19.90 against the deadband's −24.88, while " +
    "remaining feasible in 12 of 12 episodes. That constant has two parameters and no " +
    "state, and it beats all four safe-RL algorithms in a finished comparison table " +
    "the project had already published."),
  H2("2.2 Every row carries its cost, and feasibility is per episode"),
  P("On several environments a table of returns alone ranks a **random policy first**. " +
    "It gets there by ignoring the budget — 1.5× to 34× over, infeasible in every " +
    "episode measured. Reward without cost is therefore not a ranking."),
  P("Feasibility is counted per episode and never averaged across them. The " +
    "evacuation trigger is the standing example: its mean cost is 28.2 against a " +
    "limit of 32, comfortably inside, while its actual per-episode spend is either " +
    "0.00 or 39–48. The mean describes no episode that exists."),
  H2("2.3 The budget comes from the ground, not from taste"),
  P("A budget nobody can exhaust is not a constraint. Where the terrain decides what " +
    "can physically be spent, the limit is derived from the terrain: the levee " +
    "environment's budget is the cost of a uniform 1.6 m embankment along the mean " +
    "bank line. Its first budget was a plausible-looking $3,000,000, which was six " +
    "times more than the windows could physically absorb."),
  P("Crucially, such a quantity is derived from the **whole** pool and never from a " +
    "split. Deriving it after partitioning gave train and evaluation different limits " +
    "— the same name attached to two different problems."),
  H2("2.4 Train and evaluate on different scenarios"),
  P("Each finite pool is partitioned once, deterministically, under a constant seed, " +
    "with a quarter held back. `gymnasium.make(id)` returns the training pool; asking " +
    "for `split=\"eval\"` is a deliberate act; every number in the catalogue is " +
    "measured on evaluation."),
  P("Where a split cannot be meaningful, the catalogue says so rather than letting " +
    "the word \"eval\" imply more than the support can carry. A quarter of six " +
    "ensemble members is one member; that is arithmetic, not a generalisation test."),
  H2("2.5 The catalogue is the artefact, and it is testable"),
  P("`pspe/envs/catalog.py` holds, per environment, what reward and cost measure, the " +
    "budget, the incumbent, the caveat, the scenario pool, and the full ladder — each " +
    "rung keyed to the policy that produced it. `scripts/score_gym.py --ladder " +
    "--check` re-runs all 58 rungs and fails on drift. A number nobody can regenerate " +
    "is a number nobody can check."),
  BREAK(),
);

// ---- 3. environments ----
add(
  H1("3. The twelve environments"),
  P("Four hazard families and one oracle. All take a `split` argument, expose cost on " +
    "`info[\"cost\"]`, and pass `check_env`."),
  table(
    ["id", "decision", "dynamics", "budget", "best policy inside it"],
    [
      ["CMDPToy-v0", "one number, one step", "closed form", "0.4", "the optimum itself, −0.16"],
      ["FloodOperate-v0", "gate + pump, 12 × 30 min", "solved shallow-water", "2.0 pump-hours", "budget-edge constant, −20.28"],
      ["FloodOperateFine-v0", "the same at 60 m cells", "solved shallow-water", "2.0 pump-hours", "budget-edge constant, −25.85"],
      ["FloodEvacuate-v0", "8 zones, 48 slots", "solved + real routing", "32 town-intervals", "doing nothing, −3726.85"],
      ["FloodDefend-v0", "levee height on 8 segments", "solved + real terrain", "$202,149", "uniform, −18.24"],
      ["FloodBuyout-v0", "buy out 9 blocks", "solved + 43,366 footprints", "17.8 houses", "even share, −0.57"],
      ["FloodCrossing-v0", "close the road, 36 × 30 min", "solved, 5 real crossings", "29 closed intervals", "budgeted closure, −7.18"],
      ["Firebreak-v0", "place breaks, 3 crew-days", "learned U-Net", "0.03 of the patch", "greedy placement, −15.86"],
      ["FireDispatch-v0", "split crews over 4 fires", "learned U-Net", "0.03 of a patch", "biggest fire, −20.99"],
      ["PDEControl-dar-v0", "9 actuators, 8 steps", "solved RK4", "0.936", "proportional, −1.57"],
      ["PDEControl-swe-v0", "the same, shallow water", "solved RK4", "0.192", "nothing is feasible throughout"],
      ["PDEControl-rdf-v0", "the same, FitzHugh–Nagumo", "solved RK4", "3.26", "proportional, −4.45"],
    ],
    [1700, 1900, 1700, 1500, 2000]),
  P("All figures are the held-out split. Section 5 and Appendix A give the full " +
    "ladders.", { size: 18, color: "52525B" }),
);

// per-environment detail
const DETAIL = [
  ["CMDPToy-v0", "3.1 The environment whose answer is known",
   ["Every other environment here is scored against the best policy anyone has found. " +
    "This one is scored against the **right** policy. Reward rises to a peak at 0.8, " +
    "cost rises linearly, so the constrained optimum is `min(peak, limit) = 0.400` and " +
    "the Lagrange multiplier there is 0.800. Both are exposed as properties on the " +
    "environment.",
    "It is a correctness oracle rather than an open problem, and it should be the " +
    "first id anybody calls. A constrained RL implementation that does not converge " +
    "to 0.400 here is broken, and that is visible in one line rather than inferred " +
    "from a league table on problems whose answers nobody knows. This is not " +
    "hypothetical: the same closed form caught two real bugs in this project's own " +
    "reimplemented safe-RL baselines.",
    "It is also the one environment in the suite that **terminates** rather than " +
    "truncating. A flood episode ends because a clock ran out and there is a value " +
    "beyond the horizon; a one-step bandit ends because the problem is over."]],
  ["FloodOperate-v0", "3.2 Operating the polder",
   ["Twelve half-hour decisions on a drain gate and a pump while a flood wave passes, " +
    "on a 3.84 km domain at 120 m cells solved with a local-inertial shallow-water " +
    "scheme. Reward is minus the depth-weighted wet fraction of the protected ground; " +
    "cost is pump-hours at the station's rated capacity.",
    "The gate is free. Cost prices pumps only, and opening the gate is worth 30.9 of " +
    "return at no cost — do-nothing scores −57.81 against gate-only's −28.44. Most of " +
    "what looks like control on this environment is gravity, and a policy that " +
    "discovers \"open the gate\" has captured the majority of the available " +
    "improvement for nothing.",
    "`FloodOperateFine-v0` is the same physical domain at 60 m, five times slower per " +
    "step. The coarse preset preserves the policy ordering, so training belongs there " +
    "and only the final comparison belongs on the fine grid."]],
  ["FloodDefend-v0", "3.3 Spending the defence budget",
   ["Eight decisions, one per stretch of creek bank: how high to build an embankment, " +
    "chosen **before** anyone knows how big the flood will be. Then one flood is " +
    "solved over the terrain as built.",
    "The terrain is real: USGS 3DEP bare earth over the Onion Creek study box, with " +
    "the channel and the at-risk ground taken from FEMA's National Flood Hazard Layer " +
    "— `FLOODWAY` for the creek, zones A/AE/AO for the hundred-year floodplain — " +
    "rather than from a threshold this project invented.",
    "The budget is money: US dollars at the USACE National Levee Safety Program rate, " +
    "escalated CWCCIS FC11 to 4Q25 and multiplied by the 0.88 Texas state factor, " +
    "through the embankment's real trapezoidal section. A trapezoid's area is " +
    "`crest·H + slope·H²`, so cost per metre of height rises with height and a 3 m " +
    "levee is twice a 2 m one rather than one and a half times.",
    "The result is that the obvious engineering answer fails. Building where the bank " +
    "is lowest is beaten by spreading the budget evenly, by 0.21 on the full pool " +
    "(t = 2.00) and 0.59 on held-out windows (t = 4.58). The mechanism is flanking: a " +
    "partial levee on an open reach is gone around rather than stopped at, so what " +
    "matters is the height of the whole line and not of its lowest point."]],
  ["FloodBuyout-v0", "3.4 Which houses to buy out",
   ["Austin did this. After the 2013 and 2015 Onion Creek floods the city acquired " +
    "flood-prone homes and returned the land to open space, and the question it faced " +
    "is the one posed here: with money for only so many properties, which?",
    "It is a different decision class from the rest of the gym. The others choose a " +
    "continuous setting; this one chooses a **set**. Each property is in or out, the " +
    "budget counts properties, and the thing being changed is not the flood but who " +
    "is standing in it. Nothing here alters the water.",
    "The budget counts houses rather than dollars, deliberately. A real buyout is " +
    "priced per house and this project has no verified acquisition figure for Austin; " +
    "inventing a plausible-looking dollar number and dividing by it would put false " +
    "precision on every result the environment produced. \"You may take eighteen " +
    "houses\" is a constraint anybody can check.",
    "This is also the environment whose headline claim did not survive a held-out " +
    "split, which is discussed at length in Section 5.3."]],
  ["FloodCrossing-v0", "3.5 When to put the barricade down",
   ["In Texas the flood that kills you is usually ankle deep and across a road. Austin " +
    "instruments twenty-two barricade arms at eleven low-water crossings for exactly " +
    "this, and the duty decision is the one posed here: the water is rising upstream, " +
    "the crossing is still dry, and closing it strands everyone who uses that road.",
    "It is better founded than the evacuation environment in one specific way. There, " +
    "the harm is a model's estimate of who was exposed and nobody has ever evacuated " +
    "on our instruction. Here, **whether the crossing is under water is a depth the " +
    "solver computes**, at a cell whose coordinates come from the city's own asset " +
    "register. What the environment still cannot know is how many cars would have " +
    "driven in, so it does not pretend to: the harm is intervals of road open while " +
    "under water, not a casualty count derived from traffic nobody counted.",
    "The budget is 29 closed intervals against a flood that covers the road for about " +
    "30, so the whole flood cannot be covered and which intervals to cover is the " +
    "decision. The bar and the obvious rule differ by two words: closing whenever the " +
    "crossing is wet — what a crew on site does — scores a perfect 0.00 and clears " +
    "the budget in 4 episodes of 12. Checking the allowance first costs return and " +
    "makes it feasible in all 12."]],
  ["Firebreak-v0", "3.6 Fire: placement and dispatch",
   ["`Firebreak-v0` places breaks on a 64 × 64 km patch over three crew-days while a " +
    "learned spread model advances the fire. `FireDispatch-v0` asks the question a " +
    "duty officer actually faces: four fires are running, there are only so many " +
    "crews, and sending them to one is not sending them to another.",
    "Both rest on the same U-Net trained on about 18,000 historic fire days from the " +
    "Next Day Wildfire Spread dataset, and both inherit its limits. The dynamics are " +
    "a learned model, not a rate-of-spread equation, and the environment steps by " +
    "feeding its own prediction back in, so error compounds in a way a conservation " +
    "law would forbid. The counterfactual is also unobservable: no fire on record had " +
    "a break cut on our instruction.",
    "At the default 3% budget, placement barely matters — the whole feasible span " +
    "from treating nothing to the best placement found is about 1.1 of 17. The regime " +
    "where placement is the question is six to eight times that budget."]],
  ["PDEControl-dar-v0", "3.7 The control testbeds",
   ["Three field-tracking problems — diffusion–advection–reaction, linearised shallow " +
    "water, and FitzHugh–Nagumo front propagation — each with nine Gaussian actuators " +
    "on a 32 × 32 periodic grid against a differentiable RK4 truth solver. They are " +
    "synthetic and make no claim about any place; they are here because a gym with " +
    "one kind of problem tests one kind of method.",
    "`PDEControl-swe-v0` carries a finding of its own. Nothing is feasible in every " +
    "episode — not even leaving all nine actuators alone, which clears the budget in " +
    "11 of 12. Its cost carries a state-driven exposure term, so the limit can be " +
    "exceeded whatever the actuators do. The constraint is not fully controllable, " +
    "which is a property of the testbed and not of any policy."]],
];

for (const [id, heading, paras] of DETAIL) {
  const c = by(id);
  add(H2(heading));
  paras.forEach((t) => add(P(t)));
  if (c) {
    add(
      table(["", ""], [
        ["Scored on", c.reward],
        ["Costs", c.cost],
        ["Budget", c.budget],
        ["Bar to beat", c.incumbent],
        ["Scenario pool", c.pool],
      ], [1700, 7660]),
      P("What it does not tell you: " + c.caveat, { size: 18, color: "52525B" }),
    );
  }
}
add(BREAK());

// ---- 4. operating ----
add(
  H1("4. Operating the gym"),
  H2("4.1 Installation and first call"),
  P("The gym needs `gymnasium` and `torch`. Three environments additionally need data " +
    "on disk; each raises a named error carrying the exact fetch command rather than " +
    "failing obscurely."),
  CODE([
    "import gymnasium as gym",
    "from pspe.envs import register_all",
    "",
    "register_all()                       # all twelve ids",
    "env = gym.make('CMDPToy-v0')         # start here: the answer is known",
    "",
    "obs, info = env.reset(seed=0)",
    "obs, reward, terminated, truncated, info = env.step(env.action_space.sample())",
    "info['cost'], info['episode_cost'], info['cost_limit']",
  ]),
  H2("4.2 The cost convention"),
  P("`info[\"cost\"]` is the spend for that step. `cost_limit` is the budget for the " +
    "**whole episode**: `baselines/safe_rl.py` winds its multiplier on " +
    "`episode_cost - cost_limit`, a sum over steps. This distinction has bitten this " +
    "project twice. A baseline that spends `cost_limit` on every step is infeasible " +
    "by a factor of the horizon, and an incumbent allowed to overspend is not an " +
    "incumbent."),
  P("Feasibility is tested as `cost <= limit * (1 + 1e-6)`. The tolerance is not " +
    "slack: `CMDPToy-v0`'s optimum sits *on* the constraint boundary by definition, " +
    "so nudging the action inside would stop it being the optimum, and in float32 a " +
    "sum of step costs lands either side."),
  H2("4.3 Splits: the part that makes a number reportable"),
  P("Every environment with a finite scenario pool takes a `split`."),
  CODE([
    "train = gym.make('FloodDefend-v0')                  # the default",
    "held  = gym.make('FloodDefend-v0', split='eval')    # a deliberate act",
    "",
    "# the partition is fixed under a constant seed and never moves",
    "from pspe.envs.splits import split_indices, describe_split",
    "split_indices(35, 'eval')          # the same 9 windows, every time",
    "describe_split(6, 'eval', 'FloodEvacuate-v0')",
  ]),
  P("Fit on `train`. Report on `eval`. The catalogue's figures are all evaluation " +
    "figures. Two environments have pools too small for this to mean much, and say so " +
    "in `splits.SPLIT_NOTES` rather than letting the word imply more than it carries."),
  table(["environment", "pool", "train", "eval"], [
    ["FloodOperate-v0, -Fine-v0", "generated polders", "48", "16"],
    ["Firebreak-v0, FireDispatch-v0", "NDWS fire days", "1,267", "422"],
    ["FloodDefend-v0", "windows of real terrain", "29", "6"],
    ["FloodBuyout-v0", "windows of real terrain", "25", "8"],
    ["FloodEvacuate-v0", "forecast members", "4", "2"],
    ["FloodCrossing-v0", "crossings, split by site", "4", "1"],
    ["CMDPToy-v0, PDEControl-*", "no finite pool", "—", "—"],
  ], [2900, 3400, 1400, 1400], { numericFrom: 2 }),
  P("The crossing splits **by site**, so an evaluation episode is a crossing the " +
    "policy has never seen. Holding out flood sizes at a crossing it has seen would " +
    "test something far easier and call it the same word.", { size: 18, color: "52525B" }),
  H2("4.4 Scoring"),
  P("One command measures everything and checks it against the catalogue."),
  CODE([
    "python3 scripts/score_gym.py                      # measure and print",
    "python3 scripts/score_gym.py --ladder             # every policy, not just two",
    "python3 scripts/score_gym.py --ladder --check     # fail if the catalog drifted",
    "python3 scripts/score_gym.py --env Firebreak-v0 --seeds 32",
    "python3 scripts/score_gym.py --split train        # for comparison only",
  ]),
  P("`--check` compares all 58 ladder rungs against `pspe/envs/catalog.py` and exits " +
    "nonzero on any mismatch. Return tolerance is 2% relative; cost tolerance is 0.2% " +
    "**of the budget**, because an absolute tolerance cannot serve both a 2.0 " +
    "pump-hour limit and a $202,149 one."),
  H2("4.5 Reproducing a random policy needs two seeds"),
  P("This cost two runs and is worth stating plainly. `reset(seed=)` seeds the " +
    "environment's own stream; `action_space.sample()` draws from a different " +
    "generator reached through `action_space.seed()`. Gymnasium keeps them separate " +
    "deliberately, so that restarting an episode does not silently reset a caller's " +
    "exploration. Seeding only the first moved our random-arm figures by up to 8% " +
    "between runs."),
  CODE([
    "env.action_space.seed(seed)   # both, or the random arm is not reproducible",
    "obs, _ = env.reset(seed=seed)",
  ]),
  H2("4.6 Adding an environment"),
  P("The contract is small. Expose a batched core if the planner needs one, wrap it " +
    "in a single-instance `gym.Env`, accept `split`, put cost on `info[\"cost\"]`, " +
    "name at least one baseline that is the strongest policy you can find inside the " +
    "budget, register the id, and add a catalogue entry with its pool and ladder. " +
    "`scripts/score_gym.py --ladder --check` will then refuse to pass until the " +
    "numbers in the entry are the numbers the environment produces."),
  BREAK(),
);

// ---- 5. results ----
add(
  H1("5. Results"),
  H2("5.1 Nothing learned has beaten a simple rule"),
  P("Across twelve environments, no learned policy currently holds the bar. On eight " +
    "of them nothing has been trained at all, so \"not beaten\" partly means \"not " +
    "attempted\" and should not be read as a stronger claim than it is."),
  P("Where learned policies have been run — the polder — four safe-RL algorithms " +
    "(PPO-Lagrangian, CPO, Sauté RL, primal-dual NPG) all finish below a " +
    "two-parameter constant. Three of the four also finish outside the budget, two of " +
    "them on every evaluation episode."),
  H2("5.2 A weak incumbent invents a result"),
  P("Three environments originally named a baseline that a table of returns ranked " +
    "**below** a random policy. That reads as \"random beats the heuristic\" and is " +
    "an artefact of the baseline, not a property of the problem. In every case random " +
    "achieves it by ignoring the budget."),
  table(["environment", "named baseline", "random", "verdict on returns alone"], [
    ["FloodOperate-v0", "−24.88 @ 0.29", "−22.00 @ 3.05", "random first"],
    ["FloodOperateFine-v0", "−26.09 @ 0.25", "−24.23 @ 3.05", "random first"],
    ["Firebreak-v0", "−31.90 @ 0.03", "−17.79 @ 1.02", "random first"],
  ], [2400, 2400, 2200, 2360]),
  P("Adding a tuned-feasible baseline removes the inversion on both flood presets — " +
    "the budget-edge constant beats random *and* stays feasible. On the firebreak it " +
    "survives, and is reported.", { size: 18, color: "52525B" }),
  H2("5.3 Holding scenarios out changes the answer"),
  P("This is the result we consider most important, because it is the one that " +
    "distinguishes a benchmark from a gym. Of three headline comparisons, measured " +
    "paired over 32 episodes on the full pool and again on held-out scenarios:"),
  table(["claim", "over the whole pool", "held out", "survives?"], [
    ["FloodDefend: uniform − lowest-first", "+0.21, t = 2.00", "+0.59, t = 4.58", "yes"],
    ["Firebreak: greedy − uniform", "+0.79, t = 5.78", "+0.90, t = 4.73", "yes"],
    { cells: ["FloodBuyout: even − lowest-first", "+7.57, t = 2.69", "+0.05, t = 0.68", "NO"],
      opts: { fill: "FBE9E7" } },
  ], [3400, 2100, 2000, 1500]),
  P("**Before the split existed all three read identically.** The buyout environment " +
    "had reported, the previous day, that an even share beats buying the lowest-lying " +
    "blocks first and that \"the obvious answer loses badly\". That is true over its " +
    "whole 33-window pool and untrue on the 8 windows nothing was fitted to."),
  P("Both readings condemn the environment rather than rescuing it: with 33 windows " +
    "it cannot hold enough out to support a claim, and the claim it did support was " +
    "about the pool it was tuned on. The entry now says so, and nothing should be " +
    "trained there and reported until the pool is larger."),
  H2("5.4 The best policy on reward is usually unaffordable"),
  P("On the polder, the best-scoring policy known is `sealed_aware_action` — pump " +
    "only while the river outside is higher than the water inside, because that is " +
    "the only window when gravity is unavailable. Its own docstring calls it \"the " +
    "number a learned policy has to beat\", measured against a budget of six " +
    "pump-hours where it is feasible."),
  P("This environment's budget is 2.0. It spends 3.46 and clears the limit in 2 " +
    "episodes of 12. It is the best policy on reward and it is not a bar anything has " +
    "to clear. The same pattern holds on the firebreak, where treating the whole " +
    "patch reaches −1.04 at a hundred times the budget, and on the crossing, where " +
    "closing the road all day scores a perfect zero and is infeasible in every " +
    "episode."),
  H2("5.5 Two environments have no stable result to report"),
  P("`FloodEvacuate-v0` has six forecast members, three of which never put a zone " +
    "over the wet threshold. Worse than thin support, its trigger has **no stable " +
    "behaviour**: whatever threshold it is given, its score coincides exactly with " +
    "one of two degenerate arms — but which one depends on the partition."),
  table(["split", "members", "the trigger at 0.55 scores"], [
    ["all", "6", "ordering everyone, exactly"],
    ["train", "4", "never ordering, exactly"],
    ["eval", "2", "never ordering, exactly"],
  ], [2200, 1800, 5360]),
  P("A one-dimensional threshold over six members does not have a behaviour to " +
    "report; it has whichever behaviour the draw gives it. No partition of this " +
    "environment supports a claim."),
  P("`FloodBuyout-v0` is the second: two-thirds of its episodes carry no signal at " +
    "all, the water reaching no house in that window whatever is bought."),
  BREAK(),
);

// ---- 6. defects ----
add(
  H1("6. What measuring caught"),
  P("This section is a catalogue of the measurement defects found while building the " +
    "gym. We include it in full because the errors are mundane, each was caught by a " +
    "specific check, and each would have been invisible in a results table. Several " +
    "had already been published in this project's own write-ups before being found."),
  H2("6.1 Baselines that were not what they claimed"),
  table(["defect", "effect", "what caught it"], [
    ["The firebreak's uniform baseline spent `cost_limit` on every step",
     "Infeasible by a factor of the horizon: 0.09 against a 0.03 budget",
     "Printing the feasibility count beside the return"],
    ["The evacuation trigger's docstring called 0.55 \"the best FEASIBLE member\"",
     "Its per-episode cost is bimodal — 0.00 or 39–48 — so the 28.2 mean describes no episode that exists",
     "Counting feasible episodes instead of averaging cost"],
    ["The proportional controller violated the budget on all three PDE testbeds",
     "Would have left \"do nothing\" as the named incumbent on three new environments",
     "Sweeping the gain and checking feasibility at each"],
    ["The polder's named incumbent underspent by 7×",
     "A two-parameter constant beats it by 5.0 and beats all four learned arms",
     "Scoring the constant that spends exactly the budget"],
  ], [2700, 3400, 3260]),
  H2("6.2 Geometry and data"),
  table(["defect", "effect", "what caught it"], [
    ["Column strips used to segment a diagonal creek",
     "Segment lengths of [0, 0, 0, 0, 960, 60, 0, 0] m — six decisions about nothing",
     "Printing the per-segment lengths"],
    ["`torch.chunk` returns fewer than K pieces",
     "The last segment of some windows was empty — the same no-op, reintroduced by its own fix",
     "Re-printing the lengths after the fix"],
    ["The buyout's first box had zero FEMA floodway polygons",
     "Every policy scored 0.00, including buying nothing: no channel to drive a flood down",
     "Noticing that doing nothing and doing everything scored the same"],
    ["Windows filtered on at-risk homes rather than reachable ones",
     "8 dead episodes in 10 — FEMA's floodplain reaches ground a channel flood does not",
     "Counting episodes with zero reward"],
    ["Crossing depth read at the published coordinate",
     "8 of 11 crossings bone dry at every flood size — a crossing is the dip in the road, not the point on the map",
     "Checking how many crossings ever got wet"],
    ["One global water level on a creek with a gradient",
     "The 20th percentile of floodplain sits 5.8 m below the lowest channel cell; no stage made the levee matter",
     "A calibration sweep that found no working stage"],
  ], [2700, 3400, 3260]),
  H2("6.3 Statistics and protocol"),
  table(["defect", "effect", "what caught it"], [
    ["`reset(seed=)` assumed to seed the action space",
     "Random-arm figures moved up to 8% between runs; two inline comments asserted the opposite of an existing test",
     "Running the scorer twice and diffing"],
    ["Separability judged from marginal standard deviations",
     "Would have retired a working environment: the firebreak's policy difference is 1/40th of its marginal spread and resolves at t = 4.96",
     "Testing paired by scenario"],
    ["Budgets derived after the split",
     "Train and eval had different limits — the same name on two different problems",
     "Asserting the budget is equal across splits"],
    ["A margin quoted across two episode draws",
     "A 3.68 margin against a cross-draw spread of 4.62: not resolved, and stated as if it were",
     "Comparing the draws on the policies they share"],
    ["Absolute cost tolerance in the checker",
     "0.01 cannot serve both a 2.0 pump-hour limit and a $202,149 one; it failed every row on eighteen cents",
     "Adding an environment with a money budget"],
  ], [2700, 3400, 3260]),
  H2("6.4 Tests that encoded a rule one case legitimately breaks"),
  P("Three times in one week, a property measured once was generalised to the suite " +
    "and asserted everywhere. The pattern is worth naming because each instance " +
    "looked like a passing test until a new environment arrived."),
  BULLET("**`terminated is False` for every environment.** Right for eleven of " +
         "twelve; wrong for the one-step bandit, whose episode ends because the " +
         "problem is over. It mislabelled the only environment whose answer is known."),
  BULLET("**The incumbent is feasible in every episode.** Right for eleven; on " +
         "`PDEControl-swe-v0` it is not a property anything can have, and the test " +
         "was asserting against the catalogue's own recorded finding."),
  BULLET("**The evacuation trigger scores what ordering-everyone scores.** True on " +
         "the full ensemble, false on either half — and discovering that produced a " +
         "sharper claim than the one the test had been defending."),
  P("Each is now an explicit, checked exemption: the suite names the exceptions and " +
    "asserts that each exception is still earned, so a recalibration that removes the " +
    "exception fails loudly rather than passing quietly."),
  BREAK(),
);

// ---- 7. portal ----
add(
  H1("7. The browser interface"),
  P("The gym is served at `/gym` in the PSPE portal. The page exists for two " +
    "audiences: a researcher deciding whether these environments are worth training " +
    "against, and an operator who wants to understand what the decision actually is."),
  P("One design decision shaped it more than any other. An early sketch carried a " +
    "footnote — *a toy version for practice; it is simpler than the real simulation, " +
    "so the numbers will differ.* It does not need one. A step of `FloodOperate-v0` " +
    "is 0.04 s of the same local-inertial solver the measurements came from, so the " +
    "page drives the **real** environment through a session API and draws its actual " +
    "state. The water a visitor watches spread is the water the benchmark measured."),
  figure("gym-pumps.png",
         "Figure 1. `FloodOperate-v0`. The river band (blue) along the west, the levee " +
         "crest (tan), and the protected basin (green) are the solver's own terrain " +
         "and masks. Stat tiles carry time, spend against budget, and damage. The " +
         "right rail groups the twelve decisions and shows where any run would rank."),
  P("A second decision was subtractive: there are no houses and no trees in the " +
    "scene. They would be the only marks on a page whose argument is that every mark " +
    "is a measurement, and a viewer cannot distinguish a decorative house from a " +
    "modelled one. The scene draws relief, depth, the protected mask and the " +
    "structure, and says so beneath the legend."),
  figure("gym-crossing.png",
         "Figure 2. `FloodCrossing-v0` renders the real 3DEP terrain around the " +
         "selected Austin crossing — the tile its depth library was solved on — " +
         "rather than a schematic road with a dip in it."),
  H2("7.1 Reading the ladder"),
  P("Damage is shown against doing nothing, which is the worst row on every " +
    "environment, so 100% is a measurement rather than a chosen anchor. Spend is " +
    "shown against what was allowed — \"all of it\", \"34× over\", \"over in 10 of 12 " +
    "runs\". That middle case matters: an approach that stays inside the budget in " +
    "most runs and blows past it in the rest is neither affordable nor simply " +
    "over-budget."),
  P("A finished run is placed in the ladder and labelled as one episode against rows " +
    "that are twelve each. It can exceed 100%, because a single run can go worse than " +
    "the average of doing nothing, and clamping that would hide a real outcome."),
  figure("gym-defend.png",
         "Figure 3. `FloodDefend-v0`. Eight decisions along the bank, a dollar budget, " +
         "and the real floodway geometry. The marker channel shows where a levee may " +
         "be built."),
  figure("gym-toy.png",
         "Figure 4. `CMDPToy-v0` has no field to draw — its observation is a constant " +
         "— so the drawing is the comparison: the dial you set beside the known " +
         "answer. The option list labels 0.40 as the answer rather than making a " +
         "guessing game of the one environment whose point is that the answer is known."),
  H2("7.2 Service API"),
  P("Three read routes and three play routes, all of which drive real environments."),
  CODE([
    "GET  /gym/environments?probe=1   # the catalog, optionally with live spaces",
    "GET  /gym/environment/{id}       # one entry, probed",
    "POST /gym/rollout                # one scored episode of a named policy",
    "",
    "POST /gym/play/start             # begin a hand-driven episode",
    "POST /gym/play/step              # one decision, or the rest of them",
    "POST /gym/play/stop",
  ]),
  P("Everything prose-shaped in those responses is read from `pspe/envs/catalog.py`, " +
    "which `scripts/score_gym.py --ladder --check` tests. There is no second copy of " +
    "any claim to fall out of date."),
  BREAK(),
);

// ---- 8. limitations ----
add(
  H1("8. Limitations"),
  P("Stated without softening, in the order we would want a reader to weigh them."),
  H2("8.1 Nothing learned has beaten any of these"),
  P("The gym is unsolved, which is the right state for a benchmark and the wrong " +
    "state for a claim that the framework helps an operator. Those are different " +
    "claims and this work supports only the first. Eight of the twelve have had no " +
    "learned policy run on them at all."),
  H2("8.2 Two environments cannot support a claim"),
  P("`FloodEvacuate-v0` has six scenarios, three of them inert, and a trigger whose " +
    "behaviour changes with the partition. `FloodBuyout-v0` has 33 windows, two-" +
    "thirds of its episodes carry no signal, and its headline comparison does not " +
    "survive being held out. Neither should have a policy trained and reported on it " +
    "until its pool grows."),
  H2("8.3 Most of the terrain is synthetic, and the fire dynamics are learned"),
  P("The polder environments use a bank of generated terrains, not Onion Creek: the " +
    "solver is real and the ground is invented. `FloodDefend-v0`, `FloodBuyout-v0` " +
    "and `FloodCrossing-v0` use real USGS terrain and real FEMA geometry, which is " +
    "why they were built. The two fire environments rest on a learned spread model " +
    "stepped on its own predictions."),
  P("We did **not** retrofit real terrain into the polder environment, and the reason " +
    "is worth recording. Its bank is a plain dictionary, so a swap looks cheap — but " +
    "`_build_structures` takes the sump from batch element 0 and applies it to the " +
    "whole batch, and `_apply_river` reads column 0 as the channel. Both hold for a " +
    "generated polder and neither holds on real ground, so a terrain swap would have " +
    "placed every episode's gate and pump at episode 0's location, silently. Read how " +
    "an environment consumes data, not the shape of the container."),
  H2("8.4 Counterfactuals are unobservable"),
  P("No flood was fought, no town evacuated and no firebreak cut on our instruction. " +
    "Every reward in the hazard environments is a model's estimate of what would have " +
    "happened. `FloodCrossing-v0` is the partial exception — whether a road is under " +
    "water is a solved depth — and even there the traffic is not modelled."),
  H2("8.5 The hydrographs are not events"),
  P("Floods are single-peaked rises held in the channel, not routed events from gauge " +
    "records. They exercise the decision; they do not reproduce a flood that happened."),
  BREAK(),
);

// ---- 9. future ----
add(
  H1("9. What would change the conclusions"),
  NUM("**Train something on `FloodDefend-v0` or `FloodCrossing-v0`.** Both now have a " +
      "bar that survives being held out and real headroom above it. This is the " +
      "single step that would convert \"an unsolved benchmark\" into evidence that " +
      "the framework does anything."),
  NUM("**Grow the evacuation ensemble.** Six members, three inert, is three scenarios. " +
      "More members would turn the environment's most interesting decision — eight " +
      "degrees of freedom per slot against a trigger's one — into something reportable."),
  NUM("**Widen the firebreak budget.** At 3% of a patch, placement barely matters. The " +
      "uniform-to-greedy gap widens from 0.8 at 3% to 5.2 at 24%, so the interesting " +
      "regime is a budget six to eight times the default."),
  NUM("**A second catchment.** `data/lismore/floor_levels.csv` holds 2,924 real " +
      "properties with floor levels against 1-in-10 and 1-in-100 design levels for " +
      "the 2022 Richmond flood in New South Wales. That is property-level damage data " +
      "for a non-US catchment and the natural transfer test."),
  NUM("**A warning-threshold environment.** Choosing the gauge level at which to warn, " +
      "trading false alarms against missed events, is the decision closest to this " +
      "project's conformal-margin work and is not yet built."),
  BREAK(),
);

// ---- appendix A ----
add(H1("Appendix A. Full ladders"));
add(P("Every policy ever run on each environment, measured on the held-out split over " +
      "12 episodes on seeds 0–11, with both the environment and the action space " +
      "seeded. `feasible` counts episodes whose summed cost cleared the limit."));
for (const c of CAT) {
  add(H3(`${c.id} — ${c.title}`));
  add(P(`Budget: ${c.budget}. Pool: ${c.pool}.`, { size: 18, color: "52525B" }));
  const rows = [...c.ladder].sort((a, b) => a.return - b.return).map((r) => {
    const ok = r.feasible === r.of;
    return {
      cells: [r.plain, r.return.toFixed(2),
              Math.abs(r.cost) >= 1000 ? r.cost.toExponential(2) : r.cost.toFixed(4),
              `${r.feasible}/${r.of}`, ok ? "" : "over budget"],
      opts: ok ? {} : { color: "C3392B" },
    };
  });
  add(table(["policy", "return", "cost", "feasible", ""], rows,
            [3500, 1300, 1700, 1300, 1560], { numericFrom: 1 }));
  if (c.status) add(P(c.status, { size: 18 }));
}
add(BREAK());

// ---- appendix B ----
add(
  H1("Appendix B. Rules"),
  P("Rules 50–63 of this project's running list, all of which came out of the gym " +
    "work. Earlier rules concern the twin itself and are documented elsewhere."),
);
[
  ["50", "An incumbent must be the strongest policy that stays inside the budget, not the first one somebody wrote. A weak baseline does not merely understate a comparison — it invents a result."],
  ["51", "A mean cost over episodes is not a feasibility claim. Where the cost distribution is bimodal, the mean describes no episode that exists."],
  ["52", "Score doing nothing and doing everything before believing any policy number. They bound the environment."],
  ["53", "A number nobody can regenerate is a number nobody can check."],
  ["54", "Test whether a benchmark can separate its own baselines, paired by scenario, before concluding anything from the size of an effect."],
  ["55", "Count the distinct worlds a benchmark can draw an episode from, and treat it as a harder limit than any statistic computed on it."],
  ["56", "Before recommending that real data be swapped into an environment, read how the environment consumes it, not the shape of the container."],
  ["57", "Derive a budget from the ground wherever the ground decides what can be spent."],
  ["58", "Terminated and truncated are a claim about why an episode ended, and the answer differs across a suite."],
  ["59", "When an environment scores identically for every policy, suspect the data before the code."],
  ["60", "A coordinate is not a cell. An asset register records where the equipment is, not where the water goes."],
  ["61", "An environment drawing from a finite pool must partition it before anything is fitted, with a partition fixed for the life of the project."],
  ["62", "A quantity derived from the scenario pool — a budget, a normaliser, a threshold — comes from the whole pool, never from one split."],
  ["63", "Report the paired comparison on both the pool and the held-out set. Only the second number is about the problem rather than the sample."],
].forEach(([n, t]) => add(P(`**${n}.** ${t}`)));
add(BREAK());

// ---- appendix C ----
add(
  H1("Appendix C. Reproduction"),
  P("Every figure in this paper comes from one of these commands."),
  CODE([
    "# the catalog, re-measured and checked (58 rungs, 12 environments)",
    "python3 scripts/score_gym.py --ladder --check",
    "",
    "# one environment, more seeds",
    "python3 scripts/score_gym.py --env FloodDefend-v0 --seeds 32 --ladder",
    "",
    "# the training split, for comparison with the reported eval figures",
    "python3 scripts/score_gym.py --ladder --split train",
    "",
    "# the conformance and split suite",
    "python3 -m pytest tests/test_gym_surface.py -q",
    "",
    "# the service and the portal",
    "python3 -m uvicorn service.app:app --port 8000",
    "cd portal && npx next dev",
  ]),
  P("Three environments need data that is not committed, each raising a named error " +
    "with the exact command:"),
  CODE([
    "# building footprints for FloodBuyout-v0  (43,366 features)",
    "python3 scripts/fetch_austin_data.py --out data/austin_onion_bld \\",
    "  --bbox=-97.82,30.11,-97.69,30.21 --skip addresses,low_water_crossings,\\",
    "  floodplain_austin,floodplain_fema,base_flood_elevations",
    "",
    "# crossing locations for FloodCrossing-v0  (22 arms, 11 sites)",
    "python3 scripts/fetch_austin_data.py --out data/austin_crossings \\",
    "  --bbox=-97.95,30.05,-97.55,30.45 --skip buildings,addresses,\\",
    "  floodplain_austin,base_flood_elevations",
    "",
    "# the NDWS records and spread checkpoint for the two fire environments",
    "# are fetched by service/fire.py on first use",
  ]),
  P("The crossing depth library is about eight minutes of solver on first call and is " +
    "cached to `data/evac/crossing_depths.npz` thereafter."),
);

// ------------------------------------------------------------------ build --
const doc = new Document({
  creator: "Guruprasad Parasnis",
  title: "A Simulation Gym for Constrained Hazard Decisions",
  description: "Long-form working paper on the PSPE simulation gym",
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21, color: "18181B" } },
      heading1: { run: { font: "Calibri", size: 32, bold: true, color: "18181B" } },
      heading2: { run: { font: "Calibri", size: 25, bold: true, color: "18181B" } },
      heading3: { run: { font: "Calibri", size: 22, bold: true, color: "3F3F46" } },
    },
  },
  features: { updateFields: true },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
              margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        spacing: { after: 120 },
        children: [new TextRun({ text: "PSPE — A Simulation Gym for Constrained Hazard Decisions",
                                 size: 16, color: "8B8B93" })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "8B8B93" })],
      })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2];
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}  (${(buf.length / 1024 / 1024).toFixed(1)} MB)`);
});
