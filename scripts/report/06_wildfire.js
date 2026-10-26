module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("6 · The wildfire twin"),

P("The same four stages on a different hazard. The point of a second domain is not breadth — it is that the failure modes differ. Fire dynamics here are **learned** rather than solved, so model error compounds on its own output in a way the flood solver's does not."),

H2("6.1 Observation: two facts most pipelines render identically"),

...figure("docs/deck_bundle/images/wildfire-W2-burning-now.jpg",
  "Seven days of NASA FIRMS detections. The day with no satellite coverage is marked 'no pass' rather than drawn as an empty map.",
  { maxHeight: 400 }),

P("Satellites pass on an orbit, so coverage is a property of the **schedule**, not of the fire. If a day with no overpass is rendered as a map with no fire, a false negative has been inserted into the data — and not randomly, because overpass gaps are systematic in time and latitude."),

QUOTE("\"We did not look\" and \"nothing burned\" are different facts. A model trained on data that renders them identically learns that fires stop on certain days, and the symptom is a slightly worse model rather than an error."),

H3("6.1.1 Four defects in the dataset loader, all caught before a number"),

P("These are failures of *belief about data* — what a number in a file means — and they are harder than protocol failures because the code is correct and the output looks right."),

table(
  ["#", "Defect", "Effect"],
  [["23", "\"No detection\" read as \"not observed\"", "Would have deleted **37.4%** of days — exactly those where fires go out"],
   ["24", "Acquisition time read as radiative power", "`frp` held clipped HHMM clock values. Spotted because values ran 742–2142 with minutes always a multiple of 6"],
   ["25", "Downsample cropped the raster margin", "Lost up to **59%** of a fire's detections, smoothly, and only for some fires"],
   ["26", "Year fold keyed on the first date", "A one-fire \"2017\" fold, from a fire starting 30 December"]],
  [6, 34, 60]),

WHY("Defect 25 is the one to remember", "It degrades data **continuously** rather than breaking it, affects only some fires, and leaves every downstream number plausible. Nothing short of a conservation check — count detections before and after — finds it. There is no diagnostic in the model that would ever have flagged it."),

...figure("docs/figures/paper/figA3_wildfire_perception.png",
  "Perception on real fire sequences: what the satellite saw, what was inferred where it did not look, and the mask that separates the two.",
  { maxHeight: 320 }),

H2("6.2 The twin loop"),

...CODE(["û_t  =  m_t ⊙ y_t  +  (1 − m_t) ⊙ Φ_t"]),

P("Forecast with the model, observe part of the real field, and the mask `m` marks which cells were seen. The mask is a **per-cell switch**, not a blend: where there is a real observation it is taken outright, and where there is not the model's guess is kept. Then the model is adapted only on the cells that were seen."),

WHY("Why adapt only on observed cells", "So the model never trains on its own guesses. That is what keeps the loop from reinforcing its own drift, and it is the difference between a twin and a forecaster that has been run for longer."),

H3("6.2.1 The number that shrank, and why both are reported"),

table(
  ["Sample", "Improvement over open loop"],
  [["8 fires", "**8.4×**"],
   ["607 fires", "About **one third** of that, holding on 76%"]],
  [45, 55]),

P("The eight-fire figure was found first. At 75× the sample size the same benefit is a third the size. Both are reported, with the small-sample one second — this is what selection on a small sample looks like, and the same phenomenon reappears in the gym's held-out split (§7.4)."),

P("A further check: an **EnKF baseline** was run, and gain 1 turned out to be the right answer — that is, the simple replacement the loop already does is what a tuned filter converges to on this problem."),

H2("6.3 Checking the forecast: why accuracy is actively misleading"),

...figure("docs/deck_bundle/images/wildfire-W3-forecast-scored.jpg",
  "The model is shown today and never tomorrow, then scored against what actually happened. Hits and misses are reported separately.",
  { maxHeight: 400 }),

P("It found **99 of the 406** cells that burned, and called **94** that did not."),

P("Fire covers about **1% of the map**. So a model that predicts no fire anywhere is 99% accurate and has no skill whatsoever. With a base rate this extreme, accuracy is dominated by the negative class and is nearly independent of anything done to the positives."),

QUOTE("There is a sharper reason to count hits and misses separately than the usual imbalance argument: **the decision only ever consumes the positive predictions.** Crews are sent to cells predicted to burn. So recall and the false-positive count are the two operationally real quantities, they trade against each other, and collapsing them into one number destroys exactly the trade-off a planner has to make."),

H2("6.4 Planning: where to send the crews"),

...figure("docs/deck_bundle/images/wildfire-W5-where-crews-go.jpg",
  "Crews can clear a fixed fraction of ground per day. Blue is cleared ground, orange is what is alight today.",
  { maxHeight: 400 }),

H3("6.4.1 Naming the fair comparison"),

table(
  ["Comparison", "Result", "What it measures"],
  [["Against leaving it alone", "**−42%**", "Nearly uninformative — the budget is spent either way, so almost any policy beats not using the crews. It measures the *existence* of crews."],
   ["Against the same crews spread evenly", "**−33.1%**", "**The one that isolates the decision.** Same budget, only *where* varies."]],
  [28, 14, 58]),

WHY("Why the weaker number is the headline", "Because the stronger one answers a question nobody is asking. An operator already has crews; the decision is where to send them. Reporting −42% as the result would be reporting the budget, not the policy — the same error the polder environment catches in §7.3."),

H3("6.4.2 Against the operational rule"),

...figure("docs/deck_bundle/images/wildfire-W6-vs-operational.png",
  "Against forecast-then-treat-the-riskiest, on fires that actually happened. Left: the gain is largest where crews are scarce. Right: it needs a planning horizon to exist at all.",
  { maxHeight: 330 }),

P("**Twelve points better** than the operational rule — forecast the risk, treat the riskiest — on real fire records, planned through a learned spread model rather than a simulator. Population-weighted burned area, matched crew budget, paired across fires at **t = 11.15**."),

P("Why planning beats it: treating the riskiest cell today is greedy. It ignores that a break is only worth cutting where the fire will *arrive*, which depends on where the other breaks are cut. It is a sequential allocation problem being solved myopically."),

H3("6.4.3 The conditional result, which is stronger than the headline"),

table(
  ["Planning horizon", "Gain"],
  [["1 day", "**+0.6 points, not significant**"],
   ["3 days", "+12.0"],
   ["5 days", "+14.3"]],
  [50, 50]),

P("And by crew budget: +6.8 at a 1% daily budget, +16.9 at 5%."),

QUOTE("At a one-day horizon a greedy allocation is already optimal, so there is nothing for a planner to win. The gain appears at three and five days because that is when the decision becomes genuinely sequential. **A method that claimed to help at every horizon would be the suspicious one.**"),

WHY("Why the operating point belongs in the sentence", "An audit finding against our own abstract: \"twelve points better\" is the 3-day row of a horizon ablation, not a separate experiment. Quoting it without the horizon and the budget implies a result that holds across the sweep, and it does not. The headline sentence now carries both."),

H2("6.5 A distillation result that was meaningless in two ways"),

P("An attempt to distil the planner into a fast policy produced a number that would have **confirmed our own amortisation claim**, which is why it was checked."),

P("Defect 34: the distillation cloned raw actions with an MSE on a sparse target, so the policy learned the MSE-optimal constant — −1 everywhere. It treated 6×10⁻⁵ % of ground per day for a 0.0% reduction. The arm looked like a successful amortisation because the number it produced was small and stable."),

WHY("The general lesson", "A null result that agrees with your hypothesis is the most dangerous kind, because nothing prompts you to check it. This one was caught only by asking what the distilled policy actually did, rather than what it scored."),

];
};
