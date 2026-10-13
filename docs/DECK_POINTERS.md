# PSPE & the Onion Creek flood twin — slide pointers

**Audience:** mixed. People who will never read a loss function, sitting beside
people who will ask what the error bars were. The deck is built for the first
group; the second group is served by the plots and by the reserve slides at the
end.

**The rule this outline follows:** no number appears as text on a slide unless a
picture is carrying it. If a figure can show it, the figure shows it and you say
the sentence.

Each slide gives **SAY** (the line out loud), **SHOW** (the image), and where
useful **IF ASKED** (the technical backing, for the question that follows).

Images: `docs/figures/deck/*.{svg,pdf,png}` from
`python scripts/make_deck_figures.py`; portal screenshots in
`docs/images/portal-guide/`.

---

## Act I — The problem, through one creek

### 1. Title
- **PSPE — decision systems measured against reality, not against their own model.**
- Subtitle: *a flood digital twin for Onion Creek, Austin, and the method behind it*
- **SHOW:** `docs/images/portal-guide/01-forecast-open.jpg` full-bleed, no text over it.

### 2. The decision
- **SAY:** A storm is coming. Someone has to decide which roads to close, whose
  street to warn, and whether the thing you built last year will help. Today
  those are three different people looking at three different maps.
- **SHOW:** `01-forecast-open.jpg` — point at the live gauge, the blue flood, the
  red cut roads. One picture, three decisions.

### 3. Who this is actually about
- **SAY:** This is a real street in south Austin. In this storm, sixty of its
  seventy-eight houses have water above the floor — not water in the garden,
  water in the living room.
- **SHOW:** `docs/images/portal-guide/16-street-solved.jpg`
- *Let this one sit. This is the slide people remember.*

### 4. Why it is hard — the part everyone underestimates
- **SAY:** You need a model. And a model's own report card tells you almost
  nothing about whether the decision it feeds is any good.
- **SAY:** Every component in a system like this grades its own homework. The
  forecaster scores its forecast. The simulator scores its physics. All of it is
  measured *inside the model's account of the world*.
- **SHOW:** `deck_measurement`
- **SAY (the punchline):** We found six cases where a component passed its own
  test and the decision got worse. In three of them, making the component
  *better* made the decision *worse*.

---

## Act II — What PSPE is, and what it buys

### 5. The idea in one sentence
- **SAY:** Check the system against the world, on a schedule, and let that check
  overrule the model.
- **SAY:** That is the whole thesis. Everything else is how to do it without
  needing a hundred times more real data.
- **SHOW:** `pspe_system` (existing architecture figure)

### 6. What it offers
- **SAY:** Three things, and none of them is "better accuracy".
  1. It learns from **far less real-world data** than standard methods.
  2. It tells you **how often it will fail**, in advance.
  3. It **refuses** when it cannot answer, instead of guessing.
- **SHOW:** `deck_offers`
- *The third one is the one practitioners react to. Dwell there.*

### 7. How it works, mechanically
- **SAY:** We have a fast approximate model and a slow exact one. The fast one
  gives a cheap direction that is slightly wrong. The real system gives an
  expensive direction that is right but noisy. We use both, and the blend is not
  a setting we chose — it is computed to minimise the noise.
- **SHOW:** `deck_framework`
- **IF ASKED:** α\* = (V_lr − Cov) / (B² + V_pw + V_lr − 2 Cov), with B² the
  *measured* bias of the model gradient, not an assumed constant. The cost of
  measuring it is real rollouts, which is why we report it as a budget.

### 8. And a controller for the constraint
- **SAY:** There is always a budget — fuel, crew-days, how much disruption a
  city will tolerate. A controller pushes on the constraint when we exceed it
  and eases off when we do not.
- **SAY (set up Act V):** That controller has one failure mode, and we walked
  straight into it later. Remember this slide.

---

## Act III — The flood twin: what it is and how it works

### 9. What it is
- **SAY:** A live digital twin of Onion Creek. Real ground, real roads, real
  river gauges. You can ask it what a storm does, where a wall would pay, and
  who cannot get out.
- **SHOW:** `01-forecast-open.jpg`

### 10. How it works
- **SAY:** Three inputs — the shape of the ground, the rain, and the river. A
  water model pushes it forward. Out of the far side comes a depth of water
  everywhere, hour by hour, and everything else is a question asked of that.
- **SHOW:** `deck_twin_pipeline`
- **IF ASKED:** LISFLOOD-FP local-inertial shallow water on a 40 m grid, USGS
  3DEP bare-earth elevation, OpenStreetMap roads, live USGS gauge feed.

### 11. It is not a video — it is solved, live, anywhere
- **SAY:** This is not a pre-rendered scenario. Draw a box anywhere, and it
  fetches the terrain, the roads and the gauges and solves it.
- **SHOW:** `docs/images/portal-guide/05-forecast-town.jpg` or
  `docs/images/portal-guide/02-forecast-scrubbed.jpg`

### 12. The lesson that nearly broke it
- **SAY:** The default elevation data in this field includes the treetops and
  the rooftops. Water ponds on a forest canopy. Same storm, same model —
  fifty-four percent more flood, purely from the elevation data.
- **SHOW:** `deck_dem`
- **SAY:** Every number was self-consistent and wrong. This is the thesis in
  miniature, inside our own application.

### 13. Which roads close, and who cannot leave
- **SAY:** Once you have water everywhere, hour by hour, "which roads are cut"
  is a routing question — and so is "can anyone actually get out".
- **SHOW:** `docs/images/portal-guide/06-roads-open.jpg`

### 14. Would anyone know in time?
- **SAY:** A closure map does not tell you whether a warning would arrive. So we
  measured it — not modelled it — from the river-gauge record of real floods.
- **SHOW:** `deck_warning`
- **SAY:** The median looks fine. The median is also the least honest number on
  the chart: the two largest floods since that gauge was installed produced no
  record at all, and one of the four it did capture gave twenty minutes.
- **SAY:** Gauges fail in the floods worth warning for.

---

## Act IV — From twin to decision gym

### 15. Why turn it into a gym
- **SAY:** A twin tells you what *will* happen. It does not tell you what to
  *do*. So we wrapped two real decisions around it.
- **SHOW:** `deck_gym_loop`
- **SAY:** Operating a flood gate and pump. And deciding which neighbourhoods to
  evacuate, and when. Both have a budget, and the budget is what makes them
  hard.

### 16. What makes it a *real* gym
- **SAY:** The routing is real — three and a half thousand road segments,
  time-dependent, so a road that floods while you are driving on it counts. We
  precompute it so an episode costs microseconds and the realism survives.
- **IF ASKED:** Standard Gymnasium interface, Safety-Gymnasium cost convention,
  so any outside algorithm runs on it unmodified.

### 17. Operating the structures
- **SAY:** A gate drains for free when the river is low and seals when it is
  high. A pump works whenever you like and costs fuel. The entire decision is
  when to spend the fuel.
- **SHOW:** `docs/images/portal-guide/26-operate-result-panel.png` or
  `24-operate-placed-panel.png`

---

## Act V — Main results

### 18. Headline — learning from less
- **SAY:** Give every method the same amount of real-world data, and ours beats
  all four of the standard safe-learning algorithms. On both test problems.
- **SHOW:** `deck_matched_budget`
- **SAY:** The dashed line is doing nothing at all. On the right-hand problem,
  three of the four baselines cannot be told apart from it.
- **IF ASKED:** 20 seeds, bootstrap intervals. The closest competitor needs 4.6×
  more data to draw level with us.

### 19. Headline — safety you can state
- **SAY:** We cut constraint violations roughly threefold. And the mechanism is
  not what you would guess: a safety margin on its own does nothing at all. It
  only works when it is calibrated against *real* measurements.
- **SHOW:** `deck_margin`

### 20. The result we did not want
- **SAY:** On the flood gate and pump, a thermostat beats every learned method
  we tried. Including ours.
- **SHOW:** `deck_polder`
- **SAY:** Worse — three of the four published safe-learning methods finish
  *outside* the fuel budget, two of them every single time. A method that breaks
  the constraint is not a worse solution. It is not a solution.

### 21. Why our own planner lost
- **SAY:** It pushed the pump to maximum and stayed there. The safety controller
  from slide 8 screamed — and nothing happened, because a policy pinned at its
  limit cannot respond to anything.
- **SHOW:** `deck_collapse`
- **SAY:** The fix was already in our code, written months earlier for a
  different problem, with this exact failure described in the comment above it —
  and switched off by default. Every flood result we had ran with it off.
- **SAY (honest ending):** Turn it on and everything behaves: it comes inside
  the budget, the controller releases. And it still loses to the thermostat.

### 22. Why the thermostat is so hard to beat
- **SAY:** Because on this problem there is almost nothing to decide. Sweep every
  setting of the simple rule and you get three distinct outcomes. Three.
- **SHOW:** `deck_flood_frontier`
- **SAY:** So the fair question is not "can learning beat the rule". It is "can
  learning reach the outcomes *between* the rule's three steps" — and those
  steps are enormous.

---

## Act VI — What we will not claim

### 23. The trade we make
- **SAY:** We get more done, and we cross the line more often than the cautious
  methods do. We report that number rather than hiding it, because a method that
  never crosses the line is usually a method that never does anything.
- **SHOW:** `deck_matched_budget` again — point at the violation percentages.

### 24. Things that did not survive
- **SAY:** A claimed contribution — training the simulator and the planner
  together — did not replicate. We replaced the figures rather than patching them.
- **SAY:** Our explanation module produced briefs that scored well on
  faithfulness and carried **zero** information about the actual state. A
  control experiment caught it.
- **SAY:** Our best twin result shrank to a third of its size when we re-ran it
  at seventy-five times the sample.
- **SAY (the point):** Three of those we found by applying our own rule to
  ourselves. All three moved a published number down.

### 25. What is still open
- Correcting the model from the *outcome* of an intervention, not just from
  observed state — the one digital-twin claim we cannot yet make.
- A retrospective shadow: replay historical floods using only what was known at
  the time, and score the model against what actually happened.
- Whether the mixing rule helps on the flood problem once the policy is not
  pinned at its limit. Running now.

### 26. One sentence
> **Components that score well on their own metrics do not add up to a decision
> that works. Measure against reality on a schedule, and you can both recover
> the lost performance and say how often you will fail.**

---

## Reserve slides — for the technical questions

- **R1. The null floor.** `deck_null_floor` — at full data budget, three of four
  published baselines cannot be shown to beat a do-nothing policy on one of our
  families. Use if someone challenges the benchmark.
- **R2. The mixing rule, honestly.** `deck_mixing_rule` — adaptive α is a
  *variance* effect, not a mean effect. The only budget-matched comparison does
  not separate on the mean (p = 0.083). What it removes is the tail: 5 of 20
  seeds violating, down to 0.
- **R3. Evacuation at matched cost.** Learned policies come out ahead of the
  trigger by roughly +500 with a standard error of the same size — a direction,
  not a result. The earlier version of this comparison scored them against a
  rule spending 2.3× as much; fixing that flipped the sign and changed no
  conclusion.
- **R4. Conformal coverage.** Exchangeability rejected on 20 of 20 seeds, and
  coverage holds anyway.
- **R5. δ is not a dial.** Tightening the risk level costs return and buys no
  safety.
- **R6. The gauge at the wrong end.** The nearest gauge to the study box sits
  0.13 km *past* the outlet; using it meant the model was being driven by its
  own downstream answer. Found by checking position, not distance.
- **R7. Live portal.** Open it. `/warning`, the Roads tab, the street view.

---

## Figure captions

For a paper, or to read aloud. The images carry no caption of their own.

**`deck_offers`** — What the method provides: (a) the same result from
substantially less real-world interaction; (b) a violation rate stated in
advance rather than discovered; (c) an explicit refusal where the data does not
support an answer.

**`deck_framework`** — A fast differentiable model supplies a low-noise but
biased gradient; the real system supplies an unbiased but noisy one. The two are
combined at the weight that minimises the variance of the result, where the
model's bias is measured rather than assumed. A budget controller adjusts the
constraint penalty, and the resulting policy is re-measured on the real system.

**`deck_measurement`** — Six components, each of which passed its own metric
(left) while the decision it fed was worse than that metric implied (centre).
The right column names the measurement that revealed the gap.

**`deck_twin_pipeline`** — The flood twin. Ground elevation, rainfall and live
river level drive a shallow-water model on a 40 m grid; the result is a depth of
water everywhere at each hour, from which road closures, building-level flooding
and evacuation routing are derived. The live gauge corrects the model during an
event.

**`deck_dem`** — The same storm solved on two elevation datasets. A surface
model includes vegetation and buildings and ponds water on them, inflating the
flooded area by 54% relative to a bare-earth survey.

**`deck_gym_loop`** — The twin as a sequential decision problem. A controller
observes the flood state and acts; the environment returns both an outcome and a
cost, and the budget on that cost is what makes the problem non-trivial. Two
tasks are posed on it: operating a flood gate and pump, and timing a
neighbourhood evacuation.

**`deck_matched_budget`** — Improvement over a do-nothing policy on two PDE
control families, every method restricted to the same number of real-system
samples. Points are means over 20 seeds, bars are 95% bootstrap intervals, and
the percentage beside each interval is the fraction of runs that violated the
safety constraint.

**`deck_margin`** — Constraint violations over 20 seeds. A calibrated margin
alone changes nothing; periodic testing against the real system helps; together
they cut violations from 14.1% to 5.0%.

**`deck_polder`** — Controllers for a flood gate and pump against an energy
budget. Circles are within budget, crosses are not. The best admissible
controller is the simple rule already in operational use.

**`deck_collapse`** — One training run with a stabilising guard disabled (red)
and enabled (green). Without it the policy's output grows past the point where
it can respond at all, consumes the maximum possible energy, and the safety
penalty grows without limit against a quantity that can no longer move.

**`deck_flood_frontier`** — Every setting of a single-threshold evacuation rule.
The rule produces three distinct outcomes; across the shaded band its cost varies
2.3× while the outcome does not change.

**`deck_warning`** — Warning time available before each of the six largest floods
on record, measured as the lag between an upstream gauge's peak and the outlet's.
The dashed line is the minimum needed to evacuate.

**`deck_null_floor`** — The same family as the right-hand panel of
`deck_matched_budget`, with baselines given their full customary data budget.
Three of four still cannot be shown to improve on doing nothing.

**`deck_mixing_rule`** — (a) Violation rates for a fixed blend weight against two
adaptive rules. (b) The real-system samples each consumed; the third arm's zero
comes with 75% more data, so only the first two are a matched comparison.
