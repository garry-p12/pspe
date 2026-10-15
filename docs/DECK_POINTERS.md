# PSPE & the Onion Creek flood twin — slide pointers

**Audience:** mixed. People who will never read a loss function, sitting beside
people who will ask what the error bars were. The deck is built for the first
group; the second is served by the plots and the reserve slides.

**Each slide has four parts:**

- **SAY** — the headline. One line, said out loud.
- **ON SLIDE** — the bullets that go on the slide itself. Two or three, short.
- **SHOW** — the image.
- **IF ASKED** — backup detail for questions. **Speaker notes only, never on a
  slide.**

**The rule:** no statistic appears as a bullet unless a figure on the same slide
is carrying it. Bullets carry meaning; figures carry numbers.

Images: `docs/figures/deck/*.{svg,pdf,png}` from
`python scripts/make_deck_figures.py`; portal screenshots in
`docs/images/portal-guide/`.

---

## Act I — The problem, through one creek

### 1. Title
- **SAY:** PSPE — decision systems measured against reality, not against their own model.
- **ON SLIDE:**
  - A flood digital twin for Onion Creek, Austin, Texas
  - And the method behind it
- **SHOW:** `01-forecast-open.jpg` full-bleed.

### 2. The decision
- **SAY:** One picture, three decisions.
- **ON SLIDE:**
  - Which roads to close, whose street to warn, what to build
  - Today: three different people, three different maps, three different models
  - None of them can tell you whether the others are right
- **SHOW:** `01-forecast-open.jpg` — point at the live gauge, the blue water, the red cut roads.

### 3. Who this is about
- **SAY:** Sixty of its seventy-eight houses have water above the floor.
- **ON SLIDE:**
  - A real street in south Austin, in a storm this creek has already seen
  - Water **over the floor**, not over the ground outside — floor assumed
    0.17 m up, and the panel says so
  - The deepest house takes 4.85 m
- **IF ASKED:** Ground at each structure is City of Austin lidar; the floor
  height above it is an assumption, because nobody publishes it for Austin and
  FEMA elevation certificates are not bulk-public. 0.17 m is a slab-on-grade
  step. On 2,924 surveyed Lismore properties the median floor stood 1.95 m
  above its own front gate — so this assumption is the weakest link on the
  slide and the portal labels it rather than folding it into a depth.
- **SHOW:** `16-street-solved.jpg`
- *Let this sit. This is the slide people remember.*

### 4. Why it is hard
- **SAY:** Every component grades its own homework.
- **ON SLIDE:**
  - Forecast skill, reconstruction accuracy, simulator error — all measured
    *inside the model's own account of the world*
  - Six cases where a component passed its own test and the decision got worse
  - In three, making the component **better** made the decision **worse**
- **SHOW:** `deck_measurement`
- **IF ASKED:** The map from component metric to decision quality is not
  monotone, and in three measured cases it is inverted.

---

## Act II — What PSPE is, and what it buys

### 5. The idea
- **SAY:** Check the system against the world, on a schedule, and let that check overrule the model.
- **ON SLIDE:**
  - Three paths carry real measurement back into the loop
  - Probe the real system · re-anchor to observation · score on the decision
  - One path does not close, and we mark it as such
- **SHOW:** `deck_system`
- **IF ASKED:** The dashed path is intervention-outcome feedback — no
  observational record contains a counterfactual where we cut a firebreak on
  our instruction. That boundary is the honest scope of the twin claim.

### 6. What it offers
- **SAY:** Three things, and none of them is "better accuracy".
- **ON SLIDE:**
  - **Learns from less** — the same result on a fraction of the real-world data
  - **States its own failure rate** — in advance, not in hindsight
  - **Refuses** when the data cannot support an answer
- **SHOW:** `deck_offers`
- *The third is the one practitioners react to. Dwell there.*

### 7. How it works
- **SAY:** A fast approximate model and a slow exact one. Use both.
- **ON SLIDE:**
  - The fast model gives a cheap direction that is slightly wrong
  - The real system gives an expensive direction that is right but noisy
  - The blend is **computed, not chosen** — it minimises the noise in the result
- **SHOW:** `deck_framework`
- **IF ASKED:** α\* = (V_lr − Cov) / (B² + V_pw + V_lr − 2 Cov). B² is the
  *measured* bias of the model gradient, not an assumed constant — measuring it
  costs real rollouts, which is why we report it as a budget.

### 8. The constraint controller
- **SAY:** And a controller for the budget.
- **ON SLIDE:**
  - There is always a budget — fuel, crew-days, how much disruption a city tolerates
  - Over the limit, push harder; back under, ease off
  - **Remember this slide.** It has one failure mode and we walked into it
- **IF ASKED:** PID control on the Lagrange multiplier, gains scaled to the
  reward gradient. The failure mode is slide 21.

---

## Act III — The flood twin

### 9. What it is
- **SAY:** A live digital twin of Onion Creek.
- **ON SLIDE:**
  - Real ground, real roads, real river gauges — 27 stations, read live
  - Ask it what a storm does, where a wall would pay, who cannot get out
  - Not a case study: it solves, in the browser, while you watch
- **SHOW:** `01-forecast-open.jpg`

### 10. How it works
- **SAY:** Three inputs in. A depth of water everywhere, hour by hour, out.
- **ON SLIDE:**
  - Ground elevation, rainfall, river level → shallow-water physics on a 40 m grid
  - Eighteen hours of flood in about two minutes
  - Everything else — roads, buildings, evacuation — is a question asked of that depth
- **SHOW:** `deck_twin_pipeline`
- **IF ASKED:** LISFLOOD-FP local-inertial solver, USGS 3DEP bare-earth
  elevation, OpenStreetMap roads, live USGS gauge feed.

### 11. Solved anywhere
- **SAY:** It is not a video. It is solved, live, anywhere.
- **ON SLIDE:**
  - Draw a box on any part of the map and it fetches terrain, roads and gauges
  - Click any point for the depth of water on that spot
  - Nothing here was prepared in advance for this talk
- **SHOW:** `05-forecast-town.jpg`

### 12. The lesson that nearly broke it
- **SAY:** Trees and rooftops are not ground.
- **ON SLIDE:**
  - The default elevation data in this field includes vegetation and buildings
  - Water ponds on a forest canopy — same storm, same model, 54% more flood
  - A finer grid makes it **worse**: more resolution, more spurious canopy
- **SHOW:** `deck_dem`
- **IF ASKED:** Copernicus GLO-30 is a surface model; USGS 3DEP is bare earth.
  Every number was internally self-consistent and wrong. This is the thesis in
  miniature, inside our own application.

### 13. Who cannot leave
- **SAY:** Which roads close, and who cannot leave.
- **ON SLIDE:**
  - Once you have water hour by hour, both are routing questions
  - The route avoids roads that are under water **when you would reach them**
  - 117 of 891 km unusable in this storm — and a way out still exists, for now
- **SHOW:** `06-roads-open.jpg`

### 14. Would anyone know in time?
- **SAY:** A closure map does not tell you whether a warning would arrive.
- **ON SLIDE:**
  - Measured from the gauge record of real floods — not modelled
  - Median notice looks comfortable; the median is the least honest number here
  - The two largest floods since the gauge was built left **no record at all**
- **SHOW:** `deck_warning`
- **IF ASKED:** One of the four it did capture gave 20 minutes, against the 120
  the evacuation study needs. Gauges fail in the floods worth warning for.

---

## Act IV — From twin to decision gym

### 15. Why a gym
- **SAY:** A twin tells you what *will* happen. It does not tell you what to *do*.
- **ON SLIDE:**
  - Two real decisions wrapped around the same twin
  - **Operate** — when to open the gate, when to spend pump fuel
  - **Evacuate** — which neighbourhoods to warn, and when
- **SHOW:** `deck_gym_loop`
- **IF ASKED:** Both return an outcome and a cost. The budget on that cost is
  what makes either problem non-trivial.

### 16. What makes it real
- **SAY:** What makes it a *real* gym.
- **ON SLIDE:**
  - Three and a half thousand road segments, time-dependent routing
  - A road that floods while you are driving on it counts against you
  - Precomputed, so an episode costs microseconds and the realism survives
- **IF ASKED:** Standard Gymnasium interface and the Safety-Gymnasium cost
  convention, so an outside algorithm runs on it unmodified.

### 17. Operating the structures
- **SAY:** The entire decision is when to spend the fuel.
- **ON SLIDE:**
  - A gate drains for free when the river is low, and seals when it is high
  - A pump works whenever you like and costs fuel
  - On a big enough flood the honest answer is *there is no decision here* — and
    the panel says so
- **SHOW:** `26-operate-result-panel.png`

---

## Act V — Main results

### 18. Headline
- **SAY:** Same data: ours beats all four standard safe-learning algorithms.
- **ON SLIDE:**
  - Every method capped at the **same** real-world sample count
  - Two independent test problems, 20 repeats each
  - The dashed line is doing nothing at all — on the right, three of four
    baselines cannot be told apart from it
- **SHOW:** `deck_matched_budget`
- **IF ASKED:** Bootstrap intervals, seed-paired. The closest competitor needs
  4.6× more data to draw level.

### 19. Safety
- **SAY:** Safety you can state: violations cut roughly threefold.
- **ON SLIDE:**
  - A safety margin on its own does **nothing** — zero of twenty repeats improve
  - It only works when calibrated against real measurements
  - Calibrate the thing that actually fails, not the model's opinion of itself
- **SHOW:** `deck_margin`

### 20. The result we did not want
- **SAY:** On the flood gate and pump, a thermostat beats every learned method. Including ours.
- **ON SLIDE:**
  - The incumbent rule-based controller wins, on a fraction of the energy
  - Three of four published safe-learning methods finish **over budget** — two of
    them on every single run
  - A method that breaks the constraint is not a worse solution. It is not a solution
- **SHOW:** `deck_polder`

### 21. Why our planner lost
- **SAY:** It pushed the pump to maximum and stayed there.
- **ON SLIDE:**
  - A policy pinned at its limit cannot respond to anything — every gradient vanishes
  - The budget controller from slide 8 pushed harder and harder against a policy
    that could no longer move
  - The fix was already in our code, written months earlier, **switched off by default**
- **SHOW:** `deck_collapse`
- **IF ASKED:** Turn it on and every mechanism behaves — it comes inside the
  budget and the controller releases. And it still loses to the thermostat.

### 22. Why the thermostat is hard to beat
- **SAY:** Sweep every setting of the simple rule and you get three outcomes. Three.
- **ON SLIDE:**
  - One threshold applied to every neighbourhood at once has very few behaviours
  - Across a wide band of cost, the outcome does not change at all
  - So the fair question is not "can learning beat the rule" but "can it reach
    the outcomes **between** its steps"
- **SHOW:** `deck_flood_frontier`

---

## Act VI — What we will not claim

### 23. The trade
- **SAY:** We get more done, and we cross the line more often.
- **ON SLIDE:**
  - We are the least safe arm wherever the comparison is run
  - We report that number rather than burying it
  - A method that never crosses the line is usually one that never does anything
- **SHOW:** `deck_matched_budget` — point at the violation percentages.

### 24. Things that did not survive
- **SAY:** Three claims we removed, and how we found them.
- **ON SLIDE:**
  - **Joint training** — a claimed contribution that did not replicate; we
    replaced the figures rather than patching them
  - **Explanation briefs** — scored well on faithfulness, carried **zero**
    information about the actual state
  - **Our best twin result** — shrank to a third at seventy-five times the sample
- **IF ASKED:** Three of these we found by applying our own rule to ourselves.
  All three moved a published number down.

### 25. Still open
- **SAY:** What is still open.
- **ON SLIDE:**
  - **Correcting from outcomes**, not just observed state — the one twin claim
    we cannot yet make
  - **A retrospective shadow** — replay historical floods on what was known at
    the time, and score against what happened
  - **The mixing rule on floods**, now the policy is not pinned at its limit

### 26. One sentence
> **Components that score well on their own metrics do not add up to a decision
> that works. Measure against reality on a schedule, and you can both recover
> the lost performance and say how often you will fail.**

---

## Reserve slides — for the technical questions

- **R1. The null floor.** `deck_null_floor` — at full data budget, three of four
  published baselines cannot be shown to beat a do-nothing policy on one family.
  *Use if someone challenges the benchmark.*
- **R2. The mixing rule, honestly.** `deck_mixing_rule` — a variance effect, not
  a mean effect. The budget-matched comparison does not separate (p = 0.083);
  what it removes is the tail, 5 of 20 seeds violating down to 0.
- **R3. Evacuation at matched cost.** Learned policies come out ahead of the
  trigger by roughly +500 with a standard error of the same size — a direction,
  not a result. The earlier version scored them against a rule spending 2.3× as
  much; fixing that flipped the sign and changed no conclusion.
- **R4. Conformal coverage.** Exchangeability rejected 20 of 20 seeds, and
  coverage holds anyway.
- **R5. δ is not a dial.** Tightening the risk level costs return and buys no
  safety.
- **R6. The gauge at the wrong end.** The nearest gauge sits 0.13 km *past* the
  outlet; using it meant the model was driven by its own downstream answer.
  Found by checking position, not distance.
- **R7. Live portal.** Open it: the Roads tab, the warning card, the street view.

---

## Figure captions

For a paper, or to read aloud. The images carry no caption of their own.

**`deck_system`** — The four stages, the three paths that carry real
measurement back into them, and the one that cannot. The dashed path is
intervention-outcome feedback: no observational record contains a case where
an intervention was taken on the system's instruction, so that path is
model-based and is marked rather than drawn as if it closed.

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
