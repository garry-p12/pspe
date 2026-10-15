# Speaker notes — 47 slides, 30 minutes

**38 seconds per slide on average, so this is a triage document as much as a
script.** Every slide carries one of three marks:

- **SPINE** — deliver fully. Twenty-three slides, about two thirds of the
  running time. If you only land these, the talk works.
- **FAST** — one sentence, then move. Do not read the bullets; they are on
  screen and the audience reads faster than you talk.
- **CUT IF BEHIND** — nine slides you can skip live without apology. Nothing
  downstream depends on them. Say nothing about skipping; just advance.

The per-slide times sum to **30:15**, which means there is no slack: the
running totals in brackets assume you take every CUT slide at its listed time.
Drop two of them and you have a minute of air.

Running total is in brackets. If you are past the bracketed time at a SPINE
slide, cut the next CUT-IF-BEHIND slide rather than rushing the spine.

**Three delivery rules for a deck this dense:**

1. **Never read a bullet.** They are already on the slide. Say the thing the
   bullet does not say.
2. **Say the number once.** Every figure carries its own numbers; repeating
   them aloud makes the slide redundant and burns seconds you do not have.
3. **The honest slides get the same energy as the wins.** 41, 42, 44 are the
   most credible minutes in the talk. Do not hurry them to protect the pitch.

---

## ACT I — One creek  [0:00 → 3:00]

### 1. Title — **SPINE**  (0:30) [0:30]
Open on the picture. Five seconds of silence before you speak.

"This is Onion Creek, in south Austin. What you're looking at is live — those
are real river gauges reading right now, and that blue is a flood the model
solved while the page loaded."

**The claim:** "The argument is that systems like this fail in a particular
way, and the fix is to measure them against reality on a schedule rather than
against their own model."

→ *"Here's the decision it's actually for."*

### 2. The decision — **FAST**  (0:30) [1:00]
Point at three things in order: gauge, red roads, blue water.

"Three different people need three different answers from this. Today those
are three separate tools, and **none of them can tell you whether the other
two are right.**"

→ *"Before the method, see who it's about."*

### 3. Dixie Drive — **SPINE**  (1:00) [2:00]
Slow down. This is the emotional anchor and it is worth the silence.

"Seventy-eight buildings on one street. In this storm, sixty of them have
water above floor level. The worst takes four point eight five metres."

**Say the caveat here, not in questions:** "The ground under each house is
surveyed lidar. The floor height above that ground is an **assumption** —
seventeen centimetres, a slab and its step — because nobody publishes floor
heights for Austin. The panel says 'floor assumed'. It is the weakest number
on this slide and it is labelled."

→ *"So why is this hard? For a reason that isn't hydrology."*

### 4. Nothing is tested — **SPINE**  (1:00) [3:00]
The thesis slide. Everything downstream depends on it landing.

"Every component here reports on itself. The forecaster scores its forecast.
The simulator reports its own error. And every one of those numbers is
computed **inside the model's account of the world**."

Walk one row only — the first: "The planner's safety budget was satisfied,
according to the planner. Tested on the real system, seven percent of runs had
broken the limit."

**Flat delivery:** "We found six. In three, making the component *better* made
the decision *worse*. That is not a bug we fixed. That is the shape of the
problem."

→ *"Let me make that precise."*

---

## ACT II — The trap  [3:00 → 5:30]

### 5. Why the budget silently breaks — **SPINE**  (1:00) [4:00]
Point at the three terms, do not read the equation.

"Real cost is three things: what the controller sees, a hidden bias, and
run-to-run spread. The controller drives the first term to the limit and
reports success. Reality is the sum."

**The line that matters:** "For the margin to work it has to grow with sigma —
the policy's own spread. And sigma is the one quantity nothing in the loop
measures."

→ *"Which kills the obvious fix."*

### 6. Why the obvious fix fails — **SPINE**  (0:50) [4:50]
"The natural answer is: inflate the budget by how wrong the model is.
Ensembles, error bars, pessimism bonuses — they all measure model error. **None
of them measures sigma.**"

Point at the table. "Same margin. At one spread it's fine. At another it's
broken — forty-two percent."

"Three published methods that do exactly this land at twelve to thirteen
percent, against fourteen for applying **no margin at all**."

→ *"And that's a prediction, not a complaint."*

### 7. A testable prediction — **FAST**  (0:40) [5:30]
"One number governs it: rho, the policy's spread over the model's error
spread. At rho equals one the rate is exactly what you asked for. Above one it
degrades, and the formula says by how much."

**"Our experiments match that curve to within 0.003."** Then move.

→ *"So what do you do instead?"*

---

## ACT III — What PSPE is  [5:30 → 11:30]

### 8. Check against the world — **SPINE**  (0:50) [6:20]
"The answer is almost embarrassingly simple: check the system against the
world on a schedule, and when the check disagrees with the model, **the check
wins**."

Point at the three green paths, then the red dashed box: "That one does not
close. To learn from the outcome of an intervention you need a record of what
happened when someone took your advice. No such record exists. We mark that
rather than blur it."

→ *"What does it buy?"*

### 9. Characteristics — **FAST**  (0:30) [6:50]
"Three things, and none of them is better accuracy." Then dwell only on (c):

**"A system that says 'the gauge was silent, I cannot answer' is more useful
than one that produces a confident number from nothing. A refusal is a
result."**

→ *"Mechanically —"*

### 10. The goal — **CUT IF BEHIND**  (0:25) [7:15]
"Standard constrained problem: maximise reward, keep average cost under a
limit. The limit is calibrated — thirty-five percent of the way from doing
nothing to greedy — so a constrained learner has to give something up."

**One clause carries the rest of the talk:** "On average. Not every run."

### 11. The simulator — **CUT IF BEHIND**  (0:25) [7:40]
"A Fourier Neural Operator. Convert the field to frequencies, keep the twelve
lowest, learn how to mix them, convert back. Trained to be right next step,
right several steps running, and consistent with the physics."

### 12. Two gradients — **SPINE**  (1:00) [8:40]
Do not read the formula. Point once.

"Two ways to work out which direction to improve. Differentiate through the
simulator — precise, low noise, but the simulator's error becomes a **bias**.
Or try actions and do more of what worked — no bias, but very noisy."

"Standard practice is to pick one. We use both."

**The beat:** "And the blend is not a knob we tuned. It's computed, every
step, to minimise the noise in the result. The bias term in it is **measured**,
not assumed — we pay real samples to find out how wrong the model is."

→ *"One more piece."*

### 13. The budget-keeper — **SPINE**  (0:45) [9:25]
"A price on cost, set by a PID controller. Like a thermostat: over the limit,
raise the price; back under, ease off."

**Plant the flag, hard:** "Notice the catch at the bottom. The 'measured cost'
it reacts to is the **simulator's** cost. Remember this slide — we walk
straight into its failure mode in about fifteen minutes."

### 14. The fix — **SPINE**  (0:50) [10:15]
"Probe the real system periodically. Each probe gives you an episode's real
overshoot, which already contains both the bias and the spread — you don't
have to estimate them separately."

"Sort them, take the k-th. That's a conformal quantile, and it carries a
stated failure rate."

**The part people remember:** "And if you have too few runs, **no value in the
sample carries the guarantee** — so the code refuses rather than returning the
largest number it happens to have. A normal approximation always returns
something. That's the reason not to use one."

→ *"Does it work?"*

---

## ACT IV — Does the fix work  [11:30 → 13:15]

### 15. Matched budget — **SPINE**  (1:00) [11:15]
"The critical word is **matched**. Normally these baselines get far more
real-world data than we use, and people report the ratio as if it were the
result. It isn't — it's an input."

"Capped at the same real sample count, we're ahead of all four, on both
problems."

Point at the dashed line: "That's doing nothing at all. On the right-hand
problem, three of the four published baselines **cannot be distinguished from
it**."

Point at the percentages: "And those are our violation rates. We cross the
line more often than they do. We report it."

### 16. Violations cut threefold — **SPINE**  (1:00) [12:15]
"Fourteen percent down to five."

**The mechanism is the slide:** "Look at the second bar. A calibrated safety
margin, on its own, does **nothing** — zero of twenty repeats improve. It only
works calibrated against real measurements."

"If you calibrate a margin against the model's own opinion of its error,
you're asking the thing that's wrong how wrong it is."

→ *"Now — does any of this survive contact with a real place?"*

---

## ACT V — Flood twin  [13:15 → 19:00]
*Thirteen slides, six minutes. This act is where you make up time. Three are
marked CUT and two more are FAST — take them.*

### 17. A live twin — **FAST**  (0:30) [12:45]
"Real ground, real roads, twenty-seven live river gauges. You can ask it what
a storm does, where a wall would pay, and who can't get out. It solves in the
browser — nothing here was prepared for this talk."

### 18. Three inputs — **SPINE**  (0:50) [13:35]
Walk left to right, fast.

"Three inputs: the shape of the ground, the rain, the river. Shallow-water
physics on a forty-metre grid. Eighteen hours of flood in about two minutes."

**The architecture point:** "Out of the far side comes **one** thing — depth
of water everywhere, every hour. Which roads close is a routing question asked
of that. Whose floor floods is a height question asked of the same object.
There is one model here, not four."

### 19. Questions it answers — **CUT IF BEHIND**  (0:20) [13:55]
"Five questions, five places in the interface." Advance.

### 20. FEMA overlap — **FAST**  (0:40) [14:35]
"We can point it anywhere and re-solve. And we score the result against FEMA's
one-percent floodplain."

**Say what the number is and is not:** "We find about half of what FEMA maps,
and put a third of our water outside it. That's a **disagreement
measurement**, not a validation — FEMA is a different hazard definition
computed a different way. Neither is ground truth for the other. Both numbers
are on the page."

### 21. Elevation — **SPINE**  (0:50) [15:25]
Deliver as a confession, not a lecture.

"The standard global elevation product includes the treetops and the rooftops.
It's a *surface* model. So when you run water over it, the water ponds on the
forest canopy."

"Same storm, same solver, same code. **Fifty-four percent more flood**, purely
from which elevation file you loaded."

**The part that should worry them:** "And a finer grid makes it *worse*, because
more resolution means more spurious canopy. Every number was self-consistent.
The model never complained. It was just wrong about the ground."

### 22. Which roads close — **FAST**  (0:30) [15:55]
"Those four numbers across the top are wading depths — how deep water can be
before that vehicle stops. Car is twenty-five centimetres. The table
recomputes against whichever you pick."

**One honest line:** "Depth is the only variable. Velocity — which is what
actually sweeps cars away — is solved and never used."

### 23. Getting out — **CUT IF BEHIND**  (0:25) [16:20]
"Routing from the centre of the box to the nearest dry junction. Free-flow
speeds, no signals, no congestion — on a page about everyone leaving at once."

### 24. The kit — **FAST**  (0:35) [16:55]
"Five things you can build, each with a real cost model."

**Say only this about provenance:** "The rates are USACE's levee cost
programme, escalated to 2025 and multiplied by the Texas factor. Construction
only — design, land and contingency roughly double it, and we print that
factor and deliberately never multiply it in."

Point at 2.97×: "A two-metre levee costs three times a one-metre levee, not
twice. Fill grows with the square of height."

### 25. Banks and placing — **CUT IF BEHIND**  (0:25) [17:20]
"It looks for dry cells touching wet ones that are within two metres of being
overtopped — places where a little height changes the outcome."

### 26. Before / with plan — **FAST**  (0:25) [17:45]
"Two pills swap between doing nothing and the best solved option. Both from
the same grid."

**The detail worth the slide:** "Change a piece and the readout **retires**. A
result never sits beside a stale plan."

### 27. Cross-section — **CUT IF BEHIND**  (0:25) [18:10]
"Cut a line, drag the levee, see the crest it needs and what that costs.
Note the last tile: homes kept dry — **not counted**. We don't have that
number, so we don't show one."

### 28. Whose floor — **FAST**  (0:30) [18:40]
"Two settings. At the gauged discharge, thirty-one of seventy-eight houses go
over the floor. At the calibrated two-and-a-half times, sixty."

**Why the multiplier exists:** "That factor stands in for **missing physics** —
a fourteen-kilometre box that truncates the catchment, no bridges, no
culverts. It's calibrated against 2013, which flooded this block badly enough
that the city bought most of it out."

### 29. Gauges fail — **SPINE**  (0:50) [19:30]
"Everything so far assumes somebody is warned in time. So we measured that —
not modelled it. The lag between the upstream gauge's peak and the outlet's,
on the six largest floods on record."

Pause, point at the top two bars.

**"The two largest floods since that gauge was installed produced no usable
record at all. And one of the four it did capture gave twenty minutes, against
the two hours the evacuation needs."**

"Gauges fail in the floods worth warning for. Averaging over the days an
instrument worked describes the warning available during the events that
needed it least."

→ *"Everything so far has been one creek."*

---

## ACT VI — Wildfire  [19:30 → 24:00]

### 30. If it only worked on water — **FAST**  (0:25) [19:55]
"One application can't tell you whether the method works or whether that
hazard happened to suit it. So — same machinery, pointed at fire. And it's
where our best number on observed data comes from."

### 31. Wildfire twin — **SPINE**  (0:50) [20:45]
"Two NASA instruments: VIIRS on Suomi-NPP, and MODIS on Terra and Aqua. They
report what's alight as they pass overhead. Seven days here, pulled live."

Point at the grey panel.

**"That one tile is the design philosophy. The satellites didn't cover that
day. It would have been easier to draw an empty map — and an empty map says
'nothing burned', which we do not know."**

"Same distinction cost us a real number on the flood side: the river gauge was
silent on the two largest floods."

### 32. The twin loop — **CUT IF BEHIND**  (0:30) [21:15]
"Forecast, observe part of the real field, replace the guess where you have a
real value, adapt the model on what you saw."

**If you keep it, keep this clause:** "Syncing beats never looking again by
eight times on eight fires — and only about a third of that on six hundred
and seven. The bigger sample moved our own number down."

### 33. Forecast, scored — **SPINE**  (0:50) [22:05]
"Three panels: what was burning, what actually burned the next day, and what
the model expected. It only ever saw the first."

"It caught about a quarter of tomorrow's fire."

**Why that isn't as bad as it sounds:** "Fire is one percent of this map. A
forecaster that says 'no fire anywhere' is right ninety-nine percent of the
time and useless — which is why the panel counts what it found and what it
missed instead of printing an accuracy."

### 34. Trust the forecaster — **FAST**  (0:25) [22:30]
"Better than the forecaster published with this dataset. Not as good as the
best one anyone has built."

**"We did not set out to win at forecasting, and we didn't. A better spread
model would make every plan better without changing a word we claim. It's on a
slide because a tool that shows you a forecast without saying how good the
forecaster is invites you to assume it's good."**

### 35. Where the crews go — **SPINE**  (0:50) [23:20]
"Crews can clear only so much ground in a day. The decision is *where*."

"Forty-two percent less burns than doing nothing. That's the number people
quote and the one I trust least."

**The real comparison:** "Beating 'do nothing' only proves clearing fuel works
at all. Against spreading the same crews out evenly, it's thirty-three percent
better. Same crews, same budget, chosen places."

### 36. Twelve points — **FAST**  (0:30) [23:50]
"Against the forecast-then-treat rule crews actually use, at the same crew
budget, on real fire records: **twelve percentage points** of burned-area
reduction. Paired t of eleven."

Say *points*, not percent — it's 22% against 34%.

### 37. Result improves with days — **SPINE**  (0:50) [24:40]
"Run it at different planning horizons. One day ahead, the advantage
**vanishes** — six tenths of a point, not significant."

Pause.

**"That's not a weakness. It's the strongest evidence in the deck. At one day
a greedy allocation is provably optimal, so there is nothing for a planner to
find. The theory requires the gap to vanish there, and it does. A method
showing a gain at one day would be telling us something was wrong."**

"Then it opens as the decision becomes genuinely sequential."

### 38. No fire had a break cut — **SPINE**  (0:40) [25:20]
"Everything on those planning slides is in-model. The burn figures are what
the spread model predicts under each plan."

**"No fire on record ever had a break cut on our instruction, so nobody can
observe what would really have happened. Work built on fire simulators can
evaluate that counterfactual exactly, because the simulator *defines* the
intervention — at the price of a simulator gap. We fit to what real fires did,
and pay the opposite price. A trade, not a win."**

→ *"Two hazards, same method. Now — what if you ask either one to decide
repeatedly?"*

---

## ACT VII — From twins to decisions  [25:20 → 27:45]

### 39. A twin doesn't tell you what to do — **FAST**  (0:30) [25:50]
"A twin is a predictor. Prediction isn't a decision. So we wrapped two real
ones around it: running a gate and pump, and timing a multi-neighbourhood
evacuation. Both have a budget, and the budget is what makes them hard."

### 40. When to spend the fuel — **CUT IF BEHIND**  (0:25) [26:15]
"A gate drains for free when the river is low and seals when it's high. A pump
works any time and costs fuel."

**If you keep it, point at the refusal box:** "On the 2013 flood this gate and
pump make no difference at all — every schedule gives the same answer. So the
panel refuses to show a comparison. That's slide nine in practice."

### 41. A thermostat beats us — **SPINE**  (0:50) [27:05]
Same energy as slide 15. Do not hedge.

"On the gate and pump, a thermostat beats every learned method we tried.
Including ours."

"The incumbent is a level deadband — pump when the sump is deep, stop when
it's shallow, no model of anything. It wins, on a fraction of the energy."

**The sharper half:** "Worse: three of the four published safe-learning
methods finish **outside** the fuel budget, two of them every single run. A
method that breaks the constraint isn't a worse solution. It isn't a solution.
Those crosses shouldn't be ranked at all."

### 42. Our planner pushed to maximum — **SPINE**  (0:50) [27:55]
**"Remember that controller from slide thirteen."**

Point at (a), red: "That's how hard the policy pushes. It grows without bound,
past the point where its own output saturates. Beyond that dotted line every
gradient through it is zero — the policy has stopped being able to respond."

Point at (c): "And that's the budget controller, pushing harder and harder
against something that can no longer move. It fought, and it lost."

**The uncomfortable part:** "The fix was already in our codebase. Written
months earlier for a different problem, with this exact failure described in
the comment above it — and **switched off by default**. Every flood result we
had ran with it off."

Point at green: "Turn it on and everything behaves. It comes inside the
budget. And it still loses to the thermostat."

### 43. Three behaviours — **FAST**  (0:30) [28:25]
"Is the thermostat just very good? Partly. But mostly there's almost nothing
to decide."

"Every setting of that rule, swept across its whole range, produces **three**
distinct outcomes. Across the shaded band you can vary the cost by more than
double and the outcome doesn't change."

**The reframe:** "So the question isn't 'can learning beat the rule'. It's
whether it can reach the outcomes *between* those three steps."

---

## ACT VIII — What we will not claim  [28:25 → 30:00]

### 44. Three we found ourselves — **SPINE**  (0:45) [29:10]
Brisk, three items, no dwelling.

"Joint training of simulator and planner was a claimed contribution. It didn't
replicate — we replaced the figures rather than patching them."

"Our explanation briefs scored well on faithfulness and carried zero
state-specific information. A permutation control caught it."

"Our best twin result shrank to a third when we re-ran at seventy-five times
the sample."

**The point:** "Three of those we found by applying our own rule to ourselves.
All three moved a published number **down**. If the method's any good it
should cost us something — and it did."

### 45. Where that leaves the claim — **FAST**  (0:25) [29:35]
"And the honest label for the flood system: a shadow-capable API. Not a
digital shadow, certainly not a twin."

"Both gauges sit outside the model's own grid — one before the west edge, one
past the east — so neither can grade it. Until the box moves, FEMA's map is
the only scored extent we have."

### 46. Still open — **CUT IF BEHIND**  (0:15) [29:50]
"Three things open. The one that matters: correcting the model from the
*outcome* of an intervention, not just from observed state."

### 47. One sentence — **SPINE**  (0:25) [30:15]
Read it. Don't paraphrase. Then stop.

"Components that score well on their own metrics do not add up to a decision
that works. Measure against reality on a schedule, and you can both recover
the lost performance and say how often you will fail."

Pause. "Thank you — happy to take questions."

---

## If you are running long

Cut in this order. Each is self-contained and nothing downstream refers back:

1. **19** (questions the twin answers) — pure navigation
2. **27** (cross-section) — detail, not argument
3. **25** (banks and placing) — detail
4. **23** (getting out) — slide 29 covers the evacuation point better
5. **46** (still open) — fold its first bullet into 47
6. **11** (the FNO) — the architecture is not the contribution
7. **32** (twin loop) — but keep the "eight times → a third" clause somewhere
8. **10** (the goal) — only if desperate; it defines "on average"

**Never cut 3, 4, 29, 41, 42 or 44.** Three and twenty-nine are what people
remember; forty-one, forty-two and forty-four are why they believe the rest.

---

## Questions you should expect

**"Why is your FEMA overlap only 0.39?"**
Different hazard definition, computed a different way, on a box that truncates
eighty kilometres of catchment. It's a disagreement measurement, not a
validation, and we report both halves — what we miss and what we add.

**"Isn't 0.17 m for floor height a huge assumption?"**
Yes, and it's the weakest input on slide 3. On the one place that surveys them
— 2,924 Lismore properties — the median floor stands 1.95 m above its own front
gate, with the middle 90% spanning 4 cm to 3.3 m. Texas is overwhelmingly
slab-on-grade, which is why 0.17 is defensible here and would not be in
Queensland. The portal labels it rather than folding it into a depth.

**"Why should I believe a negative result from the people proposing the
method?"**
The right question, and it's why 41, 42 and 44 are in the main deck rather
than an appendix. Three of those negatives came from applying the method's own
rule to our own published numbers.

**"Would you deploy this?"**
The twin, yes, as a planning tool, with the elevation caveat stated. The
learned controllers, no — on the gate and pump the incumbent is better and
we'd say so. The warning-time measurement is what I'd put in front of an
emergency manager tomorrow, because it's measured rather than modelled.

**"What's the single biggest weakness?"**
That the margin is marginal, not per-step: it bounds the probability a
trajectory's total cost exceeds the limit, not cost at every timestep. And
that we are the least safe arm wherever the comparison is run.
