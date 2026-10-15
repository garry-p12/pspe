# Speaker notes — PSPE & the Onion Creek flood twin

A spoken script, one entry per slide. Not a transcript to read aloud word for
word: the **bold** line is the beat that must land, the rest is how to get
there. Timings sum to about 25 minutes, leaving room for questions.

Three habits that make this talk work:

1. **Say the number once, point at the picture.** Every figure carries its own
   numbers. Repeating them aloud makes the slide redundant and the talk slow.
2. **Do not apologise for the negative results.** Slides 20, 21 and 24 are the
   most credible part of the deck. Deliver them at the same pace and volume as
   slide 18.
3. **When you do not know, say so and point at a reserve slide.** There are
   seven, and they exist precisely so that "I don't know" is never the answer.

---

## Act I — The problem, through one creek  (4 min)

### 1. Title  (0:30)
Open on the picture, not on yourself. Give people five seconds to look at it
before you speak.

"This is Onion Creek, in south Austin. What you're looking at is live — those
are real river gauges reading right now, and that blue is a flood the model
solved while the page loaded."

**Then the title claim:** "The argument of this talk is that systems like this
fail in a particular way, and that the fix is to measure them against reality
on a schedule rather than against their own model."

→ *"Let me show you the decision this is actually for."*

### 2. The decision  (0:45)
Point at three things on the screen in order: the gauge reading top-left, the
red roads, the flooded area.

"Three different people need three different answers here. The emergency
manager needs to know which roads to close. The city needs to know whether the
detention pond they're costing is worth it. And somebody has to decide when to
tell a neighbourhood to leave."

**The beat:** "Today those are three separate tools, three separate models, and
none of them can tell you whether the other two are right."

→ *"Before the method, I want you to see who this is about."*

### 3. Who this is about  (1:00)
Slow down here. This is the emotional anchor of the talk and it is worth the
silence.

"This is Dixie Drive. Seventy-eight buildings. In the storm the model just ran,
sixty of them have water above floor level — not over the ground outside, over
the floor. The worst house takes four point eight five metres."

**Say the caveat here, do not save it for questions.** "The ground at each
house is surveyed lidar. The floor height above that ground is an assumption —
seventeen centimetres, a concrete slab and its step — because nobody publishes
floor heights for Austin. On the one place that does survey them, the spread
runs from four centimetres to over three metres. The panel on screen says
'floor assumed'. It is the weakest number on this slide and it is labelled."

Pause. Then:

**"Everything after this slide is in service of getting that number right, and
of knowing when we have got it wrong."**

→ *"So why is that hard? It is hard for a reason that is not about hydrology."*

### 4. Why it is hard  (1:30)
This is the thesis slide. Take your time; the rest of Act II depends on it
landing.

"Every component in a system like this reports on itself. The forecaster scores
its own forecast. The simulator reports its own error. The state estimator
reports how well it reconstructs the state. And every one of those numbers is
computed *inside the model's account of the world*."

Walk one row of the table — the first is the easiest to feel:

"Here, the planner's safety budget was satisfied, according to the planner. When
we tested it on the real system, seven percent of runs had broken the limit."

**The punchline, delivered flatly:** "We found six of these. In three of them,
making the component *better* made the decision *worse*. That is not a bug we
fixed. That is the shape of the problem."

→ *"So what do you do about it?"*

---

## Act II — What PSPE is, and what it buys  (4 min)

### 5. The idea  (1:00)
"The answer is almost embarrassingly simple to state: check the system against
the world on a schedule, and when the check disagrees with the model, the check
wins."

Point at the three green paths. "Probe the real system periodically. Re-anchor
the model state to what was actually observed. And score the belief on the
*decision* it produces, not on how well it reconstructs."

**Then point at the dashed red path and own it:** "That one does not close. To
learn from the outcome of an intervention, you need a record of what happened
when someone took your advice. No such record exists for wildfire. We mark that
boundary rather than blur it."

→ *"What does that buy you?"*

### 6. What it offers  (1:00)
"Three things, and I want to be clear that none of them is better accuracy."

Take the panels in order, briefly. "It learns from far less real-world
interaction. It tells you its failure rate in advance rather than discovering it
afterwards. And it refuses."

Dwell on the third:

**"That last one is the part practitioners care about. A system that says 'the
gauge was silent on this flood, I cannot answer' is more useful than one that
produces a confident number from nothing. A refusal is a result."**

→ *"Mechanically, here is how the first one works."*

### 7. How it works  (1:15)
Do not read the equation. Point at it once and move on.

"We have two ways to work out which direction to improve. A fast approximate
model gives us a cheap answer that is slightly wrong. The real system gives us
an expensive answer that is right but noisy."

"Standard practice is to pick one. We use both — and the blend between them is
not a knob we tuned. It is computed, every step, to minimise the noise in the
result."

**The bit worth saying out loud:** "The bias term in that formula is measured,
not assumed. We pay real samples to find out how wrong the fast model is. That
cost shows up later in this talk as a budget, and it is why one of our own
results has an asterisk on it."

→ *"One more component, and then I'll show you the application."*

### 8. The constraint controller  (0:45)
"Every real decision has a budget. Fuel for a pump. Crew-days for a firebreak.
How much disruption a city will tolerate before it stops believing you."

"So there's a controller on the constraint. Over the limit, push harder. Back
under, ease off. Standard idea, and it mostly works."

**Plant the flag:** "Remember this slide. It has exactly one failure mode, and
about an hour of compute from now, we walk straight into it."

→ *"Now — the application."*

---

## Act III — The flood twin  (5 min)

### 9. What it is  (0:45)
"This is a live digital twin of Onion Creek. Real ground elevation from laser
survey. Real road network. Twenty-seven river gauges, read live."

"You can ask it what a given storm does, where a wall would pay for itself, and
who can't get out. It is not a case study we prepared — it solves in the
browser."

→ *"Here is what is actually happening underneath."*

### 10. How it works  (1:15)
Walk left to right. Keep it concrete.

"Three inputs. The shape of the ground. The rain — either a design storm or live
radar. And the river level from the gauges. Those drive shallow-water physics on
a forty-metre grid, and eighteen hours of flood solves in about two minutes."

"Out of the far side comes one thing: the depth of water everywhere, at every
hour. And every question on the right is a question asked of that one object."

**The architecture point:** "Which roads close is a routing question asked of
the depth. Whose floor floods is a building-height question asked of the depth.
There is one model, not four."

→ *"And it is not limited to this creek."*

### 11. Solved anywhere  (0:45)
"Nothing here was precomputed for this talk. Draw a box anywhere on the map and
it fetches the terrain, the roads and the gauges for that box and solves it."

"Click any point and it tells you the depth of water on that spot and the ground
elevation under it."

→ *"Which brings me to the mistake that nearly invalidated all of it."*

### 12. The lesson  (1:15)
This slide is the thesis reappearing inside our own work. Deliver it as a
confession, not a lecture.

"The standard elevation dataset in this field — the one you get by default —
includes the treetops and the rooftops. It's a *surface* model. So when you run
water over it, the water ponds on the forest canopy."

"Same storm. Same solver. Same code. Fifty-four percent more flood, purely from
which elevation product you loaded."

**The part that should worry the room:** "And a finer grid makes it *worse*,
because more resolution means more spurious canopy. Every number was internally
self-consistent. The model never complained. It was just wrong about the
ground."

→ *"With the right ground, here's what the twin is for."*

### 13. Who cannot leave  (0:45)
"Once you have water hour by hour, 'which roads close' and 'can anyone get out'
are both routing questions."

"And this routing is time-dependent — it avoids roads that are under water *at
the moment you'd reach them*, not roads that are under water now. In this storm,
a hundred and seventeen of eight hundred and ninety-one kilometres are unusable,
and a route out still exists."

→ *"But a closure map leaves one question unanswered, and it's the important
one."*

### 14. Would anyone know in time?  (1:15)
"Knowing which roads flood doesn't tell you whether anyone would be warned. So
we measured that — not modelled it. We went to the gauge record and timed the
lag between the upstream gauge's peak and the outlet's, on real floods."

"The median is four hours forty-five. That sounds fine."

Pause, then point at the top two bars:

**"The median is the least honest number on this chart. The two largest floods
since that gauge was installed — the two that actually mattered — produced no
usable record at all. And one of the four it did capture gave twenty minutes,
against the two hours the evacuation study needs."**

"Gauges fail in the floods worth warning for. Averaging over the days the
instrument worked describes the warning available during the floods nobody
needed warning for."

→ *"So the twin tells you what will happen. It doesn't tell you what to do."*

---

## Act IV — From twin to decision gym  (3 min)

### 15. Why a gym  (1:00)
"A twin is a predictor. Prediction isn't a decision. So we wrapped two real
decisions around this one."

"The first is operating a flood gate and a pump station. The second is deciding
which neighbourhoods to evacuate and when."

**The framing:** "Both of them return an outcome and a cost. And it's the budget
on that cost that makes either one hard — without it, you evacuate everyone
immediately and pump at maximum forever."

→ *"Two things make this more than a toy."*

### 16. What makes it real  (0:45)
"Three and a half thousand road segments, with time-dependent routing. A road
that floods while you are driving on it counts against you — which is the entire
difficulty of evacuation timing."

"We precompute that routing, so an episode costs microseconds while the realism
survives. And it exposes a standard interface, so published algorithms run on it
unmodified. That matters for the comparison in a few slides."

→ *"Here's the first task."*

### 17. Operating the structures  (1:00)
"A gate drains for free when the river is lower than the water behind it, and
seals when the river comes up. A pump works whenever you like and costs fuel.
The whole decision is when to spend the fuel."

Point at the refusal panel.

**"And notice what it's saying here. On the 2013 flood, this gate and this pump
make no difference at all — every schedule gives the same answer to two decimal
places. So the panel refuses to show you a comparison. That is slide six in
practice: a refusal is a result."**

→ *"Right. Results."*

---

## Act V — Main results  (5 min)

### 18. Headline  (1:15)
"Here's the main quantitative claim. Four standard safe-learning algorithms,
two independent test problems, twenty repeats each."

"The critical word is *matched*. Normally these baselines get far more
real-world data than we use, and people report the ratio as if it were the
result. It isn't — it's an input. So we capped every method at the same real
sample count and asked what each one achieves."

"We're ahead of all four, on both problems."

**Then point at the dashed line:** "That line is doing nothing at all. On the
right-hand problem, three of the four published baselines cannot be
distinguished from it."

→ *"Second claim: safety."*

### 19. Safety  (1:00)
"Violations cut roughly threefold, and — more useful — a stated rate rather than
a hope."

"But the interesting part is the mechanism. Look at the second bar. A calibrated
safety margin, applied on its own, does *nothing*. Zero of twenty repeats
improve."

**"It only works when it's calibrated against real measurements. If you
calibrate the margin against the model's own opinion of its error, you're asking
the thing that's wrong how wrong it is. You have to calibrate the thing that
actually fails."**

→ *"Now the result we didn't want."*

### 20. The result we did not want  (1:15)
Same pace. Do not hedge.

"On the flood gate and pump, a thermostat beats every learned method we tried.
Including ours."

"The incumbent is a level deadband — pump when the sump is deep, stop when it's
shallow, no model of anything. It wins, on a fraction of the energy."

**And then the sharper half:** "Worse than that. Three of the four published
safe-learning methods finish *outside* the fuel budget — two of them on every
single evaluation run. A method that breaks the constraint isn't a worse
solution. It isn't a solution. Those crosses shouldn't be ranked at all."

→ *"Including ours. So let me show you why ours lost, because the reason is
instructive."*

### 21. Why our planner lost  (1:15)
"It pushed the pump to maximum and stayed there."

Point at panel (a), the red line. "That's how hard the policy is pushing. It
grows without bound — past the point where the policy's own output function
saturates. Beyond that line, every gradient through it is zero. The policy has
stopped being able to respond to anything."

Point at (c). "And this is the budget controller from slide eight, pushing
harder and harder against something that can no longer move. It fought, and it
lost."

**The uncomfortable part:** "The fix was already in our codebase. Written months
earlier, for a different problem, with this exact failure described in the
comment directly above it — and switched off by default. Every flood result we
had ran with it off."

Point at green. "Turn it on and everything behaves. It comes inside the budget,
the controller releases. And it still loses to the thermostat."

→ *"Which raises a fair question: is the thermostat just very good?"*

### 22. Why the thermostat is hard to beat  (1:00)
"Partly. But mostly, there's very little to decide."

"This is every setting of that simple rule, swept across its whole range. It
produces three distinct outcomes. Three. Across the shaded band, you can vary
the cost of the decision by a factor of more than two and the outcome doesn't
change at all."

**The reframe, which is the real contribution here:** "So 'can learning beat the
rule' is the wrong question. The right one is whether learning can reach the
outcomes *between* those three steps — and those steps are enormous. That's a
statement about the problem, and it took sweeping the whole family to see it."

→ *"Let me finish with what we won't claim."*

---

## Act VI — What we will not claim  (3 min)

### 23. The trade  (0:45)
"We do more, and we cross the line more often than the cautious methods do.
Those percentages beside each bar are our violation rates, and we are the least
safe arm wherever this comparison is run."

**"We report it rather than burying it, for a reason that isn't only honesty: a
method that never crosses the line is usually a method that never does anything.
Three of these baselines have a perfect safety record and cannot beat doing
nothing."**

→ *"And some things we published didn't survive."*

### 24. Things that did not survive  (1:00)
Deliver briskly. Three items, no dwelling.

"Joint training of the simulator and planner was a claimed contribution. It
didn't replicate. We replaced the figure set rather than patching it."

"Our explanation module produced briefs that scored well on faithfulness and
carried zero state-specific information — a permutation control caught it."

"And our best digital-twin result shrank to a third of its size when we re-ran
it at seventy-five times the sample."

**The point of the slide:** "Three of those we found by applying our own rule to
ourselves. All three moved a published number down. If the method is any good,
it should cost us something — and it did."

→ *"What's left open."*

### 25. Still open  (0:45)
"Three things. Correcting the model from the *outcome* of an intervention, not
just from observed state — that's the one digital-twin claim we can't yet make,
and it's the dashed red path from slide five."

"A retrospective shadow: replay historical floods using only what was known at
the time, and score the model against what actually happened."

"And whether the blend rule helps on the flood problem now that the policy isn't
pinned at its limit. That's running as we speak."

→ *"One sentence."*

### 26. One sentence  (0:30)
Read it. Don't paraphrase. Then stop talking.

"Components that score well on their own metrics do not add up to a decision
that works. Measure against reality on a schedule, and you can both recover the
lost performance and say how often you will fail."

Pause. "Thank you — happy to take questions."

---

## Handling questions

**Seven reserve slides exist so that "I don't know" is never the answer.** Know
which is which:

| if they ask | go to |
|---|---|
| "Is that benchmark any good?" | **R1** — three of four baselines don't beat doing nothing there at full budget |
| "Does the adaptive blend actually help?" | **R2** — it's a variance effect, not a mean effect, and the matched comparison doesn't separate |
| "What about the evacuation task?" | **R3** — ahead by roughly +500 with a standard error the same size. A direction, not a result |
| "Does conformal prediction hold here?" | **R4** — exchangeability rejected 20 of 20, coverage holds anyway |
| "Can you just tighten the risk level?" | **R5** — no, it costs return and buys no safety |
| "How do you know the gauge data is right?" | **R6** — we didn't, initially. The nearest gauge was past the outlet |
| "Can we see it?" | **R7** — open the portal |

**Three questions worth rehearsing:**

*"Isn't this just model validation?"* — Partly, and the related work says so.
The specific contribution is that the validation is on the **decision**, on a
schedule, and feeds back into training rather than sitting in a report.

*"Why should I believe a negative result from the people proposing the method?"*
— That's the right question, and it's why slides 20, 21 and 24 are in the main
deck rather than an appendix. Three of those we found by applying our own rule
to our own published numbers.

*"Would you deploy this?"* — The twin, yes, as a planning tool, with the
elevation caveat stated. The learned controllers, no — on the gate and pump the
incumbent thermostat is better and we'd say so. The warning-time measurement is
the piece I'd put in front of an emergency manager tomorrow, because it's
measured rather than modelled.
