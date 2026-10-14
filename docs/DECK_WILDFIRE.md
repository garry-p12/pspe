# Wildfire act — slide pointers

**An act, not a deck.** These slides slot into the main presentation
(`DECK_POINTERS.md`) as the second application, after the flood twin and
before the decision gyms. The ordering prompt in `DECK_REORDER_PROMPT.md`
places them.

**Why the act exists.** The flood twin shows the method working on one hazard.
A single application cannot distinguish "the method works" from "this hazard
suited us". Wildfire is the second instance: different physics, different data,
different failure modes, same four stages. It is also where the method's
strongest quantitative result on observed data lives.

Format matches the main deck:

- **SAY** — the headline.
- **ON SLIDE** — the body bullets. Two or three, short.
- **SHOW** — the image.
- **IF ASKED** — speaker notes only, never on a slide.
- **NOTES** — what to actually say, with the beat that must land in bold.

Images: `docs/images/portal-guide/3*-fire-*.jpg` (portal) and
`docs/figures/paper/*.png` (measured figures).

---

### W1. The same method, a different hazard
- **SAY:** If it only worked on water, it would be a flood tool.
- **ON SLIDE:**
  - Same four stages: observe, simulate, plan, calibrate
  - Different physics, different satellites, different ways of being wrong
  - This is where the method's strongest result on real data lives
- **NOTES (0:40):** "Everything so far has been one creek. One application
  can't tell you whether the method works or whether that hazard happened to
  suit it." Pause. **"So here is the same machinery pointed at fire — and it is
  where our best number on observed data comes from."**
  → *"Start where any fire decision starts: what is actually alight."*

### W2. What is burning, right now, anywhere
- **SAY:** This half is live, and it works anywhere on Earth.
- **ON SLIDE:**
  - Two NASA satellites report what is alight as they pass overhead
  - A day they did not cover is marked **no pass** — not drawn as an empty map
  - "We did not look" and "nothing burned" are different facts
- **SHOW:** `docs/images/portal-guide/32-fire-observe.jpg`
- **NOTES (1:00):** Point at the seven frames, then at the grey one.
  "Seven days over northern California, pulled live. Six of them the satellites
  covered. One they did not."
  **"That grey panel is the whole design philosophy in one tile. It would have
  been easier to draw an empty map — and an empty map says 'nothing burned',
  which we do not know."**
  "The same distinction cost us a real number on the flood side: the river
  gauge was silent on the two largest floods, and a system that treats silence
  as zero would have reported those as quiet days."
  → *"Knowing what is alight is not knowing where it goes."*

### W3. Where it goes tomorrow — and was it right?
- **SAY:** The model is shown today and never tomorrow. Then we check it against tomorrow.
- **ON SLIDE:**
  - It found 99 of the 406 places that burned, and called 94 that did not
  - Fire covers about 1% of the map, so "no fire anywhere" scores 99%
  - That is why hits and misses are counted separately, not scored as accuracy
- **SHOW:** `docs/images/portal-guide/30-fire-forecast.jpg`
- **IF ASKED:** AUC-PR 0.3157 on the held-out split, reproducing the recorded
  0.3162 ± 0.017. Above the dataset's originating baseline (0.284), below the
  current state of the art (0.3790).
- **NOTES (1:15):** "Three panels: what was burning, what actually burned the
  next day, and what the model expected. The model saw only the first."
  "It caught about a quarter of tomorrow's fire."
  **"A quarter sounds poor until you notice the alternative. Fire is one
  percent of this map. A forecaster that says 'no fire anywhere' is right
  ninety-nine percent of the time and useless — which is why the panel counts
  what it found and what it missed instead of printing a score."**
  → *"And the portal says plainly how much to trust it."*

### W4. How much should you trust the forecaster?
- **SAY:** Somewhat. And the page says so.
- **ON SLIDE:**
  - Better than the forecaster published with this dataset
  - **Not as good** as the best one anyone has built
  - Forecast skill is an input to the plan, not the contribution
- **NOTES (0:45):** "We did not set out to win at forecasting, and we did not.
  A better spread model would make every plan in the next slide better without
  changing a word of what we claim."
  **"The reason this is on a slide rather than in a footnote: a portal that
  shows you a forecast without telling you how good the forecaster is invites
  you to assume it is good."**
  → *"So — given a forecast of that quality, what do you do with it?"*

### W5. Where the crews go
- **SAY:** Crews can clear only so much ground in a day. The decision is *where*.
- **ON SLIDE:**
  - 42% less burns than leaving it alone — but that is the easy comparison
  - **33% better than sending the same crews out evenly**
  - Blue is cleared ground, orange is what is alight today
- **SHOW:** `docs/images/portal-guide/31-fire-plan.jpg`
- **NOTES (1:15):** "Here is the plan the optimiser produces, inside a three
  percent daily budget. Blue squares are where the crews go."
  "Against doing nothing, forty-two percent less burns. That is the number
  people quote and it is the one I trust least."
  **"Beating 'do nothing' only proves that clearing fuel works at all. The
  comparison that matters is against spreading the same crews out evenly — and
  there it is thirty-three percent better. Same crews, same budget, chosen
  places."**
  → *"That is the portal. Here is the measured result behind it."*

### W6. Against operational practice, on real fire records
- **SAY:** Twelve points better than the operational rule, on fires that actually happened.
- **ON SLIDE:**
  - Planned through a learned spread model on real records, not a simulator
  - Population-weighted burned area, matched crew budget
  - Paired across fires: **t = 11.15**
- **SHOW:** `docs/figures/paper/fig4_wildfire.png`
- **NOTES (1:00):** "This is the strongest quantitative result in the project
  on observed data. Firebreak placement through the learned spread model,
  against the forecast-then-treat heuristic crews actually use, at the same
  crew budget. Twelve points of burned-area reduction, paired t of eleven."
  → *"But the number I want you to look at is not that one."*

### W7. The advantage appears exactly where the theory says it should
- **SAY:** At one day ahead, the gain is zero. That is the result.
- **ON SLIDE:**
  - One day ahead: **+0.6 points, not significant**
  - Three days: +12.0 · Five days: +14.3
  - At one day a greedy allocation is already optimal — so there is nothing to win
- **NOTES (1:15):** "Run the same comparison at different planning horizons.
  One day ahead, the advantage vanishes — six tenths of a point, not
  significant."
  Pause. **"That is not a weakness, it is the strongest evidence in the deck
  that the planner is doing what we say. At a one-day horizon a greedy
  allocation is provably optimal, so there is nothing for a planner to find.
  The theory requires the gap to vanish there, and it does. A method that
  showed a gain at one day would be telling us something was wrong."**
  "Then it opens up as the decision becomes genuinely sequential: twelve points
  at three days, fourteen at five."
  → *"One thing we cannot do, and should say out loud."*

### W8. What no record can tell us
- **SAY:** No fire in any archive had a firebreak cut because we said so.
- **ON SLIDE:**
  - Every plan number is what the spread model predicts, not what happened
  - Simulator studies can evaluate the counterfactual exactly; we cannot
  - We buy realism and lose the counterfactual — a trade, not a win
- **NOTES (1:00):** "Everything on the planning slides is in-model. The burn
  figures are what the spread model predicts under each plan."
  **"No fire on record ever had a break cut on our instruction, so nobody can
  observe what would really have happened. Work built on fire simulators can
  evaluate that counterfactual exactly, because the simulator defines the
  intervention — at the price of a simulator gap. We fit to what real fires
  did, and pay the opposite price."**
  "That is a trade-off, stated as one. It is also the one claim the digital
  twin cannot close, and it is the dashed path on the architecture slide."
  → *"Two hazards, same method. Now — what happens when you ask either one to
  make a decision repeatedly?"*

---

## Figure captions

**`32-fire-observe`** — Seven days of NASA FIRMS active-fire detections over
northern California, pulled live. Each frame is the fire burning at satellite
overpass, not the cumulative burn scar. The day with no satellite coverage is
marked *no pass* rather than rendered as an empty map.

**`30-fire-forecast`** — A held-out fire day. Left: what was alight. Centre:
what burned the following day. Right: what the model expected, with colour
carrying its confidence. The model is given the first panel and eleven
environmental drivers, and never sees the second.

**`31-fire-plan`** — Firebreak intensities optimised through the spread model
under a 3%-per-day crew budget. Blue is treated ground, orange is what is
alight. Scored against both doing nothing and spreading the same crew-days
uniformly.

**`fig4_wildfire`** — Firebreak planning on observed fire records against the
operational forecast-then-treat heuristic, at matched crew budget, with the
horizon ablation. The advantage is absent at a one-day horizon, where a greedy
allocation is already optimal, and grows as the decision becomes sequential.

---

## What this act is for, in one line

The flood twin shows the method operating. **The wildfire twin shows it
generalising, and carries the horizon ablation — the one result that
demonstrates the planner exploits the structure it claims to exploit rather
than a tuning artefact.**
