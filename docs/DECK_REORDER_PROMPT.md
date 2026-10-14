# Prompt — reorder the PSPE deck into one story

Paste everything below the line into the slide tool, together with the current
deck, `DECK_POINTERS.md`, `DECK_WILDFIRE.md`, `SPEAKER_NOTES.md` and the
`images/` folder.

---

## What you are doing

You are given a 44-slide deck that is **correct but not sequenced**. Every
slide is accurate; the order does not tell a story. Reorder it, merge what is
redundant, cut what is duplicated, and insert a new wildfire act. Do not invent
content: everything you need is in the supplied outlines.

## What is wrong now, specifically

Fix these. They are the reason the deck feels like a list rather than an
argument.

1. **Eleven slides of mathematics sit between the problem and the method.**
   Slides 5–15 present the PDE, the objective, the Fourier operator, the two
   gradients, the dual, the failure, the fix, the closed form, the twin loop
   and the explanation score — *before* the audience has been told what the
   method is or what it is for. Slide 16 then says "check the system against
   the world, that is the whole thesis", which is the sentence the maths was
   supposed to be explaining. **The claim must precede its formalism.**

2. **The thesis is stated on slide 4 and abandoned until slide 16.** Slide 4
   says components grade their own homework. Slide 10 is that statement made
   precise. Six slides of setup sit between them, so the payoff lands after the
   audience has let go of the question.

3. **Two slides are in the wrong act.** Slide 14 (the twin loop, satellite
   sync) is wildfire material sitting in the mathematics act. Slide 15 (scoring
   the explanation, faithfulness withdrawn) states a withdrawn claim that is
   stated *again* at slide 35. **Say it once, in the honesty act.**

4. **A plant and its payoff are thirteen slides apart.** Slide 19 ends "remember
   this slide" about the budget controller. The payoff is slide 32. Across two
   act boundaries, the audience has forgotten.

5. **The flood thread is cut twice.** The flood twin runs 20–25, then the
   decision gym interrupts (26–28), then PDE testbed results (29–30), and only
   then the flood results (31–33). Flood results belong with the flood.

6. **One act contains two unrelated kinds of result.** "Main results" holds
   both the method's validation on PDE testbeds (29–30) and a negative result
   about a flood control problem (31–33). These answer different questions and
   should not sit under one heading.

7. **Slide 34 reuses slide 29's figure** for a different claim, five slides
   later, with no callback language.

8. **Slides 2 and 20 use the same screenshot** with near-identical framing.

9. **Several slides are text-only grey boxes** (19, 27, 35, 36, R3–R6) in a deck
   that is otherwise figure-led. After a run of figures they read as missing
   content.

10. **There is no wildfire, and the wildfire portal is now live.** A single
    application cannot separate "the method works" from "that hazard suited
    us".

## The order to build

Nine acts. Slide numbers in brackets refer to the **current** deck; `W*` refers
to `DECK_WILDFIRE.md`; "NEW" means build it from the outline text supplied.

**ACT I — One creek, one decision** *(~4 min)*
1. Title [1]
2. A storm is coming, someone has to decide [2]
3. Sixty of seventy-eight houses have water above the floor [3]
4. Every component grades its own homework [4]

**ACT II — The trap, made precise** *(~4 min)*
Keep only the mathematics that makes slide 4 true. The rest moves or goes.
5. The dual sees one number, reality is another [10] — *move up, directly after
   the thesis it formalises*
6. Why the obvious fix fails [11]
7. A testable prediction, matched to 0.003 [13]

**ACT III — What PSPE is** *(~4 min)*
8. Check the system against the world, on a schedule [16]
9. Three things, none of them better accuracy [17]
10. Two ways to get a gradient, mixed [18 + 8 merged] — *the concept slide and
    the α\* equation are the same idea twice; merge into one*
11. The fix: calibrate on the real overshoot [12]
12. There is always a budget [19] — *end with the plant, and make it explicit:
    "this controller has one failure mode and we hit it in Act VII"*

**ACT IV — Does the fix work** *(~3 min)*
13. Same data, we beat all four [29]
14. Violations cut threefold, and the margin alone does nothing [30]

**ACT V — Twin one: flood** *(~5 min)*
15. A live digital twin of Onion Creek [20] — *reuse of the slide 2 screenshot
    is fine here; it is a deliberate return, so say "the same picture, now
    with the method behind it"*
16. Three inputs, one depth field [21]
17. Not a pre-rendered scenario [22]
18. Fifty-four percent more flood from the elevation data alone [23]
19. Which roads are cut is a routing question [24]
20. Gauges fail in the floods worth warning for [25]

**ACT VI — Twin two: wildfire** *(~7 min)* — **NEW, from `DECK_WILDFIRE.md`**
21. The same method, a different hazard [W1]
22. What is burning, right now, anywhere [W2]
23. Where it goes tomorrow, and was it right [W3]
24. How much should you trust the forecaster [W4]
25. Where the crews go [W5]
26. Twelve points better than operational practice [W6]
27. The advantage appears where the theory says it should [W7]
28. What no record can tell us [W8]

*The twin loop currently at [14] belongs here, as satellite synchronisation
material, if you want a ninth wildfire slide. Place it between W2 and W3.*

**ACT VII — From twins to decisions** *(~5 min)*
29. A twin tells you what will happen, not what to do [26]
30. The routing is real [27] — *needs a visual; use the Roads screenshot or
    merge into 29*
31. The entire decision is when to spend the fuel [28]
32. A thermostat beats every learned method, including ours [31]
33. Our planner pushed the pump to maximum and stayed there [32] — ***this is
    the payoff to Act III slide 12. Open it with "remember that controller".***
34. Sweep the simple rule and you get three outcomes [33]

**ACT VIII — What we will not claim** *(~3 min)*
35. We cross the line more often than the cautious methods [34] — *callback, not
    a new figure: say "the same chart, now look at the percentages"*
36. Three claims we moved down ourselves [35] — *this is where the withdrawn
    faithfulness claim is stated, once*
37. What is still open [36]

**ACT IX — Close** *(~1 min)*
38. One sentence [37]

**BACKUP** — R1–R7 unchanged [38–44], plus the removed mathematics as R8–R12.

## What to cut, merge or demote

- **Demote to backup, do not delete:** the PDE definition [5], the constrained
  objective [6], the Fourier operator [7], the budget-keeper detail [9], the
  explanation score [15]. These are correct and a technical audience may ask;
  they are not the argument. Label them R8–R12.
- **Merge:** [8] into [18] — the gradient concept and its equation are one idea.
- **Move:** [14] into Act VI.
- **State once:** the withdrawn faithfulness claim. It is currently on [15] and
  [35]. Keep [35].

## Rules that do not change

- **Do not invent, round, or clean up any number.** Every figure was generated
  from measured data. If a number is not in an outline or visible in a figure,
  it does not go on a slide.
- **Do not add a summary slide claiming the method wins everywhere.** Acts VII
  and VIII deliberately contain results where it loses. Those stay, at the same
  size and prominence as the wins.
- **SAY** is the headline. **ON SLIDE** bullets go on the slide, two or three,
  as bullets. **IF ASKED** is speaker notes only, never on a slide.
- Put the matching **NOTES** entry in each slide's speaker-notes field.
- Never more than three bullets on a slide.
- No clip art, stock photography or decorative gradients. Every visual is
  supplied and is either a measured figure or a screenshot of the real system.
- Minimum 20pt body, 28pt+ headlines, 16:9, one restrained template.

## The three transitions that carry the story

If nothing else survives editing, these must:

1. **Act II → III.** "So the margin has to grow with something the model cannot
   see. That is the whole problem, and it has a solution."
2. **Act V → VI.** "Everything so far has been one creek. One application
   cannot tell you whether the method works or whether that hazard suited it."
3. **Act III → VII (the plant and its payoff).** "Remember that controller" on
   slide 33, referring back to slide 12. Without this line the deck's best
   self-criticism lands as an unrelated bug report.

## Length

38 main slides, about 36 minutes at the pacing in `SPEAKER_NOTES.md`, plus
backup. If you must cut to 30 minutes, cut Act VII to slides 29, 32 and 33,
and cut Act V slide 17. Do not cut Act VIII.
