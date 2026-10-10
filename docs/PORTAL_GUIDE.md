# The Floodplain Planner, end to end

A tour of the PSPE flood portal as it ships: Onion Creek, Austin, Texas. Every
screenshot here is a real session against the live service — no mock-ups, no
composites — and every number in the captions is the number the page printed
on the day it was taken.

The tool answers five questions, in the order a flood actually forces them:


| #   | Question                                              | Where                    |
| --- | ----------------------------------------------------- | ------------------------ |
| 1   | What is happening, and what would a storm do here?    | **Forecast** tab         |
| 2   | Can people get out?                                   | **Roads** tab            |
| 3   | Where would a wall pay for itself?                    | **Plan → Build a plan**  |
| 4   | How high does it have to be, and what does that cost? | **Plan → Cross-section** |
| 5   | Whose floor does the water get over?                  | **Street** view          |


Read it in that order the first time. Each step narrows the question, and each
one costs more computer time than the one before it.

---



## Before you start

Two processes. The solver is a FastAPI service; the portal is a Next.js app
that talks to it.

```bash
# the solver — leave it running
cd ~/pspe
python3 -m uvicorn service.app:app --port 8000

# the portal, in another shell
cd ~/pspe/portal
npm run dev            # http://localhost:3000
```

Check the service is up before you open the page:

```bash
curl -s localhost:8000/health
```

The portal works without it — the shipped study is a static file — but every
button that solves something will fail.

**What loads on first paint.** `public/data/austin/study.json`, 2.9 MB (1.3 MB
gzipped): the solved depth field, the terrain, nine instantaneous frames of the
run, 3,624 OpenStreetMap road features and the quantised grids behind the
click-anywhere readout. Until it arrives the map pane is a pulsing grey block.
There is no retry — if it fails you get a "No study is built" banner.

**Deep links.** `?view=forecast` · `?view=roads` · `?view=plan` ·
`?view=plan&budget=35`. The budget is in US$ millions and is clamped to 2–80.
Note that opening straight into `?view=plan` skips the Forecast panel, so the
bank search is sent with the name `Area` instead of the place name, and the
live gauge feed is never fetched at all — nothing on any tab calls it except
the Forecast tab's Layers panel.

---



## The shape of the screen

![The portal on first load](images/portal-guide/01-forecast-open.jpg)

Left: one scrolling panel, with three tabs at the top — **Forecast**, **Roads**,
**Plan**. Right: the map. Four chips sit over the map's top-left corner —
**Whole valley**, **Town**, **Street**, **Water depth** — and they are not tabs:
they change the camera and the layers, not the question.


| Chip             | What it does                                                                                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Whole valley** | Frames the whole solved box. The default.                                                                                                                                                                             |
| **Town**         | One zoom step in on the same centre — the neighbourhood rather than the catchment.                                                                                                                                    |
| **Street**       | Opens the street view (§5). The left column is replaced.                                                                                                                                                              |
| **Water depth**  | Draws the flood as a **body** of water standing on the ground, and tilts the camera so you can see it as one. Off by default: a flood map is read as a map, north up and flat, and this is the opt-in second reading. |


![Town](images/portal-guide/05-forecast-town.jpg)

*The **Town** chip: the same flood, one zoom step in.*

Note that **Water depth** appears twice, meaning two different things: the chip
over the map builds the solid body, and the switch in the Layers list paints
the flat blue wash on the terrain. The legend's heading is a third.

Top right: `LIVE · 27 gauges · read 11:45`, and **How this works**, which
leads to the research pages (§7). The badge names the time of the reading on
screen, not the publisher's cadence — the panel re-fetches every five minutes
against a feed USGS updates every fifteen.

### Where the detail went

The panels carry the verdict, the numbers and anything that changes what you
would do. Everything else — how a figure was derived, where the data came
from, why a default is what it is — sits behind a small **(i)** next to the
thing it explains. Click it, read it, press Escape.

That is a deliberate trade, and it is measurable: the five views carried 8,251
characters of text and now carry 4,953, with the explanatory share down from
52% to 30%. Nothing that changes a decision was moved. If you want the full
account of what this tool cannot tell you, it is §8 — written out in prose,
not hidden behind anything.

> **"LIVE" now means polled.** The gauge feed re-fetches every five minutes
> against a USGS cadence of fifteen, so the badge, the reading and the
> hydrograph are at most one cycle behind, and the badge states the time of
> the reading it is showing. It used to be fetched once at page load and
> went stale in silence while printing the publisher's cadence as though it
> were its own. The rain radar is still fetched only when you switch it on.

---



## 1 · Forecast — what is happening, and what a storm would do

The default tab. It stacks four things: a live gauge reading, the layer
switches, the roads this flood closes, and the controls to re-solve somewhere
else.

![The Forecast panel](images/portal-guide/01-forecast-open-panel.png)

### The gauge headline

```
observed                                          4:50 PM
River steady
Colorado Rv at Austin · 3.87 m · −0 cm/h · 4.8 m³/s
3.75 m below the action level · 3-day high 4.37 m
Forecast 4.27 m by Fri 11 PM · NWS
```


| Phrase                            | What it is                                                                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `River rising / falling / steady` | Sign of the stage trend at ±1 cm/h                                                                                                                    |
| `3.87 m`                          | Latest USGS instantaneous stage, feet × 0.3048                                                                                                        |
| `−0 cm/h`                         | Latest minus the fifth-latest sample × 100. That is four reporting intervals, which equals an hour only on a 15-minute site                           |
| `4.8 m³/s`                        | Latest discharge, cfs × 0.0283168466                                                                                                                  |
| `3.75 m below the action level`   | Position in the NOAA NWPS category ladder (action / minor / moderate / major)                                                                         |
| `Forecast 4.27 m by Fri 11 PM`    | The NWS forecast **peak** stage and its valid time                                                                                                    |
| the **LIVE · 27 gauges** badge    | Every USGS site in the study box **padded by 0.12°** — about ±13 km, because the gauge that matters is usually upstream and outside the modelled area |


The sparkline to the right of the reading is min–max normalised with no
absolute scale; two gauges' sparklines are not comparable. The chart below it
is the three-day record thinned to roughly hourly, with the published flood
levels drawn as dashed lines where they fit the window — when they do not, it
says so: `action 7.62 m, 3.72 m above now — off this scale`.

**The gauge is the study's own reach.** The feed is ranked by name match
before anything else — the same matcher the Plan tab's flood record uses — so
a study of Onion Creek reports `Onion Ck at US Hwy 183` and not the Colorado
River, which is two orders of magnitude larger and merely shares a city. The
headline and the chart beneath it take the same gauge; they used to pick by
different rules and could describe two different creeks.

### Layers


| Switch           | Default  | What it gates                                                                                                 |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| **Water depth**  | on       | The blue wash painted on the terrain, and the depth legend. Sub-line: `solved at 40 m · 18.71 km² over 10 cm` |
| **River gauges** | on       | The white gauge dots and up to six reading labels on the map                                                  |
| **Rain radar**   | off      | One NOAA base-reflectivity image stretched over the padded box                                                |
| **Road status**  | on       | The red/amber road list in this panel                                                                         |
| **Cameras**      | disabled | Nothing — "no verified public feed wired up yet"                                                              |


![Rain radar over the valley](images/portal-guide/03-forecast-radar.jpg)

Two cautions worth carrying:

- **Rain radar is fetched once.** "every 6 min" is the product's cadence, not
this layer's. The image URL carries no time parameter and the layer only
rebuilds if the bounds change, so what you see is whatever the service
returned when you flipped the switch.
- **The depth legend is linear; the paint is not.** The PNG ramp is
`(d / 8.0) ** 0.42`. Four metres renders about three-quarters of the way
along the bar, not at the middle tick. Read the legend for *deeper/shallower*,
and click the map when you want a number.



### Playing the run

![Hour 4.5 of the run](images/portal-guide/02-forecast-scrubbed.jpg)

The bar across the bottom of the map is the **run scrubber**: nine solved
snapshots, 2.25 model hours apart, 0 → 18 h. Press play and it steps through
them at 650 ms a frame — about five seconds for the whole flood — then returns
to the peak.

- `peak of 18 h · 18.7 km² · wet at any time` — the **envelope**: every cell
that is wet at any point in the run.
- `hour 4.5 of 18 · 17.8 km² · wet at this hour` — one instant.

The envelope is always the larger of the two, and the difference is not a
trend. The shaded block on the little chart ends at `rain stops` (hour 6); the
tick reads `17.8 km² at hour 4.5`, the worst single hour.

**Hour 0 is when the storm starts, not when you opened the page.** Nothing in
this model is anchored to a wall clock.

### Clicking the map

![A depth reading on the map](images/portal-guide/04-forecast-inspect.jpg)

Click anywhere inside the study box. A card appears in the bottom-left corner,
above the scrubber:

![The Here readout](images/portal-guide/04-forecast-inspect-readout.png)

```
Here                                    clear
4.39 m
of water on the ground
ground 153.0 m · 30.1856, -97.7250
```

The depth is read from an 8-bit grid — `byte / 255 × 8.0 m`, so 3.1 cm per
level — and **it is the peak over the whole run**, not the depth at the hour
the scrubber is showing. `dry` means at or under 10 cm, not zero. Outside the
study box the card simply does not appear.

### Which roads this flood closes

At the bottom of the Forecast panel:

```
Roads out of Onion Creek                    car · closed over 25 cm
[ Any water ] [ Car ] [ 4WD / truck ] [ Impassable ]
■ McKinney Falls Parkway        627 cm   Closed
■ East Slaughter Lane           524 cm   Closed
■ South Interstate 35           496 cm   Closed
…
29.7 km of 898 km cut  (i)
```

The four pills are depth thresholds: **0.10 m**, **0.25 m**, **0.45 m**,
**0.80 m**. They are four constants with one-line notes, not vehicle
engineering, and depth is the *only* variable — flow velocity, which is what
actually sweeps a car away, is solved by the model and never used here.

Two things about this list that matter when you quote it:

- The rows are **OSM way segments**, not streets. East William Cannon Drive is
125 separate features; East Slaughter Lane can appear twice with two
different depths.
- The denominator, 898 km, is every LineString in the bundle — residential and
unclassified included — not "roads out of town".



### Re-solving somewhere else

Below the roads: a place search, three driver pills (**River** / **Coastal** /
**Rainfall**), three storm pills (**Heavy rain** 20 mm/h, **Severe storm**
50 mm/h, **Extreme** 80 mm/h, all for 6 hours), and
**Re-run Austin, Texas**.

That button posts a fresh solve: the same 12 km box at 60 m over 12 hours,
about a minute. While it runs you get a seconds counter and a bar whose width
is `elapsed × 1.6%` — a timer, not progress; it parks at 96% after a minute and
stays there.

> **Two things change when you press it.** The map and the scrubber are both
> replaced — the re-run asks for its own nine snapshots, so the timeline plays
> the flood on screen rather than the shipped study's. And the PROJECTION
> badge, the FEMA skill score and the "What this answer is not" fold all
> disappear, because they belong to the shipped study and the new field has
> never been scored against anything.

Also here: **See what the satellite saw**, which runs Sentinel-1 change
detection over the box (a minute is normal) and reports two numbers —
`Observed flooding X km²` and the permanent water it excluded — with the
acquisition date and orbit. The SAR image itself is downloaded and discarded;
only the numbers reach the screen.

### What the shipped flood actually is

At the bottom, with the evidence chip:

```
● PROJECTION
Onion Creek, Austin, Texas
a 300 mm, 6-hour storm — near the 1% annual chance 6-hour depth for Travis
County, solved here on USGS 3DEP bare earth. This is our solver's answer, not
a reference flood.

Overlap with FEMA's 1% floodplain      CSI 0.39
Of it we find                                50%
Of what we call wet, outside it              37%
18.72 km² against 23.69 km² over 116 mapped polygons
```

A uniform design rainstorm — 50 mm/h for six hours over the whole box, all four
edges open — not a replay of a recorded event. It finds about half of what FEMA
maps. Both facts are on the page; neither is buried.

---



## 2 · Roads — can people get out

![Getting out](images/portal-guide/06-roads-open.jpg)

Click **Roads**. Two cards, and they answer two different questions with two
different methods.

### Getting out

```
Getting out                                 6579 road segments
From Onion Creek, Austin, Texas to the nearest dry ground.
[ Any water ] [ Car ] [ 4WD / truck ] [ Impassable ]
Water deeper than 25 cm stops this vehicle

Road out is open                                   2 min
2.6 km · via Nuckols Crossing Road, Vertex Boulevard, Unnamed road
117 of 891 km unusable  (i)
Free-flow speeds, no traffic model.
```

- **Origin** is the centre of the study box, not a place you choose.
- **Destination** is the nearest junction that is dry and at least 2 km away.
The panel tries the ten nearest such junctions in order and reports the first
one it can reach. If all ten fail it says **No way out by road** — which means
"none of ten candidates", not "nowhere". That verdict used to fire on Onion
Creek with 97% of the network open; it is now the honest ten-candidate
statement, and the harness `verify-getting-out.mjs` exists to keep it that way.
- **Time** is `length / free-flow class speed`. Nine hardcoded speeds, motorway
100 km/h down to track 20, no intersections, no signals, no congestion — on a
page about everyone leaving at once.
- **117 of 891 km** counts an edge's whole length if any sample on it is over
the threshold, because a route can only refuse a whole segment. The Forecast
tab's "29.7 km of 898 km" asks how much tarmac is wet and counts only the
sub-segments that are. Same network, same flood, same 25 cm, about 4× apart —
both right for their own question, and each now states its rule in its own
(i) rather than leaving you to discover the discrepancy.

`Getting out` scores roads against the **peak** field, so a road that is dry for
the first three hours is Closed all day. The card below it does not.

### Plan a trip

**From** and **To** are place searches (three characters, 350 ms debounce,
Nominatim through `/api/geocode`). Pick a driver, a storm, optionally type a
discharge, then **Find a way through**. A minute or two.

This one is *timed*: the service solves 13 snapshots over 12 hours and routes
against the water that is on the ground at the moment the driver reaches each
edge. It returns, in order:

- the written brief, in this shape: *decision, drive time and distance, the
roads it goes via, the deepest water on it and how far in, what it avoids and
how deep that is, and the deadline.* It is rendered from the route record and
then re-parsed and checked against it; if any checkable claim disagrees the
brief is withheld and the panel says which claims failed. There is no
language model in this path — `render` and `parse` are declared inverses;
- three tiles: drive time, distance by road, and the delay against a dry day;
- **How long this stays possible** — the last of 25 half-hourly departure slots
from which a route exists;
- **Water on the way** — every crossing over 5 cm, with the depth at the moment
the driver gets there;
- **Why not the direct road** — the crossings the water-blind route would hit;
- **Roads lost as the flood arrives** — one bar per snapshot; click a bar to
paint that instant on the map, and `back to peak` to return.

> **A discharge typed here runs as typed.** The control sends the design flag
> with the number, so Onion Creek's recorded 3,823 m³/s runs as 3,823 whatever
> storm pill is lit. Left *empty*, the estimate from the terrain is multiplied
> by the storm — 1× / 2.5× / 4×. Until recently this card sent the number
> without the flag and silently ran 9,557 m³/s at **Severe storm**; the Plan
> tab never had the bug and this card no longer does.

> **The conformal-margin blocks on this tab need a plan first.** "If the
> recommended walls are built … Stays open, 90% of the time" and "Route that
> still works if the water is deeper than estimated" are computed from the Plan
> tab's solved levee heights. On a fresh visit to Roads they are simply absent.
> Solve a plan (§3) and come back.

---



## 3 · Plan → Build a plan — where a wall would pay

![The Plan tab](images/portal-guide/07-plan-scenario.jpg)

Click **Plan**. It opens in **Build a plan**. Two things happen without a
click: the gauge's flood record is read, and a bank search starts.

### Judged against

![Judged against](images/portal-guide/07-plan-scenario-panel.png)

```
Judged against                       59 y at Onion Ck at US Hwy 183
[ 31 Oct 2013 flood · 3.82 × 10³ m³/s                             ▾ ]
A flood of this size, not that day replayed  (i)
```

The dropdown has three groups:


| Group                             | What it offers                         | Where it comes from                                                                      |
| --------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Floods this reach has carried** | up to 4 recorded peaks                 | USGS NWIS annual peak streamflow, cfs → m³/s                                             |
| **Fitted from the record**        | the 1% and 4% annual chance discharges | log-Pearson III on the same record — log-space moments, Wilson–Hilferty frequency factor |
| **The study's own storm**         | 50 mm/h for 6 h                        | the shipped design event                                                                 |


Changing it throws away the bank list, every placed piece, the solved plan and
the cross-section, and starts the search again.

The line under it is the most important sentence on the tab — **a flood of
this size, not that day replayed.** Open its (i) for why:

> The discharge is the recorded one; what the map draws is this solver's
> response to it. The flow enters at the edge of the box rather than down
> eighty kilometres of catchment, and the model recovers about half of FEMA's
> mapped floodplain.

Two caveats the page does not spell out. The fit is station-skew log-Pearson III
— **not** Bulletin 17C: no regional skew weighting, no Expected Moments
Algorithm, no low-outlier test. And the inflow is held for `storm_hours`
(6 h, inherited from the design storm) and then released.

### The kit

![The kit and the banks](images/portal-guide/08-plan-sites-panel.png)

Five pieces. Click one to arm it, click a bank to place it.


| Piece            | Geometry                              | Rate             | Cost model                                |
| ---------------- | ------------------------------------- | ---------------- | ----------------------------------------- |
| **Earth levee**  | 1.0 m high · 1V:3H batters            | US$15.2/m³ fill  | `length × (3.048·H + 3·H²) × 15.2`        |
| **Raised levee** | 2.0 m high · 1V:3H batters            | US$15.2/m³ fill  | same — so 2 m costs **2.97×** 1 m, not 2× |
| **Flood wall**   | 1.5 m · concrete T-wall               | US$11 k/m/m      | linear in height                          |
| **Floodgate**    | one 1.22 m barrel · drains one way    | US$725 k/outfall | per structure                             |
| **Pump station** | 1 m³/s firm · lifts water from inside | US$800 k/m³/s    | per structure                             |


The panel shows one line — **a 2 m levee costs 2.97× a 1 m one · construction
only** — and its (i) carries the rates: USACE NLSP (Nov 2023, Mar 2023 price
level), escalated to 2025 by CWCCIS and × 0.88 for Texas. The gate and the pump are **real solver terms**,
not line items: the gate is a one-way `max(ΔH, 0)` exchange across the bank, the
pump a 1 m³/s lift with wet-well hysteresis at 0.30 m on / 0.10 m off.

> The prices are construction only. The panel says design, land and contingency
> roughly double the bill (2.1×) — and that factor is printed, never multiplied
> in. The budget bar understates project cost by about half, deliberately and
> in writing.



### The banks

```
6 places a wall would pay
near South Pleasant Valley Road   0.40 km   needs 0.18 m
near Pinehurst Drive South        0.36 km   needs 0.37 m
Onion Creek bank                  0.28 km   needs 0.12 m
…
Height is what the lowest point needs  (i)
```

A site is a contiguous run of **dry** cells that touch wet ones and are within
2 m of being overtopped, found from a full 40 m baseline solve — about 72
seconds. They are drawn on the map as dashed white-cased alignments; a taken
one goes solid.

A piece shorter than the bank needs says so. Put a 1.0 m Earth levee on a
bank reading `needs 1.83 m` and the row reads **`0.83 m short`** in the
warning colour instead of quietly swapping to a cost. It still lets you place
it — a short levee is a legitimate thing to price — but it will not let you
believe it holds.

### Placing and solving

![Two pieces placed](images/portal-guide/09-plan-placed.jpg)

Arm **Earth levee**, click a bank; arm **Raised levee**, click another. The
header counts `2 pieces` and the budget bar fills. **Undo**, **Clear** and
**Save plan** sit at the bottom — Save writes a JSON file locally with the
scenario, the gauge fit, every piece and its cost, and the solved result.

**Solve this plan** needs at least two pieces. It enumerates a library of
candidate plans and solves one bed per plan in a single batched timestep loop
at 60 m over 12 hours — minutes, not seconds.

![What it buys](images/portal-guide/10-plan-impact-panel.png)

```
What it buys
Land flooded        4.0 km² → 4.0 km²     no change
Homes over floor        —  →  —             –
Road cut                —  →  —             –
```

That is the real readout for two levees on Onion Creek under the 2013 peak, and
it is the honest answer, not a loading state. §5.14m in the results log measured
the same thing: on this reach, under this scenario, two 2 m levees moved the
flooded area by -0.05% — slightly *worse*, not better.

- **Land flooded** is box-wide area over 10 cm, at the library's 60 m grid.
- **At least X% less flooding**, where a plan does buy something, is the change
in *mean depth over the protected strip* — the ground landward of your
alignments, a couple of km² — minus the conformal margin. It is a different
quantity over a different denominator from "Land flooded", and the panel
names the footprint underneath it: *"Per cent of mean depth over the ground
landward of 2 alignments, within 420 m of each — 1.8 km²."* A plan can move
one of these two numbers and not the other.
- Below four placed pieces there is no margin at all, and the panel explains
why: split conformal needs nine multi-site combinations before any finite
order statistic carries 90%, and two sites yield two. Where it still cannot
calibrate, it shows the service's own refusal verbatim —
`NOT ATTAINABLE: 7 calibration runs, 9 needed for delta = 0.1; no margin is applied`.
- **Homes over floor** and **Road cut** are dashes because they are not
computed, and the panel says so: "the dash is 'not computed', not
'unchanged'".

> **The grids differ and the panel says so.** The bank search runs at 40 m, the
> plan library at 60 m. The totals in this table are not the area quoted beside
> the map; the *change* is sound because both of its columns come from the same
> grid.



### Before / With plan

![Before](images/portal-guide/11-plan-before.jpg)

Once a plan is solved, a pair of pills appears over the top of the map. They
swap the painted flood between the do-nothing bed and the best solved option —
both from the library's own grid, so the pair on the map is the pair in the
table. When no solved option beats doing nothing, the map says
`no option beat doing nothing — same water` rather than repainting the same
field and letting you conclude the plan did nothing.

That is a real outcome here, not a defect: on this reach, under the design
storm, two 2 m levees made the flooded area very slightly *worse*.

Changing the pieces after a solve retires the readout and brings the solve
button back, so a result can never sit beside a plan that is no longer on the
map.

---



## 4 · Plan → Cross-section — how high, and what it costs

![Candidate lines](images/portal-guide/12-section-lines.jpg)

Click the **Cross-section** pill. Three numbered steps.

### 1 · Choose a line to protect

![Step one](images/portal-guide/12-section-lines-panel.png)

The same banks as the kit, listed by length. Click one and its precomputed
section line loads as ends **A** (town side) and **B** (river side). Or open
**Or draw a line yourself** → **Pick on the map** and click twice.

### 2 · Cut the section

![The section](images/portal-guide/13-section-cut.jpg)

**Cut the section** derives a box around A–B with at least 3 km of pad each
side, solves a 12-hour flood with a 6-hour storm, and returns ground and peak
water every ~30 m along the line. The strip under the map is the result:

```
A · town side        near South Pleasant Valley Road · elevation section        B · river side
                                                                 solved peak 167.08 m
        ╭───╮
   crest 167.38 m
0.80 km · sampled every 31 m from a 50 m model
```

The panel beside it:

```
Crest height    1.89 m        Cost         US$100 k
above the ground it stands on  over 400 m of bank
Above the solved peak  +0.30 m      Homes kept dry  —  not counted
```



### 3 · Move the wall

![Snapped to the cheapest crest](images/portal-guide/14-section-snapped.jpg)

Two ways to move it:

- **Drag the white handle on the section.** Sideways moves the wall along the
line; upwards raises the crest. One drag sets both.
- **Crest elevation** slider — height only, in metres on the DEM datum,
0.05 m steps.

**Put it at the cheapest crest that holds** moves the wall to the highest ground
on the landward side (excluding the outer 8% of the line) and sets the crest to
the highest water on the river side plus 0.3 m of freeboard. In the capture
above it changed nothing, because the hand-placed wall was already there.

The verdict card under the numbers is the part worth reading:

> **Water behind it** — The crest holds, but there is already 0.24 m of water
> behind it — the flood reaches the landward side another way, and a wall does
> not drain what is inside.

Four verdicts: **holds**, **overtops**, **wet behind**, and **in the channel** —
the last reading "a wall here is a weir — drag it onto the bank", with its (i)
explaining that the section cannot tell a bank from a riverbed by elevation.

**What this costing assumes** folds open to the crest-height × length × rate
arithmetic, the service's own note, and the `driver_note` — which is where the
discharge that actually ran is written down.

---



## 5 · Street — whose floor

![The street controls](images/portal-guide/15-street-controls.jpg)

Click the **Street** chip over the map (its accessible name is "Zoom to street
level"). The whole left column is replaced.

![Where the water comes from](images/portal-guide/15-street-controls-panel.png)

```
← Onion Creek
Dixie Dr
Water over the floor, not over the ground outside.

Where the water comes from     [ Down the channel ] [ Rain on the box ]
How big an event               [ As gauged ] [ 2.5x ] [ 4x ]
Discharge coming down the creek  [ estimate from the te ] m³/s
Onion Creek peaked near 3,400 m³/s in 2013. Reproducing that
flood here takes about 2.5× it.  (i)

                      [ Open the street view ]
```

That last line is the one to read before you quote a count off this view. Its
(i) says why:

> At 3,400 m³/s with no multiplier this model puts no house in the block over
> its floor, and 2013 flooded this neighbourhood. The gap is the model's, not
> the creek's: a 14 km box truncates a catchment draining hundreds of square
> kilometres, and nothing here knows about the bridges and culverts that backed
> the water up.

Treat the multiplier as the calibration knob it is.

### Opening it

This is the longest solve in the service: a 14 km parent catchment at 40 m with
13 snapshots over 18 hours, then a separate terrain fetch for the 1.7 km block
at 10 m, then building footprints and address points from the City of Austin.
**Two minutes is normal and cold it runs past the five minutes a browser's**
`fetch` **will wait.**

That limit is in the client, not the solver, and the page is built for it: the
work is keyed by request hash, so pressing **Look again** joins the run already
going instead of starting a second one. If it times out, ask again. Verified
end to end: two requests three seconds apart both reported `pending` at 240 s
and 243 s of one shared solve; the run finished, cached, and the next ask
returned 188 buildings instantly.

### The answer

![A solved street](images/portal-guide/16-street-solved.jpg)

![Dixie Dr](images/portal-guide/16-street-solved-panel.png)

```
Dixie Dr
78 buildings · ground 161.2–168.2 m NAVD88 · solved at 10 m

Deepest on the ground    Over floor level
16.27 m                  57 of 78
anywhere in the block    floor assumed +0.17 m
```

Click any row — or any building on the map — for the per-building card:

![A building](images/portal-guide/17-street-building-panel.png)

```
coa:336505                                          clear
slab on ground, assumed · ground 161.79 m
[ Over floor ]  4.36 m of water inside
Floor 0.17 m above the ground · 4.53 m on the ground outside
```



### Reading the model on the table

The street view is not a flood map. It is a scale model of one block, seen at
about 50°, and everything in it is drawn in one vertical frame:

- **The ground** is a matte stone slab with a visible lip — the solved 10 m
bare-earth terrain, with the satellite photograph left as context around it.
- **Each house** is an extruded box standing on *its own* ground. Bone means
the floor is above the water; salmon means the water is over the floor; the
one you clicked is near-black.
- **The water** is a surface, not a colour: a translucent mesh at the solved
water-surface elevation, cell by cell, so it carries the creek's gradient
and its edge is the shoreline. Deeper water reads darker.
- **Heights are exaggerated ×4**, and the legend says so. At true scale a 6 m
house against 2 m of water is six pixels against two — geometrically honest
and visually nothing. The *plan* is not exaggerated, and ground, houses and
water all go through the same multiplier, so the waterline sits where the
two surfaces really cross.

The legend, bottom-left, carries all four colours and the scale:

```
▢ Floor above water     ▢ Water over floor
▪ The one you clicked   ▬ Water surface, deeper darker
heights ×4 · ground, houses and water all on that scale
floor assumed 0.17 m above ground · block falls 6.6 m
```

Street names are drawn on the slab. The two buttons at the top right turn the
block 45° at a time, because the house you want is sometimes behind another
one.

> **The drawing and the panel use two different grounds, on purpose.** The
> boxes stand on the *solver's* DEM and the water is built from it, so the
> colours and the waterline agree. The panel quotes the *city survey* ground,
> which is the better number for a reader. On this block the two differ by up
> to 2.1 m, which is why the water plane is not built from the response's own
> `water_surface_m` field — that one mixes the two frames.

**Clicking behaves differently in the two places.** A click on the map selects
a house and does not move the camera — you are already looking at it. A click
in the list *navigates*, because the list is a directory sorted by depth and
its top row is often nowhere near the street on screen. On Onion Creek the
worst-hit house stands alone in open floodplain four hundred metres from the
nearest street.

![A building](images/portal-guide/17-street-building.jpg)

Selecting a house also draws a **probe**: a stick from its floor up to the
waterline, captioned with the figure. That is the picture of `+4.36 m over the floor`.

### The clock

![Part-way up](images/portal-guide/18-street-early.jpg)

**When** runs from `peak` (the default) through the thirteen solved snapshots.
Move it and the water surface rises or falls, its edge moves across the block,
the building colours change and the per-building numbers follow. On Onion Creek
the block is dry at hour 0, peaks around hour 3, and is dry again by hour 9.

> **Two things to hold on to.** The floor height is a single assumed constant,
> `FLOOR_ABOVE_GROUND_M = 0.17 m`, applied to every building, because Austin
> does not publish finished floor elevations — and it is not a control. And no
> hydraulics are solved at 10 m: the 40 m catchment's water *surface* is
> sampled onto the fine grid and depth recomputed as surface minus fine ground.
> The 10 m is the ground, not the flow.

---



## 6 · What is slow, and why


| Action                     | Typical     | Hard limit             | What you see                               |
| -------------------------- | ----------- | ---------------------- | ------------------------------------------ |
| First paint                | seconds     | none                   | grey pulsing map                           |
| Gauge feed                 | seconds     | 110 s                  | no headline at all; Layers says `loading…` |
| Re-run with this storm     | ~60–90 s    | 280 s                  | seconds counter + a timer bar              |
| See what the satellite saw | 1–2 min     | 280 s                  | button text only                           |
| Find a way through (trip)  | 1–2 min     | 580 s                  | counter in the button                      |
| Finding banks              | ~72 s       | 580 s, shared          | counter + pulsing bar                      |
| Solve this plan            | minutes     | 280 s per ask, retried | button text only                           |
| Cut the section            | ~1 min      | 580 s                  | counter in the button                      |
| Open the street view       | ~2 min cold | 580 s, shared          | counter in the button                      |


Three endpoints — `/sites`, `/plan`, `/buildings` — share in-flight work: the
first caller starts the solve, every other caller waits on the same event, and
asking again is free. `/analyse`, `/route` and `/section` do not, so two
submissions of the same long request start two solves and each makes the other
slower. Wait rather than retry on those three.

---



## 7 · How this works — the research pages

**How this works**, top right, opens four pages.

### /methodology — the overview

![The overview](images/portal-guide/19-methodology.jpg)

The four-stage loop with an evidence badge on each: Perceive and Simulate are
**VALIDATED MODEL**, Plan and Explain are **PROJECTION**. Three panels below
link out: the shipped Austin study, the Richmond validation case, and the margin
explorer.

The three badges are the portal's central honesty device, and their definitions
live in hover tooltips:

- **OBSERVED** — a satellite or survey recorded this; no model in between.
- **VALIDATED MODEL** — our model, scored against an independent reference and
reported with that score.
- **PROJECTION** — rests on a stated action model. No observational record
contains the counterfactual, so this is not validated.



### /methodology/flood — the validation case

![The archive scene](images/portal-guide/20-methodology-flood.jpg)

The February–March 2022 Northern Rivers flood, a 32.2 km reach of the lower
Richmond River — the one case where the solver was scored against a *reference
solution* rather than a regulatory map, CSI 0.979. Play the 90-frame replay,
tilt the terrain, raise the vertical exaggeration, and click any of the six
candidate levee sites.

The result that earns the page: **three of the six make flooding worse.** Each
of those is downstream or on a drainage path, so a wall there traps water
instead of excluding it. That is why the planner carries a constraint and a
calibrated margin rather than a reward alone.

Read the percentages carefully: +15.37% is 15.37% of an exposure-weighted peak
depth of 0.279 m — about 4 cm — and the absolute figure appears nowhere on the
page.

### /methodology/margin — when a guarantee is possible

![The margin explorer](images/portal-guide/21-methodology-margin.jpg)

```
b + z_δ·σ  <  f · s
```

The feasibility precondition: the calibrated margin has to fit inside the
fraction `f` of the gain the intervention was going to buy. One slider — **f**,
10% to 60% — and the crossing moves with it. At f = 0.30 the crossing is 8.2%
forecast spread: below that a δ = 0.1 margin fits inside the gain; above it the
margin eats more than the intervention was worth.

The second chart is why it fails where it matters: a 10% error in discharge
becomes a several-fold error in depth right at the threshold, because that is
where a structure's effect is most sensitive.

δ is fixed at 0.1 on this page; the only control is f. The testbed terrain is
constructed, not surveyed.

### /methodology/limits — what this cannot show

![The limits](images/portal-guide/22-methodology-limits.jpg)

Eight entries, and two of them are results this project published internally and
then withdrew after measurement contradicted them — including a synthetic result
that overstated achievable span by about 40× and a feasibility verdict that was
wrong for several hours. They are kept because the correction *is* the finding.

---



## 8 · What it will not tell you

Collected in one place, because a tool that cannot say what it is unsure of is
not a decision tool.

**About the flood**

- The shipped field is a **uniform design rainstorm**, not a recorded event, and
it finds about half of FEMA's mapped 1% floodplain (CSI 0.39).
- Manning's *n* is 0.035 everywhere — channel and floodplain alike, no
land-cover field.
- Bankfull discharge, where you do not supply one, is DEM arithmetic: bank level
from the 20th percentile of box elevation, incision clamped to 0.5–6 m, slope
floored.
- A river or coastal boundary is held only for `storm_hours` and then released.
- Culverts, bridges and storm drains are not modelled anywhere.
- Depth is quantised to 3.1 cm and clipped at 8 m for display and for every
click-anywhere reading, even where the solve itself is deeper.

**About the satellite**

- Sentinel-1 flies this box on one orbit with a twelve-day repeat, and Onion
Creek peaks and drains inside a day. **No pass in the record coincides with a
flood peak here**, so the satellite can confirm persistent inundation but
cannot grade a flash flood on this creek.
- The detector refuses when the scene's backscatter histogram carries no
water/land split, rather than reporting a number. Measured: a dry day gave
separability 0.41 and a post-event pass 0.48, against a 0.55 floor. Before
that guard existed it called 71 km² of a 138 km² box permanent water.

**About the roads**

- Depth is the only passability variable. Velocity is solved and unused.
- Travel times are free-flow class speeds with no intersections, signals or
congestion, and every edge is bidirectional — a quoted detour can be an
illegal manoeuvre.
- "Getting out" uses the peak field; "Plan a trip" uses a 13-snapshot schedule.
They can disagree, and that is the methods differing, not the roads.
- The two "km cut" figures use different accounting (see §2).

**About the plan**

- Homes over floor and road cut are not scored for a plan.
- Costs are construction only; the 2.1× all-in factor is printed, not applied.
- The published range behind each rate (earth fill is 4.64–49.9 US$/m³) is in
the code and not on the screen.
- The modelled embankment is floored at one cell wide, so at 60 m it occupies
more ground than the one you priced.
- The conformal band is calibrated on interaction residuals of multi-site
library plans; it is not a guarantee about an arbitrary plan you draw.

**About the buildings**

- One assumed floor height, 0.17 m, for every building in Austin.
- Addresses are a nearest-point join within 25 m, not a match.
- No hydraulics at 10 m — the fine grid carries the coarse solve's water
surface.

**Now behind an (i) rather than on the panel**

These are unchanged in substance; they simply no longer sit between you and a
control. Each is one click away in the place it applies.

- How the scenario discharge becomes a boundary condition, and why a recorded
flood is "a flood of this size" rather than that day replayed.
- The USACE rate provenance, the 2.1× all-in factor, and why an embankment's
cost is quadratic in height.
- How the candidate banks were found, and that they are not chosen by anyone.
- Why the street multiplier needs calibrating, and what the 10 m grid is and
is not.
- The travel-time model, the road-cut accounting, and the trip solve's own
statistics.
- That the river forecast "spread" is the day-to-day scatter of seven daily
forecast maxima, with no ensemble and no hindcast behind it.

**About "live"**

- The model can be driven by the gauge rather than by a storm preset
(`river_q_observed`), and refuses rather than substituting a design storm
when the reach has no instrument reporting.
- A dated flood can be re-run from the gauge record (`/replay`), which
refuses when the record has a hole — Onion Creek's own gauge stopped for
24 hours across the peak of 31 October 2013.
- The gauge feed polls every five minutes; nothing else does. Nothing on the
Roads or Plan tabs fetches it at all, so those tabs show whatever the
Forecast tab last read.
- The rain radar image is fetched once when you switch it on.
- A failed gauge fetch and a genuinely ungauged reach produce the same message.

---



## 9 · Checking it yourself

The portal ships its own harness. With both servers up:

```bash
cd ~/pspe/portal
node scripts/verify-austin-default.mjs    # the shipped study, the chrome, the click readout
node scripts/verify-getting-out.mjs       # the evacuation verdict is true, not just present
node scripts/verify-kit.mjs               # the levee kit end to end
node scripts/verify-section.mjs           # cut, drag, snap
node scripts/verify-section-math.mjs      # the trapezoidal volume and its cost
node scripts/verify-street.mjs            # the street view and the floor frame
node scripts/verify-before-after.mjs      # Before / With plan paint two different floods
node scripts/panel-weight.mjs            # how much text each panel puts on screen
```

and the service's own tests:

```bash
cd ~/pspe && python3 -m pytest tests/ -q
```

Each script prints one `ok` or `FAIL` line per claim and exits non-zero on any
failure. They are written against the *statements on the screen*, not against
internal state — which is why they have caught false headlines that every unit
test passed.

---

*Captured against Onion Creek, Austin, Texas — USGS 3DEP bare earth, FEMA NFHL,
USGS NWIS, NOAA NWPS, OpenStreetMap, City of Austin building footprints and
address points, Esri World Imagery.*