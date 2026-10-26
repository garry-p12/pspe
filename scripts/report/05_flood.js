module.exports = (H) => {
const { P, H1, H2, H3, BULLET, NUM, CODE, QUOTE, WHY, RULE, table, figure } = H;
return [

H1("5 · The flood twin, component by component"),

P("Onion Creek, south Austin. Real terrain, real road network, real building footprints, 27 live river gauges, and a shallow-water solver. This chapter goes through every component, what it computes, what was chosen, and what each number does and does not mean."),

...figure("docs/deck_bundle/images/slide-01-title-portal-overview.jpg",
  "The flood twin. Left: live gauge, layers, and the roads this flood closes. Right: solved water depth over satellite imagery, with road status coloured by passability.",
  { maxHeight: 420 }),

H2("5.1 Terrain — and the most expensive lesson in the project"),

P("The default global elevation product, Copernicus DEM GLO-30, is a **surface** model. It is the top of whatever is there: canopy, rooftops, everything."),

H3("5.1.1 Why that breaks a flood solver specifically"),

P("A shallow-water solver routes water down gradients of the surface it is given. Feed it a surface model and a stand of trees becomes a twenty-metre watertight block that water flows around, while a clearing becomes a basin that water ponds in. Neither happens in the world."),

...figure("docs/deck_bundle/images/slide-12-elevation-data-lesson.png",
  "The same storm, the same solver, two elevation products. 28.84 km² against 18.71 km² — 54% more flood, from the input data alone.",
  { maxHeight: 300 }),

QUOTE("**Every number in the wrong version was internally consistent.** Nothing in the model's own diagnostics would have flagged it. It took comparing against a different, independent source of ground truth — which is the thesis of this project arriving in a place we did not plan to put it."),

WHY("Why the error is worse than its size suggests", "It is not an offset that could be calibrated away. It is **structured by land cover** — largest exactly where there are trees and buildings, which is where people are. A uniform bias is recoverable; a bias correlated with the thing you care about is not."),

H3("5.1.2 Two further terrain defects"),

P("Defect 31: **DEM nodata filled with sea level.** A 247 m trench ringed every inland domain — 3.6% of cells at 0 m across 84 to 100% of each boundary — draining the entire catchment into it. Caught by a terrain analysis that reported a coast 1,500 km inland."),

P("Defect, §5.14e: **the terrain was 2.2 m too high where the houses are.** Resolved in §5.14f by solving the terrain problem rather than measuring around it, moving to USGS 3DEP bare earth."),

H2("5.2 The solver"),

P("LISFLOOD-FP local-inertial shallow water, the discretisation used as standard in government flood mapping. State is `(h, qx, qy)`: depth at cell centres, and **unit discharge at the x- and y-interfaces**. Adaptive CFL timestep, θ-weighted flux limiter."),

WHY("Why a conservation law rather than a learned model, here", "The flood twin is route (b) of §6.2 — a physics simulator standing in for reality where no counterfactual record exists. That substitution is only defensible if the simulator is a conservation law rather than an empirical fit, if its discretisation is the operational standard, and if it is independent of the surrogate being tested. All three hold. It does not close the counterfactual gap; it is the strongest available version of not closing it."),

H3("5.2.1 Two solver defects, both caught by conservation checks"),

P("Defect 19: an **unlimited explicit scheme manufactured the constrained quantity**. A `clamp(min=0)` on depth was creating water, and the solver gained 6.4% volume over a closed domain. Caught by a closed-domain mass-conservation test, and it matters more than it sounds: a scheme that cannot violate the constrained quantity cannot be used to study violating it."),

P("Defect 32: **river inflow poured in as a volume rather than held as a stage**, producing 112 m of water because the cells could not spread it as fast as it arrived. Caught by a depth with no physical meaning."),

...figure("docs/deck_bundle/images/slide-10-how-the-twin-works.png",
  "Inputs to the flood twin. Bare-earth elevation, rainfall or live radar, and river level; the dashed line is the gauge correcting the model while the storm is still running, which is what makes it a twin rather than a forecast.",
  { maxHeight: 300 }),

H2("5.3 Gauges, and the shadow"),

P("Twenty-seven live stations. The gauge does two jobs: it drives the model as a boundary condition, and it **marks the model's homework** — the shadow comparison scores a solved stage against the observed one."),

H3("5.3.1 Warning time: the strongest negative result in the flood work"),

...figure("docs/deck_bundle/images/slide-14-warning-time.png",
  "Measured warning time from the river gauge's own record, by flood. Not a simulation of gauge failure — the actual record.",
  { maxHeight: 300 }),

P("The structure of the failure is what makes it serious. It is not random downtime. **The two largest floods since the gauge was installed produced no usable record at all**, and they are the two largest *because* the water was high enough and fast enough to compromise the instrument."),

QUOTE("Gauge availability is **negatively correlated with event severity**. The probability of having data is lowest precisely when the data would matter most. Any system treating missing gauge data as missing-at-random inherits that bias and will be most confident when it should be least."),

P("Of the four floods it did capture, three gave 275 to 335 minutes — comfortably above the 120 needed to evacuate. One gave **20 minutes**."),

WHY("What this argues for, and against", "It is an argument for simulation, since the instrument cannot be relied on for the events that matter. It is equally an argument against trusting any single instrument — including the twin. Both halves belong in the sentence."),

H3("5.3.2 Two claims withdrawn here"),

P("§5.14v recorded three claims checked and two withdrawn: **the gauge is at the wrong end** of the reach for the inference being drawn from it, and **the SAR guard was backwards**. Both had been published internally before the check."),

H2("5.4 Validation against FEMA and against satellite"),

H3("5.4.1 FEMA's 1% floodplain"),

table(
  ["Metric", "Value", "What it means"],
  [["CSI", "**0.39**", "Critical success index — counts misses *and* false alarms against you"],
   ["Of FEMA's mapped area, we find", "**50%**", "Recall"],
   ["Of our wet area, outside FEMA's", "**37%**", "False-alarm share"]],
  [40, 16, 44]),

P("18.72 km² against 23.69 km², over 116 mapped polygons, under a uniform 300 mm 6-hour design storm near the 1% annual-chance depth for Travis County, on USGS 3DEP bare earth."),

WHY("Why CSI and not overlap", "Overlap is one-sided. Reported alone, \"we find half of FEMA's area\" sounds like a 50% score; it ignores that 37% of what we call wet is outside their maps. CSI penalises both directions, which is how two individually flattering numbers produce 0.39. Reporting overlap alone would let a reader — or an author — pick whichever direction looked better."),

P("What the disagreement means, stated honestly: FEMA's maps are a regulatory product built with local survey, structures and methods we do not have. Disagreement runs both ways and some of it is certainly ours. **This is a sanity check on the solver, not a validation of it.**"),

H3("5.4.2 Sentinel-1, and the reference that was not the world"),

P("Observed flood extent from same-orbit Sentinel-1 differencing. Our solver matched the published reference at CSI 0.979 — and then the reference itself was scored against the satellite and matched at **0.535**."),

QUOTE("Agreeing with a reference at 0.979 says nothing about agreeing with the world, if the reference agrees with the world at 0.535. Score the reference against observation, not only the model against the reference."),

P("Defect 21 sits inside this: **119 km² of the apparent over-prediction was the sea** and the permanent river channel. Headline CSI moved from 0.378 to 0.535 once permanent water was excluded. It was caught by drawing the map and looking at it — which is worth recording as a method."),

H2("5.5 Roads — can people get out"),

...figure("docs/deck_bundle/images/slide-13-roads-cut.jpg",
  "Road status at the car threshold. Red is cut; the list is ordered by depth at the crossing point.",
  { maxHeight: 420 }),

P("Four depth thresholds stand for vehicle classes: 0.10 m (any water), 0.25 m (car), 0.45 m (4WD/truck), 0.80 m (impassable). At the car threshold this flood cuts **29.7 km of 898 km**."),

H3("5.5.1 Three qualifications, each with a direction"),

P("A caveat without a direction invites a reader to assume the worst. Each of these has a known sign."),

table(
  ["Qualification", "What it changes", "Direction"],
  [["Rows are OSM **way segments**, not streets — East William Cannon Drive is 125 features", "How long the list looks. The kilometres are correct; summing segment lengths is valid either way", "Makes the *list* look longer than the street count"],
   ["The 898 km denominator is **every LineString in the bundle**, residential and unclassified included", "The denominator, not the numerator", "**Understates** the share of evacuation routes cut"],
   ["Passability is thresholded on **depth alone**", "Which roads count as closed", "**Understates** closures — velocity never makes a road safer"]],
  [34, 32, 34]),

QUOTE("Two of the three mean the real disruption is worse than the number on the page. We quote the optimistic end of our own layer and say so."),

WHY("On velocity specifically", "The solver's state already carries unit discharge at every interface, so velocity is computed at every step and discarded for this layer. Using it is a change to the road layer, not to the physics — it is wiring, not research, which is why it sits in future work rather than in limitations."),

H3("5.5.2 Two correct numbers that differ by 4×"),

P("The Roads tab reports **117 of 891 km** unusable; the Forecast tab reports **29.7 of 898 km** cut. Same network, same flood, same 25 cm threshold."),

P("Both are right. The routing view counts an **edge's whole length** if any sample on it exceeds the threshold, because a route can only refuse a whole segment. The forecast view counts only the **wet sub-segments**, because it is asking how much tarmac is under water. They answer different questions, and each now states its rule at the point of use rather than leaving a reader to discover the discrepancy."),

H3("5.5.3 Routing assumptions, in both directions"),

P("From the centre of the study box to the first of the ten nearest dry junctions at least 2 km away that it can reach. Time is length over free-flow class speed."),

BULLET("**No signals, no congestion** — makes evacuation look *easier* than it is, on a page about everybody leaving at once."),
BULLET("**Scored against the peak field** — a road dry for the first three hours is marked closed all day, which makes it look *harder*."),

P("These have not been netted out and we would not claim they cancel."),

H2("5.6 Buildings — whose floor the water reaches"),

...figure("docs/deck_bundle/images/slide-03-street-houses-flooded.jpg",
  "Dixie Drive. Red blocks are buildings with water over the floor; the panel lists each building's depth above its own floor.",
  { maxHeight: 420 }),

P("This is the computation readers ask about most, so it is given in full. The headline is **60 of 78 houses with water above the floor, deepest 4.85 m**. It is four layers stacked, and they have very different reliability."),

H3("5.6.1 Layer 1 — the buildings and their ground: measured"),

P("City of Austin building footprints, 585,916 of them, carrying ground elevation at the structure and roof height from 2012 lidar. Real survey. Where a footprint has no city elevation the DEM is used instead, and every building reports which via a `ground_source` field."),

P("One data trap worth recording: the sentinel for a missing value in that service is a **large negative number rather than null**, so a mean taken straight off the service returns −70,005,326."),

H3("5.6.2 Layer 2 — the water: modelled, in two stages"),

P("A parent catchment is solved at 40 m, then the street block is **downscaled** from it: the water *surface* from the coarse solve, which carries the hydraulic gradient, laid over bare-earth terrain at street resolution. Depth becomes surface minus the fine ground."),

WHY("Why a street cannot be solved directly", "Measured: 50 mm/h for six hours over a 2 km box with open edges put **10 of 576** Austin houses over their floors, because the water that floods a street arrives from upstream and a box that small has no upstream in it — the rain simply drains out of the sides. Three separate blocks sitting entirely inside FEMA's floodplain behaved the same way. The two-stage solve is not an optimisation; it is the only correct way to pose the question."),

H3("5.6.3 Layer 3 — the floor height: assumed"),

...CODE(["FLOOR_ABOVE_GROUND_M = 0.17"]),

P("**This is a guess, one constant for every building, and the code says so in those words.** Austin publishes ground at the structure and not the floor above it; the FEMA elevation certificates that carry the real number are not bulk-public. Texas construction is overwhelmingly slab-on-grade, and 0.17 m is a slab plus its step."),

P("How much that assumption can matter, from our own data elsewhere: across **2,924 surveyed Lismore properties the median floor stands 1.95 m above its own front gate**, with the middle 90% spanning 0.04 m to 3.28 m, because a Queenslander sits on stumps. So 0.17 m is defensible *for Austin specifically* and would be badly wrong almost anywhere else. Every response carries a `floor_source` field saying so, and the caller is expected to display it rather than fold it into a depth."),

H3("5.6.4 Layer 4 — the comparison: a calculation, and a non-obvious one"),

...CODE(["over  =  depth_now  −  FLOOR_ABOVE_GROUND_M"]),

P("That looks too simple, and the reason it is correct is the best detail in this code path. The intuitive form — city floor *elevation* against DEM ground plus depth — is wrong by **far more than the quantity it measures**."),

table(
  ["Quantity", "Size"],
  [["Disagreement between city survey ground and DEM ground, on bare earth", "**2.05 m RMSE**"],
   ["The floor offset being subtracted from it", "0.17 m"],
   ["Ratio", "**12×**"]],
  [66, 34]),

P("Written the intuitive way, a building under 2 mm of water came back as 0.33 m over its floor, purely because the DEM sat half a metre above the city's survey at that point. Since depth is already relative to the solver's own ground, and the floor offset is relative to ground at the building, subtracting them stays **in one vertical frame** and carries no absolute error. The city elevation is kept for display only."),

H3("5.6.5 So what is \"60 of 78\"?"),

P("Strictly, a count of grid cells, with three qualifications a reader needs:"),

BULLET("It is at **peak** — the worst instant of the run, not a simultaneous snapshot."),
BULLET("It is at the **2.5× calibrated discharge**, not the gauged one. At the gauged discharge it is **31 of 78**."),
BULLET("It is **one cell per building**, sampled at the footprint centroid — not averaged over the footprint, so a large building on a gradient gets a single value."),

H3("5.6.6 The calibration multiplier, and what it is standing in for"),

P("The 2.5× factor is calibrated against the 2013 flood, which hit this block hard enough that most of it was subsequently bought out — so the target is an **observed outcome**, not a preference. It stands in for identifiable missing physics: a 14 km box truncating the upstream catchment, and no bridges or culverts."),

P("The important negative: it is **not** absorbing uncertainty in the shape of the rainfall. That was tested separately and the forcing shape moves flooded area by about 6%, so the multiplier is not a free parameter soaking up everything else."),

P("One display defect worth recording: the stated storm multiple was formatted with `:.0f`, so the note read \"2x the forecast peak of 3,000 m³/s\" beside the 7,500 it had actually applied. Defect 38, caught by reading the note beside the number it describes."),

H2("5.7 The plan kit — where a wall would pay"),

P("Five pieces, each with a published unit rate from the USACE National Levee Safety Program cost data, escalated 2023 → 2025 and regionalised to Texas by ×0.88."),

table(
  ["Piece", "Geometry", "Rate", "Cost model"],
  [["Earth levee", "1.0 m high, 1V:3H batters", "US$15.2/m³ fill", "length × (3.048·H + 3·H²) × 15.2"],
   ["Raised levee", "2.0 m high, 1V:3H batters", "US$15.2/m³ fill", "same, so 2 m costs **2.97×** 1 m"],
   ["Flood wall", "1.5 m concrete T-wall", "US$11 k/m/m", "linear in height"],
   ["Floodgate", "one 1.22 m barrel, one-way", "US$725 k/outfall", "per structure"],
   ["Pump station", "1 m³/s firm lift", "US$800 k/m³/s", "per structure"]],
  [16, 28, 20, 36]),

WHY("Why the 2.97× matters more than it looks", "Fill volume grows with the square of height because the batters widen as the levee rises, so doubling a levee nearly triples it. That non-linearity is why \"build it a bit higher everywhere\" is usually the wrong plan, and it is the entire reason the placement question is worth asking."),

P("The 2.1× multiplier for design, land and contingency is **printed and never applied**, so every figure on screen is bare construction and is an underestimate of what a city would actually spend."),

H3("5.7.1 Three defects in the planning path"),

P("Defect 20: **levee size specified in cells, not metres.** The same site looked six times less effective at a different grid resolution. Caught by re-deriving the physical width at two resolutions; the rule that followed is that an intervention's physical dimensions are specified in physical units."),

P("Defect 28: **the surrogate was discontinuous between one measure and two.** Adding a levee that helps *lowered* the prediction, and the planner responded by refusing to spend half of every budget. Caught by building a planner that asks \"what if I add one more?\""),

P("Defect 37: **the headline reduction was read from a surrogate extrapolated at the point the greedy search chose** — 111.7% displayed against a solved 99.81%, past the physical maximum. The margin was unattainable there so nothing caught it. Found by solving the recommended plan in order to answer a question about a road."),

P("§5.14m records the levee that was **a round hill fighting the wrong flood**, and §5.14d a levee priced off a cross-section with a dam that was nearly shipped."),

H2("5.8 Cross-section, and the benefit column that stays empty"),

P("Cut a section anywhere and the tool sizes the crest: 1.89 m above its own ground, +0.30 m above the solved peak, US$100 k over 400 m of bank."),

P("**Homes kept dry is a dash.** It is not counted, because the benefit side needs a depth-damage curve that has not been validated here. Reporting a cost without a benefit is incomplete; inventing a benefit would be worse."),

H2("5.9 Solving anywhere"),

...figure("docs/deck_bundle/images/slide-11-solved-anywhere.jpg",
  "Two place names in: the twin derives a bounding box, picks a resolution, fetches terrain, solves, pulls the road network, builds a routable graph and routes. About 104 s end to end.",
  { maxHeight: 420 }),

P("Junction extraction is verified at **287.4 km against a published 288.0 km**. Defect 29 is why this matters: plan, margin and attribution had been keyed to a **precomputed library**, so the framework worked in one valley and everywhere else was a bare solve. Caught by entering another location and finding only a levee slider."),

...figure("docs/figures/paper/fig10_flood_precondition.png",
  "Where a guaranteed margin fits inside the benefit a levee buys. Real terrain is 2.3x more demanding than the synthetic testbed, and operational forecast skill does not reach the threshold.",
  { maxHeight: 340 }),

H3("5.9.1 The precondition, measured before claiming usefulness"),

P("A guaranteed margin fits inside the benefit a levee buys only below about **3.6% rainfall forecast spread** on real Australian terrain — 8.2% on the synthetic testbed, so real terrain is **2.3× more demanding**."),

QUOTE("Operational precipitation forecasting does not achieve 3.6% at 48-hour lead. **On this task, with today's forecast skill, a δ = 0.1 margin does not fit.** We report that rather than the synthetic figure."),

];
};
