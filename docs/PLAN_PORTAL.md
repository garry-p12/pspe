# Plan: a digital-twin portal for PSPE (wildfire + flood)

Draft for review, 2026-09-26. **Nothing built yet.**

## 0. The constraint that shapes everything else

A demo portal is exactly where careful work gets overclaimed. This session has
spent most of its effort narrowing claims — withdrawing a span result, re-scoping
the synthetic flood numbers as *fluvial*, tracing a dry bias to our own boundary
condition. A slick interface reading "PSPE cuts flood damage 18%" would undo all
of it in one screenshot.

So the portal's central design feature is an **evidence badge** carried by every
panel, with three states:

| badge | meaning | example |
|---|---|---|
| **OBSERVED** | satellite or survey record | FIRMS detections; FloodCastBench reference depths |
| **VALIDATED MODEL** | our model, scored against an independent reference | our solver at **CSI 0.979**; NDWS forecast vs held-out truth |
| **PROJECTION** | rests on a stated action model, *not* validated | every intervention effect (§6.2) |

Nothing in the UI shows an intervention outcome without the PROJECTION badge and
a one-line statement of what is assumed. That is not a disclaimer bolted on; it is
what a real decision-support tool for public safety would have to do, and it makes
the portal a more honest artefact than most research demos.

---

## 1. Stack

| layer | choice | why |
|---|---|---|
| framework | **Next.js 15**, App Router, TypeScript | asked for; static export means it hosts anywhere |
| map | **MapLibre GL JS** | open source, **no API key**, unlike Mapbox |
| geo 3D | **deck.gl** (`TerrainLayer`, `BitmapLayer`, `ScatterplotLayer`) | purpose-built for exactly this; composes with MapLibre |
| charts | **visx** or Recharts | the margin/precondition plots need custom axes |
| UI | Tailwind + shadcn/ui | professional chrome without hand-rolling components |
| state | Zustand | timeline scrubbing and layer toggles across panels |
| deploy | `output: 'export'` → static | no server; works on Vercel, GitHub Pages or a TACC web dir |

**Prerequisite: Node is not installed on this machine** (`node`, `npm`, `npx` all
absent). Installing it via Homebrew or nvm is step zero and needs your go-ahead,
since it touches your system outside the repo.

---

## 2. Data pipeline — the real work

Raw assets are far too large to ship: the Australia event alone is 2,881 frames at
1073², and the archive is 19 GB on `$WORK`. A precompute step on Vista produces a
small web bundle.

| asset | source | web form | size |
|---|---|---|---|
| Australia DEM | `Study regions/Australia_DEM.tif` | 16-bit PNG heightmap, 1073² | ~1 MB |
| flood depth sequence | 2,881 frames | ~90 frames, 268², packed into PNG sprite sheets | ~4 MB |
| levee site screen | `runs/flood_sites/sites.json` | 933-site controllability grid → JSON + PNG overlay | ~200 KB |
| solver validation | `runs/floodcast_validate/validate.json` | CSI/RMSE curves | ~20 KB |
| wildfire sequences | FIRMS / NDWS patches | 64² masks, PNG sprites + GeoJSON | ~2 MB |
| all metrics | `runs/**/*.json` | one consolidated `results.json` | ~100 KB |

Total ≈ **10 MB**, which loads fast and can live in the repo.

**One open question to resolve before map work:** the depth rasters carry **no
georeferencing** (§5.6). The DEM has a UTM tiepoint, so the terrain can be placed
on a real basemap — but I must read the DEM's CRS and confirm the event's actual
location before drawing anything on a world map. If the CRS cannot be resolved,
the flood view falls back to a local 3D coordinate frame (still compelling, just
not a slippy map). **I will not label a location I have not verified.**

---

## 3. What the portal contains

### Page 1 — The loop
The four modules as a live diagram: Perceive → Simulate → Plan → Explain, each
clickable through to its evidence. Plain statement of scope: the *sync* half is
validated on real data, the *act* half rests on an action model.

### Page 2 — Flood twin (Australia 2022) — **the centrepiece**
* 3D terrain from the real DEM, water rising over 10 days on a scrubber
* **Our solver vs the reference, side by side**, with the live CSI badge (0.979)
* **Levee placement:** click perimeter sectors, re-run a precomputed plan, watch
  the flood redirect
* **The backwater demo:** two of six sites make flooding *worse* (−18.5%). The
  user will place one expecting to help. That is the most valuable thing the
  portal can teach, and it is a real measured result
* **The 933-site screen** as a heatmap: where a levee can help at all — and the
  3× swing between sites a kilometre apart

### Page 3 — Wildfire twin
* FIRMS fire sequences on a map, observed vs forecast
* **Occlusion slider:** hide part of the state, compare beliefs (blind / persist /
  perceive), and — the point — show the *decision* consequence, not just
  reconstruction error
* Firebreak allocation under budget

### Page 4 — The safety margin
* Interactive δ → margin → violation rate
* The precondition `σ/s < f/z_δ` as a live feasibility check with the crossing at
  8.2%
* The amplification curve (40× at the crest)

### Page 5 — Limits
Not a footnote. The counterfactual gap (§6.2), the fluvial-vs-pluvial scoping, the
wall-clock caveats, and what would be needed to close each.

---

## 4. Scope options

| option | what | effort |
|---|---|---|
| **A — Flood centrepiece** | Pages 1, 2, 5 only. The 3D flood twin, levee demo, backwater finding, evidence badges | **2–3 days** |
| **B — Full portal** | All five pages, both hazards, margin explorer | **5–7 days** |
| **C — Narrative deck** | Static site wrapping the 11 existing figures with text. No interactivity | **~1 day** |

**My recommendation: A, then extend to B if it earns its keep.** The flood page is
where the tangible digital-twin story lives — real terrain, real water, a decision
that visibly changes an outcome, and a trap that teaches the backwater lesson.
Wildfire adds breadth but not a new kind of claim.

---

## 5. Risks

* **Node install** touches your system; needs approval.
* **19 GiB free** locally. `node_modules` is ~500 MB — fine, but not roomy.
* **Georeferencing may not resolve**, in which case the map becomes a 3D scene.
  Stated above; I will not fake a location.
* **Phase 3 is still running.** Portal work is CPU-light and does not consume SUs,
  so it genuinely runs in parallel — but my attention is not parallel, and the
  allocation expires **2026-09-30**. If a science decision arrives, it preempts.
* **Scope creep toward prettiness.** The evidence badges and the backwater demo
  are the substance; animation polish is not.

---

## 6. What I need decided

1. **Scope: A, B or C?** (I recommend A.)
2. **May I install Node** via Homebrew/nvm on this machine?
3. **Where should it live** — `pspe/portal/` inside the repo, or a separate one?
4. **Is it for you, or to be shown?** A portal for collaborators can stay rough;
   one for a committee, funder or ICML reviewer needs the chrome, and that
   changes the effort estimate more than any feature does.
