# Floodplain Planner

A flood mitigation planning tool. It answers three questions a council asks:

* **Is anything coming this week?** — live rainfall and river forecast for the district.
* **What gets cut, and where?** — flood extent over real terrain, with the road
  network and a depth threshold you set by vehicle type.
* **What should we build, and does it work?** — mitigation options with cost,
  road-kilometres kept open, and a warning where an option moves risk instead of
  removing it.

It works for the Richmond Valley from a prepared dataset, and for **anywhere on
Earth** by fetching terrain on demand and solving in about a minute.

## Running it

Two processes. The portal is a client; the physics lives in the service.

```bash
# 1. the modelling service (solver, terrain fetch, forcing)
cd ..                       # repo root
python3 -m uvicorn service.app:app --host 127.0.0.1 --port 8000

# 2. the portal
cd portal
npm run build && npx next start -p 3100
```

Then open <http://localhost:3100>.

`npm run dev` works for development, but use the production build for anything
you are showing: the terrain mesh is noticeably slower to first paint in dev.

**Restart the portal after every build.** `next start` serves the build it was
launched with; rebuilding underneath a running server leaves it 500-ing on the
stylesheet, which renders as an unstyled page.

## Where the numbers come from

| layer | source | key needed |
|---|---|---|
| Terrain, anywhere | Copernicus DEM GLO-30 (public S3, COG range reads) | no |
| Terrain, Richmond | FloodCastBench (Xu et al., Sci Data 2025) | no |
| Rainfall + river forecast | Open-Meteo / GloFAS | no |
| Roads | OpenStreetMap via Overpass | no |
| Aerial imagery | Esri World Imagery | no |
| Place search | Nominatim | no |

Nothing here requires an account.

## What the service does

`service/app.py` exposes `POST /analyse`: given a bounding box, it fetches
Copernicus terrain, reprojects to the local UTM zone, applies any levees, and
solves the 2-D shallow water equations (LISFLOOD-FP local-inertial scheme). A
13 km town at 60 m over a 12 h storm takes about 35 seconds end to end, which is
why the portal can answer on demand rather than queueing.

## Honesty in the interface

Numbers are labelled by what stands behind them:

* Recorded inundation and terrain are **observations**.
* Our solver against the published reference scores **CSI 0.979** — it is a
  validated model, and that is what "verified" means here.
* **Every intervention effect is a projection.** No observational record contains
  the counterfactual: nobody built the levee, so nobody recorded what it would
  have done. The tool says so wherever it shows one.
* Fast estimates for combinations are shown as a **range**, never a point, and a
  plan that is adopted should be confirmed by a full solver run.

"How this works" in the header opens the methodology, including results that were
measured, withdrawn and corrected.
