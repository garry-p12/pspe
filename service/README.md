# Flood modelling service

The solver, terrain fetch and forcing. The portal talks to this; it holds no
model code of its own.

```bash
python3 -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

`GET  /health`  — liveness and cache size.

`POST /analyse` — model a bounding box:

```json
{"west": 153.22, "south": -28.86, "east": 153.36, "north": -28.75,
 "name": "Lismore", "dx": 60, "rain_mm_h": 50,
 "storm_hours": 6, "run_hours": 12,
 "levees": [{"lat": -28.80, "lon": 153.28, "height_m": 2.5, "width_m": 600}]}
```

Returns flooded area, peak depth, and base64 PNG overlays for terrain and depth,
plus timings. Results are cached by request hash.

Guards worth knowing about:

* `max_cells` (default 700) refuses a box that would queue a solve of hours.
* A box spanning more than six DEM tiles is refused.
* Ocean reads as nodata in Copernicus and is set to sea level, not left as a
  hole — a hole would act as an infinite sink and silently drain the domain.
* Levee width is in **metres**. Specifying it in cells made a structure's size an
  accident of the grid, which cost a sixfold error before it was caught.
