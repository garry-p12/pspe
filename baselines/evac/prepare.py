"""Build the two inputs the evacuation experiments stand on.

    python3 -m baselines.evac.prepare

Writes `roads.json` and `schedules.npy` into `$PSPE_EVAC_DATA` (default
`data/evac/`). Both are regenerated rather than committed: the schedules are
7.6 MB of solver output, and a result that depends on a solver should be
reproducible by running it.

Two choices here are load-bearing and were each wrong once:

* **15-minute snapshots, not hourly.** `Schedule.at` takes the MAX of the two
  bracketing snapshots, which is conservative in the only direction that
  matters -- it never calls a road passable during an interval it did not
  look at. With hourly snapshots and twelve-minute trips, every journey is
  scored against the next HOUR's flood, and this network goes from 0 to
  between 105 and 604 cut edges inside that first hour. Escape became
  impossible even at t=0 on a dry network and every arm scored an identical
  -434.44.
* **`source="auto"`**, which is USGS 3DEP bare earth inside the US.
  `build_domain` DEFAULTS to Copernicus GLO-30, a SURFACE model that reads
  metres high over trees and buildings and ponds water bare earth drains.
  Solved on it, the shipped design storm floods 28.84 km2 against the
  portal's 18.71, and a finer grid makes it worse because more resolution
  means more spurious canopy.

The ensemble is multiples of the SHIPPED DESIGN STORM -- rain on the box at
50 mm/h for 6 h -- not of the 2013 river peak. The portal's Roads tab reports
117 of 891 km unusable on that storm, so the x1.00 member is checkable
against a published figure, and the print below does exactly that check.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from pspe.plan import routes
from pspe.simulate.flood import FloodConfig, FloodSolver
from pspe.simulate.real.anywhere import build_domain
from service.app import PlanRequest, build_forcing, fetch_roads

from . import table as T

BOX = dict(west=-97.82, south=30.11, east=-97.69, north=30.21)
DX = 40.0
SNAP_H = [t * T.INTERVAL / 60.0 for t in range(T.T)]
MEMBERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
RAIN_MM_H = 50.0
STORM_H = 6.0


def main() -> int:
    T.DATA.mkdir(parents=True, exist_ok=True)
    roads_path = T.DATA / "roads.json"

    if roads_path.exists():
        fc = json.loads(roads_path.read_text())
        print(f"roads.json already present: {len(fc['features'])} features")
    else:
        fc = fetch_roads(**BOX)
        if not fc.get("features"):
            raise SystemExit("Overpass returned no roads; try again later. "
                             "The flood half of this does not depend on it, "
                             "but the evacuation half is entirely about roads.")
        roads_path.write_text(json.dumps(fc))
        print(f"wrote {roads_path}: {len(fc['features'])} features")

    g = routes.build_graph(fc["features"])
    print(f"graph {len(g.nodes)} nodes {len(g.edges)} edges {g.total_km:.0f} km")

    t0 = time.time()
    dom = build_domain(BOX["west"], BOX["south"], BOX["east"], BOX["north"],
                       dx=DX, name="onion-evac", source="auto")
    print(f"domain {dom.z.shape} at {dom.dx:.0f} m ({time.time() - t0:.0f}s)")

    flat, offs = routes.edge_cells(g, dom.bounds, dom.z.shape)
    z = torch.as_tensor(dom.z, dtype=torch.float32)[None]
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=1e-4, dt_max=300.0)
    at = sorted(float(x) * 3600.0 for x in SNAP_H)
    run_s = max(at) + 3600.0
    lens = np.array([e.length_m for e in g.edges]) / 1000.0

    scheds = []
    for mult in MEMBERS:
        t1 = time.time()
        req = PlanRequest(**BOX, name="evac", dx=dom.dx, driver="rain",
                          rain_mm_h=RAIN_MM_H * mult, storm_hours=STORM_H,
                          run_hours=run_s / 3600.0, sites=[])
        fx = build_forcing(dom, req)
        solver = FloodSolver(z, cfg)
        h = torch.zeros_like(z)
        qx, qy = solver.zeros_flux(1)
        src, sea_mask, sea_level = fx["source"], fx["sea_mask"], fx["sea_level"]
        rows: list = []
        nxt, tt = 0, 0.0
        while tt < run_s:
            dt = min(solver.adaptive_dt(h), run_s - tt)
            if dt <= 0:
                break
            on = tt < STORM_H * 3600.0
            r = (src if on else 0.0) if src is not None else 0.0
            h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=z)
            if sea_mask is not None and on:
                lvl = torch.clamp(sea_level - z, min=0.0)
                h = torch.where(sea_mask.expand_as(h), torch.maximum(h, lvl), h)
            tt += dt
            while nxt < len(at) and tt >= at[nxt]:
                rows.append(routes.depths_from_field(h[0].numpy(), flat, offs))
                nxt += 1
        while nxt < len(at):
            rows.append(routes.depths_from_field(h[0].numpy(), flat, offs))
            nxt += 1
        scheds.append(rows)

        arr = np.asarray(rows)
        # km-weighted at the car threshold, which is what the portal reports.
        # An edge COUNT is not that number and must not be compared to it.
        cut = arr.max(axis=0) > 0.25
        print(f"member x{mult:.2f} ({RAIN_MM_H * mult:3.0f} mm/h): peak "
              f"{arr.max():.2f} m, {100 * cut.mean():.0f}% edges, "
              f"{lens[cut].sum():.0f} of {lens.sum():.0f} km cut "
              f"({time.time() - t1:.0f}s)")

    out = T.DATA / "schedules.npy"
    np.save(out, np.asarray(scheds, dtype=np.float32))
    print(f"wrote {out}  shape {np.asarray(scheds).shape}")
    print("\nnext: python3 -m baselines.evac.table   (routes the ensemble, ~8 min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
