"""Build the egress table once; every experiment loads it.

The table is real time-dependent routing over the Onion Creek network -- M
members x K zones x T slots of `fastest_at` -- and it is by far the most
expensive thing in these experiments, around eight minutes. Two scripts were
rebuilding it concurrently at one point, which is the whole cost twice for
one answer.

Cached as `.npz` plus `.json` rather than pickle: the arrays are arrays and
the rest is scalars, so there is no reason to use a format that executes code
on load.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pspe.envs.evacuate import EgressTable, build_egress_table
from pspe.plan import routes

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PSPE_EVAC_DATA", ROOT / "data" / "evac"))

#: 48 slots of 15 minutes is twelve hours, which covers the Onion Creek
#: hydrograph from the first rise to well past the crest.
T, INTERVAL = 48, 15.0
#: How far ahead the operator's signal leads the closures, in slots. Eight is
#: 120 minutes, the lead at which the study stops finding a route out for
#: everyone; below it no policy of any kind helps.
LEAD = 8
#: The zones, in people. Eight neighbourhoods drawn south of the creek axis.
POP = np.array([1200., 900., 1500., 700., 1100., 800., 1300., 600.])

MISSING = (
    "missing {what} at {path}.\n"
    "These are solver output and road geometry, not fixtures, so they are not\n"
    "committed. Build them with:\n"
    "    python3 -m baselines.evac.prepare\n"
    "or point PSPE_EVAC_DATA at a directory that already has them.")


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(MISSING.format(what=what, path=path))
    return path


def cache_paths(lead: int = LEAD) -> tuple[Path, Path]:
    return (DATA / f"table_lead{lead}.npz", DATA / f"table_lead{lead}.json")


def build(lead: int = LEAD) -> EgressTable:
    """Route the whole ensemble. Deterministic given the two input files."""
    g = routes.build_graph(
        json.loads(_need(DATA / "roads.json", "the road network").read_text())
        ["features"])
    arr = np.load(_need(DATA / "schedules.npy", "the solved flood ensemble"))
    sch = [routes.Schedule(times=[t * INTERVAL for t in range(T)],
                           depths=[r.tolist() for r in arr[m]])
           for m in range(arr.shape[0])]

    # Zones south of the creek, a safe node well north of it. The axis is the
    # median latitude of the graph, which is the creek, because the network
    # was clipped to the study box.
    lat = np.array([n[1] for n in g.nodes])
    axis = float(np.median(lat))
    south = np.flatnonzero(lat < axis - 0.012)
    north = np.flatnonzero(lat > axis + 0.025)
    safe = int(north[len(north) // 2])

    # Candidate zones must be reachable on a DRY network, or the episode is
    # scoring a routing failure as a flood casualty.
    dry = routes.Schedule(times=[t * INTERVAL for t in range(T)],
                          depths=[[0.0] * arr.shape[2] for _ in range(T)])
    rng = np.random.default_rng(0)
    zones: list[int] = []
    for c in rng.permutation(south):
        r = routes.fastest_at(g, int(c), safe, dry, 0.25, depart_min=0.0)
        if r is not None and np.isfinite(getattr(r, "minutes", np.inf)):
            zones.append(int(c))
        if len(zones) == len(POP):
            break
    return build_egress_table(g, sch, zones, safe, POP, limit_m=0.25, slots=T,
                              interval_min=INTERVAL, max_travel_min=45.0,
                              lead_slots=lead)


def save(tab: EgressTable, lead: int = LEAD) -> None:
    npz, meta = cache_paths(lead)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, ok=tab.ok, minutes=tab.minutes,
                        exposure=tab.exposure, signal=tab.signal,
                        deadline=tab.deadline, population=tab.population)
    meta.write_text(json.dumps(
        {"zone_nodes": list(tab.zone_nodes), "safe_node": tab.safe_node,
         "interval_min": tab.interval_min,
         "zone_rc": [list(rc) for rc in tab.zone_rc], "grid": tab.grid}))


def load(lead: int = LEAD) -> EgressTable:
    """The cached table, building and saving it the first time."""
    npz, meta = cache_paths(lead)
    if not (npz.exists() and meta.exists()):
        tab = build(lead)
        save(tab, lead)
        return tab
    z = np.load(npz)
    m = json.loads(meta.read_text())
    return EgressTable(
        ok=z["ok"], minutes=z["minutes"], exposure=z["exposure"],
        signal=z["signal"], deadline=z["deadline"],
        population=z["population"],
        zone_nodes=[int(n) for n in m["zone_nodes"]],
        safe_node=int(m["safe_node"]), interval_min=float(m["interval_min"]),
        zone_rc=[tuple(rc) for rc in m["zone_rc"]], grid=int(m["grid"]))


if __name__ == "__main__":
    import time
    t0 = time.time()
    tab = build()
    save(tab)
    print(f"built and cached in {time.time() - t0:.0f}s: "
          f"{tab.members} members x {tab.zones} zones x {tab.slots} slots")
    r = load()
    assert np.array_equal(r.ok, tab.ok) and np.allclose(r.exposure, tab.exposure)
    assert np.allclose(r.signal, tab.signal) and r.zone_nodes == tab.zone_nodes
    print("reload matches the build exactly")
