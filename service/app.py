"""Flood modelling service.

The portal is a client; this is where the physics lives. It takes a place on
Earth, fetches terrain and forecast, solves the shallow-water equations, and
returns consequences — flood depth as an image, roads cut, area inundated.

Kept deliberately small: one solver, one domain builder, no state beyond a
result cache. A request for a 13 km town at 60 m over a 12 h design storm
completes in about 35 seconds, which is why this can answer on demand rather
than queueing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pspe.observe import sar  # noqa: E402
from pspe.simulate.real import forcing  # noqa: E402
from pspe.plan import routes, surrogate  # noqa: E402
from pspe.explain import route_brief  # noqa: E402
from pspe.simulate.flood import FloodConfig, FloodSolver, Structures  # noqa: E402
from pspe.simulate.real.anywhere import (  # noqa: E402
    build_domain, forcing_for,
)

app = FastAPI(title="PSPE flood service")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CACHE: dict[str, dict[str, Any]] = {}
DEPTH_VMAX = 8.0


class ObserveRequest(BaseModel):
    """What the satellite last saw over this box."""
    west: float
    south: float
    east: float
    north: float
    name: str = "Selected area"
    days_back: int = Field(45, ge=12, le=365)
    size: int = Field(512, ge=128, le=1024)


class PlanRequest(BaseModel):
    """Plan an arbitrary place under a budget, with a calibrated guarantee."""
    west: float
    south: float
    east: float
    north: float
    name: str = "Selected area"
    dx: float = Field(90.0, ge=30.0, le=240.0)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    # Candidate levee locations. Four or more are needed before a 90% margin is
    # attainable, and the response says so rather than quietly dropping to a
    # point estimate.
    sites: list[dict] = []
    max_height_m: float = Field(3.0, ge=0.5, le=10.0)
    # US dollars. The study is in Texas and the unit rates behind it are
    # USACE figures escalated to 2025 and adjusted for Texas; see the portal's
    # `lib/rates`, which is where the per-measure rates now come from and
    # which is the only place they are written down.
    budget_usd: float = Field(20e6, ge=0.0)
    delta: float = Field(0.1, ge=0.01, le=0.5)
    protect: Optional[dict] = None   # {"lat":..,"lon":..,"radius_m":..}
    # What drives the flood. "rain" spreads the storm over the whole box, which
    # is a real question and the wrong one for a river or coastal city: a levee
    # has nothing to block when the water falls on both sides of it.
    driver: str = Field("rain", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None    # None -> take it from the flood API
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    surge_m: float = Field(2.5, ge=0.0, le=10.0)


class AnalyseRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float
    name: str = "Selected area"
    dx: float = Field(60.0, ge=30.0, le=240.0)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    levees: list[dict] = []
    # The baseline must be driven the same way the plan is, or the two disagree
    # about what "doing nothing" costs.
    driver: str = Field("rain", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    surge_m: float = Field(2.5, ge=0.0, le=10.0)
    # Instantaneous depth fields through the run, for the scrubber under the
    # map. Zero means peak only, which is what the study bundle was built
    # with before this existed.
    snapshots: int = Field(0, ge=0, le=24)


# What counts as flooded, in metres, for everything that is drawn or counted.
#
# There were two numbers. The overlay painted every cell over 1 cm and the area
# metric counted cells over 10 cm, so on the Austin study the map showed 63 km2
# of water while the panel beside it said 18.71 km2 -- both honest, three times
# apart, and the first thing a visitor sees. One centimetre over a 40 m cell is
# 16 cubic metres; it is wet ground, not a flood. The solver keeps its own
# `depth_min` of 1 mm, which is a numerical floor for wet/dry interfaces and a
# different thing from a reporting threshold.
WET_M = 0.10


def depth_png(d: np.ndarray, vmax: float) -> str:
    """Depth as a base64 PNG overlay, on the same ramp the portal uses."""
    x = np.clip(np.nan_to_num(d) / max(vmax, 1e-6), 0.0, 1.0) ** 0.42
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (96 * (1 - x) + 4 * x).astype(np.uint8)
    out[..., 1] = (200 * (1 - x) + 28 * x).astype(np.uint8)
    out[..., 2] = (255 * (1 - x) + 128 * x).astype(np.uint8)
    a = np.where(np.nan_to_num(d) > WET_M, 132 + 118 * x, 0.0)
    out[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def terrain_png(z: np.ndarray) -> str:
    """Terrain-RGB for the 3D view, matching the portal's decoder."""
    v = np.nan_to_num(z, nan=0.0)
    enc = np.clip((v + 10000.0) / 0.1, 0, 256 ** 3 - 1).astype(np.uint32)
    out = np.zeros(v.shape + (3,), dtype=np.uint8)
    out[..., 0] = (enc >> 16) & 255
    out[..., 1] = (enc >> 8) & 255
    out[..., 2] = enc & 255
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def solve(dom, rain_mm_h: float, storm_h: float, run_h: float,
          levees: list[dict], req=None,
          snap_hours: "Sequence[float] | None" = None,
          ) -> tuple[np.ndarray, dict, "np.ndarray | None"]:
    z = torch.as_tensor(dom.z, dtype=torch.float32)
    h_, w_ = z.shape

    # Levees raise the bed, which is what makes them gate flow rather than add
    # water. Through `site_masks`, so a levee is the same object here as in the
    # plan solver: a crest swept along an alignment, floored at one cell
    # because that is the narrowest thing the grid can block with. This used
    # to be its own Gaussian ridge, which meant the two endpoints disagreed
    # about what a levee IS -- a round 600 m mound here, something else there.
    if levees:
        hs = [float(lv.get("height_m", 2.0)) for lv in levees]
        masks = site_masks(dom, levees, max(hs) if hs else 2.0)
        for i, hgt in enumerate(hs):
            z = z + hgt * masks[i]

    rows = z.mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (h_ * dom.dx))), 1e-5)
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=slope, dt_max=300.0)
    solver = FloodSolver(z[None], cfg)
    h = torch.zeros(1, h_, w_)
    qx, qy = solver.zeros_flux(1)
    peak = h.clone()
    fx = build_forcing(dom, req) if req is not None else {
        "source": rain_mm_h / 3.6e6, "sea_mask": None, "sea_level": None,
        "driver": "rain", "note": "", "q_m3s": None, "stage_m": None}
    src, sea_mask, sea_level = fx["source"], fx["sea_mask"], fx["sea_level"]
    # Instantaneous depth at a few times, not the running peak. The peak field
    # answers "was this ever wet", which is the wrong question for anyone
    # watching the water arrive. The run already computes every one of these
    # and threw them all away; keeping a handful is what lets the strip under
    # the map scrub the flood instead of scrubbing nothing.
    snap_at = sorted(float(x) * 3600.0 for x in (snap_hours or []))
    snaps: list["torch.Tensor"] = []
    nxt = 0
    t, dur, steps = 0.0, run_h * 3600.0, 0
    t0 = time.time()
    while t < dur:
        dt = min(solver.adaptive_dt(h), dur - t)
        if dt <= 0:
            break
        on = t < storm_h * 3600.0
        r = (src if on else 0.0) if src is not None else 0.0
        h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=z[None])
        if sea_mask is not None and on:
            want = torch.clamp(sea_level - z, min=0.0)
            h = torch.where(sea_mask[None].expand_as(h), torch.maximum(h, want), h)
        peak = torch.maximum(peak, h)
        t += dt; steps += 1
        while nxt < len(snap_at) and t >= snap_at[nxt]:
            snaps.append(h[0].clone())
            nxt += 1
    # A run that stops early still owes a schedule of the right length, or the
    # client reads a shorter horizon than it asked about.
    while nxt < len(snap_at):
        snaps.append(h[0].clone())
        nxt += 1

    return (peak[0].numpy(),
            {"driver": fx["driver"], "driver_note": fx["note"],
             "steps": steps, "seconds": round(time.time() - t0, 1)},
            (torch.stack(snaps).numpy() if snaps else None))


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cached": len(CACHE)}


def observed_png(flooded: np.ndarray, permanent: np.ndarray) -> str:
    """Observed flood as a base64 PNG: new water solid, permanent water faint.

    The two are drawn differently on purpose. A change detector answers "what is
    newly wet", and painting permanent river and sea in the same colour as new
    flooding is how a viewer is misled into thinking the model over-predicts
    (defect 21).
    """
    h, w = flooded.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Permanent water recedes into the base; new water is bright. The two must
    # stay distinguishable without hue, because conflating them is defect 21.
    rgba[permanent] = (90, 130, 170, 90)
    rgba[flooded] = (37, 99, 168, 205)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/observe")
def observe(req: ObserveRequest) -> dict:
    """The Perceive stage: the most recent Sentinel-1 view of this box.

    This is what makes the tool a twin rather than a simulator. The model says
    what it believes is flooded; this says what the instrument actually saw on
    its last usable overpass, so the two can be put side by side and disagree.

    Two honesty constraints are enforced here rather than left to the caller:
    the flood and baseline scenes must share a relative orbit, because
    differencing across viewing geometries manufactures flooding wherever the
    incidence angle changed; and the answer carries the acquisition date, since
    Sentinel-1 repeats every 12 days and a twin that presents a week-old
    overpass as "now" is lying about its own freshness.
    """
    key = "obs:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    t0 = time.time()
    bbox = [req.west, req.south, req.east, req.north]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=req.days_back)
    try:
        scenes = sar.search(bbox, start.isoformat(), end.isoformat())
    except Exception as exc:
        return {"ok": False, "reason": f"scene search failed: {exc}"}
    if len(scenes) < 2:
        return {"ok": False, "reason":
                f"only {len(scenes)} Sentinel-1 scenes over this box in "
                f"{req.days_back} days; nothing to difference against"}

    latest = scenes[-1].datetime[:10]
    try:
        flood, baselines = sar.pick_pair(scenes, latest)
    except Exception as exc:
        return {"ok": False, "reason": f"no same-orbit baseline: {exc}"}
    if not baselines:
        return {"ok": False, "reason":
                "found a recent scene but no earlier one on the same relative "
                "orbit; differencing across tracks would invent flooding"}

    try:
        fm = sar.detect(flood, baselines, bbox, out_shape=(req.size, req.size))
    except Exception as exc:
        return {"ok": False, "reason": f"detection failed: {exc}"}

    out = {
        "ok": True,
        "name": req.name,
        "acquired": flood.datetime,
        "days_old": (end - date.fromisoformat(flood.datetime[:10])).days,
        "baselines": len(baselines),
        "baseline_dates": [b.datetime[:10] for b in baselines],
        "relative_orbit": flood.relative_orbit,
        "orbit_state": flood.orbit_state,
        "flooded_km2": round(fm.flooded_km2, 2),
        "permanent_km2": round(float(fm.permanent.sum())
                               * abs(fm.transform[0] * fm.transform[4]) / 1e6, 2),
        "threshold_db": round(fm.threshold_db, 2),
        "observed_png": observed_png(fm.flooded, fm.permanent),
        "bounds": bbox,
        "seconds": round(time.time() - t0, 1),
        "note": ("Sentinel-1 repeats every 12 days and cannot see standing "
                 "water under canopy or among buildings, so an empty result "
                 "is a statement about the overpass, not about the ground."),
    }
    CACHE[key] = out
    return {**out, "cached": False}


# Overpass is free, public and frequently overloaded -- the main instance
# returns 504 often enough that a single endpoint is not a dependency, it is a
# coin flip. Tried in order; the first that answers wins.
# Bare earth where a bare-earth model exists.
#
# Copernicus GLO-30 is a SURFACE model: measured against 37,624 City of Austin
# building ground elevations it reads a median 3.54 m high and is high at 93.8%
# of them, and against 2,512 council-surveyed levels at Lismore, 2.21 m high at
# 97.2% (sections 5.14e-f). In a town that is more than the difference between
# a flood that enters a house and one that does not. "auto" takes USGS 3DEP
# bare earth inside the United States, where the same comparison gives a median
# of +0.14 m, and falls back to Copernicus everywhere else.
TERRAIN = os.environ.get("PSPE_TERRAIN", "auto")
TERRAIN_LABEL = {
    "copernicus": "Copernicus DEM GLO-30 (a SURFACE model: includes canopy "
                  "and buildings)",
    "3dep": "USGS 3DEP bare earth, 1 m",
}

OVERPASS_MIRRORS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential")


def _first_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    parts = [r.strip() for r in ref.split(";") if r.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else f"{parts[0]} (+{len(parts) - 1})"


def _seg_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle-ish length of a short segment, in km."""
    dlat = (b[1] - a[1]) * 111.0
    dlon = (b[0] - a[0]) * 111.0 * math.cos(math.radians((a[1] + b[1]) / 2))
    return math.hypot(dlat, dlon)


# OpenStreetMap, kept on disk.
#
# Three things in this service come from Overpass -- the channel, the road
# network, and the names on the map -- and all three are free public mirrors
# that fail. Measured in one sitting: kumi.systems and private.coffee both
# returned 500 while overpass-api.de answered, and an hour earlier the
# reverse. Each failure has a fallback, and each fallback is worse in a way
# the reader cannot see: without the mapped channel the inflow comes from a
# terrain heuristic that picks the wrong corner of a tilted catchment, and
# the same box then floods 2.56 km2 instead of 12.31. Without the names, six
# candidate levees are called "unnamed bank".
#
# So a good answer is written to disk and kept. OSM geometry for a river or a
# street changes on a timescale of years; a thirty-day cache is not staleness,
# it is the difference between a tool that works and a tool that works when
# a volunteer-run mirror happens to be up.
_OSM_DIR = Path(os.environ.get("PSPE_OSM_CACHE",
                               str(ROOT / "data" / "osm_cache")))
_OSM_TTL_S = 30 * 86400


def _osm_disk(kind: str, key: tuple) -> Path:
    h = hashlib.sha1(json.dumps([kind, list(key)], sort_keys=True).encode())
    return _OSM_DIR / f"{kind}-{h.hexdigest()[:16]}.json"


def _osm_load(kind: str, key: tuple):
    f = _osm_disk(kind, key)
    try:
        if f.is_file() and time.time() - f.stat().st_mtime < _OSM_TTL_S:
            return json.loads(f.read_text())
    except Exception:                                      # noqa: BLE001
        return None
    return None


def _osm_save(kind: str, key: tuple, value) -> None:
    # Only a non-empty answer. An empty one is a failed fetch, and writing it
    # down turns one 504 into a month of them (rule 83).
    if not value:
        return
    try:
        _OSM_DIR.mkdir(parents=True, exist_ok=True)
        _osm_disk(kind, key).write_text(json.dumps(value))
    except Exception:                                      # noqa: BLE001
        pass


_WATERWAY_CACHE: dict = {}


def fetch_waterways(west: float, south: float, east: float, north: float,
                    timeout: float = 60.0, rounds: int = 2) -> list[dict]:
    """Mapped rivers and streams over this box, from OpenStreetMap.

    Used to find where the channel crosses the boundary, which the terrain
    alone gets wrong on a tilted catchment. Cached per box and allowed to fail:
    Overpass is a free public service and a 504 from it must not take a flood
    solve down, so the caller falls back to the edge heuristic.
    """
    key = (round(west, 4), round(south, 4), round(east, 4), round(north, 4))
    if key in _WATERWAY_CACHE:
        return _WATERWAY_CACHE[key]
    disk = _osm_load("waterways", key)
    if disk:
        _WATERWAY_CACHE[key] = disk
        return disk
    q = (f"[out:json][timeout:{int(timeout)}];"
         f'way["waterway"~"^(river|stream)$"]'
         f"({south},{west},{north},{east});out geom;")
    payload = None
    for url in [u for _ in range(rounds) for u in OVERPASS_MIRRORS]:
        try:
            req = urllib.request.Request(
                url, data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode())
            break
        except Exception:                                  # noqa: BLE001
            continue
    els = (payload or {}).get("elements", [])
    # A failed fetch is not a box with no rivers in it, and caching it as one
    # makes the failure permanent. Measured: two identical /sites calls
    # minutes apart returned different floods -- 6.18 km2 against 12.31 --
    # and different candidate banks, because one found the mapped channel and
    # the other fell back to the terrain-edge heuristic, which on a catchment
    # that tilts east puts the inflow on the wrong corner.
    if not els:
        return []
    if len(_WATERWAY_CACHE) > 32:
        _WATERWAY_CACHE.pop(next(iter(_WATERWAY_CACHE)))
    _WATERWAY_CACHE[key] = els
    _osm_save("waterways", key, els)
    return els


def fetch_roads(west: float, south: float, east: float, north: float,
                timeout: float = 90.0, rounds: int = 2) -> dict:
    """The road network over this box, from OpenStreetMap via Overpass.

    The district's roads were prepared in advance, which is why road impact
    only worked there. Nothing about the analysis needs that: the geometry is
    public and the query is small. Fetched live, cached per box, and returned
    in the same GeoJSON shape the district uses so one piece of client code
    scores both.

    A failure here is not a failure of the model. The flood answer stands
    without roads, so this returns an empty collection and says why rather
    than taking the whole request down with it.
    """
    q = (f"[out:json][timeout:{int(timeout)}];"
         f'way["highway"~"^({"|".join(ROAD_CLASSES)})$"]'
         f"({south},{west},{north},{east});out geom;")
    payload, last = None, None
    for url in [u for _ in range(rounds) for u in OVERPASS_MIRRORS]:
        try:
            req = urllib.request.Request(
                url, data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode())
            break
        except Exception as exc:                       # noqa: BLE001
            last = exc
            continue
    if payload is None:
        raise RuntimeError(f"every Overpass mirror failed ({last})")

    feats = []
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = [[g["lon"], g["lat"]] for g in geom]
        length_km = sum(_seg_km(tuple(coords[i]), tuple(coords[i + 1]))
                        for i in range(len(coords) - 1))
        tags = el.get("tags", {})
        feats.append({
            "type": "Feature",
            "properties": {
                "id": el["id"],
                # OSM packs concurrent designations into one ref field as
                # "US 30;US 151;US 218". A worst-affected list is read, not
                # parsed, so take the first and note the rest.
                "name": (tags.get("name")
                         or _first_ref(tags.get("ref"))
                         or "Unnamed road"),
                "class": tags.get("highway", "unclassified"),
                "len_m": int(length_km * 1000),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": feats}


def inspect_grids(peak: np.ndarray, z: np.ndarray, vmax: float) -> dict:
    """Depth and ground as the quantised grids the client already knows how to
    sample. Same encoding as the district's `inspect.json` + `.u8` / `.u16`, so
    the point query and the road scoring are the same code in both places."""
    d = np.clip(np.nan_to_num(peak) / max(vmax, 1e-6), 0.0, 1.0)
    depth_u8 = (d * 255).astype(np.uint8)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    span = max(zmax - zmin, 1e-6)
    elev_u16 = (np.clip((np.nan_to_num(z) - zmin) / span, 0, 1) * 65535).astype(np.uint16)
    return {
        "meta": {
            "grid": [int(peak.shape[0]), int(peak.shape[1])],
            "depth_vmax_m": vmax,
            "elev_min_m": zmin,
            "elev_max_m": zmax,
        },
        "depth_u8": base64.b64encode(depth_u8.tobytes()).decode(),
        "elev_u16": base64.b64encode(elev_u16.tobytes()).decode(),
    }


def roads_and_grids(dom, peak: np.ndarray, west: float, south: float,
                    east: float, north: float) -> dict:
    """The samplable grids. Roads are fetched separately, on purpose.

    Overpass is free, public, and takes 20-60 s for a city-sized box when it
    answers at all. Fetching it inside the solve put a flaky third party on the
    critical path of a result that already takes two minutes: a slow lookup
    delayed the flood answer and a failed one wasted the whole wait. It is now
    its own endpoint, called only when the roads view is actually opened.
    """
    return {"inspect": inspect_grids(peak, dom.z, DEPTH_VMAX)}


class RoadsRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float


# How far a habitable floor sits above the ground outside it, in metres.
#
# A guess, and labelled as one everywhere it is used. Texas construction is
# overwhelmingly slab-on-grade and 0.17 m is a slab plus its step. The number
# matters: on 2,924 surveyed Lismore properties the median floor stands 1.95 m
# above its own front gate, p5 to p95 spanning 0.04 m to 3.28 m, because a
# Queenslander sits on stumps. Nowhere publishes it for Austin -- the FEMA
# elevation certificates that carry it are not bulk-public -- so every response
# that uses it says so rather than folding it into a depth.
FLOOR_ABOVE_GROUND_M = 0.17

# City of Austin's building footprints: 585,916 of them, carrying the ground
# elevation at the structure and its roof height from 2012 lidar, in US survey
# feet. The sentinel for a missing value is a large negative rather than null,
# so a mean taken straight off the service returns -70,005,326.
AUSTIN_BUILDINGS = ("https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/"
                    "services/STRUCTURE_building_footprints_2013/FeatureServer/0/query")
_NAMED_CACHE: dict = {}


def fetch_named_lines(west: float, south: float, east: float, north: float,
                      timeout: float = 90.0, rounds: int = 2) -> list[dict]:
    """Named roads and waterways, for calling a place something findable.

    "Site 3" is not something anyone can locate on the ground. Optional and
    cached: a failure here costs a name, not an answer.

    The highway filter is `ROAD_CLASSES`, the same list the road layer uses,
    and that is not tidiness. A bare `way["highway"]["name"]` over the 13 by
    11 km Austin box asks Overpass for every driveway, parking aisle, footway
    and service road in south Austin before the `out geom 400` cap throws
    nearly all of them away: both public mirrors returned 500 and 504, the
    fetch failed silently, and all six candidate sites came back unnamed.
    """
    key = (round(west, 4), round(south, 4), round(east, 4), round(north, 4))
    if key in _NAMED_CACHE:
        return _NAMED_CACHE[key]
    disk = _osm_load("named", key)
    if disk:
        _NAMED_CACHE[key] = disk
        return disk
    q = (f"[out:json][timeout:{int(timeout)}];("
         f'way["waterway"~"^(river|stream|canal|drain)$"]["name"]'
         f"({south},{west},{north},{east});"
         f'way["highway"~"^({"|".join(ROAD_CLASSES)})$"]["name"]'
         f"({south},{west},{north},{east});"
         f");out geom;")
    payload = None
    for url in [u for _ in range(rounds) for u in OVERPASS_MIRRORS]:
        try:
            r = urllib.request.Request(
                url, data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
            with urllib.request.urlopen(r, timeout=timeout) as f:
                payload = json.loads(f.read().decode())
            break
        except Exception:                                  # noqa: BLE001
            continue
    # An empty answer is a failure, not a place with no names, and caching it
    # would make the failure permanent for the life of the process.
    if not (payload or {}).get("elements"):
        return []
    out = []
    for el in (payload or {}).get("elements", []):
        nm = (el.get("tags") or {}).get("name")
        geom = el.get("geometry") or []
        if not nm or not geom:
            continue
        kind = "waterway" if (el.get("tags") or {}).get("waterway") else "road"
        out.append({"name": nm, "kind": kind,
                    "pts": [(g["lat"], g["lon"]) for g in geom[::3]] or
                           [(geom[0]["lat"], geom[0]["lon"])]})
    if len(_NAMED_CACHE) > 24:
        _NAMED_CACHE.pop(next(iter(_NAMED_CACHE)))
    _NAMED_CACHE[key] = out
    _osm_save("named", key, out)
    return out


def nearest_name(lines: list[dict], lat: float, lon: float,
                 max_km: float = 0.6) -> str | None:
    """The nearest named line, preferring a waterway: a levee follows a bank."""
    best, bestd, bestkind = None, 1e9, ""
    for ln in lines:
        for plat, plon in ln["pts"]:
            dy = (plat - lat) * 110.574
            dx = (plon - lon) * 111.320 * math.cos(math.radians(lat))
            dkm = math.hypot(dx, dy)
            # A waterway at the same distance wins: the bank is the thing
            # being raised, and the road beside it is just the road beside it.
            score = dkm * (0.6 if ln["kind"] == "waterway" else 1.0)
            if score < bestd:
                best, bestd, bestkind = ln["name"], score, ln["kind"]
    if best is None or bestd > max_km:
        return None
    return f"{best} bank" if bestkind == "waterway" else f"near {best}"


_FOOTPRINT_CACHE: dict = {}
US_FT = 0.3048


def _rings_from_geojson(geom: dict) -> list[list[list[float]]]:
    if not geom:
        return []
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


AUSTIN_ADDRESSES = ("https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/"
                    "services/LOCATION_address_points/FeatureServer/0/query")


def fetch_address_points(west: float, south: float, east: float, north: float,
                         limit: int = 2000) -> list[dict]:
    """City address points, so a building can be called by its address.

    Without this the street view lists `coa:336505` and a resident cannot find
    their own house in it. Austin publishes 465,977 of these; the footprints
    layer carries no address field of its own, so the two are joined by
    position at request time.
    """
    try:
        q = urllib.parse.urlencode({
            "f": "geojson", "where": "1=1",
            "outFields": ("ADDRESS,PREFIX_DIRECTION,STREET_NAME,STREET_TYPE,"
                          "SUFFIX_DIRECTION,FULL_STREET_NAME"),
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "true",
            "resultRecordCount": limit,
        })
        r = urllib.request.Request(f"{AUSTIN_ADDRESSES}?{q}",
                                   headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
        with urllib.request.urlopen(r, timeout=60) as f:
            d = json.loads(f.read().decode())
    except Exception:                                      # noqa: BLE001
        return []
    out = []
    for ft in d.get("features", []):
        g = ft.get("geometry") or {}
        a = ft.get("properties") or {}
        if g.get("type") != "Point":
            continue
        num = str(a.get("ADDRESS") or "").strip()
        street = str(a.get("FULL_STREET_NAME") or "").strip()
        if not street:
            street = " ".join(str(a.get(k) or "").strip() for k in (
                "PREFIX_DIRECTION", "STREET_NAME", "STREET_TYPE",
                "SUFFIX_DIRECTION")).strip()
        street = " ".join(street.split())
        # FULL_STREET_NAME already carries the house number, so prepending
        # ADDRESS to it produced "7320 7320 LAKE CHARLES DR".
        if num and street.startswith(num + " "):
            label = street
        else:
            label = f"{num} {street}".strip() if num else street
        # The city stores addresses in capitals. "7320 LAKE CHARLES DR" shouts
        # at a reader in a list of two hundred; the directional and the street
        # type keep their conventional short forms.
        keep = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
        label = " ".join(w if (w in keep or not w.isalpha()) else w.capitalize()
                         for w in label.split())
        if label:
            out.append({"lon": g["coordinates"][0], "lat": g["coordinates"][1],
                        "address": label})
    return out


def fetch_footprints(west: float, south: float, east: float, north: float,
                     limit: int = 500) -> list[dict]:
    """Building outlines for a block, with a ground elevation where published.

    Two sources, and they are not equivalent. The City of Austin publishes the
    ground AT the structure from lidar, which is the number a floor height
    should be measured from. OpenStreetMap publishes the outline and sometimes
    a storey count, and no elevation at all -- so outside Austin the ground
    comes from the DEM and the caller is told which it got.
    """
    key = (round(west, 5), round(south, 5), round(east, 5), round(north, 5), limit)
    if key in _FOOTPRINT_CACHE:
        return _FOOTPRINT_CACHE[key]

    out: list[dict] = []
    # Austin first, where the data is richer. The bounding box is the city's
    # rough extent; outside it the service simply returns nothing and the OSM
    # branch takes over, so this only has to be approximately right.
    if -98.2 < (west + east) / 2 < -97.3 and 29.9 < (south + north) / 2 < 30.7:
        try:
            q = urllib.parse.urlencode({
                "f": "geojson", "where": "1=1",
                "outFields": "OBJECTID,MAX_HEIGHT,BASE_ELEVATION",
                "geometry": f"{west},{south},{east},{north}",
                "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
                "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "true",
                "resultRecordCount": limit,
            })
            r = urllib.request.Request(f"{AUSTIN_BUILDINGS}?{q}",
                                       headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
            with urllib.request.urlopen(r, timeout=60) as f:
                d = json.loads(f.read().decode())
            for ft in d.get("features", []):
                a = ft.get("properties") or {}
                for ring in _rings_from_geojson(ft.get("geometry")):
                    g = a.get("BASE_ELEVATION")
                    h = a.get("MAX_HEIGHT")
                    out.append({
                        "id": f"coa:{a.get('OBJECTID')}",
                        "ring": ring,
                        "address": None,
                        "ground_m": round(g * US_FT, 2) if g is not None and g > -1e6 else None,
                        "height_m": round(h * US_FT, 2) if h is not None and 0 < h < 1e6 else None,
                    })
        except Exception:                                  # noqa: BLE001
            out = []

    if not out:
        q = (f"[out:json][timeout:60];way[\"building\"]"
             f"({south},{west},{north},{east});out geom {int(limit)};")
        payload = None
        for url in OVERPASS_MIRRORS:
            try:
                req = urllib.request.Request(
                    url, data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
                with urllib.request.urlopen(req, timeout=90) as f:
                    payload = json.loads(f.read().decode())
                break
            except Exception:                              # noqa: BLE001
                continue
        for el in (payload or {}).get("elements", []):
            geom = el.get("geometry") or []
            if len(geom) < 4:
                continue
            t = el.get("tags") or {}
            lv = t.get("building:levels")
            try:
                height = float(lv) * 3.0 if lv else None
            except ValueError:
                height = None
            addr = None
            if t.get("addr:housenumber") and t.get("addr:street"):
                addr = f"{t['addr:housenumber']} {t['addr:street']}"
            out.append({
                "id": f"osm:{el['id']}",
                "ring": [[g["lon"], g["lat"]] for g in geom],
                "address": addr,
                "ground_m": None,
                "height_m": height,
            })

    out = out[:limit]

    # Attach an address to each footprint, by nearest point inside it. A city
    # address point sits on the parcel, not the structure, so the match is a
    # proximity one and is capped: 25 m is a house and its own address, 80 m is
    # the neighbour's.
    pts = fetch_address_points(west, south, east, north)
    if pts:
        for b in out:
            ring = b["ring"]
            lon = sum(c[0] for c in ring) / len(ring)
            lat = sum(c[1] for c in ring) / len(ring)
            kx = 111_320.0 * math.cos(math.radians(lat))
            best, bd = None, 1e9
            for q in pts:
                d2 = ((q["lon"] - lon) * kx) ** 2 + ((q["lat"] - lat) * 110_540.0) ** 2
                if d2 < bd:
                    bd, best = d2, q["address"]
            if best is not None and bd <= 25.0 ** 2:
                b["address"] = best

    if len(_FOOTPRINT_CACHE) > 24:
        _FOOTPRINT_CACHE.pop(next(iter(_FOOTPRINT_CACHE)))
    _FOOTPRINT_CACHE[key] = out
    return out


_GAUGE_CACHE: dict = {}
FT_M = 0.3048
CFS_M3S = 0.0283168466


def _get(url: str, timeout: float = 45.0) -> str:
    """One GET, decoded. Module scope because three fetchers want it."""
    r = urllib.request.Request(
        url, headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.read().decode()


_FORECAST_CACHE: dict = {}


def _nwps_forecast_series(lid: str, timeout: float = 30.0) -> list[dict] | None:
    """The NWS river forecast for one gauge, as a curve.

    The gauge detail endpoint carries only the forecast PEAK -- one stage and
    one time -- which is enough for a headline and not enough to draw. The
    stageflow endpoint carries the hourly trace the forecaster actually
    issued, typically five days of it, and that is what the strip under the
    map needs to show where the river is going rather than only where it has
    been.

    Returned in metres with the issue time attached, because a forecast is a
    statement made at a moment and one issued nine hours ago is a different
    claim from one issued now. Optional: a failure here costs the band, not
    the reading.
    """
    hit = _FORECAST_CACHE.get(lid)
    if hit and time.time() - hit["at"] < 600:
        return hit["series"]
    try:
        d = json.loads(_get(
            f"https://api.water.noaa.gov/nwps/v1/gauges/{lid}/stageflow",
            timeout=timeout))
    except Exception:                                      # noqa: BLE001
        return None
    fc = d.get("forecast") or {}
    rows = fc.get("data") or []
    if not rows:
        return None
    k = FT_M if str(fc.get("primaryUnits") or "ft").lower().startswith("ft") else 1.0
    out = []
    for r in rows:
        v, t = r.get("primary"), r.get("validTime")
        if not isinstance(v, (int, float)) or not t or not (-500 < v < 500):
            continue
        out.append({"t": t, "m": round(float(v) * k, 3)})
    if len(out) < 2:
        return None
    series = {"issued": fc.get("issuedTime"), "points": out}
    if len(_FORECAST_CACHE) > 24:
        _FORECAST_CACHE.pop(next(iter(_FORECAST_CACHE)))
    _FORECAST_CACHE[lid] = {"at": time.time(), "series": series}
    return series


class PeaksRequest(BaseModel):
    """Annual peak flows for the gauges on this reach."""
    west: float
    south: float
    east: float
    north: float
    pad_deg: float = Field(0.05, ge=0.0, le=0.5)
    # The watercourse the model is about, so the scenario list leads with the
    # gauge on it. Without this the Austin box leads with the Colorado River,
    # whose peaks are four times Onion Creek's and would be nonsense poured
    # into Onion Creek's channel.
    reach: str = ""


_PEAKS_CACHE: dict = {}

# Exceedance probabilities worth naming. A planner thinks in return periods
# and a hydrologist in annual exceedance probability; they are the same number
# said twice, so both are returned and neither has to be computed on screen.
AEP = (0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.002)


def _lp3_quantiles(q_cfs: list[float]) -> tuple[list[dict], dict]:
    """Flood frequency by log-Pearson III, the way Bulletin 17 does it.

    The annual maximum series is fitted in log10 space with a Pearson type III
    distribution -- the standard American method for this, and the one the
    regulatory discharges in FEMA studies come from. Frequency factors by the
    Wilson-Hilferty approximation, which is what Bulletin 17B tabulates.

    **What this is not.** Bulletin 17C weights the station skew against a
    regional skew map, applies the Expected Moments Algorithm to historical
    and censored peaks, and tests for low outliers. None of that is here: this
    is the station skew alone, on the published record, and the response says
    so. It is enough to put a named design flood behind a scenario button and
    not enough to design a levee to.
    """
    import statistics as st
    xs = [math.log10(q) for q in q_cfs if q and q > 0]
    n = len(xs)
    if n < 10:
        return [], {"n": n, "note": "Too short a record to fit a distribution."}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    if sd < 1e-9:
        return [], {"n": n, "note": "The record has no variance to fit."}
    g = (n * sum((x - m) ** 3 for x in xs)
         / ((n - 1) * (n - 2) * sd ** 3)) if n > 2 else 0.0
    nd = st.NormalDist()
    out = []
    for p in AEP:
        z = nd.inv_cdf(1.0 - p)
        if abs(g) < 1e-6:
            k = z
        else:
            k = (2.0 / g) * ((1.0 + g * z / 6.0 - g * g / 36.0) ** 3 - 1.0)
        out.append({
            "aep_pct": round(p * 100, 2),
            "return_period_y": int(round(1.0 / p)),
            "q_m3s": round(10.0 ** (m + k * sd) * CFS_M3S, 1),
        })
    return out, {"n": n, "log_mean": round(m, 4), "log_sd": round(sd, 4),
                 "skew": round(g, 4),
                 "note": ("Log-Pearson III on the station's own annual maxima, "
                          f"{n} years, station skew {g:+.2f} and no regional "
                          "weighting. Bulletin 17C would weight the skew and "
                          "handle historical peaks; this does neither.")}


def _usgs_peaks(site: str, timeout: float = 45.0) -> dict | None:
    """The published annual peak streamflow record for one USGS site."""
    hit = _PEAKS_CACHE.get(site)
    if hit and time.time() - hit["at"] < 86400:
        return hit["val"]
    try:
        rdb = _get("https://nwis.waterdata.usgs.gov/nwis/peak"
                   f"?site_no={site}&agency_cd=USGS&format=rdb", timeout=timeout)
    except Exception:                                      # noqa: BLE001
        return None
    rows, cols = [], None
    for line in rdb.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        if parts and parts[0].startswith(("5s", "15s")):
            continue                                        # the format row
        rows.append(dict(zip(cols, parts)))
    peaks = []
    for r in rows:
        try:
            q = float(r.get("peak_va") or "")
        except ValueError:
            continue
        if q <= 0:
            continue
        gh = r.get("gage_ht") or ""
        peaks.append({
            "date": r.get("peak_dt") or "",
            "q_m3s": round(q * CFS_M3S, 1),
            "q_cfs": q,
            "stage_m": round(float(gh) * FT_M, 2) if gh else None,
        })
    if len(peaks) < 5:
        return None
    quant, fit = _lp3_quantiles([p["q_cfs"] for p in peaks])
    peaks.sort(key=lambda p: -p["q_m3s"])
    val = {"site": site, "peaks": peaks, "quantiles": quant, "fit": fit,
           "years": len(peaks),
           "first": min(p["date"] for p in peaks if p["date"]),
           "last": max(p["date"] for p in peaks if p["date"])}
    if len(_PEAKS_CACHE) > 24:
        _PEAKS_CACHE.pop(next(iter(_PEAKS_CACHE)))
    _PEAKS_CACHE[site] = {"at": time.time(), "val": val}
    return val


@app.post("/peaks")
def peaks(req: PeaksRequest) -> dict:
    """What this creek has actually done, and what it does at 1% a year.

    The plan tab was solving a design rainstorm, and under a rainstorm falling
    evenly on both sides of it a levee holds water in rather than out --
    measured, two 2 m walls on Onion Creek made the flooded area 0.05% WORSE.
    Driving it from the live gauge instead gave 0.13 km2 of water, because
    Onion Creek is not in flood today. Neither is a question a levee has an
    answer to.

    The question a levee has an answer to is a named flood. Onion Creek has
    an instrumented record of those going back to 1921: 135,000 cfs on 31
    October 2013, 122,000 on 30 October 2015, 138,000 in September 1921. Those
    are discharges this reach has carried, published by the USGS, and they are
    what the scenarios on the Plan tab are. The fitted quantiles beside them
    put a 1%-annual-chance number next to the recorded ones.

    The gauge is found the same way `/gauges` finds it, over a box padded
    outwards: the reach's own gauge frequently sits just outside the modelled
    rectangle, and a scenario list that is empty because the gauge is 600 m
    east of the edge is a worse answer than one that looks a little wider.
    """
    key = "peaks:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}
    pad = req.pad_deg
    try:
        rdb = _get("https://waterservices.usgs.gov/nwis/site/?format=rdb"
                   f"&bBox={req.west - pad:.4f},{req.south - pad:.4f},"
                   f"{req.east + pad:.4f},{req.north + pad:.4f}"
                   "&siteType=ST&outputDataTypeCd=pk")
    except Exception as exc:                               # noqa: BLE001
        return {"ok": False, "reason": f"USGS did not answer ({str(exc)[:80]})."}
    cols, sites = None, []
    for line in rdb.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if cols is None:
            cols = parts
            continue
        if parts and parts[0].startswith(("5s", "15s")):
            continue
        r = dict(zip(cols, parts))
        if r.get("site_no"):
            sites.append({"site": r["site_no"], "name": r.get("station_nm", ""),
                          "lat": r.get("dec_lat_va"), "lon": r.get("dec_long_va")})
    if not sites:
        return {"ok": False, "reason": "No USGS site here publishes annual peaks."}

    # The reach's own gauge is frequently well down the alphabetical site
    # list the USGS returns -- Onion Ck at US Hwy 183 is the eighteenth site
    # in the Austin box -- so every site is fetched, not the first handful.
    # Each is a cached day-long GET.
    out = []
    for s in sites[:24]:
        rec = _usgs_peaks(s["site"])
        if rec:
            out.append({**s, **rec})
    if not out:
        return {"ok": False, "reason":
                "The gauges here have no usable annual peak record."}

    # Gauges ON the modelled watercourse first, then longest record. Length
    # alone put the Colorado River at the top of a study about Onion Creek,
    # and its 15,574 m3/s poured down Onion Creek's channel is not a scenario,
    # it is a category error.
    # Strip the punctuation BEFORE testing the stop list, not after. "Austin,"
    # is not "austin", so the place name survived the filter and then matched
    # every gauge in Austin -- which put the Colorado River back on top of a
    # study about Onion Creek.
    STOP = {"creek", "river", "bayou", "brook", "near", "at", "the"}
    want = set()
    for w in req.reach.replace(",", " ").split():
        t = w.lower().strip(".")
        if len(t) > 3 and t not in STOP:
            want.add(t)
    # A place name is not a watercourse. Everything after the first comma of
    # "Onion Creek, Austin, Texas" names the city and the state, and both
    # appear in every station name here.
    for part in req.reach.split(",")[1:]:
        for w in part.split():
            want.discard(w.lower().strip("."))
    def rank(r: dict) -> tuple:
        nm = r.get("name", "").lower()
        return (0 if want and any(w in nm for w in want) else 1, -r["years"])
    out.sort(key=rank)
    res = {"ok": True, "gauges": out,
           "source": "USGS annual peak streamflow (NWIS); log-Pearson III fit"}
    if len(CACHE) > 24:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = res
    return {**res, "cached": False}


@app.post("/gauges")
def gauges(req: RoadsRequest) -> dict:
    """Real-time river gauges over this box, with the last three days of stage.

    The interface has had nothing live in it. Depth came from a solve, the
    storm came from a preset, and the only thing on screen that moved was the
    scrubber -- which moved through a model, not through the world. USGS
    publishes instantaneous values for every gauged stream in the United
    States on a 15-minute cadence, free and without a key, and Austin has 34 of
    them inside one metro box: Shoal, Waller, Barton, Walnut, Boggy, Bear,
    Slaughter and Onion creeks, and the Colorado.

    Returned in metres and cubic metres per second. USGS serves feet and cubic
    feet per second, and a portal that puts a gauge reading beside a solved
    depth in metres has to convert or it is comparing two different quantities
    with the same label.

    This is United States only, which is a real limit and is said in the
    response rather than left for the caller to discover: `supported` is false
    outside it, and the Richmond study has no equivalent feed.
    """
    key = (round(req.west, 3), round(req.south, 3),
           round(req.east, 3), round(req.north, 3))
    hit = _GAUGE_CACHE.get(key)
    if hit and time.time() - hit["at"] < 300:
        return {**hit["payload"], "cached": True}

    lat = (req.south + req.north) / 2
    lon = (req.west + req.east) / 2
    in_us = ((24 <= lat <= 50 and -125 <= lon <= -66)
             or (51 <= lat <= 72 and -170 <= lon <= -129)
             or (18 <= lat <= 23 and -161 <= lon <= -154))
    if not in_us:
        return {"supported": False, "gauges": [], "cadence_min": None,
                "note": ("Live gauge readings come from USGS, which covers the "
                         "United States. There is no equivalent open feed for "
                         "this area.")}

    base = "https://waterservices.usgs.gov/nwis"
    try:
        rdb = _get(f"{base}/site/?format=rdb&bBox={req.west:.4f},{req.south:.4f},"
                   f"{req.east:.4f},{req.north:.4f}"
                   "&siteType=ST&siteStatus=active&hasDataTypeCd=iv")
    except Exception as exc:                               # noqa: BLE001
        return {"supported": True, "gauges": [], "cadence_min": 15,
                "note": f"USGS did not answer ({str(exc)[:80]})."}

    sites = {}
    for line in rdb.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 6 and parts[1].isdigit():
            sites[parts[1]] = {"site": parts[1], "name": parts[2],
                               "lat": float(parts[4]), "lon": float(parts[5])}
    if not sites:
        return {"supported": True, "gauges": [], "cadence_min": 15,
                "note": "No active real-time stream gauge sits in this area."}

    out = {s: {**v, "stage_m": None, "q_m3s": None, "at": None,
               "trend_cm_h": None, "series": [], "flood_levels_m": None,
               "flood_category": None, "forecast": None,
               "forecast_series": None, "lid": None}
           for s, v in sites.items()}
    try:
        d = json.loads(_get(
            f"{base}/iv/?format=json&sites={','.join(sites)}"
            "&parameterCd=00060,00065&period=P3D", timeout=90.0))
    except Exception:                                      # noqa: BLE001
        d = {"value": {"timeSeries": []}}

    for t in d.get("value", {}).get("timeSeries", []):
        sid = t["sourceInfo"]["siteCode"][0]["value"]
        g = out.get(sid)
        if g is None:
            continue
        code = t["variable"]["variableCode"][0]["value"]
        vals = [v for v in t["values"][0]["value"]
                if v.get("value") not in (None, "", "-999999")]
        if not vals:
            continue
        last = float(vals[-1]["value"])
        g["at"] = vals[-1]["dateTime"]
        if code == "00065":                                # gage height, feet
            g["stage_m"] = round(last * FT_M, 3)
            # Thinned to roughly hourly: three days at 15 minutes is 288
            # points per gauge, which is more than a 300 px sparkline can
            # show and more than the browser should be asked to parse.
            step = max(len(vals) // 72, 1)
            g["series"] = [{"t": v["dateTime"],
                            "m": round(float(v["value"]) * FT_M, 3)}
                           for v in vals[::step]]
            if len(vals) > 4:
                prev = float(vals[-5]["value"]) * FT_M
                g["trend_cm_h"] = round((g["stage_m"] - prev) * 100.0, 1)
        elif code == "00060":                              # discharge, cfs
            g["q_m3s"] = round(last * CFS_M3S, 2)

    # ---- NWS flood categories and forecast ------------------------------
    #
    # USGS says what the river is doing; it does not say what counts as a
    # flood there. "Minor", "moderate" and "major" are set per gauge by the
    # weather service, and without them the headline could only compare a
    # reading with its own recent range -- which is what it did, and said so.
    #
    # NOAA's NWPS publishes them, and the join is by POSITION rather than by
    # id: the list endpoint returns only lid, name and coordinates, and the
    # usgsId and the flood categories live on the per-gauge record. So the
    # list is matched to the USGS sites by distance and only the matches that
    # will actually be shown are fetched in full -- one request each, and
    # there are 28 NWS gauges in the Austin box.
    try:
        nw = json.loads(_get(
            "https://api.water.noaa.gov/nwps/v1/gauges?srid=EPSG_4326"
            f"&bbox.xmin={req.west:.4f}&bbox.ymin={req.south:.4f}"
            f"&bbox.xmax={req.east:.4f}&bbox.ymax={req.north:.4f}", timeout=45.0))
    except Exception:                                      # noqa: BLE001
        nw = {"gauges": []}

    pairs = []
    for ng in nw.get("gauges", []):
        try:
            nlat, nlon = float(ng["latitude"]), float(ng["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        best, bestd = None, 1e9
        for sid, g in out.items():
            dy = (g["lat"] - nlat) * 110.574
            dx = (g["lon"] - nlon) * 111.320 * math.cos(math.radians(nlat))
            d = math.hypot(dx, dy)
            if d < bestd:
                best, bestd = sid, d
        # 400 m: a gauge house and its NWS forecast point are the same
        # structure, and nothing else gauged is that close on these creeks.
        if best and bestd < 0.25:
            pairs.append((bestd, best, ng.get("lid")))
    pairs.sort()

    seen: set = set()
    for _, sid, lid in pairs:
        if sid in seen or not lid or len(seen) >= 18:
            continue
        seen.add(sid)
        try:
            det = json.loads(_get(
                f"https://api.water.noaa.gov/nwps/v1/gauges/{lid}", timeout=30.0))
        except Exception:                                  # noqa: BLE001
            continue
        g = out[sid]
        g["lid"] = lid
        cats = ((det.get("flood") or {}).get("categories") or {})
        unit = ((det.get("flood") or {}).get("stageUnits") or "ft").lower()
        k = FT_M if unit.startswith("ft") else 1.0
        levels = {}
        for name in ("action", "minor", "moderate", "major"):
            v = (cats.get(name) or {}).get("stage")
            # -9999 is NWPS for "not set", and it is a plausible-looking float
            # that would render as a flood level 3 km below the riverbed.
            if isinstance(v, (int, float)) and -500 < v < 500:
                levels[name] = round(v * k, 3)
        g["flood_levels_m"] = levels or None
        st = (det.get("status") or {})
        obs_cat = ((st.get("observed") or {}).get("floodCategory") or "")
        g["flood_category"] = obs_cat if obs_cat not in ("", "unknown") else None
        fc = (st.get("forecast") or {})
        fp = fc.get("primary")
        g["forecast"] = (
            {"stage_m": round(float(fp) * k, 3), "at": fc.get("validTime"),
             "category": fc.get("floodCategory")}
            if isinstance(fp, (int, float)) and fp > -500 else None)
        # The forecast CURVE, not just its peak.
        g["forecast_series"] = _nwps_forecast_series(lid)

    gauges = [g for g in out.values() if g["stage_m"] is not None or g["q_m3s"] is not None]
    gauges.sort(key=lambda g: -(g["q_m3s"] or 0))
    payload = {"supported": True, "gauges": gauges, "cadence_min": 15,
               "source": "USGS NWIS instantaneous values; "
                         "flood levels from NOAA NWPS",
               "note": f"{len(gauges)} gauged streams reporting in this area."}
    _GAUGE_CACHE[key] = {"at": time.time(), "payload": payload}
    if len(_GAUGE_CACHE) > 16:
        _GAUGE_CACHE.pop(next(iter(_GAUGE_CACHE)))
    return {**payload, "cached": False}


@app.post("/roads")
def roads(req: RoadsRequest) -> dict:
    """The road network over this box, cached per box."""
    key = f"roads:{req.west:.4f},{req.south:.4f},{req.east:.4f},{req.north:.4f}"
    if key in CACHE:
        return {"ok": True, "roads": CACHE[key], "cached": True}
    t0 = time.time()
    try:
        fc = fetch_roads(req.west, req.south, req.east, req.north)
    except Exception as exc:
        return {"ok": False,
                "roads": {"type": "FeatureCollection", "features": []},
                "reason": f"OpenStreetMap is not answering right now ({exc}). "
                          f"Flood depth and mitigation options are unaffected."}
    CACHE[key] = fc
    return {"ok": True, "roads": fc, "cached": False,
            "seconds": round(time.time() - t0, 1),
            "network_km": round(
                sum(f["properties"]["len_m"] for f in fc["features"]) / 1000, 1)}


def river_discharge(lat: float, lon: float) -> float | None:
    """Forecast peak river discharge here, m3/s, from Open-Meteo's flood API.

    The same service the forecast panel already uses. Returns None where there
    is no gauge-equivalent reach, which is most small streams.
    """
    url = (f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}"
           f"&longitude={lon}&daily=river_discharge_max&forecast_days=7")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read().decode())
        vals = [v for v in d["daily"]["river_discharge_max"] if v is not None]
        return float(max(vals)) if vals else None
    except Exception:
        return None


def build_forcing(dom, req) -> dict:
    """Turn the chosen driver into things the timestep loop can apply.

    Returns the per-cell source (m/s), the cells a surge is held against, the
    level to hold, and a sentence saying what was actually done -- which the
    interface shows, because a user cannot read a number without knowing what
    produced it.
    """
    z = np.asarray(dom.z, dtype=np.float32)
    rain_ms = req.rain_mm_h / 3.6e6
    out = {"source": None, "sea_mask": None, "sea_level": None,
           "driver": req.driver, "note": "", "q_m3s": None, "stage_m": None}

    if req.driver == "river":
        # dx matters: the channel cap is a physical width, and calling this
        # without a resolution reinstates the cell-count behaviour that made
        # the same river a channel at 200 m and flat ground at 130 m.
        # The MAPPED channel first, the edge heuristic only as a fallback.
        #
        # Asking which boundary cells sit near their own edge's minimum is a
        # guess about which low ground is a river, and it failed on Onion Creek
        # at Austin: the catchment tilts east, so the north-east corner reads
        # lower than the true western inflow and the discharge went into the
        # Colorado valley corner. The creek flooded 1.62 km2 against a 17.9 km2
        # mapped floodplain. OpenStreetMap has the channel, named, and says
        # Onion Creek enters by the west and leaves by the east.
        pair = None
        try:
            pair = forcing.channel_from_network(
                fetch_waterways(dom.bounds["west"], dom.bounds["south"],
                                dom.bounds["east"], dom.bounds["north"]),
                z, dom.bounds, dom.dx)
        except Exception:                                  # noqa: BLE001
            pair = None                                    # Overpass is optional
        inflow = pair[0] if pair else forcing.river_inflow(z, dx_m=dom.dx)
        out["channel_from"] = "mapped network" if pair else "terrain edges"
        _outflow = pair[1] if pair else forcing.river_outflow(z, dx_m=dom.dx)
        if inflow is None:
            out["driver"] = "rain"
            out["note"] = ("No through-channel crosses this box, so the storm "
                           "is applied as rainfall instead.")
            return out
        q = req.river_q_m3s
        if q is None:
            lat = (dom.bounds["south"] + dom.bounds["north"]) / 2
            lon = (dom.bounds["west"] + dom.bounds["east"]) / 2
            q = river_discharge(lat, lon)
        if not q or q <= 0:
            out["driver"] = "rain"
            out["note"] = ("No river discharge is published for this reach, so "
                           "the storm is applied as rainfall instead.")
            return out
        # A DESIGN flood, grounded in the channel rather than in this week's
        # weather.
        #
        # This used to be `forecast_peak * scale`, and that is multiplicative on
        # a quantity that is near zero most of the time. Lismore -- one of the
        # most flood-prone towns in Australia -- returned 0 km2 flooded on a dry
        # week, because its gauge read 1 m3/s and 2x nothing is nothing. The
        # archive scene never showed the bug because it carries a real design
        # event (6,036 m3/s at Cedar Rapids).
        #
        # A flood is water the channel cannot hold, so the scale that matters is
        # BANKFULL discharge: what flows when the channel is full to the top of
        # its banks. Manning gives it from geometry we already measure --
        #
        #     Q_bank = (1/n) * A * R^(2/3) * sqrt(S),  A ~ W*d,  R ~ d
        #
        # -- and the storm presets then say how many times bankfull to send
        # down it, which is how design floods are actually described. The
        # forecast peak is still used when it is the LARGER of the two, so a
        # town that really is flooding today is not talked out of it.
        # Bank level is the FLOODPLAIN, which is low ground. Taking the 60th
        # percentile of the whole box put it up on the hills, the depth pinned
        # to its cap, and bankfull came out at 4,863 m3/s for Lismore -- more
        # than the Wilsons River carried in the 2022 flood of record. A low
        # percentile tracks the floodplain the channel actually sits in.
        bank_m = float(np.percentile(z, 20))
        # Capped at 6 m because a 60 m DEM cannot resolve a channel: the bed
        # elevation it reports is already smoothed across the valley floor, so
        # "depth below bank" is an upper bound on a real incision, not a survey.
        depth_bank = max(0.5, min(6.0, bank_m - inflow.bed_m))
        width_bank = max(inflow.width_cells * dom.dx, dom.dx)
        slope_bank = max(abs(
            (inflow.bed_m - _outflow.bed_m) if _outflow else 1.0
        ) / max(max(z.shape) * dom.dx, 1.0), 1e-4)
        q_bankfull = ((1.0 / 0.035) * (width_bank * depth_bank)
                      * depth_bank ** (2.0 / 3.0) * math.sqrt(slope_bank))
        # A discharge the caller NAMED is already a design flood.
        #
        # The Plan tab sends the recorded peak for a dated event -- 3,823 m3/s
        # at Onion Ck at US Hwy 183 on 31 October 2013 -- and the heading says
        # so. Putting that through the storm multiple below sent 9,557 m3/s
        # down the channel while the panel above the map said 3,823. The
        # number on the page has to be the number that ran.
        #
        # Otherwise: heavy rain ~ bankfull, severe ~ 2x, extreme ~ 4x.
        # Multiples of bankfull are the standard way a design flood is stated,
        # and the storm presets are what select the multiple.
        named = bool(getattr(req, "river_q_is_design", False)) and bool(req.river_q_m3s)
        if named:
            scale = 1.0
            q_design = float(req.river_q_m3s)
        else:
            scale = max(1.0, req.rain_mm_h / 20.0)
            q_design = max(q * scale, q_bankfull * scale)

        # Held as a STAGE, not poured in as a volume.
        #
        # Injecting Q as a source into the few cells of a channel makes water
        # arrive faster than it can spread, and the first attempt reached 112 m
        # deep at Cedar Rapids -- a number with no physical meaning. A real
        # inflow boundary sets the water level the discharge corresponds to.
        # Manning's normal depth for Q over the channel width and the measured
        # bed fall does that with no free parameters:
        #
        #     h = (Q n / (W sqrt(S)))^(3/5)
        outflow = _outflow
        width_m = max(inflow.width_cells * dom.dx, dom.dx)
        fall = (inflow.bed_m - outflow.bed_m) if outflow else 1.0
        span = max(z.shape[0], z.shape[1]) * dom.dx
        slope = max(abs(fall) / span, 1e-4)
        stage = (q_design * 0.035 / (width_m * math.sqrt(slope))) ** 0.6
        stage = float(min(stage, 15.0))          # a river, not a reservoir

        out["sea_mask"] = torch.as_tensor(inflow.mask)
        out["sea_level"] = inflow.bed_m + stage
        out["q_m3s"] = round(q_design, 1)
        out["stage_m"] = round(stage, 2)
        out["note"] = (
            # `:g`, not `:.0f`: the storm presets give scales of 1, 2.5 and 4,
            # and `.0f` printed 2.5 as "2" -- so the note read "2x the forecast
            # peak of 3,000 m3/s" directly beside a value of 7,500. A stated
            # multiple a reader can check in their head has to be the multiple
            # that was applied.
            f"{q_design:,.0f} m3/s on the {inflow.edge} edge "
            f"(channel from the {out['channel_from']}), "
            + ("as given. " if named else f"{scale:g}x ")
            + ("" if named
               else f"the forecast peak of {q:,.0f} m3/s. "
               if q * scale >= q_bankfull * scale else
               f"bankfull for this channel (~{q_bankfull:,.0f} m3/s, estimated "
               f"from DEM geometry, not a gauged rating), since the gauge reads "
               f"only {q:,.0f} m3/s today. ")
            + f"Held as a {stage:.1f} m stage over "
            f"a {width_m:,.0f} m channel at a {slope:.4f} bed slope, draining "
            f"to the {outflow.edge if outflow else 'open'} edge. No rain on the box.")
        return out

    if req.driver == "coastal":
        sea = forcing.sea_edge(z)
        if sea is None:
            out["driver"] = "rain"
            out["note"] = ("No edge of this box is open water, so the storm is "
                           "applied as rainfall instead.")
            return out
        out["sea_mask"] = torch.as_tensor(sea.mask)
        out["sea_level"] = float(req.surge_m)
        out["note"] = (
            f"A {req.surge_m:.1f} m surge held against {sea.width_cells * dom.dx / 1000:.1f} km "
            f"of open water on the {sea.edge} edge. No rain on the box.")
        return out

    out["source"] = rain_ms
    out["note"] = (f"{req.rain_mm_h:.0f} mm/h for {req.storm_hours:.0f} h falling "
                   f"evenly over the whole box, draining at the edges.")
    return out


# An embankment's cross-section. A compacted earth levee is built with a flat
# crest wide enough to drive a maintenance vehicle along and batters at one
# vertical in three horizontal; both are standard and both are what decide how
# much ground the thing actually covers.
CREST_WIDTH_M = 3.048
SIDE_SLOPE = 3.0


def _polyline_dist(yy: "torch.Tensor", xx: "torch.Tensor",
                   pts: list[tuple[float, float]]) -> "torch.Tensor":
    """Distance in cells from every cell to the nearest point on a polyline."""
    out = None
    for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
        dr, dc = r1 - r0, c1 - c0
        l2 = dr * dr + dc * dc
        if l2 < 1e-12:
            seg = torch.sqrt((yy - r0) ** 2 + (xx - c0) ** 2)
        else:
            t = (((yy - r0) * dr + (xx - c0) * dc) / l2).clamp(0.0, 1.0)
            seg = torch.sqrt((yy - (r0 + t * dr)) ** 2 + (xx - (c0 + t * dc)) ** 2)
        out = seg if out is None else torch.minimum(out, seg)
    return out if out is not None else torch.full_like(yy, 1e9)


def _contour_line(z: "torch.Tensor", r: float, c: float,
                  half_cells: float) -> list[tuple[float, float]]:
    """A levee alignment through one point, along the local contour.

    Used only where the caller gave a point and no alignment -- a levee hand
    placed on the map. A wall across the slope is a dam and a wall along it is
    a levee, so the direction is taken from the terrain rather than guessed:
    the gradient of the bed at that cell points downhill, and the alignment
    runs at right angles to it.
    """
    h_, w_ = z.shape
    ri, ci = int(round(r)), int(round(c))
    ri = min(max(ri, 1), h_ - 2)
    ci = min(max(ci, 1), w_ - 2)
    gr = float(z[ri + 1, ci] - z[ri - 1, ci])
    gc = float(z[ri, ci + 1] - z[ri, ci - 1])
    n = math.hypot(gr, gc)
    if n < 1e-9:                       # flat: no contour to follow
        ar, ac = 0.0, 1.0
    else:
        ar, ac = -gc / n, gr / n       # perpendicular to the gradient
    return [(r - ar * half_cells, c - ac * half_cells),
            (r + ar * half_cells, c + ac * half_cells)]


def site_masks(dom, sites: list[dict], max_height_m: float = 2.0) -> "torch.Tensor":
    """One levee per candidate site, swept along the bank it defends.

    This was a Gaussian: `exp(-r**2 / sigma**2)` about the site's centroid. A
    round hill, the same shape whatever the bank beneath it looked like,
    tallest at a single point rather than along a crest, and with a skirt that
    never quite reaches zero. The plan solver was answering a question about a
    mound nobody would build.

    A levee is a line. What is swept here is the real cross-section -- a flat
    crest `CREST_WIDTH_M` wide following the site's own centreline, with
    batters at 1V:`SIDE_SLOPE`H -- along the alignment `/sites` returned for
    that run, or along the local contour where the caller gave only a point.

    The mask is dimensionless and gets multiplied by the design height, so the
    whole plan library still costs one batched solve. The price of that is
    that the footprint is sized once, for `max_height_m`: a shorter wall in
    the same footprint stands on flatter batters than it would be built with.
    The crest, which is the thing that decides whether water passes, is exact
    at every height.
    """
    h_, w_ = dom.z.shape
    yy, xx = torch.meshgrid(torch.arange(h_, dtype=torch.float32),
                            torch.arange(w_, dtype=torch.float32), indexing="ij")
    b = dom.bounds
    z = torch.as_tensor(dom.z, dtype=torch.float32)

    def rc(lon: float, lat: float) -> tuple[float, float]:
        return ((b["north"] - lat) / (b["north"] - b["south"]) * (h_ - 1),
                (lon - b["west"]) / (b["east"] - b["west"]) * (w_ - 1))

    # A levee is narrower than a cell, and a cell is the narrowest thing this
    # model can block with.
    #
    # At 60 m a 3 m crest with 2 m batters is 15 m wide: a quarter of a cell.
    # Rasterised honestly it raised almost nothing -- measured, two 2 m walls
    # on Onion Creek moved the flooded area by 0.0004%, which is a levee that
    # is not there rather than a levee that does not work. So the crest is
    # floored at half a cell either side of the alignment, which is the
    # standard sub-grid treatment: the wall raises the bed of every cell it
    # passes through, because blocking part of a cell is not something a
    # finite-volume scheme can express. The cost is that the modelled
    # footprint is a cell wide where the built one is 15 m, so the plan
    # over-states how much ground the embankment occupies and states the
    # crest, which is what decides whether water passes, exactly.
    crest_half = max(0.5 * CREST_WIDTH_M / dom.dx, 0.5)
    shoulder = max(SIDE_SLOPE * float(max_height_m) / dom.dx, 0.25)

    out = []
    for s in sites:
        pts = [rc(float(p[0]), float(p[1])) for p in (s.get("outline") or [])]
        if len(pts) < 2:
            r, c = rc(float(s["lon"]), float(s["lat"]))
            # `width_m` is half the alignment's length -- the same convention
            # the cost model uses, where `lengths` is `width_m * 2`.
            pts = _contour_line(z, r, c, float(s.get("width_m", 600.0)) / dom.dx)
        d = _polyline_dist(yy, xx, pts)
        out.append(((crest_half + shoulder - d) / shoulder).clamp(0.0, 1.0))
    return torch.stack(out) if out else torch.zeros(0, h_, w_)


# One unit of a structure, so a plan's intensity means something buildable.
#
# A gate at intensity 1.0 is one 48-inch (1.22 m) barrel, which is the size
# the USACE gated-outfall cost in `lib/rates` is quoted for; 2.0 is a twin
# barrel. A pump at intensity 1.0 is one cubic metre a second of firm
# capacity, which is the unit its cost is quoted in too. So the intensity the
# planner solves over and the intensity it prices are the same number.
GATE_UNIT_AREA_M2 = math.pi * (1.22 / 2.0) ** 2
PUMP_UNIT_Q_M3S = 1.0
# Wet-well hysteresis. A pump that starts and stops at one level chatters
# against its own drawdown; these are the start and stop depths above the
# intake, and they are operating settings rather than measurements.
PUMP_ON_M = 0.30
PUMP_OFF_M = 0.10


def build_structures(dom, sites: list[dict], intensity: "torch.Tensor"):
    """Gates and pumps for a batch of plans, from the sites they sit on.

    A levee is a bed perturbation and can only keep water OUT of a perimeter.
    A gate lets the land behind one drain when the river drops, and a pump
    lifts water that is already inside. Those are different instruments and
    the planner needs all three, because the pluvial case -- where a quarter
    of the rain falls inside the area being protected -- is one no levee
    placement can answer.

    Each structure sits ACROSS the bank it is placed on: inside cell on the
    landward side of the site's own section, outside cell on the river side,
    a cell and a half either way. That geometry comes out of `/sites`, which
    already decided which end of the section is landward by counting solved
    water on each side.

    `intensity` is (b, k) -- one row per candidate plan, one column per
    measure, the same matrix that drives the levee heights. A plan that does
    not build a structure sets its column to zero, which shuts the gate and
    stops the pump, so the whole library still solves in one timestep loop.
    """
    h_, w_ = dom.z.shape
    b = dom.bounds
    z0 = torch.as_tensor(dom.z, dtype=torch.float32)

    def rc(lon: float, lat: float) -> tuple[float, float]:
        return ((b["north"] - lat) / (b["north"] - b["south"]) * (h_ - 1),
                (lon - b["west"]) / (b["east"] - b["west"]) * (w_ - 1))

    def across(s: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Cells a cell and a half either side of the bank, landward first."""
        ln = s.get("line") or {}
        a, o = ln.get("a"), ln.get("b")
        if not a or not o:
            return None
        ra, ca = rc(float(a["lon"]), float(a["lat"]))
        rb, cb = rc(float(o["lon"]), float(o["lat"]))
        dr, dc = rb - ra, cb - ca
        n = math.hypot(dr, dc)
        if n < 1e-6:
            return None
        dr, dc = dr / n, dc / n
        rm, cm = rc(float(s["lon"]), float(s["lat"]))
        def clamp(r: float, c: float) -> tuple[int, int]:
            return (int(min(max(round(r), 0), h_ - 1)),
                    int(min(max(round(c), 0), w_ - 1)))
        return clamp(rm - dr * 1.5, cm - dc * 1.5), clamp(rm + dr * 1.5, cm + dc * 1.5)

    g_in, g_out, g_col, g_inv = [], [], [], []
    p_in, p_out, p_col, p_sump = [], [], [], []
    for i, s in enumerate(sites):
        kind = str(s.get("kind", "levee"))
        if kind not in ("gate", "pump"):
            continue
        pair = across(s)
        if pair is None:
            continue
        (ri, ci), (ro, co) = pair
        if kind == "gate":
            g_in.append([ri, ci]); g_out.append([ro, co]); g_col.append(i)
            # The sill sits on the ground it drains. Nothing below the invert
            # leaves through a gravity outfall, which is exactly why a gate
            # alone does not empty a bowl and a pump does.
            g_inv.append(float(z0[ri, ci]))
        else:
            p_in.append([ri, ci]); p_out.append([ro, co]); p_col.append(i)
            p_sump.append(float(z0[ri, ci]))

    if not g_col and not p_col:
        return None

    st = Structures()
    if g_col:
        cols = torch.tensor(g_col, dtype=torch.long)
        amp = intensity[:, cols]                           # (b, G)
        st.gate_in = torch.tensor(g_in, dtype=torch.long)
        st.gate_out = torch.tensor(g_out, dtype=torch.long)
        st.gate_area_m2 = amp * GATE_UNIT_AREA_M2
        st.gate_invert_m = torch.tensor(g_inv, dtype=torch.float32)
        st.gate_open = amp > 0
    if p_col:
        cols = torch.tensor(p_col, dtype=torch.long)
        amp = intensity[:, cols]                           # (b, P)
        st.pump_in = torch.tensor(p_in, dtype=torch.long)
        st.pump_out = torch.tensor(p_out, dtype=torch.long)
        st.pump_q_m3s = amp * PUMP_UNIT_Q_M3S
        st.pump_sump_m = torch.tensor(p_sump, dtype=torch.float32)
        st.pump_on_m = torch.full((len(p_col),), PUMP_ON_M)
        st.pump_off_m = torch.full((len(p_col),), PUMP_OFF_M)
    return st


def solve_beds(dom, req, z: "torch.Tensor", fx: dict,
               snap_hours: Sequence[float] | None = None,
               structures=None,
               ) -> tuple["torch.Tensor", int, "torch.Tensor | None"]:
    """Peak depth for a batch of beds, one timestep loop for all of them.

    Extracted so the plan the greedy search RECOMMENDS can be solved as well as
    the library it was chosen from. That matters for a claim about a road: the
    route experiment measures the surrogate's error on the deepest water along a
    route at a median 0.16-0.33 m, which is the size of the decision threshold
    itself, so a route verdict taken from the surrogate cannot be stated with a
    useful margin (see `eval/run_route_margin.py` and section 5.13). Solving the
    one extra bed costs a fraction of the library batch and replaces a predicted
    answer with a computed one.
    """
    rows = z.mean(dim=2).mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (z.shape[1] * dom.dx))), 1e-5)
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=slope, dt_max=300.0)
    solver = FloodSolver(z, cfg)
    solver.structures = structures
    b = z.shape[0]
    h = torch.zeros(b, z.shape[1], z.shape[2])
    qx, qy = solver.zeros_flux(b)
    peak = h.clone()
    src, sea_mask, sea_level = fx["source"], fx["sea_mask"], fx["sea_level"]
    # Snapshots are the INSTANTANEOUS depth at a few times, not the running
    # peak. Passability at 14:00 is the water on the road at 14:00; a peak
    # field answers "was this ever cut", which refuses roads that are fine for
    # the next hour and is the wrong question for someone deciding to drive.
    # Named `snap_at`, not `want`: the surge branch below already binds `want`
    # to the boundary stage field, and the first river step overwrote the
    # schedule with an (H, W) tensor. Same shadowing that made the SafeDreamer
    # arm score the wrong critic head -- a local name reused inside a long loop
    # is a collision waiting for the one driver that reaches it.
    snap_at = sorted(float(x) * 3600.0 for x in (snap_hours or []))
    snaps: list["torch.Tensor"] = []
    nxt = 0
    tt, dur, steps = 0.0, req.run_hours * 3600.0, 0
    while tt < dur:
        dt = min(solver.adaptive_dt(h), dur - tt)
        if dt <= 0:
            break
        # Forcing is on for the storm and off afterwards, so the run shows the
        # recession as well as the peak.
        on = tt < req.storm_hours * 3600.0
        r = (src if on else 0.0) if src is not None else 0.0
        h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=z)
        if sea_mask is not None and on:
            # A boundary stage, held not poured: set depth so the free surface
            # sits at the given level, and only ever raise it, so the boundary
            # never drains the land. Used for both a coastal surge and a river
            # inflow, which differ only in where the level comes from.
            want = torch.clamp(sea_level - z, min=0.0)
            h = torch.where(sea_mask.expand_as(h), torch.maximum(h, want), h)
        peak = torch.maximum(peak, h)
        tt += dt
        steps += 1
        while nxt < len(snap_at) and tt >= snap_at[nxt]:
            snaps.append(h.clone())
            nxt += 1
    # A run that ends before the last requested time still has to return a
    # schedule of the right length, or the client silently reads a shorter
    # horizon than it asked about.
    while nxt < len(snap_at):
        snaps.append(h.clone())
        nxt += 1
    return peak, steps, (torch.stack(snaps, dim=1) if snaps else None)


@app.post("/plan")
def plan_anywhere(req: PlanRequest) -> dict:
    """The whole framework, on terrain that is nowhere in the repository.

    The district planner reads a scenario library someone solved offline. That
    is why it only worked in one valley. Here the library is BUILT ON DEMAND:
    every candidate plan shares one batched solve, because the timestep loop is
    the expensive part and the bed is the only thing that differs between
    scenarios, so twenty plans cost roughly what one does.

    Then the same code the district uses fits the surrogate and takes the
    conformal margin -- `pspe.plan.surrogate`, not a copy of it, because a
    guarantee that means one thing in Richmond and another here would be worse
    than no guarantee (rule 13).
    """
    key = "plan:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}
    if len(req.sites) < 2:
        return {"ok": False, "reason":
                "Place at least two candidate levee sites. Four or more are "
                "needed before a 90% margin can be calibrated."}
    # One bed per candidate combination: the longest thing this service does,
    # and the one most likely to outlive the browser's patience. Shared, so
    # asking again joins the run instead of starting a second one.
    res = run_shared(key, lambda: _solve_plan(req, key))
    if res is None:
        return {"ok": False, "pending": True,
                "elapsed_s": round(pending_for(key) or 0.0),
                "reason": "Still solving every combination. Nothing is lost — "
                          "ask again and you will join the same run."}
    return res


def _solve_plan(req: PlanRequest, key: str) -> dict:
    t0 = time.time()
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name, source=TERRAIN)
    z0 = torch.as_tensor(dom.z, dtype=torch.float32)
    k = len(req.sites)

    # Three kinds of measure, one intensity vector.
    #
    # A levee's intensity is its crest height in metres. A gate's is how many
    # 48-inch barrels it has; a pump's is how many cubic metres a second it
    # lifts. The enumeration, the surrogate, the conformal margin and the
    # greedy search all see one dimensionless number per measure and do not
    # need to know which kind it is -- which is what lets a gate and a pump
    # into the plan library without a second code path. What differs is only
    # where the number is applied: a levee changes the BED, a gate and a pump
    # are exchanges between two named cells.
    kinds = [str(x.get("kind", "levee")) for x in req.sites]
    lev = [i for i, kd in enumerate(kinds) if kd not in ("gate", "pump")]

    plans = surrogate.enumerate_plans(k, req.max_height_m)
    P = torch.tensor(plans, dtype=torch.float32)
    z = z0[None].expand(P.shape[0], -1, -1).clone()
    if lev:
        masks = site_masks(dom, [req.sites[i] for i in lev], req.max_height_m)
        z = z + (P[:, lev, None, None] * masks[None]).sum(1)
    structures = build_structures(dom, req.sites, P)

    fx = build_forcing(dom, req)
    peak, steps, _ = solve_beds(dom, req, z, fx, structures=structures)
    b = z.shape[0]

    # What the plan is bought to change.
    #
    # This was a circle of invented radius: 2,000 m from one caller and
    # 2,500 m from another, so the SAME plan on the SAME terrain reported a
    # different percentage depending which tab opened it. Nothing on the
    # ground corresponds to it either -- a disc centred between two levees
    # contains the channel and the hills, and their depth dominates the mean
    # and dilutes whatever the walls did.
    #
    # Where the measures carry the alignment `/sites` found, the protected
    # set is derived instead: the cells LANDWARD of those alignments and
    # within the length of the section that was cut across them. Landward is
    # not guessed -- `_section_line` already decided which end is dry by
    # counting solved water on each side, and `line.a` is that end. The disc
    # survives only for a levee dropped on the map by hand, which carries no
    # alignment, and the response says which was used.
    h_, w_ = dom.z.shape
    yy, xx = torch.meshgrid(torch.arange(h_, dtype=torch.float32),
                            torch.arange(w_, dtype=torch.float32), indexing="ij")
    bb = dom.bounds

    def to_rc(lon: float, lat: float) -> tuple[float, float]:
        return ((bb["north"] - lat) / (bb["north"] - bb["south"]) * (h_ - 1),
                (lon - bb["west"]) / (bb["east"] - bb["west"]) * (w_ - 1))

    aligned = [sp for sp in req.sites
               if (sp.get("line") or {}).get("a") and (sp.get("outline") or [])]
    protect_basis = "whole domain"
    sel = torch.zeros(dom.z.shape, dtype=torch.bool)
    if aligned:
        for sp in aligned:
            pts = [to_rc(float(q[0]), float(q[1])) for q in sp["outline"]]
            if len(pts) < 2:
                continue
            d = _polyline_dist(yy, xx, pts)
            ln = sp["line"]
            ra, ca = to_rc(float(ln["a"]["lon"]), float(ln["a"]["lat"]))
            rb, cb = to_rc(float(ln["b"]["lon"]), float(ln["b"]["lat"]))
            da = torch.sqrt((yy - ra) ** 2 + (xx - ca) ** 2)
            db = torch.sqrt((yy - rb) ** 2 + (xx - cb) ** 2)
            reach = float(ln.get("length_m", 800.0)) / 2.0 / dom.dx
            sel |= (d <= reach) & (da < db)
        if bool(sel.any()):
            protect_basis = (f"the ground landward of {len(aligned)} alignment"
                             f"{'' if len(aligned) == 1 else 's'}, within "
                             f"{float(aligned[0]['line'].get('length_m', 800.0)) / 2:.0f} m "
                             "of each")
    if not bool(sel.any()) and req.protect:
        r, c = to_rc(float(req.protect["lon"]), float(req.protect["lat"]))
        rad = float(req.protect.get("radius_m", 2000.0)) / dom.dx
        sel = (((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2)
        protect_basis = (f"a {float(req.protect.get('radius_m', 2000.0)):.0f} m "
                         "circle, because these measures carry no alignment")
    if not bool(sel.any()):
        sel = torch.ones(dom.z.shape, dtype=torch.bool)
        protect_basis = "the whole box, because nothing named a smaller area"

    core = (peak * sel[None]).sum(dim=(1, 2)) / float(sel.sum())
    base = float(core[0])
    # Positive = flooding reduced where it matters.
    reduction = [(base - float(c)) / base * 100.0 if base > 1e-9 else 0.0
                 for c in core]

    def measure_cost(j: int, x: float) -> float:
        """What one measure costs at one intensity.

        The rate and the model come from the client, which is the only place
        they are written down (`lib/rates`): the service used to carry its own
        bare `2600.0` and so did two other files, so the figure a user saw
        depended on which of three copies was reached. An embankment is priced
        from its real trapezoidal volume rather than a flat rate per metre of
        height, because the volume goes as `crest*H + slope*H^2` and a flat
        rate therefore understates a tall wall and overstates a short one.
        """
        sp = req.sites[j]
        model = str(sp.get("cost_model", "earth"))
        unit = float(sp.get("unit_cost", 0.0))
        length = float(sp.get("width_m", 600.0)) * 2.0
        if model == "earth":
            return length * (CREST_WIDTH_M * x + SIDE_SLOPE * x * x) * unit
        if model == "wall":
            return length * x * unit
        return x * unit                       # a gate or a pump, per unit

    alpha = [0.0] * k
    for i, pl in enumerate(plans):
        act = [j for j, hh in enumerate(pl) if hh > 0]
        if len(act) == 1:
            alpha[act[0]] = reduction[i]

    # Every option that was solved, not only the one the greedy search chose.
    # The batch already paid for these -- the timestep loop ran them all -- and
    # returning just the winner made the Anywhere tab look like a weaker tool
    # than the district one when it had done the same work. Shaped to match the
    # district's option records so one component renders both.
    cell_km2 = (dom.dx ** 2) / 1e6
    wet = (peak > WET_M).sum(dim=(1, 2))
    options = [
        {
            "id": "base" if not any(h > 0 for h in pl) else
                  "o" + "".join(f"{j}@{h:g}" for j, h in enumerate(pl) if h > 0),
            "heights": list(pl),
            "cost_usd": sum(measure_cost(j, pl[j]) for j in range(k)),
            "core_reduction_pct": reduction[i],
            "area_flooded_km2": round(float(wet[i]) * cell_km2, 2),
            "peak_depth_m": round(float(peak[i].max()), 2),
            # The solved field for THIS option, quantised the way `inspect`
            # quantises the baseline. The solver already computes one field per
            # plan and we returned only peak[0], so choosing an option in the
            # portal moved a radio button and nothing else: there was no flood
            # to draw for it. One array per option serves all three uses the
            # client has -- the body of water, the paint on the terrain, and the
            # click-anywhere depth query -- which is why it is the grid and not
            # a picture.
            "depth_u8": base64.b64encode(
                (np.clip(np.nan_to_num(peak[i].numpy()) / DEPTH_VMAX, 0.0, 1.0)
                 * 255).astype(np.uint8).tobytes()).decode(),
        }
        for i, pl in enumerate(plans)
    ]

    f = surrogate.fit(alpha, plans, reduction, req.max_height_m, req.delta)
    S = f["saturation_S"]

    def cost(hv):
        return sum(measure_cost(i, hv[i]) for i in range(k))

    # Greedy allocation, same rule the district planner uses.
    hv, trace = [0.0] * k, []
    step_m = 0.5
    while True:
        cur = surrogate.effect(alpha, hv, req.max_height_m, S)
        spent_now = cost(hv)
        best = (1e-9, -1, 0.0)
        for i in range(k):
            if hv[i] + step_m > req.max_height_m + 1e-9:
                continue
            trial = list(hv); trial[i] += step_m
            spent = cost(trial)
            if spent > req.budget_usd:
                continue
            e = surrogate.effect(alpha, trial, req.max_height_m, S)
            per = (e - cur) / max(spent - spent_now, 1.0)
            if e > cur and per > best[0]:
                best = (per, i, e)
        if best[1] < 0:
            break
        hv[best[1]] += step_m
        trace.append({"site": best[1], "to_m": hv[best[1]], "effect": best[2]})

    eff = surrogate.effect(alpha, hv, req.max_height_m, S)
    active = sum(1 for x in hv if x > 0)
    band = f["band"] if active > 1 else 0.0
    attribution = []
    for i, hh in enumerate(hv):
        if hh <= 0:
            continue
        without = list(hv); without[i] = 0.0
        marginal = eff - surrogate.effect(alpha, without, req.max_height_m, S)
        attribution.append({
            "site": i, "height_m": hh, "marginal_pct": marginal,
            "alone_pct": alpha[i] * (hh / req.max_height_m),
            "overlap_pct": alpha[i] * (hh / req.max_height_m) - marginal,
            "cost_usd": measure_cost(i, hh),
        })
    attribution.sort(key=lambda a: -a["marginal_pct"])

    # The recommended plan, solved rather than predicted. The greedy search
    # walks the surrogate, so `hv` is almost never one of the library's beds;
    # the depth claim about it carries a conformal margin, but a claim about a
    # ROAD cannot (section 5.13: the surrogate's error on the deepest water
    # along a route is the size of the passability threshold). One more bed
    # through the same loop makes that answer a solve.
    chosen_u8 = None
    chosen_reduction = 0.0
    if any(h > 0 for h in hv) and not any(list(pl) == list(hv) for pl in plans):
        z_sel = z0[None] + (torch.tensor([hv], dtype=torch.float32)[:, :, None, None]
                            * masks[None]).sum(1)
        peak_sel, _, _ = solve_beds(dom, req, z_sel, fx)
        chosen_u8 = base64.b64encode(
            (np.clip(np.nan_to_num(peak_sel[0].numpy()) / DEPTH_VMAX, 0.0, 1.0)
             * 255).astype(np.uint8).tobytes()).decode()
        # What the plan actually does, next to what the surrogate said it would.
        # The surrogate is an extrapolation at the recommended point -- it is
        # fitted to the library and the greedy search walks it past every bed in
        # it -- so it can and does return a reduction above 100%, which is not a
        # bold claim but an out-of-range number. Measuring it is free once the
        # bed is solved, and the gap is the honest headline.
        core_sel = float((peak_sel[0] * sel).sum() / float(sel.sum()))
        chosen_reduction = ((base - core_sel) / base * 100.0) if base > 1e-9 else 0.0

    out = {
        "ok": True,
        "name": req.name,
        "heights": hv,
        # Present when the recommended plan was solved in its own right, which
        # is what a route verdict about it has to be read from.
        "chosen_depth_u8": chosen_u8,
        "chosen_solved": chosen_u8 is not None,
        # Solved, not predicted. Where both exist the solved one is the number
        # to report and the gap is the surrogate's error at the point the search
        # chose, which is the one point the surrogate was never fitted at.
        "chosen_reduction_pct": chosen_reduction if chosen_u8 is not None else None,
        "surrogate_overclaim_pct": (
            None if chosen_u8 is None else eff - chosen_reduction),
        "cost_usd": cost(hv),
        "budget_usd": req.budget_usd,
        "reduction_pct": eff,
        "guaranteed_pct": eff - band,
        "band_pct": band,
        "delta": req.delta,
        "confidence_pct": round(100 * (1 - req.delta)),
        "band_attainable": f["band_attainable"],
        "band_note": f["band_note"],
        "n_calibration_runs": f["n_calibration_runs"],
        "fit_rmse": f["fit_rmse"],
        "saturation_S": S,
        "alone_pct": alpha,
        "harmful": [{"site": i, "alone_pct": a} for i, a in enumerate(alpha) if a < 0],
        "attribution": attribution,
        "steps": trace,
        "scenarios_solved": b,
        # Over WHAT the reduction was averaged. The percentage is meaningless
        # without it, and it used to be an undisclosed circle.
        "protect_basis": protect_basis,
        "protect_km2": round(float(sel.sum()) * (dom.dx ** 2) / 1e6, 2),
        "driver": fx["driver"],
        "driver_note": fx["note"],
        "river_q_m3s": fx["q_m3s"],
        "river_stage_m": fx.get("stage_m"),
        "terrain": forcing.describe(np.asarray(dom.z, dtype=np.float32), dom.dx),
        "options": options,
        "base_depth_m": base,
        "base_area_km2": options[0]["area_flooded_km2"],
        "base_peak_depth_m": options[0]["peak_depth_m"],
        "max_height_m": req.max_height_m,
        # The alignment length each measure is costed over. `lengths` was a
        # list built beside the old flat-rate model and it outlived it by one
        # reference, which `/plan` then raised a NameError on -- caught by
        # verify-kit rather than by anything in the service, because the
        # response is only assembled after a two-minute solve.
        "crest_lengths_m": [float(x.get("width_m", 600.0)) * 2.0
                            for x in req.sites],
        "bounds": {"west": req.west, "south": req.south,
                   "east": req.east, "north": req.north},
        "depth_png": depth_png(peak[0].numpy(), DEPTH_VMAX),
        "terrain_png": terrain_png(dom.z),
        **roads_and_grids(dom, peak[0].numpy(), req.west, req.south,
                          req.east, req.north),
        "timing": {"seconds": round(time.time() - t0, 1), "steps": steps},
        "verify": ("Every number here is a solver run on this terrain: the "
                   "single-measure effects exactly, the combinations through a "
                   "surrogate fitted to them and bounded by a conformal margin."),
    }
    CACHE[key] = out
    return {**out, "cached": False}


class RouteRequest(BaseModel):
    """Get me from here to there, through a flood that is still moving."""
    origin: dict                      # {"lat":.., "lon":.., "name":..}
    destination: dict
    name: str = "Route"
    # What the vehicle can drive through. An ambulance, a bus and a hatchback
    # are three different questions about the same water.
    vehicle_depth_m: float = Field(0.25, ge=0.05, le=2.0)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    driver: str = Field("river", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    surge_m: float = Field(2.5, ge=0.0, le=10.0)
    # How many times the schedule is sampled. The flood is resolved to this,
    # and the deadline can only ever be as sharp as the sampling.
    snapshots: int = Field(13, ge=2, le=25)
    depart_step_min: float = Field(30.0, ge=5.0, le=120.0)
    dx: Optional[float] = None        # None -> chosen from how far apart they are


# Terrain, the solve and the road network, kept apart from the routing.
#
# The whole response is already cached by request hash, which helps only an
# identical request. But asking the same trip for a truck instead of a car
# changes nothing physical -- same terrain, same flood, same roads, a different
# threshold -- and re-solving it was two minutes for a number the service
# already had everything to compute. Keyed on what each stage actually depends
# on, so changing the vehicle re-routes and nothing else.
SOLVE_CACHE: dict[str, dict] = {}
ROADS_CACHE: dict[str, dict] = {}
_CACHE_MAX = 6


def _cache_put(store: dict, key: str, value) -> None:
    if len(store) >= _CACHE_MAX:
        store.pop(next(iter(store)))
    store[key] = value


def _km_between(a: dict, b: dict) -> float:
    lat = math.radians((float(a["lat"]) + float(b["lat"])) / 2)
    dx = (float(b["lon"]) - float(a["lon"])) * 111.320 * math.cos(lat)
    dy = (float(b["lat"]) - float(a["lat"])) * 110.540
    return math.sqrt(dx * dx + dy * dy)


class SectionRequest(BaseModel):
    """An elevation section across the floodplain, and what a levee on it buys.

    The portal's levee planner works at six surveyed sites with heights chosen
    in 0.5 m steps. This answers a different question an engineer actually asks
    first: along THIS line, where is the ground high enough to build on, how
    high would the crest have to be, and does it clear the flood we solved?

    Everything returned is measured. The elevation profile is sampled from the
    Copernicus DEM the solve itself ran on, and the water level is the solved
    peak along the same line -- not an assumed design stage.
    """
    a: dict                            # {"lat":.., "lon":..} one end of the line
    b: dict                            # the other end
    name: str = "Section"
    samples: int = Field(120, ge=16, le=400)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    driver: str = Field("river", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    dx: Optional[float] = None


def _street_vmax(d: np.ndarray) -> float:
    """The top of the depth ramp for a street block.

    Not the maximum, and not a high percentile of the whole block either. A
    block that touches the creek contains its channel, and the channel is
    fifteen metres deep: a ramp scaled to that puts every depth anyone walks
    through into its first tenth, and the neighbourhood draws as one flat
    blue. Taken over the WET cells only, at the 90th percentile, so the ramp
    spans the water on the streets rather than the water in the river.
    """
    wet = d[d > WET_M]
    if wet.size < 8:
        return 1.0
    return float(min(max(np.percentile(wet, 90), 0.5), 6.0))


class BuildingsRequest(BaseModel):
    """Buildings in a block, with the solved water against each floor."""
    west: float
    south: float
    east: float
    north: float
    name: str = "Street"
    dx: float = Field(20.0, ge=5.0, le=60.0)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    driver: str = Field("rain", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    surge_m: float = Field(2.5, ge=0.0, le=10.0)
    snapshots: int = Field(13, ge=2, le=25)
    max_buildings: int = Field(500, ge=1, le=2000)
    parent_km: float = Field(12.0, ge=2.0, le=40.0)
    parent_dx: float = Field(40.0, ge=20.0, le=120.0)


class SitesRequest(BaseModel):
    """Where a levee would pay, taken from the terrain rather than clicked."""
    west: float
    south: float
    east: float
    north: float
    name: str = "Area"
    dx: float = Field(40.0, ge=20.0, le=120.0)
    rain_mm_h: float = Field(50.0, ge=0.0, le=300.0)
    storm_hours: float = Field(6.0, ge=0.5, le=48.0)
    run_hours: float = Field(12.0, ge=1.0, le=72.0)
    driver: str = Field("rain", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None
    # True when `river_q_m3s` is a design discharge the caller named -- a
    # recorded flood peak or a fitted quantile -- rather than a base figure
    # for the storm presets to multiply. See `build_forcing`.
    river_q_is_design: bool = False
    surge_m: float = Field(2.5, ge=0.0, le=10.0)
    max_sites: int = Field(6, ge=1, le=12)
    shortfall_m: float = Field(2.0, ge=0.2, le=8.0)


def _centreline(rs, cs, bounds: dict, shape: tuple, bins: int = 24) -> list[list[float]]:
    """The alignment a wall would follow along a bank run, in lon/lat order.

    `cells` comes out of the flood fill in the order the stack happened to pop
    them, which is a scatter and not a path: joined up it zig-zags across the
    bank and back, and decimating it does not help. Projecting the cells onto
    the run's own principal axis -- the same axis `_section_line` cuts across
    -- and averaging within bins along that axis gives the line itself.

    This is what the map draws as the site's outline and what the plan solver
    sweeps a levee along, so a scatter here is not cosmetic: it was the
    geometry the whole plan was costed and solved against.
    """
    h_, w_ = shape
    r0, c0 = float(rs.mean()), float(cs.mean())
    dr, dc = rs - r0, cs - c0
    cov = np.array([[float((dr * dr).sum()), float((dr * dc).sum())],
                    [float((dr * dc).sum()), float((dc * dc).sum())]])
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    t = dr * axis[0] + dc * axis[1]
    lo, hi = float(t.min()), float(t.max())
    n = max(2, min(bins, int(round((hi - lo) / 2.0)) + 1))
    edges = np.linspace(lo, hi + 1e-9, n + 1)
    pts: list[tuple[float, float]] = []
    for i in range(n):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if not m.any():
            continue
        pts.append((float(rs[m].mean()), float(cs[m].mean())))
    if len(pts) < 2:
        pts = [(r0, c0), (r0 + 1e-3, c0 + 1e-3)]
    return [[round(float(bounds["west"] + (bounds["east"] - bounds["west"])
                         * c / max(w_ - 1, 1)), 6),
             round(float(bounds["north"] - (bounds["north"] - bounds["south"])
                         * r / max(h_ - 1, 1)), 6)]
            for r, c in pts]


def _section_line(rs, cs, wet, bounds: dict, shape: tuple, dx: float,
                  half_m: float = 400.0) -> dict:
    """A line across a bank run, landward end first.

    The run's principal axis is the bank; the section crosses it. Orientation
    is decided by counting solved water on each side, because the wet side is
    the river side by definition -- using elevation instead gets it backwards
    wherever a levee already exists, which is precisely where one is being
    considered.
    """
    H, W = shape
    r0, c0 = float(rs.mean()), float(cs.mean())
    # Principal axis by the covariance of the run's own cells.
    dr, dc = rs - r0, cs - c0
    cov = np.array([[float((dr * dr).sum()), float((dr * dc).sum())],
                    [float((dr * dc).sum()), float((dc * dc).sum())]])
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]                 # (drow, dcol) of the bank
    perp = np.array([-axis[1], axis[0]])                 # across it
    n = half_m / max(dx, 1e-6)

    def side_wet(sign: float) -> int:
        tot = 0
        for t in np.linspace(0.15, 1.0, 8):
            rr = int(round(r0 + sign * perp[0] * n * t))
            cc = int(round(c0 + sign * perp[1] * n * t))
            if 0 <= rr < H and 0 <= cc < W and wet[rr, cc]:
                tot += 1
        return tot

    # A is the drier end.
    sign_a = -1.0 if side_wet(-1.0) <= side_wet(1.0) else 1.0

    def at(sign: float) -> dict:
        rr = min(max(r0 + sign * perp[0] * n, 0.0), H - 1.0)
        cc = min(max(c0 + sign * perp[1] * n, 0.0), W - 1.0)
        return {
            "lat": round(float(bounds["north"] - (bounds["north"] - bounds["south"])
                               * rr / max(H - 1, 1)), 6),
            "lon": round(float(bounds["west"] + (bounds["east"] - bounds["west"])
                               * cc / max(W - 1, 1)), 6),
        }

    return {"a": at(sign_a), "b": at(-sign_a),
            "length_m": round(2 * half_m)}


_PENDING: dict = {}
_PENDING_LOCK = threading.Lock()
# How long one HTTP call will sit on a solve before handing back "still
# working". Four minutes, because the thing waiting at the other end is a
# browser whose fetch stops listening for response headers at five.
SHARE_WAIT_S = 240.0


def run_shared(key: str, fn, wait_s: float = SHARE_WAIT_S):
    """Run `fn` once per key, however many callers ask for it.

    A site search on Onion Creek is two to five minutes of timestepping, and
    the browser at the other end stops waiting for response headers at five
    — a limit in `fetch` itself, not something the page can raise. Past it
    the user sees an error over work that is still running and will finish.

    Retrying is the obvious answer and the wrong one on its own: a second
    POST with the same body started a SECOND solve, so two impatient clicks
    doubled the load and neither call got there sooner. So the work is keyed.
    The first caller starts it on a thread; every caller, first or not, waits
    on the same event for up to `wait_s` and then either takes the result or
    is told it is still coming. Asking again is free and joins the same run.

    Returns the value, or `None` when it is not ready yet. Raises whatever
    the work raised, to whoever is waiting when it fails.
    """
    with _PENDING_LOCK:
        slot = _PENDING.get(key)
        fresh = slot is None
        if fresh:
            slot = {"event": threading.Event(), "value": None, "error": None,
                    "started": time.time()}
            _PENDING[key] = slot

    if fresh:
        def work() -> None:
            try:
                slot["value"] = fn()
            except BaseException as exc:                   # noqa: BLE001
                slot["error"] = exc
            finally:
                slot["event"].set()
        threading.Thread(target=work, daemon=True, name=f"solve:{key}").start()

    if not slot["event"].wait(wait_s):
        return None
    with _PENDING_LOCK:
        _PENDING.pop(key, None)
    if slot["error"] is not None:
        raise slot["error"]
    return slot["value"]


def pending_for(key: str) -> float | None:
    """Seconds the shared run for `key` has been going, if one is."""
    with _PENDING_LOCK:
        slot = _PENDING.get(key)
    return None if slot is None else time.time() - slot["started"]


@app.post("/sites")
def sites(req: SitesRequest) -> dict:
    """Candidate levee alignments, from where the ground only just fails.

    The district's six sites were surveyed by hand and the Anywhere tab made
    the user click them, which asks a planner to guess at the one question the
    terrain can answer for them. A levee pays where the bank is ALMOST high
    enough: raising ground that is four metres under water buys nothing for the
    money, and raising ground already above the flood buys nothing at all. The
    band in between is short, and it is computable.

    So: solve the baseline, take the water surface, and find the dry cells that
    touch wet ones and sit within `shortfall_m` of being overtopped. Those are
    the places a wall of a metre or two changes the answer. Contiguous runs of
    them become alignments, each carrying the height it would actually need --
    the deepest shortfall along it, not the average, because a levee is only as
    good as its lowest point.

    Named from the nearest mapped road or waterway where there is one, because
    "site 3" is not something anyone can find on the ground.
    """
    key = "sites:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}
    res = run_shared(key, lambda: _solve_sites(req, key))
    if res is None:
        return {"ok": False, "pending": True,
                "elapsed_s": round(pending_for(key) or 0.0),
                "reason": "Still solving this flood. Nothing is lost — ask "
                          "again and you will join the same run."}
    return res


def _solve_sites(req: SitesRequest, key: str) -> dict:
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name, source=TERRAIN)
    fx = build_forcing(dom, req)
    z0 = torch.as_tensor(dom.z, dtype=torch.float32)[None]
    peak, steps, _ = solve_beds(dom, req, z0, fx, snap_hours=[])
    z = np.asarray(dom.z, dtype=np.float32)
    d = np.asarray(peak[0].numpy() if hasattr(peak[0], "numpy") else peak[0],
                   dtype=np.float32)
    wet = d > WET_M
    if not wet.any():
        return {"ok": False, "reason":
                "Nothing floods here under this storm, so there is nowhere a "
                "levee would change the answer."}

    surface = z + d
    H, W = z.shape
    # The highest water touching each cell, from its four neighbours.
    near_surface = np.full_like(surface, -1e9)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.full_like(surface, -1e9)
        rs = slice(max(dr, 0), H + min(dr, 0))
        rt = slice(max(-dr, 0), H + min(-dr, 0))
        cs = slice(max(dc, 0), W + min(dc, 0))
        ct = slice(max(-dc, 0), W + min(-dc, 0))
        sh[rt, ct] = np.where(wet[rs, cs], surface[rs, cs], -1e9)
        near_surface = np.maximum(near_surface, sh)

    shortfall = near_surface - z            # how far under the water this bank is
    cand = (~wet) & (near_surface > -1e8) & (shortfall > -0.05) & (shortfall <= req.shortfall_m)

    # Contiguous runs, four-connected, by flood fill.
    lab = np.zeros((H, W), dtype=np.int32)
    runs = []
    nxt = 0
    idx = np.argwhere(cand)
    for r0, c0 in idx:
        if lab[r0, c0]:
            continue
        nxt += 1
        stack = [(int(r0), int(c0))]
        lab[r0, c0] = nxt
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            # Eight-connected, and that is not a detail. A bank traced on a
            # grid steps diagonally wherever it runs off-axis, so four
            # connectivity cuts it at every one of those steps: measured, it
            # returned six "sites" of 160 to 320 m needing 0.09 to 0.46 m,
            # which are fragments of one bank rather than alignments anyone
            # would build.
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and cand[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = nxt
                    stack.append((rr, cc))
        if len(cells) >= 3:
            runs.append(cells)

    if not runs:
        return {"ok": False, "reason":
                f"No bank here sits within {req.shortfall_m:.1f} m of being "
                "overtopped. Everything either stays dry or goes far under, "
                "and a levee changes neither."}

    b = dom.bounds
    named = fetch_named_lines(req.west, req.south, req.east, req.north)
    out = []
    for cells in sorted(runs, key=len, reverse=True)[:req.max_sites]:
        rs = np.array([c[0] for c in cells])
        cs = np.array([c[1] for c in cells])
        need = float(shortfall[rs, cs].max())
        lat = float(b["north"] - (b["north"] - b["south"]) * rs.mean() / max(H - 1, 1))
        lon = float(b["west"] + (b["east"] - b["west"]) * cs.mean() / max(W - 1, 1))
        # Length along the run. The cell count is right for a bank one cell
        # thick and too big where it thickens, so it is capped against the
        # run's own diagonal extent -- a wall cannot be much longer than the
        # box it fits in without doubling back.
        extent = math.hypot(float(rs.max() - rs.min()), float(cs.max() - cs.min())) + 1.0
        span = min(float(len(cells)), extent * 1.6)
        out.append({
            "lat": round(lat, 6), "lon": round(lon, 6),
            "cells": len(cells),
            "length_m": round(float(span) * dom.dx),
            "extent_m": round(float(extent) * dom.dx),
            "needs_m": round(max(need, 0.0), 2),
            "width_m": round(float(span) * dom.dx / 2),
            "name": nearest_name(named, lat, lon),
            "outline": _centreline(rs, cs, b, z.shape),
            # A section ACROSS the bank, which is the line an engineer draws
            # to size a wall. Perpendicular to the run's own principal axis --
            # a levee follows the bank, so the useful section crosses it -- and
            # oriented so A is the landward end. Which end that is comes from
            # the solved water, not from the ground: the dry side of a bank is
            # not always the higher one, and on a levee it is deliberately not.
            "line": _section_line(rs, cs, wet, b, z.shape, dom.dx),
        })

    cell_km2 = (dom.dx ** 2) / 1e6
    res = {"ok": True, "sites": out, "dx_m": dom.dx,
           "bounds": b, "driver": fx["driver"], "driver_note": fx["note"],
           "shortfall_m": req.shortfall_m,
           # The flood these sites were found in.
           #
           # Without this the Plan tab drew the study's own design storm
           # while its scenario picker said "9 Sep 1921 flood": the panel
           # named one flood and the map underneath it showed another. The
           # solve has already happened to find the banks; returning its
           # field costs one PNG.
           "depth_png": depth_png(d, DEPTH_VMAX),
           "depth_vmax_m": DEPTH_VMAX,
           "flooded_km2": round(float(wet.sum()) * cell_km2, 2),
           "peak_depth_m": round(float(d.max()), 2),
           "river_q_m3s": fx.get("q_m3s"),
           "note": ("Each site is a run of ground that touches the flood and "
                    "sits within "
                    f"{req.shortfall_m:.1f} m of being overtopped. The height "
                    "shown is what the lowest point along it would need, "
                    "because a levee is only as good as its lowest point.")}
    if len(CACHE) > 24:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = res
    return {**res, "cached": False}


@app.post("/buildings")
def buildings(req: BuildingsRequest) -> dict:
    """Every building in the block, and whether the water got over its floor.

    Depth above the GROUND is what a flood model produces and depth above the
    FLOOR is what anyone actually asks, and the gap between them is not a
    constant: measured across 2,924 surveyed Lismore properties the median
    floor stands 1.95 m above its own front gate, with the middle 90% spanning
    0.04 m to 3.28 m (section 5.14e).

    **So the floor height is an assumption here, and it is returned as one.**
    Austin publishes the ground at the structure and not the floor above it,
    and FEMA elevation certificates, which carry the real number, are not
    bulk-public. Texas construction is overwhelmingly slab-on-grade, so the
    default is a slab: 0.17 m. Every building carries `floor_source` saying so,
    and the caller is expected to show it rather than fold it into a depth.

    Ground comes from the city survey where there is one and from the DEM
    otherwise, but the WATER SURFACE is always the solver's own ground plus its
    own depth. Mixing the two -- city floor against a DEM-derived surface --
    is the comparison that matters, and doing it the other way round would
    silently charge the DEM's error to the building.
    """
    key = "bld:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    span_km = _km_between({"lat": req.south, "lon": req.west},
                          {"lat": req.north, "lon": req.east})
    if span_km > 12.0:
        return {"ok": False, "reason":
                f"That block is {span_km:.0f} km across. The street view solves "
                "at a few metres; ask for something under 12 km."}

    # ---- the catchment, then the street ---------------------------------
    #
    # A street cannot be rain-forced. Measured: 50 mm/h for six hours over a
    # 2 km box with open edges put 10 of 576 Austin houses over their floors,
    # because the water that floods a street arrives from upstream and a box
    # that small has no upstream in it -- the rain simply drains out of the
    # sides. Three separate blocks sitting entirely inside FEMA's 1% floodplain
    # behaved the same way.
    #
    # So the catchment is solved at its own scale and the street is DOWNSCALED
    # from it: the water SURFACE from the coarse solve, which carries the
    # hydraulic gradient, laid over bare-earth terrain at street resolution.
    # Depth becomes surface minus the fine ground, which is the standard way a
    # flood surface is mapped onto better terrain, and it is honest about which
    # half came from where.
    mid_lat = (req.south + req.north) / 2
    mid_lon = (req.west + req.east) / 2
    half_lat = req.parent_km / 2 / 110.540
    half_lon = half_lat / max(math.cos(math.radians(mid_lat)), 0.2)
    pbox = {"west": mid_lon - half_lon, "east": mid_lon + half_lon,
            "south": mid_lat - half_lat, "north": mid_lat + half_lat}
    pdom = build_domain(pbox["west"], pbox["south"], pbox["east"], pbox["north"],
                        dx=req.parent_dx, name=f"{req.name} catchment",
                        source=TERRAIN)
    fx = build_forcing(pdom, req)
    pz0 = torch.as_tensor(pdom.z, dtype=torch.float32)[None]
    run = float(req.run_hours)
    snap_at = [run * i / (req.snapshots - 1) for i in range(req.snapshots)]
    ppeak, steps, psnaps = solve_beds(pdom, req, pz0, fx, snap_hours=snap_at)

    pz = np.asarray(pdom.z, dtype=np.float32)
    _pk = ppeak[0]
    pdepth = np.asarray(_pk.numpy() if hasattr(_pk, "numpy") else _pk, dtype=np.float32)
    psurf = pz + pdepth                                  # water surface elevation
    pseries = None
    if psnaps is not None:
        sn = psnaps.numpy() if hasattr(psnaps, "numpy") else psnaps
        arr = np.asarray(sn, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]                                 # (snapshots, H, W)
        pseries = pz[None] + arr

    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name, source=TERRAIN)
    z = np.asarray(dom.z, dtype=np.float32)
    H, W = z.shape
    b = dom.bounds

    def _on_fine(field: np.ndarray) -> np.ndarray:
        """A coarse field sampled onto the fine grid, nearest cell."""
        ph, pw = field.shape
        lons = np.linspace(b["west"], b["east"], W)
        lats = np.linspace(b["north"], b["south"], H)
        cols = np.clip(((lons - pbox["west"]) / (pbox["east"] - pbox["west"])
                        * (pw - 1)).round().astype(int), 0, pw - 1)
        rows = np.clip(((pbox["north"] - lats) / (pbox["north"] - pbox["south"])
                        * (ph - 1)).round().astype(int), 0, ph - 1)
        return field[np.ix_(rows, cols)]

    # Only where the coarse solve put water at all: a surface interpolated into
    # a dry neighbourhood would "flood" every hollow in it.
    wet_coarse = _on_fine((pdepth > WET_M).astype(np.float32)) > 0.5
    d_peak = np.where(wet_coarse, np.maximum(_on_fine(psurf) - z, 0.0), 0.0
                      ).astype(np.float32)
    series = None
    if pseries is not None:
        series = np.stack([
            np.where(_on_fine((pseries[i] - pz) > WET_M),
                     np.maximum(_on_fine(pseries[i]) - z, 0.0), 0.0)
            for i in range(pseries.shape[0])]).astype(np.float32)

    feats = fetch_footprints(req.west, req.south, req.east, req.north,
                             limit=req.max_buildings)
    out = []
    for f in feats:
        ring = f["ring"]
        lon = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
        col = int(round((lon - b["west"]) / max(b["east"] - b["west"], 1e-12) * (W - 1)))
        row = int(round((b["north"] - lat) / max(b["north"] - b["south"], 1e-12) * (H - 1)))
        if not (0 <= col < W and 0 <= row < H):
            continue
        dem_ground = float(z[row, col])
        if not np.isfinite(dem_ground):
            continue
        city_ground = f.get("ground_m")
        ground = float(city_ground) if city_ground is not None else dem_ground
        floor = ground + FLOOR_ABOVE_GROUND_M
        depth_now = float(d_peak[row, col])
        # Water over the floor is computed IN ONE VERTICAL FRAME, the solver's.
        #
        # The obvious form -- city floor elevation against DEM ground plus
        # depth -- is wrong, and wrong by far more than the quantity it
        # measures. Those two grounds disagree by 2.05 m RMSE even on bare
        # earth (section 5.14f), which is twelve times the 0.17 m floor offset
        # it is being subtracted from. Written that way, a building under 2 mm
        # of water came back as 0.33 m over its floor, purely because the DEM
        # sat half a metre above the city's survey at that point.
        #
        # Depth is already relative to the solver's own ground and the floor
        # offset is relative to the ground at the building, so the comparison
        # that carries no absolute error is simply depth minus offset. The city
        # elevation is kept for display, and for saying where the ground came
        # from, but not for this.
        over = depth_now - FLOOR_ABOVE_GROUND_M
        timeline = []
        if series is not None:
            timeline = [round(float(series[i, row, col]) - FLOOR_ABOVE_GROUND_M, 3)
                        for i in range(series.shape[0])]
        out.append({
            "id": f.get("id"),
            "ring": [[round(c[0], 6), round(c[1], 6)] for c in ring],
            "address": f.get("address"),
            "ground_m": round(ground, 2),
            "ground_source": "city survey" if f.get("ground_m") is not None else "DEM",
            "height_m": round(float(f["height_m"]), 1) if f.get("height_m") else None,
            "floor_m": round(floor, 2),
            "depth_on_ground_m": round(max(depth_now, 0.0), 3),
            "water_surface_m": round(ground + depth_now, 2) if depth_now > 1e-4 else None,
            "dem_ground_m": round(dem_ground, 2),
            "over_floor_m": round(over, 3) if over > 0 else None,
            "over_floor_series": timeline,
        })

    n_over = sum(1 for x in out if x["over_floor_m"] is not None)
    road = float(np.nanmax(d_peak)) if d_peak.size else 0.0
    res = {
        "ok": True,
        "name": req.name,
        "bounds": b,
        "dx_m": dom.dx,
        "grid": list(z.shape),
        "terrain_source": dom.source,
        "buildings": out,
        "count": len(out),
        "over_floor": n_over,
        "deepest_on_ground_m": round(road, 2),
        "snapshot_hours": [round(h, 2) for h in snap_at] if series is not None else [],
        # The block's own water, so the map draws the street's flood rather
        # than the parent study's. Without it the only thing carrying water on
        # screen is the colour of the buildings.
        "depth_png": depth_png(d_peak, _street_vmax(d_peak)),
        "terrain_png": terrain_png(z),
        "depth_vmax_m": round(_street_vmax(d_peak), 2),
        "frames": ([{"t_h": round(snap_at[i], 2),
                     "depth_png": depth_png(series[i], _street_vmax(d_peak))}
                    for i in range(series.shape[0])] if series is not None else []),
        "floor_above_ground_m": FLOOR_ABOVE_GROUND_M,
        "floor_source": (
            "ASSUMED. The floor height above ground is not published for this "
            f"area; {FLOOR_ABOVE_GROUND_M:.2f} m is a slab-on-grade default. "
            "Measured on 2,924 surveyed Lismore properties the median floor "
            "stands 1.95 m above its own gate, so this assumption is worth "
            "metres of depth where houses are not on slabs."),
        "driver": fx["driver"],
        "driver_note": fx["note"],
        "timing": {"steps": steps},
    }
    if len(CACHE) > 24:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = res
    return {**res, "cached": False}


@app.post("/section")
def section(req: SectionRequest):
    """Ground elevation and solved peak water level along a line."""
    a, b = req.a, req.b
    try:
        alat, alon = float(a["lat"]), float(a["lon"])
        blat, blon = float(b["lat"]), float(b["lon"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "Both ends need a lat and a lon."}

    span_km = _km_between(a, b)
    if span_km < 0.05:
        return {"ok": False, "reason":
                "The two ends are less than 50 m apart; that is not a section."}

    # Same box derivation as /route, so a section and a trip over the same
    # ground hit the same cached solve instead of paying for it twice.
    pad_km = max(0.3 * span_km, 3.0)
    half_lat = (span_km / 2 + pad_km) / 110.540
    mid_lat = (alat + blat) / 2
    mid_lon = (alon + blon) / 2
    half_lon = half_lat / max(math.cos(math.radians(mid_lat)), 0.2)
    half_lat = max(half_lat, abs(alat - blat) / 2 + pad_km / 110.540)
    half_lon = max(half_lon, abs(alon - blon) / 2
                   + pad_km / (111.320 * max(math.cos(math.radians(mid_lat)), 0.2)))
    box = {"west": mid_lon - half_lon, "east": mid_lon + half_lon,
           "south": mid_lat - half_lat, "north": mid_lat + half_lat}
    width_m = (box["east"] - box["west"]) * 111_320 * math.cos(math.radians(mid_lat))
    derived = 10.0 * round(width_m / 160.0 / 10.0)
    if derived > 240.0:
        return {"ok": False, "reason":
                f"Those ends are {span_km:.0f} km apart; one domain covering "
                "both would need cells coarser than 240 m. Pick a shorter line."}
    dx = (min(240.0, max(30.0, float(req.dx))) if req.dx
          else min(240.0, max(50.0, derived)))

    rr = RouteRequest(origin=a, destination=b, name=req.name,
                      rain_mm_h=req.rain_mm_h, storm_hours=req.storm_hours,
                      run_hours=req.run_hours, driver=req.driver,
                      river_q_m3s=req.river_q_m3s, dx=dx, snapshots=2)
    skey = hashlib.sha1(json.dumps([
        {k: round(v, 6) for k, v in box.items()}, dx, rr.driver, rr.rain_mm_h,
        rr.storm_hours, rr.run_hours, rr.river_q_m3s, rr.surge_m, rr.snapshots,
    ], sort_keys=True).encode()).hexdigest()[:16]
    hit = SOLVE_CACHE.get(skey)
    if hit is None:
        dom = build_domain(box["west"], box["south"], box["east"], box["north"],
                           dx=dx, name=req.name, source=TERRAIN)
        fx = build_forcing(dom, rr)
        z0 = torch.as_tensor(dom.z, dtype=torch.float32)[None]
        peak, steps, snaps = solve_beds(dom, rr, z0, fx, snap_hours=[])
        hit = {"dom": dom, "fx": fx, "peak": peak, "snaps": snaps, "steps": steps}
        _cache_put(SOLVE_CACHE, skey, hit)
    dom, peak, fx = hit["dom"], hit["peak"], hit["fx"]
    z = np.asarray(dom.z, dtype=np.float32)
    pk = peak[0]
    d_np = np.asarray(pk.numpy() if hasattr(pk, "numpy") else pk, dtype=np.float32)
    H, W = z.shape

    # Nearest cell rather than bilinear: the DEM is already the limiting
    # resolution and interpolating between posts would invent a smoothness the
    # terrain does not have.
    n = int(req.samples)
    prof = []
    for i in range(n):
        f = i / (n - 1)
        lat, lon = alat + (blat - alat) * f, alon + (blon - alon) * f
        col = int(round((lon - box["west"]) / max(box["east"] - box["west"], 1e-12) * (W - 1)))
        row = int(round((box["north"] - lat) / max(box["north"] - box["south"], 1e-12) * (H - 1)))
        col, row = min(max(col, 0), W - 1), min(max(row, 0), H - 1)
        g, w = float(z[row, col]), float(d_np[row, col])
        prof.append({
            "m": round(span_km * 1000.0 * f, 1),
            "ground_m": round(g, 2),
            "depth_m": round(max(w, 0.0), 3),
            # Water SURFACE elevation: what a crest actually has to clear.
            "water_m": round(g + w, 2) if w > 1e-4 else None,
        })

    wet = [q for q in prof if q["water_m"] is not None]
    peak_surface = max((q["water_m"] for q in wet), default=None)
    highest = max(prof, key=lambda q: q["ground_m"])

    return {
        "ok": True,
        "a": a, "b": b, "name": req.name,
        "length_m": round(span_km * 1000.0, 1),
        "dx_m": dom.dx,
        "samples": n,
        "profile": prof,
        "peak_water_m": peak_surface,
        "wet_fraction": round(len(wet) / max(len(prof), 1), 3),
        # Where a levee should start. Building from the low point is how you
        # pay for height you do not need.
        "highest_ground": {"m": highest["m"], "ground_m": highest["ground_m"]},
        # No rate here. The section used to carry its own `2600.0` in
        # Australian dollars, a third copy of a figure with no source on it;
        # the client prices the wall from `lib/rates`, where the USACE
        # embankment and T-wall rates are written down once with their
        # provenance.
        # Which forcing actually ran. A river driver that found no channel
        # falls back to rainfall, and a dry section then means "this rain did
        # not reach here" rather than "this river does not reach here" -- two
        # different answers that look identical in a profile.
        "driver": fx["driver"],
        "driver_note": fx["note"],
        "river_q_m3s": fx.get("q_m3s"),
        "note": (f"Ground sampled from the {TERRAIN_LABEL.get(dom.source, dom.source)} "
                 "this solve ran on, at "
                 f"{dom.dx:.0f} m. The water level is the solved peak along "
                 "this line, not an assumed design stage."),
    }


@app.post("/route")
def route_between(req: RouteRequest) -> dict:
    """A route from A to B that avoids the flood, and the reasons for it.

    The domain is not given here, it is DERIVED: a box containing both ends
    with room to detour, at whatever resolution keeps the solve interactive.
    That is the difference between a tool that answers for one valley and one
    that answers for a trip someone actually has to make.

    Routing happens against a SCHEDULE rather than a peak field, because the
    traveller is on each road at the time they reach it. The answers that come
    out of that are ones a peak map cannot give: leave now and you get through;
    leave after 14:45 and you do not; the direct road is the one that goes
    under, and here is the one that does not.
    """
    key = "route:" + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    t0 = time.time()
    o, d = req.origin, req.destination
    span_km = _km_between(o, d)
    if span_km < 0.3:
        return {"ok": False, "reason":
                "Those two points are less than 300 m apart. Pick a "
                "destination further away than the width of a street."}

    # Room to detour. A box drawn tight around the two ends would make the
    # straight line the only option, and a flood route is a detour by nature.
    pad_km = max(0.3 * span_km, 3.0)
    half_lat = (span_km / 2 + pad_km) / 110.540
    mid_lat = (float(o["lat"]) + float(d["lat"])) / 2
    mid_lon = (float(o["lon"]) + float(d["lon"])) / 2
    half_lon = half_lat / max(math.cos(math.radians(mid_lat)), 0.2)
    # Cover both ends even when the trip runs mostly east-west or north-south.
    half_lat = max(half_lat, abs(float(o["lat"]) - float(d["lat"])) / 2 + pad_km / 110.540)
    half_lon = max(half_lon, abs(float(o["lon"]) - float(d["lon"])) / 2
                   + pad_km / (111.320 * max(math.cos(math.radians(mid_lat)), 0.2)))
    box = {"west": mid_lon - half_lon, "east": mid_lon + half_lon,
           "south": mid_lat - half_lat, "north": mid_lat + half_lat}

    # Resolution follows the trip. A 12 km trip is solved at 50 m; a 60 km one
    # cannot be, and saying so in the response is better than quietly taking
    # twenty minutes or quietly returning a grid too coarse to mean anything.
    width_m = (box["east"] - box["west"]) * 111_320 * math.cos(math.radians(mid_lat))
    target_cells = 250
    # Two bounds, and the floor is the one that is easy to miss. Solve cost goes
    # as cells^2 per step times a step count that rises as the timestep falls,
    # and the CFL timestep is proportional to dx -- so with the cell count
    # capped, cost goes as 1/dx and a FINER grid is slower. dx falls with the
    # box, so without a floor the worst case is a box sitting right on it,
    # which is exactly a short trip across a town: a 2.6 km trip to the
    # hospital was heading for ~10 minutes while a 20 km trip took 104 s. The
    # 50 m floor puts that case at about two minutes, and 50 m is still finer
    # than anything else in this tool resolves a flood at.
    derived = 10.0 * round(width_m / target_cells / 10.0)
    if derived > 240.0:
        return {"ok": False, "reason":
                f"Those points are {span_km:.0f} km apart. One domain covering "
                f"both would need cells coarser than 240 m, at which a 2 m flood "
                f"on a road is a rounding error. Route between closer points, or "
                f"split the trip."}
    # An explicit ask is honoured down to 30 m; the derived value keeps the
    # floor, because nobody asked for a ten-minute wait on a 2 km trip.
    dx = (min(240.0, max(30.0, float(req.dx))) if req.dx
          else min(240.0, max(50.0, derived)))

    snap_h = [req.run_hours * i / (req.snapshots - 1) for i in range(req.snapshots)]
    # Everything the flood depends on, and nothing the vehicle does.
    skey = hashlib.sha1(json.dumps([
        {k: round(v, 6) for k, v in box.items()}, dx, req.driver, req.rain_mm_h,
        req.storm_hours, req.run_hours, req.river_q_m3s, req.surge_m,
        req.snapshots,
    ], sort_keys=True).encode()).hexdigest()[:16]
    hit = SOLVE_CACHE.get(skey)
    reused = hit is not None
    if hit is None:
        dom = build_domain(box["west"], box["south"], box["east"], box["north"],
                           dx=dx, name=req.name, source=TERRAIN)
        fx = build_forcing(dom, req)
        z0 = torch.as_tensor(dom.z, dtype=torch.float32)[None]
        peak, steps, snaps = solve_beds(dom, req, z0, fx, snap_hours=snap_h)
        hit = {"dom": dom, "fx": fx, "peak": peak, "snaps": snaps, "steps": steps}
        _cache_put(SOLVE_CACHE, skey, hit)
    dom, fx, peak, snaps, steps = (hit["dom"], hit["fx"], hit["peak"],
                                   hit["snaps"], hit["steps"])
    z = np.asarray(dom.z, dtype=np.float32)
    peak_np = peak[0].numpy()

    rkey = hashlib.sha1(json.dumps(
        {k: round(v, 5) for k, v in box.items()}, sort_keys=True).encode()
    ).hexdigest()[:16]
    net = ROADS_CACHE.get(rkey)
    if net is None:
        # `fetch_roads` raises when every mirror fails, and Overpass does fail
        # -- it rate-limits, and a 33 km box is a large query. Letting that
        # propagate threw away a solve that had just taken two minutes and
        # returned a 500 with nothing in it. The solve is already cached by the
        # time we get here, so the honest answer is to say which half failed
        # and that retrying is now cheap.
        try:
            net = fetch_roads(box["west"], box["south"], box["east"], box["north"])
        except Exception as exc:                            # noqa: BLE001
            return {"ok": False, "reason":
                    f"The flood is solved for this trip, but OpenStreetMap is "
                    f"not answering, so there is no road network to route on "
                    f"({exc}). The solve is kept: trying again will not re-run "
                    f"it.",
                    "solved": True, "bounds": box, "dx_m": dom.dx,
                    "depth_png": depth_png(peak_np, DEPTH_VMAX),
                    "terrain_png": terrain_png(dom.z),
                    "driver": fx["driver"], "driver_note": fx["note"]}
        _cache_put(ROADS_CACHE, rkey, net)
    g = routes.build_graph(net.get("features", []))
    if len(g.edges) < 2:
        return {"ok": False, "reason":
                "OpenStreetMap returned almost no roads for this area, so "
                "there is no network to route on."}

    flat, offs = routes.edge_cells(g, box, z.shape)
    sched = routes.Schedule(
        times=[h * 60.0 for h in snap_h],
        depths=[routes.depths_from_field(snaps[0, k].numpy(), flat, offs)
                for k in range(snaps.shape[1])],
    )
    peak_edges = sched.peak()
    limit = req.vehicle_depth_m

    # The snapshots the schedule is built from, returned as overlays.
    #
    # The solver resolves the flood at `req.snapshots` instants and the road
    # schedule is built from them -- then every field except the peak was
    # discarded, so the client could only ever draw the union of states that
    # never coexist. That union is exactly what fails 96 of 96 evacuation
    # decisions where a time-resolved solve succeeds, which makes "show the
    # water move" an argument the tool was already paying for and throwing
    # away rather than a decoration.
    #
    # Instantaneous depth, not the running peak: what is on the road at 14:00
    # is the water at 14:00.
    frames = [
        {"t_min": round(h * 60.0, 1),
         "depth_png": depth_png(snaps[0, k].numpy(), DEPTH_VMAX)}
        for k, h in enumerate(snap_h)
    ]

    src = routes.nearest_node(g, float(o["lon"]), float(o["lat"]))
    dst = routes.nearest_node(g, float(d["lon"]), float(d["lat"]))
    if src is None or dst is None or src == dst:
        return {"ok": False, "reason":
                "Both ends snap to the same junction, or no junction at all. "
                "Pick points on different roads."}
    snap_km = {
        "origin": round(_km_between(o, {"lat": g.nodes[src][1], "lon": g.nodes[src][0]}), 2),
        "destination": round(_km_between(d, {"lat": g.nodes[dst][1], "lon": g.nodes[dst][0]}), 2),
    }

    # Four answers, because the comparison is the explanation.
    dry = routes.fastest(g, src, dst, [0.0] * len(g.edges), limit)
    now = routes.fastest_at(g, src, dst, sched, limit, depart_min=0.0)
    static = routes.fastest(g, src, dst, peak_edges, limit)
    deadline, at_deadline = routes.latest_departure(
        g, src, dst, sched, limit, step_min=req.depart_step_min)

    chosen = now if now.ok else (at_deadline if at_deadline and at_deadline.ok else None)
    depart = 0.0 if now.ok else (deadline or 0.0)

    # What the direct road does, which is what "avoids the flooding" means:
    # the quickest way ignoring water, and where that way goes under. Evaluated
    # at the departure being reported, because a road that is dry for someone
    # leaving now and 70 cm deep for someone leaving at the deadline is the
    # whole reason this is scheduled rather than static -- quoting the t=0
    # answer beside a later departure would explain the wrong trip.
    def cuts_on_direct(t_depart: float) -> list[dict]:
        if not dry.ok:
            return []
        return [c for c in routes.cut_at(g, dry, sched, limit, t_depart)
                if not c["passable"]]

    direct_cuts = cuts_on_direct(depart)
    direct_cuts_at_deadline = (cuts_on_direct(deadline)
                               if deadline is not None else [])

    def route_out(r, t_depart: float) -> dict | None:
        if r is None or not r.ok:
            return None
        return {
            "ok": True,
            "minutes": round(r.minutes, 1),
            "km": round(r.km, 2),
            "max_depth_m": round(r.max_depth_m, 3),
            "via": r.names(g)[:8],
            "coords": [[round(x, 6), round(y, 6)] for x, y in r.coords(g)],
            "depart_min": round(t_depart, 1),
            "water_on_the_way": routes.cut_at(g, r, sched, limit, t_depart),
        }

    cut_timeline = [
        {"minutes": round(t, 0),
         "cut_km": round(routes.cut_km(g, sched.depths[k], limit), 1)}
        for k, t in enumerate(sched.times)
    ]

    out = {
        "ok": True,
        "origin": o, "destination": d,
        "span_km": round(span_km, 1),
        "bounds": box,
        "grid": [int(z.shape[0]), int(z.shape[1])],
        "dx_m": dom.dx,
        "vehicle_depth_m": limit,
        "network_km": round(g.total_km, 1),
        "edges": len(g.edges),
        "snap_km": snap_km,
        # The trip itself.
        "route": route_out(chosen, depart),
        "dry_route": route_out(dry, 0.0),
        # What a peak-depth map would have told you, which is the comparison
        # that justifies carrying a schedule at all.
        "static_route": route_out(static, 0.0),
        # The solved instants behind that comparison, so the client can show
        # the water arrive and recede instead of a single union of states.
        "frames": frames,
        "run_hours": req.run_hours,
        "leave_now": now.ok,
        "deadline_min": None if deadline is None else round(deadline, 0),
        "deadline_route": route_out(at_deadline, deadline or 0.0),
        "direct_road_cuts": direct_cuts,
        # The same question asked of the last possible departure, which is where
        # the detour actually shows up: by then the quick road is under water.
        "direct_road_cuts_at_deadline": direct_cuts_at_deadline,
        "cut_km_now": round(routes.cut_km(g, sched.depths[0], limit), 1),
        "cut_km_peak": round(routes.cut_km(g, peak_edges, limit), 1),
        "cut_timeline": cut_timeline,
        "snapshot_minutes": [round(t, 0) for t in sched.times],
        # Provenance, in the same shape the other endpoints use.
        "driver": fx["driver"], "driver_note": fx["note"],
        "river_q_m3s": fx["q_m3s"], "river_stage_m": fx.get("stage_m"),
        "storm": {"rain_mm_h": req.rain_mm_h, "storm_hours": req.storm_hours,
                  "run_hours": req.run_hours},
        "depth_png": depth_png(peak_np, DEPTH_VMAX),
        "terrain_png": terrain_png(dom.z),
        **roads_and_grids(dom, peak_np, box["west"], box["south"],
                          box["east"], box["north"]),
        "roads": net,
        # Says which it was, because "solved in 3 s" and "re-used a solve from
        # 3 minutes ago" are different claims about the same number.
        "timing": {"seconds": round(time.time() - t0, 1), "steps": steps,
                   "solve_reused": reused},
    }
    # The brief is scored against the record it describes, and withheld if it
    # does not match. The score travels with the response so the guarantee is
    # auditable from the outside rather than asserted here.
    out["brief"], out["brief_check"] = route_brief.checked(out)
    CACHE[key] = out
    return {**out, "cached": False}


@app.post("/analyse")
def analyse(req: AnalyseRequest) -> dict:
    key = hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    t0 = time.time()
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name, source=TERRAIN)
    t_dem = time.time() - t0

    lat = (req.south + req.north) / 2
    lon = (req.west + req.east) / 2
    try:
        fc = forcing_for(lat, lon)
    except Exception:
        fc = {"total_mm": None, "peak_discharge_m3s": None}

    snap_h = ([req.run_hours * i / (req.snapshots - 1)
               for i in range(req.snapshots)] if req.snapshots > 1 else [])
    peak, stats, snaps = solve(dom, req.rain_mm_h, req.storm_hours,
                               req.run_hours, req.levees, req,
                               snap_hours=snap_h)

    # What fraction of this flood simply fell where it lies?
    #
    # Reported as a fact, not a verdict. An earlier version of this used it to
    # classify the flood and advise whether a barrier would help; that was
    # wrong. The share distinguishes water that fell in place from water that
    # flowed in, but a levee can only block CHANNELISED inflow, not water
    # converging off surrounding ground from every direction. The test measured
    # 25% direct and called a barrier useful, while an actual levee run changed
    # the flooded area by 0.03 of 41 km2. The honest answer comes from running
    # the levee, which this service does on request.
    direct_m = req.rain_mm_h / 1000.0 * req.storm_hours
    wet_mask = peak > 0.10
    mean_wet = float(peak[wet_mask].mean()) if wet_mask.any() else 0.0
    direct_share = min(1.0, direct_m / mean_wet) if mean_wet > 1e-6 else 1.0

    cell_km2 = (dom.dx ** 2) / 1e6
    wet = peak > WET_M
    out = {
        "name": req.name,
        "bounds": dom.bounds,
        "grid": list(dom.shape),
        "dx_m": dom.dx,
        "epsg": dom.epsg,
        "extent_km": [round(v, 1) for v in dom.extent_km],
        "elevation_range_m": [float(np.nanmin(dom.z)), float(np.nanmax(dom.z))],
        "flooded_km2": round(float(wet.sum()) * cell_km2, 2),
        "wet_threshold_m": WET_M,
        "peak_depth_m": round(float(peak.max()), 2),
        "mean_depth_wet_m": round(float(peak[wet].mean()) if wet.any() else 0.0, 2),
        "depth_vmax_m": DEPTH_VMAX,
        "depth_png": depth_png(peak, DEPTH_VMAX),
        "terrain_png": terrain_png(dom.z),
        # The flood arriving, hour by hour. Same colour ramp as the peak, so
        # the scrubber and the still frame are on one scale and a frame that
        # looks shallower IS shallower.
        "frames": ([{"t_h": round(snap_h[i], 2),
                     "flooded_km2": round(float((snaps[i] > WET_M).sum())
                                          * (dom.dx ** 2) / 1e6, 2),
                     "peak_depth_m": round(float(snaps[i].max()), 2),
                     "depth_png": depth_png(snaps[i], DEPTH_VMAX)}
                    for i in range(len(snap_h))] if snaps is not None else []),
        **roads_and_grids(dom, peak, req.west, req.south, req.east, req.north),
        "terrain": forcing.describe(np.asarray(dom.z, dtype=np.float32), dom.dx),
        "driver": stats.get("driver", req.driver),
        "driver_note": stats.get("driver_note", ""),
        "forecast": fc,
        "timing": {"dem_seconds": round(t_dem, 1), **stats},
        "storm": {"rain_mm_h": req.rain_mm_h, "storm_hours": req.storm_hours,
                  "run_hours": req.run_hours,
                  "total_mm": req.rain_mm_h * req.storm_hours},
        "levees": req.levees,
        "mechanism": {
            "direct_rain_m": round(direct_m, 3),
            "mean_wet_depth_m": round(mean_wet, 3),
            "direct_share": round(direct_share, 3),
            "note": (
                f"{round(100 * direct_share)}% of a typical flooded depth is rain "
                "that fell on the spot; no barrier can exclude that. Whether a "
                "levee helps with the rest depends on the water arriving through "
                "a path one can block, which is measured by running it."
            ),
        },
        "terrain_source": dom.source,
        "terrain_source_label": TERRAIN_LABEL.get(dom.source, dom.source),
        "sources": [f"Terrain: {TERRAIN_LABEL.get(dom.source, dom.source)}",
                    "Forecast: Open-Meteo / GloFAS"],
    }
    if len(CACHE) > 24:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = out
    return {**out, "cached": False}
