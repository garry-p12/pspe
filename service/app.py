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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.observe import sar  # noqa: E402
from pspe.simulate.real import forcing  # noqa: E402
from pspe.plan import routes, surrogate  # noqa: E402
from pspe.explain import route_brief  # noqa: E402
from pspe.simulate.flood import FloodConfig, FloodSolver  # noqa: E402
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
    budget_aud: float = Field(20e6, ge=0.0)
    cost_per_m_per_m: float = 2600.0
    delta: float = Field(0.1, ge=0.01, le=0.5)
    protect: Optional[dict] = None   # {"lat":..,"lon":..,"radius_m":..}
    # What drives the flood. "rain" spreads the storm over the whole box, which
    # is a real question and the wrong one for a river or coastal city: a levee
    # has nothing to block when the water falls on both sides of it.
    driver: str = Field("rain", pattern="^(rain|river|coastal)$")
    river_q_m3s: Optional[float] = None    # None -> take it from the flood API
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
    surge_m: float = Field(2.5, ge=0.0, le=10.0)


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
          levees: list[dict], req=None) -> tuple[np.ndarray, dict]:
    z = torch.as_tensor(dom.z, dtype=torch.float32)
    h_, w_ = z.shape

    # Levees raise the bed, which is what makes them gate flow rather than add
    # water. Each is a Gaussian ridge of a stated width in METRES (defect 20).
    if levees:
        yy, xx = torch.meshgrid(torch.arange(h_), torch.arange(w_), indexing="ij")
        for lv in levees:
            r = (dom.bounds["north"] - lv["lat"]) / (
                dom.bounds["north"] - dom.bounds["south"]) * h_
            c = (lv["lon"] - dom.bounds["west"]) / (
                dom.bounds["east"] - dom.bounds["west"]) * w_
            sigma = max(1.0, float(lv.get("width_m", 600.0)) / dom.dx)
            g = torch.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / sigma ** 2))
            z = z + float(lv.get("height_m", 2.0)) * g

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

    return peak[0].numpy(), {"driver": fx["driver"], "driver_note": fx["note"],
                             "steps": steps, "seconds": round(time.time() - t0, 1)}


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


_WATERWAY_CACHE: dict = {}


def fetch_waterways(west: float, south: float, east: float, north: float,
                    timeout: float = 60.0) -> list[dict]:
    """Mapped rivers and streams over this box, from OpenStreetMap.

    Used to find where the channel crosses the boundary, which the terrain
    alone gets wrong on a tilted catchment. Cached per box and allowed to fail:
    Overpass is a free public service and a 504 from it must not take a flood
    solve down, so the caller falls back to the edge heuristic.
    """
    key = (round(west, 4), round(south, 4), round(east, 4), round(north, 4))
    if key in _WATERWAY_CACHE:
        return _WATERWAY_CACHE[key]
    q = (f"[out:json][timeout:{int(timeout)}];"
         f'way["waterway"~"^(river|stream)$"]'
         f"({south},{west},{north},{east});out geom;")
    payload = None
    for url in OVERPASS_MIRRORS:
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
    if len(_WATERWAY_CACHE) > 32:
        _WATERWAY_CACHE.pop(next(iter(_WATERWAY_CACHE)))
    _WATERWAY_CACHE[key] = els
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
        # Heavy rain ~ bankfull, severe ~ 2x, extreme ~ 4x. Multiples of
        # bankfull are the standard way a design flood is stated.
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
            f"(channel from the {out['channel_from']}), {scale:g}x "
            + (f"the forecast peak of {q:,.0f} m3/s. "
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


def site_masks(dom, sites: list[dict]) -> "torch.Tensor":
    """One Gaussian ridge per candidate site, in METRES (defect 20)."""
    h_, w_ = dom.z.shape
    yy, xx = torch.meshgrid(torch.arange(h_), torch.arange(w_), indexing="ij")
    out = []
    for s in sites:
        r = (dom.bounds["north"] - s["lat"]) / (
            dom.bounds["north"] - dom.bounds["south"]) * h_
        c = (s["lon"] - dom.bounds["west"]) / (
            dom.bounds["east"] - dom.bounds["west"]) * w_
        sigma = max(1.0, float(s.get("width_m", 600.0)) / dom.dx)
        out.append(torch.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / sigma ** 2)))
    return torch.stack(out) if out else torch.zeros(0, h_, w_)


def solve_beds(dom, req, z: "torch.Tensor", fx: dict,
               snap_hours: Sequence[float] | None = None
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

    t0 = time.time()
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name, source=TERRAIN)
    z0 = torch.as_tensor(dom.z, dtype=torch.float32)
    masks = site_masks(dom, req.sites)
    k = len(req.sites)

    plans = surrogate.enumerate_plans(k, req.max_height_m)
    P = torch.tensor(plans, dtype=torch.float32)
    # One bed per plan; one timestep loop for all of them.
    z = z0[None] + (P[:, :, None, None] * masks[None]).sum(1)

    fx = build_forcing(dom, req)
    peak, steps, _ = solve_beds(dom, req, z, fx)
    b = z.shape[0]

    # What the plan is bought to change: depth over the protected area, or the
    # whole domain when nothing is named.
    if req.protect:
        h_, w_ = dom.z.shape
        yy, xx = torch.meshgrid(torch.arange(h_), torch.arange(w_), indexing="ij")
        r = (dom.bounds["north"] - req.protect["lat"]) / (
            dom.bounds["north"] - dom.bounds["south"]) * h_
        c = (req.protect["lon"] - dom.bounds["west"]) / (
            dom.bounds["east"] - dom.bounds["west"]) * w_
        rad = float(req.protect.get("radius_m", 2000.0)) / dom.dx
        sel = (((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2)
    else:
        sel = torch.ones(dom.z.shape, dtype=torch.bool)
    if not bool(sel.any()):
        sel = torch.ones(dom.z.shape, dtype=torch.bool)

    core = (peak * sel[None]).sum(dim=(1, 2)) / float(sel.sum())
    base = float(core[0])
    # Positive = flooding reduced where it matters.
    reduction = [(base - float(c)) / base * 100.0 if base > 1e-9 else 0.0
                 for c in core]

    lengths = [float(s.get("width_m", 600.0)) * 2.0 for s in req.sites]
    unit = req.cost_per_m_per_m

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
            "cost_aud": sum(pl[j] * lengths[j] * unit for j in range(k)),
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
        return sum(hv[i] * lengths[i] * unit for i in range(k))

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
            if spent > req.budget_aud:
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
            "cost_aud": hh * lengths[i] * unit,
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
        "cost_aud": cost(hv),
        "budget_aud": req.budget_aud,
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
        "crest_lengths_m": lengths,
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
    dx: Optional[float] = None


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
        "cost_per_m_per_m_aud": 2600.0,
        # Which forcing actually ran. A river driver that found no channel
        # falls back to rainfall, and a dry section then means "this rain did
        # not reach here" rather than "this river does not reach here" -- two
        # different answers that look identical in a profile.
        "driver": fx["driver"],
        "driver_note": fx["note"],
        "river_q_m3s": fx.get("q_m3s"),
        "note": ("Ground sampled from the Copernicus DEM this solve ran on, at "
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

    peak, stats = solve(dom, req.rain_mm_h, req.storm_hours, req.run_hours,
                        req.levees, req)

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
