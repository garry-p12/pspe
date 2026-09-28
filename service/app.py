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
import urllib.parse
import urllib.request
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

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
from pspe.plan import surrogate  # noqa: E402
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


def depth_png(d: np.ndarray, vmax: float) -> str:
    """Depth as a base64 PNG overlay, on the same ramp the portal uses."""
    x = np.clip(np.nan_to_num(d) / max(vmax, 1e-6), 0.0, 1.0) ** 0.42
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (96 * (1 - x) + 4 * x).astype(np.uint8)
    out[..., 1] = (200 * (1 - x) + 28 * x).astype(np.uint8)
    out[..., 2] = (255 * (1 - x) + 128 * x).astype(np.uint8)
    a = np.where(np.nan_to_num(d) > 0.01, 132 + 118 * x, 0.0)
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
          levees: list[dict]) -> tuple[np.ndarray, dict]:
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
    rain = rain_mm_h / 3.6e6
    t, dur, steps = 0.0, run_h * 3600.0, 0
    t0 = time.time()
    while t < dur:
        dt = min(solver.adaptive_dt(h), dur - t)
        if dt <= 0:
            break
        h, qx, qy = solver.step(h, qx, qy, dt,
                                rain=rain if t < storm_h * 3600 else 0.0)
        peak = torch.maximum(peak, h)
        t += dt
        steps += 1
    return peak[0].numpy(), {"steps": steps, "seconds": round(time.time() - t0, 1)}


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
                       dx=req.dx, name=req.name)
    z0 = torch.as_tensor(dom.z, dtype=torch.float32)
    masks = site_masks(dom, req.sites)
    k = len(req.sites)

    plans = surrogate.enumerate_plans(k, req.max_height_m)
    P = torch.tensor(plans, dtype=torch.float32)
    # One bed per plan; one timestep loop for all of them.
    z = z0[None] + (P[:, :, None, None] * masks[None]).sum(1)

    rows = z.mean(dim=2).mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (z.shape[1] * dom.dx))), 1e-5)
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=slope, dt_max=300.0)
    solver = FloodSolver(z, cfg)
    b = z.shape[0]
    h = torch.zeros(b, z.shape[1], z.shape[2])
    qx, qy = solver.zeros_flux(b)
    peak = h.clone()
    rain = req.rain_mm_h / 3.6e6
    tt, dur, steps = 0.0, req.run_hours * 3600.0, 0
    while tt < dur:
        dt = min(solver.adaptive_dt(h), dur - tt)
        if dt <= 0:
            break
        r = rain if tt < req.storm_hours * 3600.0 else 0.0
        h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=z)
        peak = torch.maximum(peak, h)
        tt += dt; steps += 1

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
    wet = (peak > 0.10).sum(dim=(1, 2))
    options = [
        {
            "id": "base" if not any(h > 0 for h in pl) else
                  "o" + "".join(f"{j}@{h:g}" for j, h in enumerate(pl) if h > 0),
            "heights": list(pl),
            "cost_aud": sum(pl[j] * lengths[j] * unit for j in range(k)),
            "core_reduction_pct": reduction[i],
            "area_flooded_km2": round(float(wet[i]) * cell_km2, 2),
            "peak_depth_m": round(float(peak[i].max()), 2),
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

    out = {
        "ok": True,
        "name": req.name,
        "heights": hv,
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


@app.post("/analyse")
def analyse(req: AnalyseRequest) -> dict:
    key = hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:16]
    if key in CACHE:
        return {**CACHE[key], "cached": True}

    t0 = time.time()
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name=req.name)
    t_dem = time.time() - t0

    lat = (req.south + req.north) / 2
    lon = (req.west + req.east) / 2
    try:
        fc = forcing_for(lat, lon)
    except Exception:
        fc = {"total_mm": None, "peak_discharge_m3s": None}

    peak, stats = solve(dom, req.rain_mm_h, req.storm_hours, req.run_hours,
                        req.levees)

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
    wet = peak > 0.10
    out = {
        "name": req.name,
        "bounds": dom.bounds,
        "grid": list(dom.shape),
        "dx_m": dom.dx,
        "epsg": dom.epsg,
        "extent_km": [round(v, 1) for v in dom.extent_km],
        "elevation_range_m": [float(np.nanmin(dom.z)), float(np.nanmax(dom.z))],
        "flooded_km2": round(float(wet.sum()) * cell_km2, 2),
        "peak_depth_m": round(float(peak.max()), 2),
        "mean_depth_wet_m": round(float(peak[wet].mean()) if wet.any() else 0.0, 2),
        "depth_vmax_m": DEPTH_VMAX,
        "depth_png": depth_png(peak, DEPTH_VMAX),
        "terrain_png": terrain_png(dom.z),
        **roads_and_grids(dom, peak, req.west, req.south, req.east, req.north),
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
        "sources": ["Terrain: Copernicus DEM GLO-30",
                    "Forecast: Open-Meteo / GloFAS"],
    }
    if len(CACHE) > 24:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = out
    return {**out, "cached": False}
