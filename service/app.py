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
import math
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
