"""The flood model, marked by an instrument it did not use.

This is the module that moves the system from "shadow-capable API" to
something a record can grade, so it is worth stating exactly what makes that
possible and what does not.

## Why the upstream gauge cannot grade anything

`08158827` at Twin Creeks sits at the top of the reach and its discharge is
what DRIVES the solve. Comparing the model to it is circular: the model was
told that number. A system that reports agreement there is reporting that
arithmetic works.

## Why the outlet gauge can

`08159000` at US Hwy 183 sits 12 km downstream of the inflow, past the whole
floodplain. Nothing about it enters the solve. The stage there is a genuine
prediction -- the model routes a hydrograph across real terrain and says how
high the water stands at a point it was never told about -- and USGS has been
measuring that point every fifteen minutes for decades.

Until 2026-10-16 that gauge sat 0.13 km PAST the east edge of the study box,
so there was no solved cell at it and the comparison could not be made at all.
`build_austin_study.py` now ends the box at -97.6678, which puts the gauge
2 km inside the interior rather than on the free-outflow boundary.

## The datum, which is the part that makes it honest

A gauge reports STAGE: height above its own local datum, an arbitrary number
chosen when the instrument was installed. The model reports ELEVATION above
NAVD88. Those are not the same quantity and subtracting one from the other
would produce a confident, meaningless error.

USGS publishes the conversion. For 08159000 the gauge datum is 443.28 ft
NAVD88, so

    observed water surface (m, NAVD88) = (alt_va + gauge_height_ft) * 0.3048

and the DEM is USGS 3DEP, also NAVD88, in metres. They are then the same
quantity and the difference means something.

Two limits travel with that and are reported rather than buried: the published
datum has an accuracy of 0.17 ft (5 cm), which is the floor on any agreement
this can show; and a 40 m cell averages the channel with its banks, so the
model's "stage" at a gauge is a cell mean, not a thalweg reading.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/shadow", tags=["shadow"])

FT_TO_M = 0.3048
#: USGS parameter codes. 00065 is gauge height in feet, 00060 is discharge.
P_STAGE, P_Q = "00065", "00060"


class DatumUnavailable(Exception):
    """The gauge publishes no datum, so its stage cannot be put on NAVD88."""


@lru_cache(maxsize=32)
def gauge_datum(site: str) -> dict:
    """Altitude of the gauge datum, from the USGS site service.

    Without this a stage reading cannot be compared to a modelled elevation.
    Raises rather than defaulting: a guessed datum turns a 4 m disagreement
    into a 0.2 m agreement and nothing in the output would show it.
    """
    from .app import _get                                      # noqa: PLC0415

    rdb = _get("https://waterservices.usgs.gov/nwis/site/?format=rdb"
               f"&sites={site}&siteOutput=expanded")
    rows = [ln.split("\t") for ln in rdb.splitlines()
            if ln and not ln.startswith("#")]
    if len(rows) < 3:
        raise DatumUnavailable(f"no site record for {site}")
    hdr, val = rows[0], rows[2]
    rec = dict(zip(hdr, val))
    alt = rec.get("alt_va", "").strip()
    if not alt:
        raise DatumUnavailable(
            f"USGS publishes no gauge datum (alt_va) for {site}, so its stage "
            f"cannot be placed on NAVD88 and the model cannot be compared "
            f"against it.")
    return {
        "site": site,
        "name": rec.get("station_nm", "").strip(),
        "lat": float(rec["dec_lat_va"]), "lon": float(rec["dec_long_va"]),
        "datum_ft": float(alt),
        "datum_m": float(alt) * FT_TO_M,
        "datum_name": rec.get("alt_datum_cd", "").strip(),
        "datum_accuracy_ft": float(rec.get("alt_acy_va", "0") or 0),
        "drainage_sq_mi": float(rec.get("drain_area_va", "0") or 0),
    }


def cell_of(bounds: dict, shape: tuple[int, int],
            lat: float, lon: float) -> "tuple[int, int] | None":
    """Grid cell containing a point, or None if it falls outside.

    Row 0 is the NORTH edge, matching the image convention the rest of the
    service uses for depth rasters.
    """
    h, w = shape
    if not (bounds["west"] <= lon <= bounds["east"]
            and bounds["south"] <= lat <= bounds["north"]):
        return None
    fx = (lon - bounds["west"]) / (bounds["east"] - bounds["west"])
    fy = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    return (min(max(int(fy * h), 0), h - 1), min(max(int(fx * w), 0), w - 1))


def edge_distance_km(bounds: dict, lat: float, lon: float) -> float:
    """How far inside the box a point sits. Negative means outside.

    Reported because a gauge one cell from a free-outflow boundary is carrying
    the boundary condition rather than a routed stage, and a comparison made
    there would be marking the boundary treatment, not the model.
    """
    km_lon = 111.32 * float(np.cos(np.radians(lat)))
    return min((lon - bounds["west"]) * km_lon,
               (bounds["east"] - lon) * km_lon,
               (lat - bounds["south"]) * 111.32,
               (bounds["north"] - lat) * 111.32)


class ShadowRequest(BaseModel):
    """Run the model over a past event and mark it against the outlet gauge."""
    west: float = -97.820
    south: float = 30.110
    east: float = -97.6678
    north: float = 30.210
    #: Day of the event, YYYY-MM-DD. The window runs from the previous
    #: midnight so the rise is included rather than only the recession.
    day: str
    #: Which gauge drives the solve, and which one marks it. They must differ:
    #: a model told a number cannot be graded by that number.
    inflow_site: str = "08158827"
    outlet_site: str = "08159000"
    dx: float = Field(40.0, ge=20.0, le=120.0)
    run_hours: float = Field(24.0, ge=6.0, le=48.0)


@router.post("/stage")
def stage(req: ShadowRequest) -> dict:
    """Predicted stage at the outlet gauge against what the gauge recorded.

    The honest shape of this endpoint is a list of refusals before a number.
    It will decline when the two sites are the same, when either publishes no
    datum, when the outlet gauge is outside the box or sitting on its
    boundary, and when the gauge has no record on the day asked for. Each of
    those produces a wrong answer rather than no answer if it is skipped.
    """
    from .app import (                                         # noqa: PLC0415
        PlanRequest, build_domain, build_forcing, discharge_series_day, solve,
    )

    t0 = time.time()
    if req.inflow_site == req.outlet_site:
        return {"ok": False, "reason": (
            "The driving gauge and the marking gauge are the same site. The "
            "model is told the inflow, so comparing it there measures "
            "arithmetic rather than the model.")}

    try:
        inflow = gauge_datum(req.inflow_site)
        outlet = gauge_datum(req.outlet_site)
    except DatumUnavailable as exc:
        return {"ok": False, "reason": str(exc)}

    bounds = {"west": req.west, "south": req.south,
              "east": req.east, "north": req.north}
    inside_km = edge_distance_km(bounds, outlet["lat"], outlet["lon"])
    if inside_km <= 0:
        return {"ok": False, "reason": (
            f"{outlet['name']} is {abs(inside_km):.2f} km OUTSIDE this box, so "
            f"there is no solved cell at it. Widen the box until the gauge is "
            f"inside the interior.")}
    if inside_km < 0.5:
        return {"ok": False, "reason": (
            f"{outlet['name']} is only {inside_km:.2f} km inside the edge. The "
            f"boundary is free-outflow, so a cell that close carries the "
            f"outflow condition rather than a routed stage; marking the model "
            f"there would be marking the boundary treatment.")}

    # What the creek actually did that day, at both ends.
    drive = discharge_series_day(req.inflow_site, req.day)
    obs = discharge_series_day(req.outlet_site, req.day)
    if len(drive) < 8:
        return {"ok": False, "reason": (
            f"{inflow['name']} has no usable record on {req.day}, so there is "
            f"nothing to drive the model with.")}

    peak_q = max(r["q_m3s"] for r in drive)
    if peak_q <= 0:
        return {"ok": False, "reason": (
            f"The driving gauge read zero all day on {req.day}.")}

    # Solve the reach with that peak as a steady river inflow. A full unsteady
    # replay of the hydrograph is the right experiment and is not this one;
    # this asks whether the model puts the water at the right HEIGHT at the
    # far end when the creek is carrying a known discharge.
    dom = build_domain(req.west, req.south, req.east, req.north,
                       dx=req.dx, name="shadow", source="auto")
    cell = cell_of(bounds, dom.z.shape, outlet["lat"], outlet["lon"])
    if cell is None:
        return {"ok": False, "reason": "the outlet gauge fell outside the grid"}

    plan = PlanRequest(west=req.west, south=req.south, east=req.east,
                       north=req.north, name="shadow", dx=dom.dx,
                       driver="river", river_q_m3s=peak_q,
                       storm_hours=6.0, run_hours=req.run_hours, sites=[])
    peak, info, _ = solve(dom, 0.0, 6.0, req.run_hours, [], req=plan)

    r, c = cell
    bed_m = float(dom.z[r, c])
    depth_m = float(peak[r, c])
    model_wse_m = bed_m + depth_m

    # Observed water surface, on the same datum. This is the conversion the
    # module docstring exists for.
    stage_rows = _stage_series(req.outlet_site, req.day)
    if not stage_rows:
        return {"ok": False, "reason": (
            f"{outlet['name']} recorded no gauge height on {req.day}, so there "
            f"is nothing to mark the model against. It may record discharge "
            f"without stage, which cannot be compared to an elevation.")}
    peak_stage_ft = max(r_["ft"] for r_ in stage_rows)
    obs_wse_m = (outlet["datum_ft"] + peak_stage_ft) * FT_TO_M

    err = model_wse_m - obs_wse_m
    floor_m = outlet["datum_accuracy_ft"] * FT_TO_M

    return {
        "ok": True,
        "day": req.day,
        "driven_by": {"site": inflow["site"], "name": inflow["name"],
                      "peak_q_m3s": round(peak_q, 1)},
        "marked_by": {"site": outlet["site"], "name": outlet["name"],
                      "km_inside_box": round(inside_km, 2),
                      "datum_m_navd88": round(outlet["datum_m"], 2),
                      "datum_accuracy_m": round(floor_m, 3)},
        "model": {"bed_m": round(bed_m, 2), "depth_m": round(depth_m, 2),
                  "water_surface_m": round(model_wse_m, 2),
                  "cell": [r, c], "dx_m": dom.dx},
        "observed": {"peak_stage_ft": round(peak_stage_ft, 2),
                     "water_surface_m": round(obs_wse_m, 2),
                     "readings": len(stage_rows)},
        "error_m": round(err, 2),
        "within_gauge_accuracy": bool(abs(err) <= floor_m),
        "peak_q_at_outlet_m3s": (round(max(r_["q_m3s"] for r_ in obs), 1)
                                 if obs else None),
        "seconds": round(time.time() - t0, 1),
        "note": ("The model is driven by the UPSTREAM gauge and marked by the "
                 "DOWNSTREAM one, which it was never told about. Both are on "
                 "NAVD88: the gauge's own stage is converted with the datum "
                 "USGS publishes for it. Two limits travel with the number -- "
                 f"the published datum is accurate to {floor_m:.2f} m, which "
                 f"is the floor on any agreement, and a {dom.dx:.0f} m cell "
                 "averages the channel with its banks, so this is a cell mean "
                 "rather than a thalweg reading."),
    }


def _stage_series(site: str, day: str) -> list[dict]:
    """Gauge height in feet over one day. Separate from discharge."""
    from .app import _get                                      # noqa: PLC0415

    try:
        import json as _json
        url = ("https://waterservices.usgs.gov/nwis/iv/?format=json"
               f"&sites={site}&startDT={day}T00:00&endDT={day}T23:59"
               f"&parameterCd={P_STAGE}")
        ts = _json.loads(_get(url))["value"]["timeSeries"]
        if not ts:
            return []
        out = []
        for v in ts[0]["values"][0]["value"]:
            try:
                ft = float(v["value"])
            except (TypeError, ValueError):
                continue
            if ft > -100:
                out.append({"at": v["dateTime"][:16], "ft": ft})
        return out
    except Exception:                                          # noqa: BLE001
        return []
