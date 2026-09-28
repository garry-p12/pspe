"""Build a flood modelling domain for any land area on Earth.

The Richmond work was anchored to one archive because that is where the
reference depths were. Nothing in the solver is: shallow water does not care
where it is. What was missing was terrain and forcing for an arbitrary place,
and both are freely available without an API key:

    terrain   Copernicus DEM GLO-30, public S3, cloud-optimised GeoTIFF, global
    rainfall  Open-Meteo forecast API, global
    discharge Open-Meteo flood API (GloFAS), global

This module turns a bounding box into everything `FloodSolver` needs: a bed
elevation grid in metres on a square grid, with a stated cell size and a
projection chosen for the location.

The DEM tiles are COGs, so only the window actually needed is read over HTTP
rather than the whole 28 MB tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

S3 = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
            f"{ew}{abs(lon):03d}_00_DEM")


def tiles_for(west: float, south: float, east: float, north: float) -> list[str]:
    """Every 1x1 degree Copernicus tile touching the box."""
    out = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            out.append(tile_name(lat, lon))
    return out


def utm_epsg(lat: float, lon: float) -> int:
    """The UTM zone for a point, so the solver works in metres."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Domain:
    """A modelling domain ready for the solver."""

    z: np.ndarray            # (H, W) bed elevation, metres
    dx: float                # cell size, metres
    bounds: dict             # WGS84 bounds of the grid
    epsg: int                # projected CRS used
    name: str
    nodata_frac: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def extent_km(self) -> tuple[float, float]:
        h, w = self.z.shape
        return (w * self.dx / 1000.0, h * self.dx / 1000.0)


def build_domain(
    west: float, south: float, east: float, north: float,
    dx: float = 60.0,
    name: str = "domain",
    max_cells: int = 700,
) -> Domain:
    """Fetch terrain for a box and resample it to a square metre grid.

    `max_cells` caps the grid so a request cannot silently queue a solve that
    would take days: a 700x700 grid at 60 m is a 42 km domain, which is already
    at the edge of what one node solves in an hour.
    """
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from rasterio.merge import merge

    epsg = utm_epsg((south + north) / 2, (west + east) / 2)
    names = tiles_for(west, south, east, north)
    if len(names) > 6:
        raise ValueError(f"box spans {len(names)} DEM tiles; request a smaller area")

    srcs = []
    for n in names:
        url = f"{S3}/{n}/{n}.tif"
        try:
            srcs.append(rasterio.open(url))
        except Exception:
            # Ocean and some polar tiles simply do not exist.
            continue
    if not srcs:
        raise ValueError("no Copernicus DEM tiles cover that area (all ocean?)")

    mosaic, transform = merge(srcs, bounds=(west, south, east, north))
    src_crs = srcs[0].crs
    profile = srcs[0].profile
    for s in srcs:
        s.close()

    band = mosaic[0]
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, f"EPSG:{epsg}", band.shape[1], band.shape[0],
        left=west, bottom=south, right=east, top=north, resolution=dx,
    )
    if max(dst_w, dst_h) > max_cells:
        raise ValueError(
            f"that area needs a {dst_w}x{dst_h} grid at {dx} m. "
            f"Request a smaller box or a coarser cell size."
        )

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=band, destination=dst,
        src_transform=transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=f"EPSG:{epsg}",
        resampling=Resampling.bilinear,
        src_nodata=profile.get("nodata"), dst_nodata=np.nan,
    )

    nodata_frac = float(np.isnan(dst).mean())
    # Nodata is UNKNOWN GROUND, not sea level.
    #
    # Filling it with zero was right for a coastal box -- Copernicus reads the
    # ocean as nodata -- and catastrophic anywhere else. Reprojecting WGS84
    # tiles into UTM leaves wedge-shaped gaps along the edges, and at Cedar
    # Rapids, which sits at 247 m, those became a 247 m trench ringing the
    # domain: 3.6% of cells at exactly 0 m, covering 84-100% of every boundary,
    # an artificial sink for the whole catchment to drain into.
    #
    # Nearest-neighbour fill instead. It extends the real terrain outward, so
    # inland ground stays inland, and a genuine coastline -- where the nearest
    # valid cells are already near zero -- still fills to about sea level.
    if np.isnan(dst).any():
        from scipy import ndimage
        holes = np.isnan(dst)
        if holes.all():
            raise RuntimeError("DEM covers none of this box")
        idx = ndimage.distance_transform_edt(
            holes, return_distances=False, return_indices=True)
        dst = dst[tuple(idx)]

    return Domain(
        z=dst, dx=dx,
        bounds={"west": west, "south": south, "east": east, "north": north},
        epsg=epsg, name=name, nodata_frac=nodata_frac,
    )


def forcing_for(lat: float, lon: float, days: int = 7) -> dict:
    """Live rainfall and river-discharge forecast for a location."""
    import json
    import urllib.request

    def get(u: str):
        req = urllib.request.Request(u, headers={"User-Agent": "PSPE/1.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)

    met = get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
              f"&hourly=precipitation&forecast_days={days}&timezone=auto")
    fl = get(f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}"
             f"&daily=river_discharge_max&forecast_days={days}")
    pr = [p or 0.0 for p in met["hourly"]["precipitation"]]
    q = [v or 0.0 for v in fl.get("daily", {}).get("river_discharge_max", [])]
    return {
        "hourly_precip_mm": pr,
        "total_mm": sum(pr),
        "peak_discharge_m3s": max(q) if q else 0.0,
        "timezone": met.get("timezone"),
    }
