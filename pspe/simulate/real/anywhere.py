"""Build a flood modelling domain for any land area on Earth.

The Richmond work was anchored to one archive because that is where the
reference depths were. Nothing in the solver is: shallow water does not care
where it is. What was missing was terrain and forcing for an arbitrary place,
and both are freely available without an API key:

    terrain   Copernicus DEM GLO-30, public S3, cloud-optimised GeoTIFF, global
              USGS 3DEP bare earth, public REST, 1 m, United States only
    rainfall  Open-Meteo forecast API, global
    discharge Open-Meteo flood API (GloFAS), global

This module turns a bounding box into everything `FloodSolver` needs: a bed
elevation grid in metres on a square grid, with a stated cell size and a
projection chosen for the location.

The DEM tiles are COGs, so only the window actually needed is read over HTTP
rather than the whole 28 MB tile.

## Two terrain sources, and they are not interchangeable

Copernicus GLO-30 is a Digital SURFACE Model: it includes canopy and buildings.
Measured against 2,512 council-surveyed ground levels at Lismore it sits a
median 2.21 m above the ground and is high at 97.2% of points (section 5.14e),
and measured against USGS bare earth over downtown Austin it is a median 2.99 m
high at 87.9% of cells. In a town the difference between a flood that enters a
house and one that does not is less than that, so where a bare-earth model
exists it should be used.

`source="3dep"` is that model for the United States: the USGS 3D Elevation
Program's bare-earth DEM, served as a dynamic image service at 1 m, free and
without a key. It also arrives already projected and already on the requested
grid, because the service resamples server-side -- so the tile merge and
reprojection the Copernicus path needs do not happen at all.

Copernicus remains the default because it is global and 3DEP is not.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
# The USGS 3DEP bare-earth dynamic image service. Image, Metadata and Catalog
# capabilities; 8000 px is the server's own cap per request, which is a 240 km
# box at 30 m and an 8 km one at 1 m.
TDEP = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage")
TDEP_MAX_PX = 8000


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
    # Which elevation model this came from. Carried on the Domain rather than
    # inferred by the caller, because the caller asking for "auto" does not
    # know which one it got, and a response that names the wrong DEM is a
    # provenance claim that is simply false.
    source: str = "copernicus"

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def extent_km(self) -> tuple[float, float]:
        h, w = self.z.shape
        return (w * self.dx / 1000.0, h * self.dx / 1000.0)


def build_domain_3dep(
    west: float, south: float, east: float, north: float,
    dx: float = 60.0,
    name: str = "domain",
    max_cells: int = 700,
    timeout: float = 180.0,
) -> Domain:
    """Bare-earth terrain from USGS 3DEP, already on the grid the solver wants.

    The service reprojects and resamples server-side, so this asks for the UTM
    grid directly and gets back exactly that: no tile list, no mosaic, no
    `calculate_default_transform`, and no reprojection wedges along the edges
    to fill. The whole Copernicus nodata-trench problem (a 247 m trench ringing
    Cedar Rapids, see below) does not arise here because nothing is reprojected
    locally.

    United States only. Outside its footprint the service returns nodata rather
    than an error, so an empty return is checked for and named.
    """
    import rasterio                                       # noqa: PLC0415
    import urllib.parse                                   # noqa: PLC0415
    import urllib.request                                 # noqa: PLC0415

    epsg = utm_epsg((south + north) / 2, (west + east) / 2)
    mid = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320.0 * math.cos(mid)
    height_m = (north - south) * 110_540.0
    w = max(int(round(width_m / dx)), 1)
    h = max(int(round(height_m / dx)), 1)
    if max(w, h) > max_cells:
        raise ValueError(
            f"that area needs a {w}x{h} grid at {dx} m. "
            f"Request a smaller box or a coarser cell size.")
    if max(w, h) > TDEP_MAX_PX:
        raise ValueError(
            f"3DEP serves at most {TDEP_MAX_PX} px per request; "
            f"this asks for {w}x{h}.")

    q = urllib.parse.urlencode({
        "bbox": f"{west},{south},{east},{north}", "bboxSR": 4326,
        "size": f"{w},{h}", "imageSR": epsg,
        "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "noData": -3.4028235e38, "f": "image",
    })
    req = urllib.request.Request(f"{TDEP}?{q}",
                                 headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = r.read()
    if blob[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"3DEP did not return a GeoTIFF: {blob[:200]!r}")

    with rasterio.io.MemoryFile(blob) as mem, mem.open() as src:
        z = src.read(1).astype(np.float32)
    # The service codes "outside the footprint" as a large negative, and the
    # bottom of the real range in the United States is Death Valley at -86 m.
    z[z < -500.0] = np.nan
    nodata_frac = float(np.isnan(z).mean())
    if nodata_frac > 0.999:
        raise ValueError(
            "3DEP returned no data for that box; it covers the United States "
            "only. Use source='copernicus' elsewhere.")
    if np.isnan(z).any():
        from scipy import ndimage                         # noqa: PLC0415
        holes = np.isnan(z)
        idx = ndimage.distance_transform_edt(
            holes, return_distances=False, return_indices=True)
        z = z[tuple(idx)]

    return Domain(
        z=z, dx=dx,
        bounds={"west": west, "south": south, "east": east, "north": north},
        epsg=epsg, name=name, nodata_frac=nodata_frac, source="3dep",
    )


# Terrain, kept.
#
# `build_domain` fetched the raster over HTTP every single time it was called,
# and the portal calls it once per site search, once per plan solve and once
# per section — on the same box, at the same resolution, for ground that has
# not moved.
#
# **It is not where the time goes.** Measured on the Austin box at 40 m: a
# /sites request takes 58 s cold and 55 s warm, so the fetch is about three
# seconds of it and the timestepping is the rest. What the cache buys is
# therefore not speed but reproducibility and independence — the same box
# gives byte-identical ground every time, and a USGS outage stops being able
# to change an answer or fail a solve that has already been run once. The
# wide spread seen earlier (72 s against over 300 s for the same request) was
# CPU contention between concurrent solves, not the download.
#
# Keyed on everything that changes the array and nothing that does not: the
# box, the cell size, the cap and the source — NOT the name, which is a label
# the caller chose.
_DEM_MEM: dict = {}
_DEM_DIR = Path(os.environ.get(
    "PSPE_DEM_CACHE",
    str(Path(__file__).resolve().parents[3] / "data" / "dem_cache")))


def _dem_key(west, south, east, north, dx, max_cells, source) -> str:
    raw = json.dumps([round(west, 6), round(south, 6), round(east, 6),
                      round(north, 6), round(float(dx), 3), int(max_cells),
                      source], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _dem_load(key: str, name: str) -> "Domain | None":
    hit = _DEM_MEM.get(key)
    if hit is not None:
        return replace(hit, name=name)
    f = _DEM_DIR / f"{key}.npz"
    try:
        if not f.is_file():
            return None
        with np.load(f, allow_pickle=False) as d:
            dom = Domain(
                z=d["z"], dx=float(d["dx"]), bounds=json.loads(str(d["bounds"])),
                epsg=int(d["epsg"]), name=name,
                nodata_frac=float(d["nodata_frac"]), source=str(d["source"]))
    except Exception:                                      # noqa: BLE001
        return None
    _DEM_MEM[key] = dom
    return dom


def _dem_save(key: str, dom: "Domain") -> None:
    _DEM_MEM[key] = dom
    if len(_DEM_MEM) > 8:
        _DEM_MEM.pop(next(iter(_DEM_MEM)))
    try:
        _DEM_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _DEM_DIR / f"{key}.npz", z=dom.z, dx=dom.dx,
            bounds=json.dumps(dom.bounds), epsg=dom.epsg,
            nodata_frac=dom.nodata_frac, source=dom.source)
    except Exception:                                      # noqa: BLE001
        pass


def build_domain(
    west: float, south: float, east: float, north: float,
    dx: float = 60.0,
    name: str = "domain",
    max_cells: int = 700,
    source: str = "copernicus",
) -> Domain:
    """Fetch terrain for a box and resample it to a square metre grid.

    `max_cells` caps the grid so a request cannot silently queue a solve that
    would take days: a 700x700 grid at 60 m is a 42 km domain, which is already
    at the edge of what one node solves in an hour.

    `source` picks the elevation model. "copernicus" is GLO-30, global, and a
    SURFACE model that reads metres high where there are trees and buildings.
    "3dep" is USGS bare earth at 1 m and is the right choice inside the United
    States; "auto" uses it there and falls back to Copernicus elsewhere, which
    is the setting an interface should pass. See the module docstring.
    """
    key = _dem_key(west, south, east, north, dx, max_cells, source)
    hit = _dem_load(key, name)
    if hit is not None:
        return hit

    def keep(d: Domain) -> Domain:
        _dem_save(key, d)
        return d

    if source == "3dep":
        return keep(build_domain_3dep(west, south, east, north, dx, name, max_cells))
    if source == "auto":
        # Cheap check: the conterminous US, Alaska and Hawaii boxes. A box that
        # passes but turns out to be outside the footprint raises, and the
        # fallback catches it, so this only has to be roughly right.
        lat, lon = (south + north) / 2, (west + east) / 2
        in_us = ((24 <= lat <= 50 and -125 <= lon <= -66)
                 or (51 <= lat <= 72 and -170 <= lon <= -129)
                 or (18 <= lat <= 23 and -161 <= lon <= -154))
        if in_us:
            try:
                return keep(build_domain_3dep(west, south, east, north, dx,
                                              name, max_cells))
            except Exception:
                pass
    elif source != "copernicus":
        raise ValueError(f"unknown terrain source {source!r}")
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
        epsg=epsg, name=name, nodata_frac=nodata_frac, source="copernicus",
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
