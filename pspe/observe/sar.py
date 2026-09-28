"""Observed flood extent from Sentinel-1 radar.

Every flood number this project has produced so far was scored against another
*model*. The solver reaches CSI 0.979 against FloodCastBench's reference, but
that reference is itself a shallow-water solution: two models agreeing is not
evidence either matches the world. This module supplies the missing term — what
a satellite actually saw.

Radar is the right instrument because floods come with cloud. Sentinel-1 sees
through it, and open water is almost black to radar: a smooth surface reflects
the pulse away from the sensor instead of scattering it back.

Products are taken as **RTC** (radiometrically terrain corrected) from Planetary
Computer, which means the two hardest preprocessing steps — radiometric
calibration and terrain correction against a DEM — are already applied. What is
left is the part that carries judgement, and that is what lives here.

Two rules the detection follows, both of which matter:

1. **Compare like with like.** Sentinel-1 repeats on a 12-day cycle, and the
   same ground is seen from different tracks at different incidence angles.
   Backscatter depends on that angle, so a flood scene is only ever differenced
   against a baseline from the SAME relative orbit. Ignoring this produces
   "flooding" wherever the geometry changed.

2. **Separate flood from permanent water.** A river is dark before the flood as
   well as during it. Flooding is where water is present now and was not before,
   which needs both scenes, not a threshold on one.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION = "sentinel-1-rtc"


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=90) as r:
        return json.load(r)


_token: dict[str, Any] = {}


def sas_token(collection: str = COLLECTION) -> str:
    """A signing token, cached until shortly before it expires."""
    now = datetime.now(timezone.utc)
    t = _token.get(collection)
    if t and t["expiry"] > now:
        return t["token"]
    d = _get(f"{SAS}/{collection}")
    exp = datetime.fromisoformat(d["msft:expiry"].replace("Z", "+00:00"))
    _token[collection] = {"token": d["token"], "expiry": exp}
    return d["token"]


@dataclass
class Scene:
    id: str
    datetime: str
    orbit_state: str
    relative_orbit: int | None
    assets: dict

    @property
    def date(self) -> str:
        return self.datetime[:10]

    def href(self, band: str = "vv") -> str:
        return f"{self.assets[band]['href']}?{sas_token()}"


def search(bbox: list[float], start: str, end: str, limit: int = 100) -> list[Scene]:
    """Scenes intersecting a box in a date range, oldest first."""
    d = _post(f"{STAC}/search", {
        "collections": [COLLECTION], "bbox": bbox,
        "datetime": f"{start}/{end}", "limit": limit,
    })
    out = []
    for f in d.get("features", []):
        p = f["properties"]
        out.append(Scene(
            id=f["id"], datetime=p["datetime"],
            orbit_state=p.get("sat:orbit_state", "?"),
            relative_orbit=p.get("sat:relative_orbit"),
            assets=f["assets"],
        ))
    return sorted(out, key=lambda s: s.datetime)


def same_track(a: Scene, b: Scene) -> bool:
    """Whether two scenes share a viewing geometry.

    Prefers the relative orbit number. Where that is absent, acquisition
    time-of-day is a reliable stand-in: a given track always crosses at very
    nearly the same local time.
    """
    if a.relative_orbit is not None and b.relative_orbit is not None:
        return a.relative_orbit == b.relative_orbit
    if a.orbit_state != b.orbit_state:
        return False
    ta = datetime.fromisoformat(a.datetime.replace("Z", "+00:00"))
    tb = datetime.fromisoformat(b.datetime.replace("Z", "+00:00"))
    return abs((ta.hour * 60 + ta.minute) - (tb.hour * 60 + tb.minute)) <= 3


def pick_pair(scenes: list[Scene], flood_date: str,
              min_days_before: int = 6) -> tuple[Scene, list[Scene]]:
    """The scene nearest the flood, and the same-track scenes before it.

    Several baselines are returned rather than one because a single pre-image
    carries its own speckle and its own weather. Taking a median across them
    gives a far steadier reference for what "normally dry" looks like.
    """
    target = datetime.fromisoformat(flood_date + "T00:00:00+00:00")
    flood = min(scenes, key=lambda s: abs(
        (datetime.fromisoformat(s.datetime.replace("Z", "+00:00")) - target).total_seconds()))
    ft = datetime.fromisoformat(flood.datetime.replace("Z", "+00:00"))
    base = [
        s for s in scenes
        if same_track(s, flood)
        and (ft - datetime.fromisoformat(s.datetime.replace("Z", "+00:00"))).days >= min_days_before
    ]
    return flood, base


def read_bbox(scene: Scene, bbox: list[float], band: str = "vv",
              out_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, Any]:
    """Read only the window covering `bbox` (WGS84) from a scene.

    Scenes are ~30000x23000 pixels; reading whole ones to look at a town would
    be absurd. These are COGs, so the window is fetched over HTTP directly.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(scene.href(band)) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(left, bottom, right, top, src.transform)
        shape = out_shape or (int(win.height), int(win.width))
        arr = src.read(1, window=win, out_shape=shape,
                       resampling=rasterio.enums.Resampling.average,
                       boundless=True, fill_value=np.nan)
        tf = src.window_transform(win)
        # Rescale the transform when the read was decimated.
        sx = win.width / shape[1]
        sy = win.height / shape[0]
        tf = tf * rasterio.Affine.scale(sx, sy)
        return arr.astype(np.float32), (tf, src.crs)


def to_db(gamma0: np.ndarray) -> np.ndarray:
    """Backscatter in decibels. RTC products are linear power."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(gamma0 > 0, gamma0, np.nan))


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold, for splitting a bimodal backscatter histogram.

    Flooded scenes are bimodal by nature — dark water against brighter land —
    so the split is found from the data rather than fixed in advance. A fixed
    dB threshold travels badly between land covers and incidence angles.
    """
    v = values[np.isfinite(values)]
    if v.size < 100:
        return float("nan")
    lo, hi = np.percentile(v, [1, 99])
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    w = np.cumsum(hist)
    total = w[-1]
    if total <= 0:
        return float("nan")
    centres = (edges[:-1] + edges[1:]) / 2
    cw = np.cumsum(hist * centres)
    mean_total = cw[-1] / total
    w0 = w / total
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = cw / w
        m1 = (cw[-1] - cw) / (total - w)
    var = w0 * w1 * (m0 - m1) ** 2
    var[~np.isfinite(var)] = -1
    return float(centres[int(np.argmax(var))])


@dataclass
class FloodMap:
    water_now: np.ndarray       # bool: open water in the flood scene
    permanent: np.ndarray       # bool: water present before as well
    flooded: np.ndarray         # bool: water now that was not there before
    threshold_db: float
    flood_db: np.ndarray
    baseline_db: np.ndarray
    transform: Any
    crs: Any
    meta: dict

    @property
    def flooded_km2(self) -> float:
        px = abs(self.transform[0] * self.transform[4]) / 1e6
        return float(self.flooded.sum()) * px


def detect(flood: Scene, baselines: list[Scene], bbox: list[float],
           band: str = "vv", out_shape: tuple[int, int] | None = None,
           drop_db: float = 3.0, threshold_db: float | None = None) -> FloodMap:
    """Flood extent as the water that is present now and was not before.

    Two conditions have to hold together. The pixel must be dark *now*, by a
    threshold Otsu picks from this scene's own histogram; and it must have
    fallen by `drop_db` against the pre-flood median. The first alone marks
    every smooth surface — tarmac, sand, the ocean — as flood. The second alone
    marks speckle. Requiring both is what leaves standing water.
    """
    f_lin, (tf, crs) = read_bbox(flood, bbox, band, out_shape)
    shape = f_lin.shape
    stack = []
    for b in baselines:
        arr, _ = read_bbox(b, bbox, band, shape)
        stack.append(arr)
    if not stack:
        raise ValueError("no same-track baseline scenes; cannot separate flood "
                         "from permanent water")
    base_lin = np.nanmedian(np.stack(stack), axis=0)

    f_db = to_db(f_lin)
    b_db = to_db(base_lin)
    thr = threshold_db if threshold_db is not None else otsu(f_db)

    water_now = np.isfinite(f_db) & (f_db < thr)
    water_before = np.isfinite(b_db) & (b_db < thr)
    dropped = np.isfinite(f_db) & np.isfinite(b_db) & ((b_db - f_db) > drop_db)

    flooded = water_now & dropped & ~water_before
    permanent = water_now & water_before

    return FloodMap(
        water_now=water_now, permanent=permanent, flooded=flooded,
        threshold_db=thr, flood_db=f_db, baseline_db=b_db,
        transform=tf, crs=crs,
        meta={
            "flood_scene": flood.id, "flood_datetime": flood.datetime,
            "baseline_scenes": [b.id for b in baselines],
            "n_baselines": len(baselines),
            "band": band, "drop_db": drop_db,
            "threshold_db": thr, "shape": list(shape),
            "orbit_state": flood.orbit_state,
            "relative_orbit": flood.relative_orbit,
        },
    )


# --------------------------------------------------------------------------- #
# Where radar can and cannot see
# --------------------------------------------------------------------------- #
# ESA WorldCover classes, 10 m.
WORLDCOVER = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 70: "snow/ice", 80: "permanent water",
    90: "herbaceous wetland", 95: "mangroves", 100: "moss/lichen",
}

# C-band radar cannot reliably see standing water under a canopy, and in built-up
# areas flooded structures produce a double-bounce that makes them BRIGHTER
# rather than darker. Comparing a model against radar over these covers charges
# the model for error the instrument cannot adjudicate, so they are reported
# separately rather than silently counted as disagreement.
SAR_BLIND = {10, 50, 95}                  # tree cover, built-up, mangroves
SAR_VISIBLE = {20, 30, 40, 60, 90, 100}   # open ground the sensor can judge


def read_worldcover(bbox: list[float], out_shape: tuple[int, int],
                    dst_crs: Any = None, dst_transform: Any = None) -> np.ndarray:
    """ESA WorldCover over a box, on a grid you specify.

    Returned as class codes, not a mask, so the caller decides which covers to
    trust rather than having that judgement baked in here.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds

    d = _post(f"{STAC}/search", {
        "collections": ["esa-worldcover"], "bbox": bbox, "limit": 10,
    })
    feats = d.get("features", [])
    if not feats:
        raise ValueError("no WorldCover tile covers that box")
    tok = sas_token("esa-worldcover")

    out = np.zeros(out_shape, dtype=np.uint8)
    for f in feats:
        href = f["assets"]["map"]["href"] + "?" + tok
        with rasterio.open(href) as src:
            if dst_crs is None:
                left, bottom, right, top = bbox
                win = from_bounds(left, bottom, right, top, src.transform)
                a = src.read(1, window=win, out_shape=out_shape,
                             resampling=Resampling.nearest,
                             boundless=True, fill_value=0)
                out = np.where(out == 0, a, out)
            else:
                tmp = np.zeros(out_shape, dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, 1), destination=tmp,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=Resampling.nearest, dst_nodata=0,
                )
                out = np.where(out == 0, tmp, out)
    return out


def stratify(model_mask: np.ndarray, sar_mask: np.ndarray,
             cover: np.ndarray) -> dict:
    """Score a model against radar separately where radar can see.

    The headline comparison mixes two very different situations: ground the
    sensor can adjudicate, and ground where its silence means nothing. Splitting
    them is the difference between "the model over-predicts" and "the model
    over-predicts where we can actually check".
    """
    def scores(sel):
        m = model_mask & sel
        s = sar_mask & sel
        h = int(np.logical_and(m, s).sum())
        miss = int(np.logical_and(~m, s).sum())
        fa = int(np.logical_and(m, ~s).sum())
        denom = h + miss + fa
        return {
            "pixels": int(sel.sum()), "hits": h, "misses": miss,
            "false_alarms": fa,
            "csi": h / denom if denom else float("nan"),
            "pod": h / (h + miss) if (h + miss) else float("nan"),
            "far": fa / (h + fa) if (h + fa) else float("nan"),
        }

    visible = np.isin(cover, list(SAR_VISIBLE))
    blind = np.isin(cover, list(SAR_BLIND))
    per_class = {}
    for code, name in WORLDCOVER.items():
        sel = cover == code
        if sel.sum() > 500:
            per_class[name] = scores(sel)
    return {
        "all": scores(np.ones_like(cover, dtype=bool)),
        "sar_visible": scores(visible),
        "sar_blind": scores(blind),
        "per_class": per_class,
        "visible_frac": float(visible.mean()),
        "blind_frac": float(blind.mean()),
    }
