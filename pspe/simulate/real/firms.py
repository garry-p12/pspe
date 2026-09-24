"""NASA FIRMS active-fire detections, gridded into daily rasters.

NDWS gives independent (day t, day t+1) pairs with no date, no fire id and no
location, so nothing in it can be chained. A digital twin's defining loop —
observe, estimate, forecast, observe again, correct — needs a *sequence* of
real observations of the *same* fire, which is what this module builds.

FIRMS serves per-detection records (lat, lon, acquisition date/time, brightness,
fire radiative power) from VIIRS and MODIS. `fetch_fire` pulls a bounding box
over a date range and `grid_sequence` rasterises each day onto a fixed grid, so
a fire becomes a (days, H, W) stack directly comparable to an NDWS FireMask.

Two properties of this data matter and are preserved rather than smoothed away:

* **Days go missing.** A satellite overpass can be cloud-blocked or simply not
  cover the box. The Dixie Fire's 2021-07-27 has zero detections between days
  with 1,558 and 491. Those days are returned as `observed=False` rather than
  as empty fire, because "we did not look" and "nothing burned" are different
  facts and a twin that confuses them will drift.
* **Detections are instantaneous, not cumulative.** A pixel is reported while
  it is actively burning at overpass, so the raster is a snapshot of the front,
  not the burn scar. That is the same quantity NDWS's FireMask carries.

The API key is read from `FIRMS_MAP_KEY` or `~/.firms_map_key` and is never
written into the repository.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAYS_PER_CALL = 5          # the endpoint refuses more
CACHE = Path(__file__).resolve().parents[3] / "data" / "firms"


def map_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        return key.strip()
    path = Path.home() / ".firms_map_key"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        "No FIRMS key. Set FIRMS_MAP_KEY or put the key in ~/.firms_map_key "
        "(free from https://firms.modaps.eosdis.nasa.gov/api/map_key/)."
    )


@dataclass(frozen=True)
class FireSpec:
    """A named fire: where to look, when, and for how long."""
    name: str
    west: float
    south: float
    east: float
    north: float
    start: str                  # YYYY-MM-DD
    days: int
    source: str = "VIIRS_SNPP_SP"


# Large, long-burning, well-documented US fires. Boxes are drawn generously so
# the front stays inside the frame for the whole window.
FIRES: tuple[FireSpec, ...] = (
    FireSpec("dixie",        -121.8, 39.7, -120.2, 40.8, "2021-07-14", 20),
    FireSpec("bootleg",      -121.6, 42.2, -120.4, 42.9, "2021-07-07", 20),
    FireSpec("caldor",       -120.7, 38.5, -119.8, 39.0, "2021-08-15", 20),
    FireSpec("august_cplx",  -123.6, 39.2, -122.4, 40.4, "2020-08-17", 20),
    FireSpec("creek",        -119.7, 36.9, -118.9, 37.5, "2020-09-05", 20),
    FireSpec("cameron_peak", -106.1, 40.3, -105.2, 40.9, "2020-08-14", 20),
    FireSpec("camp",         -121.8, 39.6, -121.2, 39.95, "2018-11-08", 14),
    FireSpec("mosquito",     -121.0, 38.8, -120.3, 39.3, "2022-09-06", 18),
)


def _fetch_chunk(spec: FireSpec, start: date, days: int, key: str) -> list[dict]:
    box = f"{spec.west},{spec.south},{spec.east},{spec.north}"
    url = f"{API}/{key}/{spec.source}/{box}/{days}/{start.isoformat()}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = resp.read().decode("utf-8", "replace")
    if not text.lstrip().lower().startswith("latitude"):
        raise RuntimeError(f"FIRMS returned: {text.strip()[:160]}")
    return list(csv.DictReader(io.StringIO(text)))


def fetch_fire(spec: FireSpec, cache: bool = True) -> list[dict]:
    """Every detection in the box over the window, oldest first."""
    path = CACHE / f"{spec.name}_{spec.source}_{spec.start}_{spec.days}.csv"
    if cache and path.exists():
        with path.open() as fh:
            return list(csv.DictReader(fh))

    key = map_key()
    begin = date.fromisoformat(spec.start)
    rows: list[dict] = []
    for offset in range(0, spec.days, MAX_DAYS_PER_CALL):
        n = min(MAX_DAYS_PER_CALL, spec.days - offset)
        rows += _fetch_chunk(spec, begin + timedelta(days=offset), n, key)
    rows.sort(key=lambda r: (r["acq_date"], r["acq_time"]))

    if cache and rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows


@dataclass
class FireSequence:
    """One fire as daily rasters, plus which days were actually observed."""
    name: str
    dates: list[str]
    fire: np.ndarray            # (T, H, W) float32 in [0, 1], fraction-of-cell burning
    frp: np.ndarray             # (T, H, W) float32, fire radiative power (MW)
    observed: np.ndarray        # (T,) bool — False means no overpass/no data that day
    bounds: tuple[float, float, float, float] = field(default=(0, 0, 0, 0))

    @property
    def grid(self) -> int:
        return self.fire.shape[-1]


def grid_sequence(spec: FireSpec, rows: list[dict], grid: int = 64,
                  min_confidence: str = "n") -> FireSequence:
    """Rasterise detections onto `grid` x `grid` cells, one frame per day.

    VIIRS confidence is categorical (l/n/h). Low-confidence detections are
    dropped by default because they include a lot of sun glint and hot bare
    ground; keeping them roughly doubles the apparent fire area.
    """
    order = {"l": 0, "n": 1, "h": 2}
    keep_from = order.get(min_confidence, 1)

    begin = date.fromisoformat(spec.start)
    days = [(begin + timedelta(days=i)).isoformat() for i in range(spec.days)]
    index = {d: i for i, d in enumerate(days)}

    fire = np.zeros((len(days), grid, grid), dtype=np.float32)
    frp = np.zeros((len(days), grid, grid), dtype=np.float32)
    seen = np.zeros(len(days), dtype=bool)

    for r in rows:
        t = index.get(r["acq_date"])
        if t is None:
            continue
        conf = str(r.get("confidence", "n")).strip().lower()[:1]
        if order.get(conf, 1) < keep_from:
            continue
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if not (spec.south <= lat <= spec.north and spec.west <= lon <= spec.east):
            continue
        # Row 0 is the north edge, matching image convention.
        y = int((spec.north - lat) / (spec.north - spec.south) * grid)
        x = int((lon - spec.west) / (spec.east - spec.west) * grid)
        y, x = min(max(y, 0), grid - 1), min(max(x, 0), grid - 1)
        fire[t, y, x] += 1.0
        try:
            frp[t, y, x] += float(r.get("frp") or 0.0)
        except ValueError:
            pass
        seen[t] = True

    # Several detections can land in one cell; saturate rather than sum, so the
    # channel stays comparable to NDWS's binary-ish mask.
    fire = np.clip(fire, 0.0, 3.0) / 3.0
    return FireSequence(spec.name, days, fire, frp, seen,
                        (spec.west, spec.south, spec.east, spec.north))


def load_sequences(specs=FIRES, grid: int = 64, cache: bool = True) -> list[FireSequence]:
    out = []
    for spec in specs:
        try:
            rows = fetch_fire(spec, cache=cache)
        except Exception as exc:                      # one bad fire must not sink the set
            print(f"[firms] {spec.name}: {exc}")
            continue
        seq = grid_sequence(spec, rows, grid=grid)
        if seq.observed.sum() >= 6:                   # needs a usable run of days
            out.append(seq)
        else:
            print(f"[firms] {spec.name}: only {int(seq.observed.sum())} observed days, skipped")
    return out
