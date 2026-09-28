"""WildfireSpreadTS as daily fire sequences, in the shape the twin loop expects.

    from pspe.simulate.real.wsts import load_sequences
    seqs = load_sequences(root="data/wsts/WildfireSpreadTS", grid=64)

Why this exists: `firms.py` builds sequences from NASA FIRMS because NDWS cannot
be chained. It works, but it covers eight fires. WildfireSpreadTS
[Gerard et al., NeurIPS 2023 D&B] was built for exactly this and covers **607
fires, 13,607 daily images, 2018-2021**, so the paired test that §5.2 reports on
8 fires can be run at roughly 75x the sample size on a benchmark reviewers
already know.

The output is `firms.FireSequence`, deliberately: the twin loop then runs
unchanged and the two datasets are provably scored by the same code.

Layout, from the dataset documentation: one folder per fire, one GeoTIFF per
day named by date, 23 channels in a fixed order with **active fire last**.
Everything else in the file is there because the archive is real data:

* Fires differ in raster size, so each is block-averaged onto a common grid.
  Averaging a binary detection mask gives fraction-of-cell-burning, which is
  what `FireSequence.fire` means for FIRMS too.
* **NaN in the active-fire channel means "not detected", not "not observed".**
  Measured over 890 days sampled across 40 fires: the VIIRS reflectance and
  terrain bands are finite on 100% of pixels on a normal day, the fire channel
  is finite on 0.0-0.2% of them, and **37.4% of days carry zero fire pixels
  while being perfectly good observations**. Deriving observability from the
  fire channel would therefore discard a third of the record -- specifically
  the days a fire went out, which is the part of carrying state forward that is
  actually hard. Observability is read from the reflectance bands instead, and
  only **2.2%** of days are genuinely missing (cloud or sensor failure).
* Dates come from the filename and are sorted, because a fire's folder is not
  ordered on disk and an out-of-order sequence would make the loop meaningless.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np

from .firms import FireSequence

# The 23 channels in file order, from the dataset documentation. Only the last
# is read here; the rest are named so a later experiment can index them without
# re-deriving the order.
CHANNELS = (
    "VIIRS M11", "VIIRS I2", "VIIRS I1", "NDVI", "EVI2",
    "total precipitation", "wind speed", "wind direction",
    "min temperature", "max temperature", "energy release component",
    "specific humidity", "slope", "aspect", "elevation",
    "Palmer drought severity index", "landcover class",
    "forecast total precipitation", "forecast wind speed",
    "forecast wind direction", "forecast temperature",
    "forecast specific humidity", "active fire",
)
FIRE_BAND = len(CHANNELS)          # rasterio bands are 1-indexed
# VIIRS surface reflectance. These are what the fire product is derived from, so
# their absence -- not the fire channel's -- is what "the satellite did not see
# this place today" means.
REFLECTANCE_BANDS = (1, 2, 3)
MIN_REFLECTANCE_COVER = 0.5
DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def _parse_date(p: Path) -> date | None:
    m = DATE_RE.search(p.stem)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def fire_folders(root: str | Path, years: tuple[int, ...] | None = None) -> list[Path]:
    """Every folder holding at least two dated GeoTIFFs, sorted for determinism.

    Two is the minimum a sequence can be scored on: the loop forecasts from one
    day onto the next, so a single-frame fire contributes nothing and would only
    add a name to the table.
    """
    root = Path(root)
    found: list[Path] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        tifs = [f for f in d.glob("*.tif") if _parse_date(f) is not None]
        if len(tifs) < 2:
            continue
        if years is not None:
            yrs = {_parse_date(f).year for f in tifs}          # type: ignore[union-attr]
            if not yrs & set(years):
                continue
        found.append(d)
    return found


def _block_reduce(a: np.ndarray, grid: int) -> np.ndarray:
    """NaN-aware mean onto a grid x grid raster, covering the WHOLE extent.

    The obvious implementation -- reshape into blocks of h//grid by w//grid --
    silently throws away the right and bottom margins whenever the raster is
    not a multiple of `grid`. WildfireSpreadTS rasters are around 300x250, so
    at grid 64 that discards roughly a fifth of each axis, and measured against
    the native detection count it lost between 0% and **59%** of a fire's
    detections depending on where in the frame the fire sat. A fire near an
    edge would simply have been smaller in the data than in the world.

    Index binning instead: every native pixel is assigned to a cell, so nothing
    is dropped wherever the raster shape falls. Averaging a binary detection
    mask still gives fraction-of-cell-burning, which is what FIRMS means too.
    """
    h, w = a.shape
    ri = np.arange(h) * grid // h
    ci = np.arange(w) * grid // w
    idx = (ri[:, None] * grid + ci[None, :]).ravel()
    flat = a.ravel()
    good = np.isfinite(flat)
    n = grid * grid
    total = np.bincount(idx[good], weights=flat[good], minlength=n)
    count = np.bincount(idx[good], minlength=n)
    out = np.full(n, np.nan, dtype=np.float64)
    hit = count > 0
    out[hit] = total[hit] / count[hit]
    return out.reshape(grid, grid).astype(np.float32)


def load_fire(folder: str | Path, grid: int = 64) -> FireSequence | None:
    """One fire's folder as a FireSequence, or None if nothing is usable."""
    import rasterio

    folder = Path(folder)
    files = sorted((f for f in folder.glob("*.tif") if _parse_date(f) is not None),
                   key=lambda f: _parse_date(f))                # type: ignore[arg-type,return-value]
    if len(files) < 2:
        return None

    dates, fires, frps, observed = [], [], [], []
    bounds = (0.0, 0.0, 0.0, 0.0)
    for f in files:
        try:
            with rasterio.open(f) as src:
                if src.count < FIRE_BAND:
                    continue
                band = src.read(FIRE_BAND).astype(np.float32)
                refl = src.read(list(REFLECTANCE_BANDS)).astype(np.float32)
                nodata = src.nodatavals[FIRE_BAND - 1]
                if bounds == (0.0, 0.0, 0.0, 0.0):
                    b = src.bounds
                    bounds = (float(b.left), float(b.bottom),
                              float(b.right), float(b.top))
        except Exception:
            continue
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)

        # Whether the satellite saw this place today, from the reflectance the
        # fire product is derived from. A day the fire simply was not burning
        # is a real observation and must be scored as one.
        seen = bool(np.isfinite(refl).mean() >= MIN_REFLECTANCE_COVER)
        # In the fire channel, finite means "detected"; NaN is the overwhelming
        # majority and means "not detected". The VALUE is the VIIRS acquisition
        # time as HHMM, not radiative power -- measured over 7,030 detections
        # it runs 742 to 2142, the minutes are always a multiple of the 6-minute
        # granule and never reach 60. See `frp` below.
        finite = np.isfinite(band)
        detect = np.where(finite & (band > 0), 1.0, 0.0)

        dates.append(_parse_date(f).isoformat())                # type: ignore[union-attr]
        fires.append(np.clip(_block_reduce(detect, grid), 0.0, 1.0))
        observed.append(seen)

    if len(dates) < 2 or not any(observed):
        return None
    fire = np.nan_to_num(np.stack(fires)).astype(np.float32)
    # WildfireSpreadTS ships no radiative power, so `frp` is zeros rather than a
    # plausible-looking number derived from the wrong column. The twin loop does
    # not read it; anything that starts to should fail loudly here rather than
    # quietly regress on clock times.
    frp = np.zeros_like(fire)
    return FireSequence(name=folder.name, dates=dates, fire=fire, frp=frp,
                        observed=np.asarray(observed, dtype=bool), bounds=bounds)


def load_sequences(root: str | Path, grid: int = 64, max_fires: int | None = None,
                   years: tuple[int, ...] | None = None,
                   min_observed: int = 4, min_burning: float = 0.0,
                   verbose: bool = True) -> list[FireSequence]:
    """Every usable fire, as FireSequence, filtered to sequences worth scoring.

    `min_observed` drops fires the satellite barely saw: a sequence with two
    observed days carries almost no information about carrying state forward,
    and including it would inflate n without inflating evidence -- which is the
    exact criticism §9.9 makes of the 8-fire version.
    """
    folders = fire_folders(root, years)
    out: list[FireSequence] = []
    skipped = {"unreadable": 0, "too_few_observed": 0, "no_fire": 0}
    for d in folders:
        seq = load_fire(d, grid=grid)
        if seq is None:
            skipped["unreadable"] += 1
            continue
        if int(seq.observed.sum()) < min_observed:
            skipped["too_few_observed"] += 1
            continue
        if float(seq.fire.max()) <= min_burning:
            skipped["no_fire"] += 1
            continue
        out.append(seq)
        if max_fires is not None and len(out) >= max_fires:
            break
    if verbose:
        print(f"[wsts] {len(out)} sequences from {len(folders)} folders "
              f"(skipped {skipped})", flush=True)
    return out
