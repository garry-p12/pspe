"""FloodCastBench (Xu et al., Scientific Data 2025) -> the repo's schema.

Real flood dynamics for four events, as sequential water-depth grids at a 300 s
timestep, produced by a finite-difference solution of the 2-D shallow water
equations on a staggered grid and validated by Critical Success Index against
SAR-based flood maps at 0.01 m and 0.05 m.

Why this dataset earns its place. Every planning number in Part 5 was scored
against a surrogate written in this repository, perturbed to stand in for
reality. NDWS fixed that for *dynamics* but not for *interventions*: no
observational record contains the counterfactual (6.2). FloodCastBench does not
contain it either -- but it ships a DEM and an accepted solver's output, so an
intervention can be evaluated by re-solving accepted physics with a levee in
place. That is 6.2's route (b), taken deliberately: one model standing in for
the world, but a conservation-law model that is the operational standard and is
independent of the surrogate under test.

**Two limitations of the published archive, found by inspection, not assumed.**
The data paper describes a "relevant data" folder holding DEM, land use,
rainfall, georeferencing and initial conditions. The archive ships
`Study regions/` with **only** DEMs. So:

  * the event's own rainfall forcing cannot be replayed, and validation has to
    run on a window where the forcing is negligible -- measured from the
    reference's own volume budget, never assumed;
  * roughness cannot be taken from land cover, so Manning's n is a single
    literature value. It is not fitted to the reference, because a solver tuned
    to reproduce its comparison target says nothing about independence from it.

Depth TIFFs carry no georeferencing tags at all, which matters more than it
sounds: only one event's DEM grid matches its depth rasters, so only that event
can be placed. See `GRID_MATCHED`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The archive's real layout:
#   Study regions/<Region>_DEM.tif                        30 m, georeferenced
#   Low-fidelity flood forecasting/480m/<Region>/<s>.tif
#   High-fidelity flood forecasting/{30m,60m}/<Region>/<s>.tif
# Depth frames are named by ELAPSED SECONDS, in multiples of 300.
EVENTS: dict[str, dict[str, object]] = {
    "pakistan2022": {"region": "Pakistan", "folder": "Low-fidelity flood forecasting",
                     "res": "480m", "dx": 480.0, "days": 14.0},
    "mozambique2019": {"region": "Mozambique", "folder": "Low-fidelity flood forecasting",
                       "res": "480m", "dx": 480.0, "days": 6.0},
    "australia2022": {"region": "Australia", "folder": "High-fidelity flood forecasting",
                      "res": "30m", "dx": 30.0, "days": 10.0},
    "uk2015": {"region": "UK", "folder": "High-fidelity flood forecasting",
               "res": "30m", "dx": 30.0, "days": 3.0},
}

# Only `australia2022` has a DEM whose grid matches its depth rasters exactly
# (1073x1073). Measured with a physical test -- flooded cells must sit in low
# ground, so a correct placement puts them well below the median elevation:
#
#   Australia 30 m : wet cells at elevation percentile 0.253  (chance 0.5)
#   Pakistan 480 m : wet cells at elevation percentile 0.452 across nine
#                    candidate 16x block-average offsets, none distinguishable
#                    from chance -- the placement is simply not recoverable
#
# So solver validation runs on Australia. Any other event needs the authors'
# preprocessing to place its grid, and guessing would yield a plausible but
# meaningless comparison.
GRID_MATCHED: frozenset[str] = frozenset({"australia2022"})

NODATA = -1e30


MANNING_BY_LULC: dict[int, float] = {
    1: 0.035,   # water
    2: 0.100,   # trees
    4: 0.050,   # flooded vegetation
    5: 0.035,   # crops
    7: 0.080,   # built area
    8: 0.025,   # bare ground
    9: 0.025,   # snow/ice
    10: 0.030,  # clouds
    11: 0.045,  # rangeland
}
MANNING_DEFAULT = 0.035


def _read_tiff(path: Path) -> np.ndarray:
    """Read one TIFF as a float array. tifffile is already a repo dependency."""
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if arr.ndim == 3:                     # (H, W, C) or (C, H, W) -> first band
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr


def _numbered(paths: list[Path]) -> list[Path]:
    """Sort sequentially-numbered frames by their integer, not lexically.

    `10.tif` sorts before `9.tif` as a string, which would silently shuffle a
    flood's time axis and is exactly the kind of fault that produces a plausible
    but wrong dynamics score.
    """
    def key(p: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return (int(digits) if digits else -1, p.stem)

    return sorted(paths, key=key)


@dataclass
class FloodEvent:
    """One event's static fields and a lazy handle on its depth sequence."""

    name: str
    dx: float
    dem: np.ndarray
    manning: np.ndarray
    initial_depth: np.ndarray | None
    depth_frames: list[Path]
    rain_frames: list[Path]
    dt: float = 300.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape

    def depth(self, index: int) -> np.ndarray:
        return _read_tiff(self.depth_frames[index])

    def rain(self, index: int) -> np.ndarray:
        """Rainfall in m/s. GPM-IMERG is published as mm/h."""
        return _read_tiff(self.rain_frames[index]) / 3.6e6

    def depth_series(self, stride: int = 1, limit: int | None = None) -> np.ndarray:
        frames = self.depth_frames[::stride]
        if limit is not None:
            frames = frames[:limit]
        return np.stack([_read_tiff(p) for p in frames])


def _find(root: Path, *keywords: str) -> list[Path]:
    """Directories whose path contains all keywords, case- and separator-insensitive."""
    hits = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        flat = str(p).lower().replace("_", " ").replace("-", " ")
        if all(k in flat for k in keywords):
            hits.append(p)
    return hits


def load_event(
    root: str | Path,
    event: str = "australia2022",
    resolution: str | None = None,
) -> FloodEvent:
    """Load one event from the extracted archive.

    Raises rather than guessing when the DEM grid does not match the depth
    rasters, because the depth TIFFs carry no georeferencing and a wrong
    correspondence would produce a plausible but meaningless comparison.
    """
    if event not in EVENTS:
        raise KeyError(f"unknown event {event!r}; known: {sorted(EVENTS)}")
    spec = EVENTS[event]
    root = Path(root)
    if root.name != "FloodCastBench" and (root / "FloodCastBench").is_dir():
        root = root / "FloodCastBench"
    if not root.exists():
        raise FileNotFoundError(root)

    region = str(spec["region"])
    res = resolution or str(spec["res"])
    dx = float(spec["dx"]) if resolution is None else float(res.rstrip("m"))

    dem_path = root / "Study regions" / f"{region}_DEM.tif"
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)
    dem = _read_tiff(dem_path)
    dem[dem < NODATA] = float("nan")

    depth_dir = root / str(spec["folder"]) / res / region
    frames = _numbered([p for p in depth_dir.glob("*.tif")])
    if not frames:
        raise FileNotFoundError(f"no depth frames under {depth_dir}")

    first = _read_tiff(frames[0])
    if first.shape != dem.shape:
        raise SystemExit(
            f"{event} at {res}: DEM grid {dem.shape} does not match the depth "
            f"grid {first.shape}, and the depth rasters carry no georeferencing, "
            f"so the correspondence cannot be established from the archive. "
            f"Grid-matched events: {sorted(GRID_MATCHED)}."
        )

    # No land cover ships with the archive, so roughness is a single literature
    # value rather than a map. Stated, not hidden: it is the one calibration knob
    # a hydrodynamic model has, and we are declining to tune it.
    manning = np.full(dem.shape, MANNING_DEFAULT, dtype=np.float32)

    return FloodEvent(
        name=event, dx=dx, dem=dem, manning=manning,
        initial_depth=None, depth_frames=frames, rain_frames=[],
    )


def frame_seconds(path: Path) -> int:
    """Elapsed seconds encoded in a frame's filename."""
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    return int(digits) if digits else -1


def critical_success_index(pred: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """CSI = hits / (hits + misses + false alarms), the record's own metric.

    Reported at 0.01 m and 0.05 m in the data paper, so our solver is compared
    on the same footing rather than on a metric chosen after the fact.
    """
    p, o = pred > threshold, obs > threshold
    hits = float(np.logical_and(p, o).sum())
    misses = float(np.logical_and(~p, o).sum())
    false_alarms = float(np.logical_and(p, ~o).sum())
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else float("nan")


def describe(root: str | Path) -> dict[str, object]:
    """What is actually in an extracted archive: for checking before trusting."""
    root = Path(root)
    tifs = [p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
    shps = [p for p in root.rglob("*.shp")]
    top = sorted({p.relative_to(root).parts[0] for p in root.rglob("*") if p.is_dir()})
    return {
        "root": str(root),
        "top_level": top[:20],
        "n_tiff": len(tifs),
        "n_shp": len(shps),
        "sample_tiffs": [str(p.relative_to(root)) for p in tifs[:10]],
        "fidelity_dirs": [str(p.relative_to(root)) for p in _find(root, "fidelity")][:10],
    }
