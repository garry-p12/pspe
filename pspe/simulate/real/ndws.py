"""Next Day Wildfire Spread (Huot et al., IEEE TGRS 2022) -> the repo's schema.

Real wildfire observations: ~18k samples, each a 64x64 km patch of the United
States at 1 km resolution, with the fire mask on day t, twelve explanatory
channels, and the fire mask on day t+1. It is the closest thing to a community
benchmark for fire spread, and it is already at this repository's grid size.

Why this dataset earns its place: every dynamics number in this repo so far has
been scored against a solver written in this repo. That cannot detect an error
shared by the solver and the surrogate. `dar`/`swe`/`rdf` are simulated truth;
this is observed truth.

What it is not: an intervention dataset. Nobody recorded what the fire would
have done under a different suppression plan, so the controls are zero and this
supports Perceive/Simulate claims only. Planning must be evaluated in a
simulator calibrated to these events, and labelled as such.

Fire mask encoding, which matters and bites:

    1   fire
    0   no fire
    -1  unlabelled (cloud, missing, outside the labelled window)

The -1 cells are *not* "no fire". Averaging over them, or feeding them to a
regression loss as if they were zeros, silently trains the model to predict
absence where the label is merely missing. `load_ndws` returns a validity mask
and never fills -1 with a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tfrecord import iter_examples

# The twelve explanatory channels, in the dataset's own names.
DRIVER_FEATURES: tuple[str, ...] = (
    "elevation",    # terrain
    "th",           # wind direction
    "vs",           # wind speed
    "tmmn",         # min temperature
    "tmmx",         # max temperature
    "sph",          # specific humidity
    "pr",           # precipitation
    "pdsi",         # Palmer drought severity index
    "NDVI",         # vegetation
    "population",   # population density
    "erc",          # energy release component
)
PREV_MASK = "PrevFireMask"
NEXT_MASK = "FireMask"

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "ndws"

# Kaggle: fantineh/next-day-wildfire-spread. Files are named by split, e.g.
# next_day_wildfire_spread_train_00.tfrecord.
SPLIT_PREFIX = {"train": "next_day_wildfire_spread_train",
                "eval": "next_day_wildfire_spread_eval",
                "test": "next_day_wildfire_spread_test"}


@dataclass
class NDWSConfig:
    root: Path | str = DATA_ROOT
    split: str = "train"
    grid: int = 64
    n_samples: int | None = None      # None = every sample in the split
    drivers: tuple[str, ...] = field(default_factory=lambda: DRIVER_FEATURES)
    verify_crc: bool = False


def split_files(root: Path | str, split: str) -> list[Path]:
    root = Path(root)
    prefix = SPLIT_PREFIX.get(split)
    if prefix is None:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_PREFIX)}")
    files = sorted(root.glob(f"{prefix}*.tfrecord"))
    if not files:  # fall back to any tfrecord, for a hand-placed subset
        files = sorted(root.glob("*.tfrecord"))
    return files


def load_ndws(cfg: NDWSConfig | None = None) -> dict[str, np.ndarray]:
    """Load NDWS into `(states, controls)` plus drivers and a validity mask.

    Returns
        states     (N, 2, 1, H, W)  [fire mask at t, fire mask at t+1]
        controls   (N, 2, 1, H, W)  zeros — the dataset records no intervention
        valid      (N, 2, 1, H, W)  False where the label was -1 (unlabelled)
        drivers    (N, C, H, W)     the explanatory channels, in `cfg.drivers`
        driver_names, source
    """
    cfg = cfg or NDWSConfig()
    files = split_files(cfg.root, cfg.split)
    if not files:
        raise FileNotFoundError(
            f"no .tfrecord files under {cfg.root}. Download with "
            f"`python scripts/download_ndws.py --split {cfg.split}` "
            "(needs a Kaggle API token)."
        )

    grid = cfg.grid
    states, drivers, valid = [], [], []
    for path in files:
        for example in iter_examples(path, verify=cfg.verify_crc):
            if PREV_MASK not in example or NEXT_MASK not in example:
                continue
            prev = np.asarray(example[PREV_MASK], dtype=np.float32).reshape(grid, grid)
            nxt = np.asarray(example[NEXT_MASK], dtype=np.float32).reshape(grid, grid)

            pair = np.stack([prev, nxt])                      # (2, H, W)
            # -1 marks unlabelled cells. Keep the mask; leave the values alone
            # so a caller that ignores `valid` gets an obviously wrong number
            # rather than a plausible one.
            valid.append((pair >= 0.0)[:, None, :, :])
            states.append(pair[:, None, :, :])                # (2, 1, H, W)

            chans = [
                np.asarray(example[name], dtype=np.float32).reshape(grid, grid)
                for name in cfg.drivers if name in example
            ]
            drivers.append(np.stack(chans) if chans else np.zeros((0, grid, grid), np.float32))

            if cfg.n_samples is not None and len(states) >= cfg.n_samples:
                break
        if cfg.n_samples is not None and len(states) >= cfg.n_samples:
            break

    if not states:
        raise ValueError(f"{len(files)} file(s) parsed but no usable examples found")

    state_arr = np.stack(states)
    return {
        "states": state_arr,
        "controls": np.zeros_like(state_arr),
        "valid": np.stack(valid),
        "drivers": np.stack(drivers),
        "driver_names": np.asarray([n for n in cfg.drivers], dtype=object),
        "source": np.asarray("ndws", dtype=object),
    }


def driver_stats(data: dict[str, np.ndarray], clip: float = 5.0) -> dict[str, np.ndarray]:
    """Per-channel mean/std for the explanatory rasters.

    These channels are in wildly different physical units — elevation in metres
    (~10^3), NDVI scaled by 10^4, population density, drought index near zero.
    Fed raw to a network, the large-magnitude channels dominate the first layer
    and the rest contribute noise. Measured on NDWS: without this, the model
    beat the prevalence floor by only ~3x and predicted fire almost everywhere.

    Robust statistics (median / IQR-derived scale) rather than mean / std: the
    rasters carry sentinel values and heavy tails, and one bad tile should not
    set the scale for a whole channel.
    """
    drivers = data["drivers"]                       # (N, C, H, W)
    flat = drivers.reshape(drivers.shape[0], drivers.shape[1], -1)
    median = np.median(flat, axis=(0, 2))
    q1, q3 = np.percentile(flat, [25, 75], axis=(0, 2))
    scale = np.maximum((q3 - q1) / 1.349, 1e-6)     # IQR -> sigma for a normal
    return {"center": median.astype(np.float32),
            "scale": scale.astype(np.float32),
            "clip": np.float32(clip)}


def normalize_drivers(drivers: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Standardise and clip. Clipping bounds the damage sentinel values can do."""
    center = stats["center"][None, :, None, None]
    scale = stats["scale"][None, :, None, None]
    clip = float(stats["clip"])
    return np.clip((drivers - center) / scale, -clip, clip).astype(np.float32)


def fire_stats(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Class balance and labelling coverage — both decide how to read a score.

    Next-day fire is a rare-positive problem: a model predicting "no fire
    everywhere" scores well on accuracy and on any error metric dominated by
    the background. Report prevalence next to any headline number.
    """
    states, valid = data["states"], data["valid"]
    next_mask, next_valid = states[:, 1], valid[:, 1]
    labelled = next_valid.sum()
    positives = ((next_mask > 0.5) & next_valid).sum()
    return {
        "samples": float(states.shape[0]),
        "labelled_fraction": float(labelled / next_valid.size),
        "fire_prevalence": float(positives / max(labelled, 1)),
    }
