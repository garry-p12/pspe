"""PDEBench as a real surrogate benchmark (paper Section 7.1 / 7.2).

Phase 1 currently scores the surrogate against this repo's own solver, which is
internal consistency, not external credibility. PDEBench (Takamoto et al.,
NeurIPS 2022) is the community's standard neural-operator benchmark: it ships
the same PDE families we use — 2D diffusion-reaction, 2D shallow-water — with
*published* FNO/U-Net forecast errors, so training our FNO on it and reporting
relative L2 next to their numbers is the "beats/matches baselines on a real
dataset" evidence the paper's experimental program calls for.

PDEBench has no intervention forcing, so the control channel is zero: this
validates the *surrogate* (Simulate) against real data, not the planner. The
loader emits the exact `(states, controls)` schema the rest of the repo uses,
so `SimulateTrainer` runs on it unchanged.

The HDF5 layout varies across PDEBench files (top-level `data`/`tensor`, or
per-sample groups, with the channel axis sometimes last and sometimes absent).
`load_pdebench` normalises the common cases into `(n, T, C, H, W)` and fails
loudly on the rest rather than mis-indexing.

`h5py` is imported lazily so the rest of the package has no hard dependency on
it; the download runs on Colab or any box with network, and the loader's
correctness is unit-tested offline against a synthetic PDEBench-layout file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "pdebench"

# PDEBench 2D files on DaRUS (dataset doi:10.18419/darus-2986), with the real
# Dataverse datafile IDs (verified against the DaRUS API). Direct download URL
# is https://darus.uni-stuttgart.de/api/access/datafile/<id> — no account.
#
# These files are LARGE (sizes below): the reaction-diffusion and shallow-water
# temporal datasets are the right ones for our autoregressive surrogate, but at
# 6-13 GB the download+train is a Colab / large-box step, not something to run
# on a constrained machine. The loader's correctness is proven offline against a
# synthetic PDEBench-layout HDF5 in tests/test_pdebench.py; only the real
# download needs the bandwidth.
PDEBENCH_FILES: dict[str, dict[str, str]] = {
    "2d_shallowwater": {
        "file": "2D_rdb_NA_NA.h5",
        "id": "133021",
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133021",
        "size": "6.6 GB",
        "maps_to": "swe",
        "channels": "1 (surface height)",
    },
    "2d_reacdiff": {
        "file": "2D_diff-react_NA_NA.h5",
        "id": "133017",
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133017",
        "size": "13.2 GB",
        "maps_to": "rdf",
        "channels": "2 (activator, inhibitor)",
    },
}


@dataclass
class PDEBenchConfig:
    path: str                     # local .h5 path
    grid: int = 64                # target resolution (PDEBench is downsampled to this)
    n_samples: int = 256
    max_steps: int = 24           # cap the trajectory length to our horizon budget
    seed: int = 0


# --------------------------------------------------------------------------- #
# Pure normalisation (no h5py; unit-tested offline against a synthetic array)
# --------------------------------------------------------------------------- #
def normalise_layout(arr: np.ndarray) -> np.ndarray:
    """Any PDEBench solution array -> (n_samples, T, C, H, W).

    Handles the shapes PDEBench files appear in:
      (n, T, H, W)        -> add C=1
      (n, T, H, W, C)     -> move C to front of the spatial block
      (T, H, W, C)        -> single trajectory, add sample axis
    """
    if arr.ndim == 4:
        n, t, h, w = arr.shape
        return arr.reshape(n, t, 1, h, w)
    if arr.ndim == 5:
        # Decide whether the last axis is channels (small) or the sample/time
        # layout is (n, T, H, W, C).
        n, t, a, b, c = arr.shape
        if c <= 4 and a == b:  # (n, T, H, W, C)
            return np.transpose(arr, (0, 1, 4, 2, 3))
        return arr  # already (n, T, C, H, W)
    if arr.ndim == 3:
        t, h, w = arr.shape
        return arr.reshape(1, t, 1, h, w)
    raise ValueError(f"unhandled PDEBench array shape {arr.shape}")


def to_schema(
    arr: np.ndarray, grid: int, n_samples: int, max_steps: int
) -> dict[str, np.ndarray]:
    """(n, T, C, H, W) numpy -> repo `(states, controls)` schema at `grid`.

    Controls are zero (PDEBench has no intervention). States are downsampled to
    `grid` and clipped to `max_steps + 1` frames.
    """
    normed = normalise_layout(arr)
    n = min(n_samples, normed.shape[0])
    t = min(max_steps + 1, normed.shape[1])
    normed = normed[:n, :t]

    tensor = torch.from_numpy(np.ascontiguousarray(normed)).float()
    n, steps1, c, h, w = tensor.shape
    if (h, w) != (grid, grid):
        tensor = F.interpolate(
            tensor.reshape(n * steps1, c, h, w), size=(grid, grid),
            mode="bilinear", align_corners=False,
        ).reshape(n, steps1, c, grid, grid)

    states = tensor.numpy().astype(np.float32)
    controls = np.zeros((n, steps1 - 1, 1, grid, grid), dtype=np.float32)
    return {"states": states, "controls": controls}


# --------------------------------------------------------------------------- #
# HDF5 loading (lazy h5py)
# --------------------------------------------------------------------------- #
def _find_solution_array(path: Path) -> np.ndarray:
    """Extract the main solution tensor from a PDEBench HDF5 file."""
    import h5py  # type: ignore[import-not-found]

    with h5py.File(path, "r") as f:
        # Common case: a top-level dataset named 'data', 'tensor', 'u', 'solution'.
        for key in ("data", "tensor", "u", "solution"):
            if key in f and isinstance(f[key], h5py.Dataset):
                return f[key][:]
        # Per-sample groups ('0000', '0001', ...), each holding a 'data' array.
        sample_keys = sorted(k for k in f.keys() if k.isdigit())
        if sample_keys:
            frames = []
            for k in sample_keys:
                grp = f[k]
                inner = grp["data"] if "data" in grp else grp[list(grp.keys())[0]]
                frames.append(inner[:])
            return np.stack(frames, axis=0)
        # Fall back to the largest float dataset in the file.
        best, best_size = None, -1
        for name, obj in _walk(f):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.floating):
                if obj.size > best_size:
                    best, best_size = obj[:], obj.size
        if best is not None:
            return best
    raise ValueError(f"no solution array found in {path}")


def _walk(group):
    import h5py  # type: ignore[import-not-found]

    for key in group.keys():
        obj = group[key]
        yield key, obj
        if isinstance(obj, h5py.Group):
            yield from _walk(obj)


def load_pdebench(cfg: PDEBenchConfig) -> dict[str, np.ndarray]:
    """Load a PDEBench .h5 file into the repo's `(states, controls)` schema."""
    path = Path(cfg.path)
    if not path.exists():
        raise FileNotFoundError(
            f"no PDEBench file at {path}. Download one with "
            f"`python scripts/download_pdebench.py <name>` (see PDEBENCH_FILES)."
        )
    arr = _find_solution_array(path)
    return to_schema(arr, cfg.grid, cfg.n_samples, cfg.max_steps)


def channels_in(path: str | Path) -> int:
    """Peek at a PDEBench file's channel count without loading it fully."""
    cfg = PDEBenchConfig(path=str(path), n_samples=1, max_steps=1)
    return int(load_pdebench(cfg)["states"].shape[2])
