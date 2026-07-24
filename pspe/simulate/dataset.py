"""PDE trajectory dataset generation and loading.

Trajectories are generated *with* randomly excited control fields. A surrogate
trained on uncontrolled data cannot answer the planner's counterfactual ("what
if I actuate here?"), so the control channel has to be excited at data-generation
time, not only at planning time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..envs.actuators import GaussianActuatorBasis
from .solvers import make_testbed

Tensor = torch.Tensor

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "pde_testbeds"


@dataclass
class DataConfig:
    testbed: str = "dar"
    n_trajectories: int = 256
    steps: int = 24
    grid: int = 64
    n_actuators: int = 9
    control_scale: float = 0.5
    control_hold: int = 4  # steps an actuator setting is held before resampling
    seed: int = 0


def generate_dataset(
    cfg: DataConfig,
    device: torch.device | str = "cpu",
    out_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Roll out the numerical solver and save (states, controls) to `.npz`."""
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    testbed = make_testbed(cfg.testbed, grid=cfg.grid, device=device)
    basis = GaussianActuatorBasis(cfg.n_actuators, cfg.grid, device=device)

    state = testbed.initial_condition(cfg.n_trajectories, generator).to(device)
    states = [state]
    controls = []
    amplitudes: Tensor | None = None

    for t in range(cfg.steps):
        if t % cfg.control_hold == 0:
            amps = basis.sample_amplitudes(cfg.n_trajectories, cfg.control_scale, generator)
            amplitudes = amps.to(device)
        assert amplitudes is not None
        control = basis.field(amplitudes)
        state = testbed.step(state, control)
        if not torch.isfinite(state).all():
            raise RuntimeError(
                f"solver diverged at step {t} on testbed {cfg.testbed!r}; "
                "reduce dt or increase substeps in DEFAULT_INTEGRATION"
            )
        states.append(state)
        controls.append(control)

    data = {
        "states": torch.stack(states, dim=1).cpu().numpy().astype(np.float32),
        "controls": torch.stack(controls, dim=1).cpu().numpy().astype(np.float32),
        "amplitudes": np.zeros(0, dtype=np.float32),  # placeholder, kept for schema stability
    }
    meta = {**asdict(cfg), "n_channels": testbed.n_channels, "dt": testbed.dt}

    path = Path(out_path) if out_path else dataset_path(cfg.testbed, cfg.grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, meta=np.array(str(meta)))
    return data


def dataset_path(testbed: str, grid: int = 64, root: Path | str | None = None) -> Path:
    """Datasets are grid-scoped, so a 32px run cannot overwrite a 64px one."""
    return Path(root or DATA_ROOT) / f"{testbed}_{grid}.npz"


def load_dataset(
    testbed: str, grid: int = 64, root: Path | str | None = None
) -> dict[str, np.ndarray]:
    path = dataset_path(testbed, grid, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset at {path}. Generate it with: "
            f"python scripts/generate_data.py testbed={testbed} data.grid={grid}"
        )
    with np.load(path, allow_pickle=True) as blob:
        return {"states": blob["states"], "controls": blob["controls"]}


def ensure_dataset(
    testbed: str,
    grid: int = 64,
    n_trajectories: int = 256,
    steps: int = 24,
    seed: int = 0,
    root: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Load the testbed's dataset at this resolution, generating it if absent."""
    path = dataset_path(testbed, grid, root)
    if not path.exists():
        print(f"[{testbed}] no {grid}px dataset; generating {path.name}")
        generate_dataset(
            DataConfig(
                testbed=testbed, grid=grid, n_trajectories=n_trajectories,
                steps=steps, seed=seed,
            ),
            device=device,
            out_path=path,
        )
    return load_dataset(testbed, grid, root)


class PDEDataset(Dataset):
    """Windows of length `horizon + 1` sampled from stored trajectories."""

    def __init__(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        horizon: int = 4,
        split: str = "train",
        train_frac: float = 0.8,
    ) -> None:
        n_traj = states.shape[0]
        cut = max(1, int(train_frac * n_traj))
        sl = slice(0, cut) if split == "train" else slice(cut, n_traj)
        self.states = torch.from_numpy(states[sl])
        self.controls = torch.from_numpy(controls[sl])
        self.horizon = horizon
        self.n_traj = self.states.shape[0]
        self.n_windows = self.states.shape[1] - horizon  # windows per trajectory
        if self.n_traj == 0 or self.n_windows <= 0:
            raise ValueError(
                f"split {split!r} is empty: {self.n_traj} trajectories, "
                f"{self.states.shape[1]} steps, horizon {horizon}"
            )

    def __len__(self) -> int:
        return self.n_traj * self.n_windows

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        traj_idx, t0 = divmod(idx, self.n_windows)
        window = self.states[traj_idx, t0 : t0 + self.horizon + 1]
        control = self.controls[traj_idx, t0 : t0 + self.horizon]
        return window, control
