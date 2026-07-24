"""Phase 1 trainer for any surrogate that implements (state, control) -> next.

Shared by the FNO, DeepONet and GNOT runs so the operator comparison changes
only the model, and by the ablation runner so `use_physics=False` is the same
code path with one flag flipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.common import count_parameters, peak_memory_mb, timer
from ..utils.logging import RunLogger
from .dataset import PDEDataset, ensure_dataset
from .losses import SurrogateLossWeights, surrogate_loss
from .rollout import surrogate_fidelity
from .solvers import make_testbed

Tensor = torch.Tensor


@dataclass
class SimulateTrainConfig:
    testbed: str = "dar"
    surrogate: str = "fno"
    epochs: int = 20
    batch: int = 16
    horizon: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_physics: bool = True         # ablation switch: physics-informed loss on/off
    padded: bool = False             # channel-padded head, for cross-family transfer
    physics_weight: float = 0.1
    rollout_weight: float = 1.0
    eval_horizon: int = 16
    grid: int = 64
    amp: bool = True                 # mixed precision (CUDA only; no-op elsewhere)
    # Score the rollout against the *stored* held-out trajectory instead of the
    # analytic solver. Required for real datasets (PDEBench): there is no
    # matching in-tree solver to roll as ground truth, only the recorded frames.
    eval_vs_data: bool = False
    seed: int = 0
    log_dir: str = "runs/simulate"


class SimulateTrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: SimulateTrainConfig | None = None,
        data: dict[str, np.ndarray] | None = None,
        logger: RunLogger | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or SimulateTrainConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.testbed = make_testbed(self.cfg.testbed, grid=self.cfg.grid, device=self.device)
        blob = data or ensure_dataset(self.cfg.testbed, grid=self.cfg.grid)
        data_grid = blob["states"].shape[-1]
        if data_grid != self.cfg.grid:
            raise ValueError(
                f"dataset for {self.cfg.testbed!r} is {data_grid}x{data_grid} but the "
                f"config asks for {self.cfg.grid}x{self.cfg.grid}. Regenerate it: "
                f"python scripts/generate_data.py testbed={self.cfg.testbed} "
                f"data.grid={self.cfg.grid}"
            )
        self.train_set = PDEDataset(blob["states"], blob["controls"], self.cfg.horizon, "train")
        self.val_set = PDEDataset(blob["states"], blob["controls"], self.cfg.horizon, "val")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.weights = SurrogateLossWeights(
            physics=self.cfg.physics_weight, rollout=self.cfg.rollout_weight
        )
        self.logger = logger or RunLogger(self.cfg.log_dir)
        self._use_amp = self.cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

    def train(self) -> dict[str, float]:
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch, shuffle=True)
        step = 0
        with timer() as clock:
            for epoch in range(self.cfg.epochs):
                self.model.train()
                for traj, controls in loader:
                    traj = traj.to(self.device)
                    controls = controls.to(self.device)
                    with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                        loss, components = surrogate_loss(
                            self.model, self.testbed, traj, controls,
                            self.cfg.horizon, self.weights, self.cfg.use_physics,
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    step += 1
                    self.logger.log(step, epoch=epoch, **components)

        metrics = self.evaluate()
        trainable, total = count_parameters(self.model)
        summary = {
            **metrics,
            "surrogate": self.cfg.surrogate,
            "testbed": self.cfg.testbed,
            "use_physics": float(self.cfg.use_physics),
            "params/trainable": float(trainable),
            "params/total": float(total),
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
        }
        self.logger.log_summary(**summary)
        return summary

    @torch.no_grad()
    def evaluate(self, horizon: int | None = None) -> dict[str, float]:
        """Relative-L2 rollout error against ground truth.

        Ground truth is the analytic solver for the synthetic testbeds, or the
        stored held-out frames when `eval_vs_data` is set (real datasets).
        """
        horizon = horizon or self.cfg.eval_horizon
        states = self.val_set.states.to(self.device)
        controls = self.val_set.controls.to(self.device)
        steps = min(horizon, controls.shape[1])

        if self.cfg.eval_vs_data:
            report = self._data_fidelity(states, controls, steps)
        else:
            report = surrogate_fidelity(self.model, self.testbed, states[:, 0], controls, steps)
        return {
            "rel_l2_1step": float(report["rel_l2_1step"]),
            "rel_l2_final": float(report["rel_l2_final"]),
            "rel_l2_mean": float(report["rel_l2_mean"]),
            "eval_horizon": float(steps),
        }

    @torch.no_grad()
    def _data_fidelity(self, states: Tensor, controls: Tensor, steps: int) -> dict[str, float]:
        """Roll the surrogate and compare to the recorded trajectory, not a solver."""
        from .losses import relative_l2
        from .rollout import rollout_surrogate

        pred = rollout_surrogate(self.model, states[:, 0], controls, steps, detach=True)
        per_step = [float(relative_l2(pred[:, t], states[:, t])) for t in range(1, steps + 1)]
        return {
            "rel_l2_1step": per_step[0],
            "rel_l2_final": per_step[-1],
            "rel_l2_mean": sum(per_step) / len(per_step),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": self.cfg.__dict__}, path)


def load_surrogate(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    """Rebuild a saved surrogate (used by Phases 2 and 5)."""
    from .operators import make_surrogate

    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    testbed = make_testbed(cfg["testbed"], grid=cfg["grid"])
    model = make_surrogate(
        cfg["surrogate"], testbed.n_channels, grid=cfg["grid"],
        padded=cfg.get("padded", False),
    )
    model.load_state_dict(blob["model"])
    return model.to(device)
