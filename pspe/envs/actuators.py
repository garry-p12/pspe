"""Actuator basis: low-dimensional action vector -> intervention field s(x).

The planner acts with a field, but optimising a full 64x64 field per step is
both ill-posed and outside the compute budget. Actions are therefore amplitudes
on a fixed bank of Gaussian actuators at known locations - a physically
meaningful low-rank parameterisation (think K deployable interventions), and
the same basis the Explain module names in its briefs.
"""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


class GaussianActuatorBasis:
    """`n_actuators` fixed isotropic Gaussians on a periodic unit square."""

    def __init__(
        self,
        n_actuators: int = 9,
        grid: int = 64,
        sigma: float = 0.08,
        max_amplitude: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_actuators = n_actuators
        self.grid = grid
        self.sigma = sigma
        self.max_amplitude = max_amplitude
        self.device = torch.device(device)
        self.dtype = dtype
        self.centers = self._layout(n_actuators)
        self.basis = self._build_basis()  # (K, H, W), peak-normalised

    # -- construction ------------------------------------------------------- #
    def _layout(self, k: int) -> Tensor:
        """Near-square lattice of centres, inset from the domain edges."""
        side = int(math.ceil(math.sqrt(k)))
        coords = []
        for i in range(side):
            for j in range(side):
                if len(coords) == k:
                    break
                coords.append(((i + 0.5) / side, (j + 0.5) / side))
        return torch.tensor(coords, device=self.device, dtype=self.dtype)

    def _build_basis(self) -> Tensor:
        lin = torch.linspace(0, 1, self.grid + 1, device=self.device, dtype=self.dtype)[:-1]
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        fields = []
        for cx, cy in self.centers:
            # Periodic distance, so an actuator near the boundary wraps.
            dx = torch.remainder(xx - cx + 0.5, 1.0) - 0.5
            dy = torch.remainder(yy - cy + 0.5, 1.0) - 0.5
            fields.append(torch.exp(-(dx**2 + dy**2) / (2 * self.sigma**2)))
        return torch.stack(fields, dim=0)

    # -- use ---------------------------------------------------------------- #
    def to(self, device: torch.device | str) -> "GaussianActuatorBasis":
        self.device = torch.device(device)
        self.basis = self.basis.to(self.device)
        self.centers = self.centers.to(self.device)
        return self

    def field(self, amplitudes: Tensor) -> Tensor:
        """(B, K) amplitudes -> (B, 1, H, W) intervention field.

        Differentiable in `amplitudes`, which is what the pathwise branch of the
        hybrid gradient estimator flows through.
        """
        if amplitudes.ndim == 1:
            amplitudes = amplitudes.unsqueeze(0)
        amplitudes = amplitudes * self.max_amplitude
        field = torch.einsum("bk,khw->bhw", amplitudes, self.basis.to(amplitudes.dtype))
        return field.unsqueeze(1)

    def sample_amplitudes(
        self, batch: int, scale: float = 0.5, generator: torch.Generator | None = None
    ) -> Tensor:
        """Random amplitudes for dataset generation (excites the control channel).

        Drawn on CPU then moved, so one CPU generator seeds runs on any device.
        """
        amps = torch.randn(batch, self.n_actuators, dtype=self.dtype, generator=generator)
        return (scale * amps).to(self.device)

    def describe(self, amplitudes: Tensor) -> list[dict[str, float]]:
        """Human/LLM-readable actuator summary, consumed by the Explain module."""
        amps = amplitudes.detach().flatten().tolist()
        return [
            {"actuator": i, "x": float(self.centers[i, 0]), "y": float(self.centers[i, 1]),
             "amplitude": float(a)}
            for i, a in enumerate(amps)
        ]
