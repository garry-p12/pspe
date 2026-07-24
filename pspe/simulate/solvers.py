"""Ground-truth PDE solvers for the three PSPE testbeds.

Finite-difference solvers on a periodic square domain, written in PyTorch rather
than NumPy for three reasons:

1. the same `rhs` defines both the ground truth and the physics-informed
   residual used to regularise the surrogate (no second, drifting copy);
2. the solver is differentiable, so it can act as the high-fidelity reference
   inside a pathwise-gradient rollout;
3. it runs on CPU/MPS/CUDA unchanged, which is what keeps the free-tier
   compute story intact.

Grid resolution is kept modest (64x64 by default) per the proposal's compute
budget. Control enters every testbed as an additive source field `s(x, t)` on
the first channel, which is the "intervention field" the planner acts with.

`py-pde` can substitute for these solvers (`pip install -e ".[extras]"`), but
nothing in the repo requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Periodic finite-difference stencils. All operate on (B, C, H, W).
# --------------------------------------------------------------------------- #
def d_dx(u: Tensor, dx: float) -> Tensor:
    return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)


def d_dy(u: Tensor, dx: float) -> Tensor:
    return (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2.0 * dx)


def laplacian(u: Tensor, dx: float) -> Tensor:
    return (
        torch.roll(u, 1, dims=-1)
        + torch.roll(u, -1, dims=-1)
        + torch.roll(u, 1, dims=-2)
        + torch.roll(u, -1, dims=-2)
        - 4.0 * u
    ) / (dx * dx)


@dataclass
class TestbedSpec:
    """Static description of a testbed, used for config and shape checks."""

    name: str
    n_channels: int
    grid: int
    dt: float
    substeps: int
    params: dict[str, float] = field(default_factory=dict)


class PDETestbed:
    """Base class: an explicit-RK4 integrator around a testbed-specific `rhs`."""

    name: str = "base"
    n_channels: int = 1

    def __init__(
        self,
        grid: int = 64,
        dt: float = 0.01,
        substeps: int = 4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **params: float,
    ) -> None:
        self.grid = grid
        self.dt = dt
        self.substeps = substeps
        self.dx = 1.0 / grid
        self.device = torch.device(device)
        self.dtype = dtype
        self.params = {**self.default_params(), **params}

    # -- interface ---------------------------------------------------------- #
    @staticmethod
    def default_params() -> dict[str, float]:
        return {}

    def rhs(self, u: Tensor, control: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def initial_condition(self, batch: int, generator: torch.Generator | None = None) -> Tensor:
        raise NotImplementedError

    # -- integration -------------------------------------------------------- #
    def step(self, u: Tensor, control: Tensor | None = None) -> Tensor:
        """Advance one macro step `dt` using `substeps` RK4 micro steps.

        The macro step is what the surrogate learns; the micro steps only exist
        to keep the explicit scheme stable at the diffusion numbers used here.
        """
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.rhs(u, control)
            k2 = self.rhs(u + 0.5 * h * k1, control)
            k3 = self.rhs(u + 0.5 * h * k2, control)
            k4 = self.rhs(u + h * k3, control)
            u = u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    def rollout(
        self,
        u0: Tensor,
        steps: int,
        control_fn: Callable[[int, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Return the trajectory (B, steps + 1, C, H, W) including `u0`."""
        traj = [u0]
        u = u0
        for t in range(steps):
            control = control_fn(t, u) if control_fn is not None else None
            u = self.step(u, control)
            traj.append(u)
        return torch.stack(traj, dim=1)

    def residual(self, u_t: Tensor, u_next: Tensor, control: Tensor | None = None) -> Tensor:
        """PDE residual of a one-macro-step transition.

        Uses the midpoint state so the residual is second-order accurate in
        `dt`; a first-order (forward-difference) residual would penalise a
        perfectly correct surrogate at O(dt), swamping the model error.
        """
        u_mid = 0.5 * (u_t + u_next)
        return (u_next - u_t) / self.dt - self.rhs(u_mid, control)

    # -- helpers ------------------------------------------------------------ #
    def spec(self) -> TestbedSpec:
        return TestbedSpec(
            name=self.name,
            n_channels=self.n_channels,
            grid=self.grid,
            dt=self.dt,
            substeps=self.substeps,
            params=dict(self.params),
        )

    def _coords(self) -> tuple[Tensor, Tensor]:
        lin = torch.linspace(0, 1, self.grid + 1, device=self.device, dtype=self.dtype)[:-1]
        return torch.meshgrid(lin, lin, indexing="ij")

    def _random_field(
        self,
        batch: int,
        n_modes: int = 4,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Smooth periodic random field: a truncated random Fourier sum in [0, 1].

        Random draws happen on CPU and are moved afterwards: a CPU `generator`
        cannot seed a CUDA/MPS allocation, and seeding has to stay reproducible
        across devices.
        """
        yy, xx = self._coords()
        out = torch.zeros(batch, self.grid, self.grid, device=self.device, dtype=self.dtype)
        for kx in range(n_modes):
            for ky in range(n_modes):
                if kx == 0 and ky == 0:
                    continue
                amp = torch.randn(
                    batch, 1, 1, dtype=self.dtype, generator=generator
                ).to(self.device)
                phase = (
                    torch.rand(batch, 1, 1, dtype=self.dtype, generator=generator).to(self.device)
                    * 2.0
                    * torch.pi
                )
                decay = 1.0 / (1.0 + kx * kx + ky * ky)
                out = out + amp * decay * torch.sin(
                    2 * torch.pi * (kx * xx + ky * yy) + phase
                )
        out = out - out.amin(dim=(-2, -1), keepdim=True)
        denom = out.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return out / denom


class DiffusionAdvectionReaction(PDETestbed):
    """u_t = D * lap(u) - v . grad(u) + r * u * (1 - u) + s.

    The proposal's primary testbed: a contaminant/heat field that spreads,
    drifts with a background flow, grows logistically, and can be pushed by an
    intervention field `s`.
    """

    name = "dar"
    n_channels = 1

    @staticmethod
    def default_params() -> dict[str, float]:
        return {"D": 0.01, "vx": 0.35, "vy": 0.2, "r": 0.6}

    def rhs(self, u: Tensor, control: Tensor | None = None) -> Tensor:
        p = self.params
        du = (
            p["D"] * laplacian(u, self.dx)
            - p["vx"] * d_dx(u, self.dx)
            - p["vy"] * d_dy(u, self.dx)
            + p["r"] * u * (1.0 - u)
        )
        if control is not None:
            du = du + control
        return du

    def initial_condition(self, batch: int, generator: torch.Generator | None = None) -> Tensor:
        return (0.8 * self._random_field(batch, 4, generator)).unsqueeze(1)


class ShallowWaterTransport(PDETestbed):
    """Linearised shallow water with linear drag, channels (h, u, v).

    h_t = -H (u_x + v_y) - c_h * h + s
    u_t = -g * h_x - c * u
    v_t = -g * h_y - c * v

    Linearised rather than fully nonlinear so an explicit scheme stays stable at
    64x64 on free-tier compute; it keeps the wave-transport character that makes
    this testbed a different PDE family from the two parabolic ones.
    """

    name = "swe"
    n_channels = 3

    @staticmethod
    def default_params() -> dict[str, float]:
        return {"g": 1.0, "H": 1.0, "drag": 0.1, "h_damp": 0.02}

    def rhs(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        p = self.params
        h, u, v = state[:, 0:1], state[:, 1:2], state[:, 2:3]
        dh = -p["H"] * (d_dx(u, self.dx) + d_dy(v, self.dx)) - p["h_damp"] * h
        du = -p["g"] * d_dx(h, self.dx) - p["drag"] * u
        dv = -p["g"] * d_dy(h, self.dx) - p["drag"] * v
        out = torch.cat([dh, du, dv], dim=1)
        if control is not None:
            # Control forces the surface height only; momentum responds through
            # the pressure-gradient term.
            pad = torch.zeros_like(out)
            pad[:, 0:1] = control[:, 0:1]
            out = out + pad
        return out

    def initial_condition(self, batch: int, generator: torch.Generator | None = None) -> Tensor:
        h = 0.3 * (self._random_field(batch, 3, generator) - 0.5).unsqueeze(1)
        zeros = torch.zeros_like(h)
        return torch.cat([h, zeros, zeros], dim=1)


class ReactionDiffusionFront(PDETestbed):
    """FitzHugh-Nagumo front propagation, channels (u, v).

    u_t = Du * lap(u) + u - u^3 / 3 - v + s
    v_t = Dv * lap(v) + eps * (u + a - b * v)

    Excitable-medium dynamics: travelling fronts and spiral waves, a genuinely
    different transfer target from the advective testbed.
    """

    name = "rdf"
    n_channels = 2

    @staticmethod
    def default_params() -> dict[str, float]:
        return {"Du": 0.005, "Dv": 0.002, "eps": 0.25, "a": 0.7, "b": 0.8}

    def rhs(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        p = self.params
        u, v = state[:, 0:1], state[:, 1:2]
        du = p["Du"] * laplacian(u, self.dx) + u - (u**3) / 3.0 - v
        dv = p["Dv"] * laplacian(v, self.dx) + p["eps"] * (u + p["a"] - p["b"] * v)
        out = torch.cat([du, dv], dim=1)
        if control is not None:
            pad = torch.zeros_like(out)
            pad[:, 0:1] = control[:, 0:1]
            out = out + pad
        return out

    def initial_condition(self, batch: int, generator: torch.Generator | None = None) -> Tensor:
        u = (2.0 * self._random_field(batch, 3, generator) - 1.0).unsqueeze(1)
        v = (0.4 * self._random_field(batch, 2, generator) - 0.2).unsqueeze(1)
        return torch.cat([u, v], dim=1)


TESTBEDS: dict[str, type[PDETestbed]] = {
    "dar": DiffusionAdvectionReaction,
    "swe": ShallowWaterTransport,
    "rdf": ReactionDiffusionFront,
}

# Per-testbed integration settings that keep the explicit scheme stable at the
# default 64x64 resolution.
DEFAULT_INTEGRATION: dict[str, dict[str, float | int]] = {
    "dar": {"dt": 0.02, "substeps": 4},
    "swe": {"dt": 0.01, "substeps": 8},
    "rdf": {"dt": 0.05, "substeps": 4},
}


def make_testbed(name: str, **kwargs: object) -> PDETestbed:
    if name not in TESTBEDS:
        raise KeyError(f"unknown testbed {name!r}; expected one of {sorted(TESTBEDS)}")
    settings: dict[str, object] = dict(DEFAULT_INTEGRATION[name])
    settings.update({k: v for k, v in kwargs.items() if v is not None})
    return TESTBEDS[name](**settings)  # type: ignore[arg-type]
