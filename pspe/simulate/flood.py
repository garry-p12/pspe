"""Local-inertial shallow-water solver for flood inundation, with topography.

Why a new solver rather than reusing `ShallowWaterTransport` (`swe`): that
testbed is *linearised* and *has no topography* --

    h_t = -H (u_x + v_y) - c_h h + s

so its control adds water to the surface uniformly and there is no geometry for
an intervention to exploit. That is the measured reason `swe` was retired: the
achievable band was ~7% and methods could not separate (defect 12).

A real flood intervention is geometric. A levee raises the bed `z`, and because
the pressure gradient acts on the *free surface* `h + z`, raising `z` redirects
flow instead of adding to it. This module therefore implements the scheme
FloodCastBench itself uses as ground truth: the LISFLOOD-FP local-inertial
("subcritical") formulation of the shallow-water equations.

Momentum, per interface, with Manning friction:

    q_{t+1} = (q_t - g h_f dt d(h+z)/dx) / (1 + g dt n^2 |q_t| / h_f^{7/3})

Mass, per cell:

    h_{t+1} = h_t + dt/dx (sum q_in - sum q_out) + r dt

The interface depth uses the standard upwind construction

    h_f = max(h_i + z_i, h_j + z_j) - max(z_i, z_j)

which is what gives correct wetting and drying: `h_f` collapses to zero across a
dry interface or a levee crest that the water has not topped, so no flux crosses
it. Boundaries are free-outflow, not periodic -- water must be able to leave the
domain, and `torch.roll` would wrap a flood wave back onto the floodplain.

Units are physical throughout: metres, seconds, m^2/s for unit discharge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor

G = 9.81


@dataclass
class FloodConfig:
    """Solver settings. `alpha` and `theta` follow FloodCastBench (0.7, 0.7)."""

    dx: float = 480.0          # metres, the FloodCastBench low-fidelity grid
    manning: float = 0.03      # floodplain roughness, s/m^(1/3)
    alpha: float = 0.7         # adaptive-timestep safety factor
    theta: float = 0.7         # flux-limiter weight (q smoothing)
    depth_min: float = 1.0e-3  # metres; below this an interface is dry
    dt_max: float = 300.0      # seconds, the dataset's temporal resolution
    open_edges: tuple[str, ...] = ("south",)  # edges that discharge out of the domain
    bed_slope: float = 1.0e-3  # used for the normal-depth outflow at open edges


class FloodSolver:
    """Local-inertial shallow water over a fixed bed `z`.

    State is `(h, qx, qy)`: depth at cell centres, unit discharge at the x- and
    y-interfaces. `qx[..., i, j]` is the flux from cell `j` to cell `j+1`, so it
    carries one fewer column than `h`; `qy` likewise one fewer row.
    """

    def __init__(
        self,
        z: Tensor,
        cfg: FloodConfig | None = None,
        manning: Tensor | None = None,
    ) -> None:
        self.cfg = cfg or FloodConfig()
        if z.dim() == 2:
            z = z.unsqueeze(0)
        self.z = z
        self.device = z.device
        self.dtype = z.dtype
        # Roughness may be a field. It is the one calibration knob a
        # hydrodynamic model has, so when a land-cover map is available it is
        # taken from there rather than fitted -- which keeps the solver
        # independent of the reference depths it is compared against. Friction
        # acts at interfaces, so average the two adjacent cells.
        if manning is None:
            self.nx = self.ny = None
            self.n_cell = None
        else:
            if manning.dim() == 2:
                manning = manning.unsqueeze(0)
            self.n_cell = manning
            self.nx = 0.5 * (manning[..., :, :-1] + manning[..., :, 1:])
            self.ny = 0.5 * (manning[..., :-1, :] + manning[..., 1:, :])

    # -- helpers ------------------------------------------------------------ #
    def zeros_flux(self, batch: int) -> tuple[Tensor, Tensor]:
        _, h_, w_ = self.z.shape
        qx = torch.zeros(batch, h_, w_ - 1, device=self.device, dtype=self.dtype)
        qy = torch.zeros(batch, h_ - 1, w_, device=self.device, dtype=self.dtype)
        return qx, qy

    def adaptive_dt(self, h: Tensor) -> float:
        """CFL for the local-inertial scheme: alpha dx / sqrt(g h_max)."""
        hmax = float(h.max().clamp(min=self.cfg.depth_min))
        dt = self.cfg.alpha * self.cfg.dx / (G * hmax) ** 0.5
        return min(dt, self.cfg.dt_max)

    def _interface_depth(self, wse: Tensor, dim: int) -> tuple[Tensor, Tensor]:
        """Upwind interface depth and free-surface slope along `dim`."""
        z = self.z
        if dim == -1:
            wl, wr = wse[..., :, :-1], wse[..., :, 1:]
            zl, zr = z[..., :, :-1], z[..., :, 1:]
        else:
            wl, wr = wse[..., :-1, :], wse[..., 1:, :]
            zl, zr = z[..., :-1, :], z[..., 1:, :]
        h_f = torch.maximum(wl, wr) - torch.maximum(zl, zr)
        h_f = h_f.clamp(min=0.0)
        slope = (wr - wl) / self.cfg.dx
        return h_f, slope

    def _momentum(
        self, q: Tensor, h_f: Tensor, slope: Tensor, dt: float,
        n: Tensor | None = None,
    ) -> Tensor:
        cfg = self.cfg
        n_sq = cfg.manning**2 if n is None else n**2
        wet = h_f > cfg.depth_min
        h_safe = h_f.clamp(min=cfg.depth_min)
        # theta-weighted q for stability (LISFLOOD-FP flux limiter).
        if cfg.theta < 1.0:
            q_bar = cfg.theta * q + 0.5 * (1.0 - cfg.theta) * (
                torch.roll(q, 1, dims=-1) + torch.roll(q, -1, dims=-1)
            )
        else:
            q_bar = q
        num = q_bar - G * h_safe * dt * slope
        den = 1.0 + G * dt * n_sq * q_bar.abs() / h_safe ** (7.0 / 3.0)
        return torch.where(wet, num / den, torch.zeros_like(q))

    def _edge_outflow(self, h: Tensor) -> Tensor:
        """Normal-depth discharge out of each open edge, as a per-cell sink.

        q = (1/n) h^(5/3) sqrt(S0) is the Manning normal-depth relation. Without
        it a clipped domain with an upstream inflow simply fills up, because the
        interior interfaces alone cannot move water past the last cell.
        """
        cfg = self.cfg
        out = torch.zeros_like(h)
        if not cfg.open_edges:
            return out
        n_edge = cfg.manning if self.n_cell is None else self.n_cell
        coef = (cfg.bed_slope**0.5) / n_edge
        q = coef * h.clamp(min=0.0) ** (5.0 / 3.0)
        if "south" in cfg.open_edges:
            out[..., -1, :] += q[..., -1, :]
        if "north" in cfg.open_edges:
            out[..., 0, :] += q[..., 0, :]
        if "east" in cfg.open_edges:
            out[..., :, -1] += q[..., :, -1]
        if "west" in cfg.open_edges:
            out[..., :, 0] += q[..., :, 0]
        return out

    # -- stepping ----------------------------------------------------------- #
    def step(
        self,
        h: Tensor,
        qx: Tensor,
        qy: Tensor,
        dt: float,
        rain: Tensor | float = 0.0,
        z: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One explicit step. `rain` is a source in m/s. `z` overrides the bed."""
        bed = self.z if z is None else (z.unsqueeze(0) if z.dim() == 2 else z)
        saved, self.z = self.z, bed
        try:
            wse = h + bed
            hfx, sx = self._interface_depth(wse, -1)
            hfy, sy = self._interface_depth(wse, -2)
            qx = self._momentum(qx, hfx, sx, dt, self.nx)
            qy = self._momentum(qy, hfy, sy, dt, self.ny)

            # Boundary outflow at the open edges, as normal depth (Manning).
            edge = self._edge_outflow(h)

            # Mass balance, with a flux limiter. An explicit scheme can ask an
            # interface for more water than the donor cell holds; clamping the
            # result at zero would *create* mass (measured: +6.4% on a dam
            # break). So scale every outflow from a cell by the fraction of its
            # depth that is actually available, before applying the divergence.
            k = dt / self.cfg.dx
            out = edge * k
            out[..., :, :-1] += qx.clamp(min=0.0) * k
            out[..., :, 1:] += (-qx).clamp(min=0.0) * k
            out[..., :-1, :] += qy.clamp(min=0.0) * k
            out[..., 1:, :] += (-qy).clamp(min=0.0) * k
            scale = torch.where(out > h, h / out.clamp(min=1e-12), torch.ones_like(h))

            qx = torch.where(qx > 0, qx * scale[..., :, :-1], qx * scale[..., :, 1:])
            qy = torch.where(qy > 0, qy * scale[..., :-1, :], qy * scale[..., 1:, :])
            edge = edge * scale

            div = torch.zeros_like(h)
            div[..., :, :-1] -= qx
            div[..., :, 1:] += qx
            div[..., :-1, :] -= qy
            div[..., 1:, :] += qy
            div -= edge
            h = h + k * div
            if isinstance(rain, Tensor) or rain != 0.0:
                h = h + dt * rain
            self.last_clamped = float((-h.clamp(max=0.0)).sum())
            return h.clamp(min=0.0), qx, qy
        finally:
            self.z = saved

    def rollout(
        self,
        h: Tensor,
        duration: float,
        rain: Tensor | float = 0.0,
        z: Tensor | None = None,
        inflow: tuple[slice, slice, float] | None = None,
        record_every: float | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Integrate for `duration` seconds with an adaptive timestep.

        `inflow` is `(row_slice, col_slice, m_per_s)`: a sustained source over a
        patch, which is how an upstream hydrograph enters a clipped domain.
        Returns the final depth and any recorded frames.
        """
        qx, qy = self.zeros_flux(h.shape[0])
        t = 0.0
        frames: list[Tensor] = []
        next_record = 0.0 if record_every else float("inf")
        while t < duration:
            dt = min(self.adaptive_dt(h), duration - t)
            if dt <= 0:
                break
            h, qx, qy = self.step(h, qx, qy, dt, rain=rain, z=z)
            if inflow is not None:
                rs, cs, rate = inflow
                h[..., rs, cs] = h[..., rs, cs] + dt * rate
            t += dt
            if t >= next_record:
                frames.append(h.clone())
                next_record += record_every  # type: ignore[operator]
        return h, frames


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #
def floodplain_terrain(
    grid: int = 96,
    dx: float = 480.0,
    slope: float = 1.2e-3,
    channel_depth: float = 5.0,
    channel_width: float = 5.0,
    berm_height: float = 2.2,
    gaps: tuple[tuple[float, float], ...] = ((0.30, 0.55), (0.58, 0.80), (0.80, 0.35)),
    town_centre: tuple[float, float] = (0.62, 0.24),
    town_radius: float = 0.13,
    town_drop: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """A floodplain protected from its channel by a natural berm with low gaps.

    This geometry, not a closed bowl, is what makes a levee study meaningful. An
    earlier version put the settlement in an isolated depression; it then flooded
    from rain falling *inside* the depression, which no levee can prevent, and
    channel discharge never reached it at all. Here the settlement sits on the
    floodplain west of the channel, separated by a berm that the flood wave
    over-tops only at the gaps. Closing a gap redirects the flow, which is the
    leverage a levee actually has.

    `gaps` is a tuple of `(row_fraction, depth_fraction)`: where along the berm
    each low point sits and how much of the berm height is missing there. Several
    gaps of differing size make the intervention a genuine allocation problem
    rather than a single obvious fix.

    Returns `(z, exposure)` with `exposure` summing to one over the settlement.
    """
    ys = torch.arange(grid, device=device, dtype=torch.float32) / (grid - 1)
    xs = torch.arange(grid, device=device, dtype=torch.float32) / (grid - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Down-valley tilt plus gentle cross-valley confinement.
    z = slope * dx * grid * (1.0 - yy)
    z = z + 3.0 * (2.0 * (xx - 0.5)).abs() ** 3

    # Main channel, meandering so flow is not grid-aligned.
    centre = 0.5 + 0.05 * torch.sin(3.0 * torch.pi * yy)
    z = z - channel_depth * torch.exp(-(((xx - centre) * grid / channel_width) ** 2))

    # Berm on the west bank, with low gaps.
    berm_pos = centre - 0.085
    ridge = torch.exp(-(((xx - berm_pos) * grid / 3.0) ** 2))
    missing = torch.zeros_like(yy)
    for row, frac in gaps:
        missing = torch.maximum(missing, frac * torch.exp(-(((yy - row) * grid / 4.0) ** 2)))
    z = z + berm_height * ridge * (1.0 - missing)

    # Settlement: a shallow depression on the floodplain that still drains south.
    d2 = ((yy - town_centre[0]) ** 2 + (xx - town_centre[1]) ** 2) / town_radius**2
    bowl = torch.exp(-d2)
    z = z - town_drop * bowl

    exposure = bowl / bowl.sum()
    return z, exposure


def levee_basis(
    grid: int = 96,
    dx: float = 480.0,
    sites: tuple[float, ...] = (0.30, 0.44, 0.58, 0.70, 0.80, 0.90),
    width_cells: float = 3.0,
    span_cells: float = 5.0,
    channel_width: float = 5.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Candidate levee segments along the west berm, one mask per site.

    A levee raises the bed locally. Because the momentum equation acts on the
    free surface `h + z`, raising `z` changes where the water can go -- it does
    not add or remove water. That is the difference from the `swe` actuator,
    which added an equal source everywhere and could only move the objective
    inside a 7% band.

    Returns `(K, grid, grid)` masks in [0, 1]; a plan scales each by a height.
    """
    ys = torch.arange(grid, device=device, dtype=torch.float32) / (grid - 1)
    xs = torch.arange(grid, device=device, dtype=torch.float32) / (grid - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centre = 0.5 + 0.05 * torch.sin(3.0 * torch.pi * yy)
    berm_pos = centre - 0.085
    across = torch.exp(-(((xx - berm_pos) * grid / width_cells) ** 2))
    masks = []
    for row in sites:
        along = torch.exp(-(((yy - row) * grid / span_cells) ** 2))
        masks.append(across * along)
    return torch.stack(masks)
