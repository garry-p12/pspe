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


@dataclass
class Structures:
    """Point structures: gates that let water out one way, pumps that lift it.

    Both are things a levee alone cannot do, and the distinction matters for
    what the planner can offer. A levee is a bed perturbation, so it can only
    keep water OUT of a perimeter. A pump acts on water already inside one --
    which is the case section 5.9's pluvial result had no instrument for, where
    no levee placement helped because a quarter of the rain fell inside the
    area being protected.

    Index tensors are (N, 2) of (row, col). Capacities carry a leading batch
    dimension aligned with the bed batch, so `/plan`'s enumeration -- one bed
    per candidate plan, one timestep loop for all of them -- survives: a plan
    that does not build a gate simply sets it shut.
    """
    gate_in: Tensor | None = None        # (G,2) protected-side cell
    gate_out: Tensor | None = None       # (G,2) discharge cell
    gate_area_m2: Tensor | None = None   # (G,) barrel cross-section
    gate_invert_m: Tensor | None = None  # (G,) sill elevation; nothing below drains
    gate_open: Tensor | None = None      # (b,G) bool: does this plan build it
    gate_cd: float = 0.62                # orifice discharge coefficient

    pump_in: Tensor | None = None        # (P,2) wet-well cell
    pump_out: Tensor | None = None       # (P,2), or None to discharge off-domain
    pump_q_m3s: Tensor | None = None     # (b,P) rated capacity
    pump_sump_m: Tensor | None = None    # (P,) intake elevation
    pump_on_m: Tensor | None = None      # (P,) start level above the sump
    pump_off_m: Tensor | None = None     # (P,) stop level; must be < on


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
        # Point structures, set by the caller after construction. None means
        # the step loop is byte-for-byte what it was before gates and pumps
        # existed, which is what keeps the existing solver tests meaningful.
        self.structures: Structures | None = None
        # A pump's on/off state has to persist between steps or it chatters.
        # Sized from the batch at first use, because `/plan` runs the same
        # solver code at several batch sizes.
        self._pump_running: Tensor | None = None
        # Water lifted out of the domain, in cubic metres, so a closed-domain
        # mass test can still balance when a pump discharges off-map.
        self.pumped_volume_m3: float = 0.0
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

    def _structure_demand(self, h: Tensor, wse: Tensor, dt: float
                          ) -> tuple[Tensor, dict] | tuple[None, None]:
        """Floodgates and pumps, as unit discharges in m^2/s like `_edge_outflow`.

        Both are SINKS at a cell and SOURCES at another, and both must ride the
        availability limiter in `step` rather than being added to `rain`. The
        rain term is applied after the mass balance and outside the limiter,
        with `h.clamp(min=0)` immediately after it, so a negative source there
        drives depth below zero and the clamp *creates* mass -- the same 6.4%
        dam-break error the limiter was written to kill. `last_clamped` and
        `test_flux_limiter_prevents_negative_depth` would both catch it, which
        is the only reason that shortcut is merely wrong and not invisible.

        ## The gate is not a hole in the bed

        A levee here is only an addition to `z`, and flow across it stops
        because `_interface_depth` collapses to zero: `h_f = max(wl, wr) -
        max(zl, zr)`. That expression is SYMMETRIC in the two cells, so
        lowering the bed at a crest gives a culvert that flows whichever way
        the surface tilts -- including inward when the river is higher, which
        is the one condition a floodgate exists to prevent. Direction cannot be
        expressed in the bed. It has to be an explicit directional exchange
        between a named inside cell and a named outside cell, which is what
        this is: `max(dH, 0)` IS the flap, and it seals with no branch.

        ## Both are limited twice, on purpose

        Each carries its own clamp -- a gate moves at most half the volume that
        would level its two cells this step, a pump lifts at most half the
        water above its sump -- and the solver's own `scale` is the backstop.
        Relying on `scale` alone is mass-safe and not convergent: the structure
        would then discharge whatever its cell happened to hold, and the answer
        would depend on dt. The half-volume rule is also what stops an orifice
        chattering: dQ/d(dH) goes to infinity as the head closes, so without it
        the flap flips every step and the two cells ring against each other.
        """
        st = self.structures
        if st is None:
            return None, None
        b = h.shape[0]
        demand = torch.zeros_like(h)
        book: dict = {"gate": None, "pump": None}
        cell_area = self.cfg.dx * self.cfg.dx

        if st.gate_in is not None and st.gate_in.numel():
            ri, ci = st.gate_in[:, 0], st.gate_in[:, 1]
            ro, co = st.gate_out[:, 0], st.gate_out[:, 1]
            wse_in = wse[..., ri, ci]                       # (b,G)
            wse_out = wse[..., ro, co]
            head = (wse_in - st.gate_invert_m).clamp(min=0.0)
            dH = (wse_in - wse_out).clamp(min=0.0)          # the flap
            q = st.gate_cd * st.gate_area_m2 * torch.sqrt(2.0 * G * dH)
            # A soft switch one depth_min wide, so closure is continuous
            # rather than a step function the solver can oscillate across.
            q = q * (dH / self.cfg.depth_min).clamp(max=1.0)
            q = q * (head > 0.0)
            if st.gate_open is not None:
                q = q * st.gate_open.to(q.dtype)
            q_level = 0.5 * dH * cell_area / max(dt, 1e-9)
            q = torch.minimum(q, q_level)                   # m^3/s, (b,G)
            unit = q / self.cfg.dx                          # m^2/s, (b,G)
            idx_b = torch.arange(b, device=h.device).unsqueeze(1).expand_as(unit)
            demand.index_put_((idx_b.reshape(-1),
                               ri.expand(b, -1).reshape(-1),
                               ci.expand(b, -1).reshape(-1)),
                              unit.reshape(-1), accumulate=True)
            book["gate"] = (ri, ci, ro, co, unit, idx_b)

        if st.pump_in is not None and st.pump_in.numel():
            ri, ci = st.pump_in[:, 0], st.pump_in[:, 1]
            avail = (h[..., ri, ci] - (st.pump_sump_m - self.z[..., ri, ci])
                     .clamp(min=0.0)).clamp(min=0.0)
            if self._pump_running is None or self._pump_running.shape[0] != b:
                self._pump_running = torch.zeros(b, st.pump_in.shape[0],
                                                 dtype=torch.bool, device=h.device)
            # A deadband, so a station does not chatter on and off around one
            # level. This is the solver's only genuine per-step state besides
            # `last_clamped`, and it is sized from the batch at first use
            # because `/plan` solves different batch sizes with the same code.
            self._pump_running = torch.where(
                avail > st.pump_on_m, torch.ones_like(self._pump_running),
                torch.where(avail < st.pump_off_m,
                            torch.zeros_like(self._pump_running),
                            self._pump_running))
            q_max = 0.5 * avail * cell_area / max(dt, 1e-9)
            q = torch.minimum(st.pump_q_m3s.expand(b, -1), q_max)
            q = q * self._pump_running.to(q.dtype)
            unit = q / self.cfg.dx
            idx_b = torch.arange(b, device=h.device).unsqueeze(1).expand_as(unit)
            demand.index_put_((idx_b.reshape(-1),
                               ri.expand(b, -1).reshape(-1),
                               ci.expand(b, -1).reshape(-1)),
                              unit.reshape(-1), accumulate=True)
            book["pump"] = (ri, ci,
                            None if st.pump_out is None else st.pump_out[:, 0],
                            None if st.pump_out is None else st.pump_out[:, 1],
                            unit, idx_b)
        return demand, book

    def _structure_apply(self, div: Tensor, scale: Tensor, book: dict,
                         dt: float) -> Tensor:
        """Move the structures' water, at the rate the limiter actually allowed.

        The receiving side MUST read the post-`scale` value. Crediting the
        outlet with the unlimited discharge while the donor is held back is
        exactly how a structure manufactures water, which is the failure the
        limiter exists to prevent -- and it would pass every test that only
        checks for negative depth.
        """
        for key in ("gate", "pump"):
            if book.get(key) is None:
                continue
            ri, ci, ro, co, unit, idx_b = book[key]
            taken = unit * scale[..., ri, ci]
            flat_b, flat_ri, flat_ci = (idx_b.reshape(-1),
                                        ri.expand_as(idx_b).reshape(-1),
                                        ci.expand_as(idx_b).reshape(-1))
            div.index_put_((flat_b, flat_ri, flat_ci),
                           -taken.reshape(-1), accumulate=True)
            if ro is None:
                # Discharged out of the domain. Booked in cubic metres so a
                # closed-domain mass test still balances: `taken` is a unit
                # discharge in m^2/s, so the volume it moves in this step is
                # taken * dx * dt. Nothing disappears silently.
                self.pumped_volume_m3 += float(taken.sum()) * self.cfg.dx * dt
                continue
            div.index_put_((flat_b,
                            ro.expand_as(idx_b).reshape(-1),
                            co.expand_as(idx_b).reshape(-1)),
                           taken.reshape(-1), accumulate=True)
        return div

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
            # Gates and pumps join the same bookkeeping: demanded here,
            # rationed by `scale` below, applied into `div` after it.
            struct, book = self._structure_demand(h, wse, dt)

            # Mass balance, with a flux limiter. An explicit scheme can ask an
            # interface for more water than the donor cell holds; clamping the
            # result at zero would *create* mass (measured: +6.4% on a dam
            # break). So scale every outflow from a cell by the fraction of its
            # depth that is actually available, before applying the divergence.
            k = dt / self.cfg.dx
            out = edge * k
            if struct is not None:
                out = out + struct * k
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
            if book is not None:
                div = self._structure_apply(div, scale, book, dt)
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
