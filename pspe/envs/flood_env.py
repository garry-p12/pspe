"""Operating a polder during a flood: gates and pumps, under an energy budget.

Every other planning environment in this repository decides something *before*
the event and then watches it happen. `FloodPlanEnv`'s question -- where would
a wall pay -- is a contextual bandit wearing an MDP's clothes: you build, the
flood arrives, you score. This one is the sequential problem underneath the
portal, and it is the one an operator actually has:

    the river is rising, the gate has sealed, water is pooling behind the
    levee, and the pumps cost fuel you do not have enough of. Do you spend it
    now, or hold it for the peak that has not arrived yet?

The physics is already in `FloodSolver`. `Structures` carries `gate_open` and
`pump_q_m3s` with a leading batch dimension -- written so `/plan` could solve
one bed per candidate plan in a single timestep loop -- which means B parallel
worlds with B different control settings is what that code already does. This
module only has to drive it.

## Why a gate and a pump and not a levee

A levee is a bed perturbation, so it can only keep water OUT of a perimeter.
Once rain falls *inside* the ring -- §5.9's pluvial case, where a quarter of
the rain landed in the area being protected and no levee placement helped --
the only ways out are gravity through a gate, or lift through a pump. The gate
is free and unavailable when you need it most: it is a flap, so it seals the
moment the river outside rises above the water inside. The pump works whenever
you want and costs energy.

That tension is the whole decision, and it is genuinely anticipatory. A greedy
controller that pumps whenever water is present empties its budget on the
rising limb and has nothing left when the gate seals at the peak. Doing
nothing floods the polder. The rule-based incumbent -- `_pump_running`'s
on/off deadband in `FloodSolver._structure_demand` -- is a local level
controller with no model of the hydrograph at all, and it is what
`baseline_action` reproduces so a learned policy has something real to beat.

## What is observed and what is not

The policy sees the depth field, the bed, which cells it is protecting, and
how far into the event it is. It does NOT see the future hydrograph. That is
deliberate: an operator knows the river is rising and roughly how long events
here last, and has a forecast with error. Handing the policy the exact future
stage would turn an anticipation problem into an arithmetic one.

## What this cannot tell you

The same thing `FireEnv` says about firebreaks. No observational record
contains the counterfactual: nobody logged what this polder would have done
under a pump schedule other than the one its operator ran. The dynamics are
this project's solver, so a policy trained here inherits its defects -- no
culverts or bridges, a uniform Manning's n, a clipped domain -- and a learned
controller is very good at finding them. The forecast side of the model is
scored against FEMA and against a reference solution elsewhere; the effect of
an intervention is in-model only, and saying so is not a disclaimer, it is the
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..simulate.flood import FloodConfig, FloodSolver, Structures

Tensor = torch.Tensor


@dataclass
class FloodOperateTask:
    """Reward weights and the one hard constraint.

    `cost_limit` is in pump-hours at rated capacity, normalised by the number
    of pumps, so 1.0 means "you may run every pump flat out for one hour" and
    an episode of `horizon * control_minutes` longer than that forces a
    choice. It is the CMDP's d_1 and the dual enforces it.
    """

    cost_limit: float = 1.0        # pump-hours at rated capacity, per pump
    wet_m: float = 0.10            # the depth the portal calls "wet", everywhere
    reward_scale: float = 100.0    # wet fraction in percent, so returns are O(1)
    depth_weight: float = 0.5      # how much depth matters beyond mere wetness


#: The basin's lowest cell, and the gate's sill. Shared across a bank so the
#: one gate geometry `Structures` allows is exact for every member.
SUMP_M = 9.0
#: The river bed, below the sump, so the flap can open at low stage.
RIVER_BED_M = 7.0


def make_polder_bank(
    n: int = 64,
    grid: int = 64,
    *,
    dx: float = 60.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """A bank of synthetic polders: a bounded basin behind a levee.

    Randomised so a policy has to learn the problem rather than one map. Real
    terrain is the better evaluation set and this is the better training set,
    which is the usual split -- `FloodOperateEnv` takes either, because it only
    wants `z`, a protected mask and a river mask.

    Layout, west to east: a river channel, a levee crest, floodplain, and
    inside the floodplain a basin that water cannot leave by itself. The gate
    connects the basin's sump to the river; the pump lifts from the same sump
    and discharges off-domain.

    ## Why the basin and not the whole east half

    The first version protected everything behind the levee -- 1,336 ha at
    `dx` 60 -- and the environment did not separate: doing nothing scored
    -214 and pumping flat out for six hours scored -208, a 3% difference for
    the entire actuation budget. The arithmetic says why. A real polder pump
    is sized at 1-2 L/s/ha, so 1,336 ha buys about 3 m3/s; six hours of that
    is 65,000 m3, against 400,000 m3 for even a gentle 5 mm/h of rain on that
    area. The pump was two per cent of the problem, so no policy could matter.

    `task.py` already warns about exactly this: calibrating a testbed means
    checking that the RETURN separates, not only that the cost does. A basin
    of ~370 ha at 6 m3/s -- 16 L/s/ha, urban storm pumping rather than
    agricultural drainage -- puts six hours of pumping and six hours of rain
    within a factor of two of each other, which is the regime where the
    decision is real.
    """
    g = generator
    dev = torch.device(device)

    river_w = max(3, grid // 16)
    crest_w = max(2, grid // 32)
    # The basin: a bounded depression inside the floodplain, half the grid
    # across, so the protected area is a polder and not a half-domain.
    r0, r1 = grid // 4, 3 * grid // 4
    c0, c1 = grid // 2 - grid // 8, grid - grid // 8

    protect = torch.zeros(grid, grid, dtype=torch.bool, device=dev)
    protect[r0:r1, c0:c1] = True
    river = torch.zeros(grid, grid, dtype=torch.bool, device=dev)
    river[:, :river_w] = True

    rr = torch.linspace(-1.0, 1.0, r1 - r0, device=dev)
    cc = torch.linspace(-1.0, 1.0, c1 - c0, device=dev)
    ry, cx = torch.meshgrid(rr, cc, indexing="ij")
    dish = (ry ** 2 + cx ** 2).clamp(max=1.0)

    # Elevations are not free parameters: they are what decides whether the
    # gate can ever open, and the first version got it backwards.
    #
    # A polder drains by gravity when the river outside falls BELOW the water
    # inside, so the sump has to sit above the river's low stage and below its
    # peak. Put the sump under the low stage -- as a basin floor at 8.5 m
    # against a river bed at 9.0 m did -- and the flap is shut for the whole
    # episode: measured, "gate only" scored -71.37 against "do nothing"'s
    # -71.37, to the last decimal, because the gate never once opened.
    #
    # With SUMP_M = 9.0 and a river swinging 7.5 to 10.5 m, the gate drains
    # freely early, seals somewhere on the rising limb, and reopens on the
    # recession. That window is the whole decision.
    #
    # The sump elevation is FIXED across the bank while the basin depth
    # varies, because `Structures.gate_invert_m` is per-gate and not per-batch
    # -- one geometry, per-batch capacities. Varying it here would silently
    # apply one member's sill to all of them.
    z = torch.empty(n, grid, grid, device=dev)
    for i in range(n):
        rnd = lambda: torch.rand(1, generator=g, device=dev).item()  # noqa: E731
        depth = 1.2 + 1.3 * rnd()            # how far the plain sits above the sump
        zi = torch.full((grid, grid), SUMP_M + depth, device=dev)
        zi[r0:r1, c0:c1] = SUMP_M + depth * dish
        zi[:, :river_w] = RIVER_BED_M
        zi[:, river_w:river_w + crest_w] = 12.5 + 1.5 * rnd()  # the levee
        z[i] = zi

    ha = int(protect.sum().item()) * dx * dx / 1e4
    return {"z": z, "river": river, "protect": protect, "protected_ha": ha}


class FloodOperateEnv:
    """BatchedFieldEnv-compatible: reset / step / obs_shape / action_dim.

    One `step` advances the solver by `control_minutes` of simulated time with
    the chosen gate opening and pump capacity held constant over it -- a
    control interval, not a solver timestep. The solver takes its own adaptive
    CFL steps inside that window.
    """

    #: "truth" until a surrogate is supplied, which the trainer reads to decide
    #: whether a pathwise branch exists at all.
    dynamics = "truth"

    def __init__(
        self,
        bank: dict[str, Tensor],
        task: FloodOperateTask | None = None,
        *,
        horizon: int = 12,
        control_minutes: float = 30.0,
        dx: float = 60.0,
        differentiable: bool = False,
        # Real terrain carries real structures: the gate and pump a planner
        # actually placed, built by the service's own `build_structures` from
        # the pieces on the map. Passing them in is what lets the portal run
        # the SAME environment a policy was trained in, rather than a second
        # implementation that agrees with it until it does not.
        structures: object | None = None,
        sump_rc: tuple[int, int] | None = None,
        # A `FloodIntervalSurrogate`. When present, `step` advances the
        # interior through it instead of the solver and the whole episode
        # carries a graph -- which is the only way the hybrid estimator's
        # pathwise branch has anything to differentiate. The river boundary
        # and the budget accounting are unchanged either way: one is a
        # boundary condition and the other is arithmetic on the action.
        surrogate: object | None = None,
        n_pumps: int = 1,
        pump_q_m3s: float = 6.0,
        gate_area_m2: float = 4.0,
        rain_mm_h: tuple[float, float] = (2.0, 10.0),
        river_rise_m: tuple[float, float] = (0.5, 3.5),
        device: torch.device | str = "cpu",
    ) -> None:
        self.task = task or FloodOperateTask()
        # Pathwise gradients, truncated at one control interval.
        #
        # `reward_t` stays connected to `action_t` through the solver steps
        # inside that interval, and the state is detached before the next one.
        # Keeping the graph across a whole episode is not an option: twelve
        # intervals of roughly 250 adaptive CFL steps is ~3,000 solver steps,
        # and storing the intermediates for all of them runs to gigabytes at
        # any useful batch size.
        #
        # So the pathwise branch here is BIASED by truncation -- it sees an
        # action's effect on this interval and not its effect on the next
        # one, which for a pump filling a budget is precisely the part that
        # matters. That is a statement about this estimator, not a defect to
        # hide: the hybrid estimator exists to weigh a biased low-variance
        # gradient against an unbiased high-variance one, and `alpha` is where
        # the trade shows up.
        #
        # OFF BY DEFAULT, and the reason is worth reading before turning it on.
        #
        # The two actuators have opposite gradient behaviour in the same
        # environment. Measured over eight control intervals, mean |d(reward)
        # /d(action)| per step:
        #
        #     step     gate          pump
        #       0   3.879e-03     0.000e+00
        #       2   9.314e+07     0.000e+00
        #       5   5.562e+22     2.418e-02
        #       7   4.201e+20     3.671e-02
        #
        # The pump is linear in its capacity and behaves: its gradient grows
        # smoothly as water arrives to lift. The gate is an orifice, so
        # `q = cd * A * sqrt(2 g dH)` and `dq/d(dH) -> infinity as dH -> 0`.
        # `_structure_demand` already says so in prose -- it is why the
        # half-volume limiter exists -- but the limiter was added to stop the
        # flap CHATTERING, and it does nothing about the derivative. Near
        # closure, which is exactly the operating point the decision turns on,
        # the gate's pathwise gradient spans twenty-five orders of magnitude.
        #
        # That is not a bug to route around. It is the sharpest available test
        # of the hybrid estimator's central claim: `alpha*` is the
        # variance-minimising mix, and a pathwise branch this badly conditioned
        # should drive it to zero. One caution before running that experiment
        # -- (5.6e22)^2 overflows float32, so the variance estimate must be
        # taken in float64 or the pathwise branch must be clipped before
        # mixing, or `alpha` will come back NaN rather than small.
        self.surrogate = surrogate
        # Per-instance, not the class default: `BatchedFieldEnv` and the
        # planner both read `dynamics` to decide what they are training on,
        # and an env running a learned model while reporting "truth" is the
        # one mislabel that would make an evaluation meaningless.
        self.dynamics = "surrogate" if surrogate is not None else "truth"
        # Asking for a surrogate IS asking for gradients; there is no other
        # reason to accept the model error.
        self.differentiable = differentiable or surrogate is not None
        self.external_structures = structures
        self.external_sump = sump_rc
        self._sump_rc: tuple[int, int] | None = sump_rc
        self.device = torch.device(device)
        self.z_bank = bank["z"].to(self.device)
        self.river = bank["river"].to(self.device)
        self.protect = bank["protect"].to(self.device)
        self.n = self.z_bank.shape[0]
        # Square for the synthetic bank, rectangular for a real study box.
        # `grid` is kept as the square-domain shorthand the training presets
        # use; `h_cells` and `w` are what the indexing actually reads.
        self.h_cells, self.w = int(self.z_bank.shape[-2]), int(self.z_bank.shape[-1])
        self.grid = self.w
        self.horizon = horizon
        self.control_s = control_minutes * 60.0
        self.dx = dx
        self.n_pumps = n_pumps
        self.pump_rated = pump_q_m3s
        self.gate_area_m2 = gate_area_m2
        self.rain_mm_h = rain_mm_h
        self.river_rise_m = river_rise_m

        self.cfg = FloodConfig(dx=dx, manning=0.035, open_edges=())
        self.solver: FloodSolver | None = None
        self._state: Tensor | None = None
        self.h: Tensor | None = None
        self.t = 0
        self._spent: Tensor | None = None     # pump-hours used so far, (B,)

    # -- interface ----------------------------------------------------------- #
    @property
    def action_dim(self) -> int:
        """One gate opening plus one capacity fraction per pump."""
        return 1 + self.n_pumps

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (4, self.h_cells, self.w)

    # -- episode setup ------------------------------------------------------- #
    def reset(
        self,
        batch: int = 1,
        generator: torch.Generator | None = None,
        idx: Tensor | None = None,
    ) -> Tensor:
        if idx is None:
            idx = torch.randint(0, self.n, (batch,), generator=generator)
        self._idx = idx
        b = idx.shape[0]
        z = self.z_bank[idx].clone()                              # (B,H,W)

        self.solver = FloodSolver(z, self.cfg)
        self.h = torch.zeros(b, self.h_cells, self.w, device=self.device)
        self.qx, self.qy = self.solver.zeros_flux(b)
        self.t = 0
        self._spent = torch.zeros(b, device=self.device)

        # Per-episode forcing. The policy never sees these directly; it sees
        # what they do to the water.
        r0, r1 = self.rain_mm_h
        self._rain_ms = ((r0 + (r1 - r0) * torch.rand(b, generator=generator,
                                                      device=self.device))
                         / 3.6e6)
        s0, s1 = self.river_rise_m
        self._rise = s0 + (s1 - s0) * torch.rand(b, generator=generator,
                                                 device=self.device)
        # Where the peak falls inside the episode, so the gate seals at a
        # different moment every time and a fixed schedule cannot win.
        self._peak_at = 0.25 + 0.5 * torch.rand(b, generator=generator,
                                                device=self.device)

        if self.external_structures is not None:
            self.solver.structures = self.external_structures
            self._sump_rc = self.external_sump
        else:
            self._build_structures(z, b)
        self._apply_river(0.0)
        self.state = self._observe()
        return self.state

    def _build_structures(self, z: Tensor, b: int) -> None:
        """Gate from the bowl's sump to the river; pump lifting off-domain."""
        inside = self.protect.view(-1)
        flat = z.view(b, -1).clone()
        flat[:, ~inside] = float("inf")                 # sump must be inside
        sump = flat.argmin(dim=1)
        # One geometry for the batch: the bank is built so the sump sits in
        # the same place to within a cell, and `Structures` indexes positions
        # once with per-batch capacities.
        s0 = int(sump[0].item())
        r_in, c_in = divmod(s0, self.w)
        self._sump_rc = (r_in, c_in)
        river_cols = int(self.river[0].sum().item())
        r_out, c_out = r_in, max(0, river_cols // 2)

        dev = self.device
        st = Structures(
            gate_in=torch.tensor([[r_in, c_in]], device=dev),
            gate_out=torch.tensor([[r_out, c_out]], device=dev),
            gate_area_m2=torch.tensor([self.gate_area_m2], device=dev),
            gate_invert_m=torch.tensor([SUMP_M], device=dev),
            gate_open=torch.ones(b, 1, device=dev),
            pump_in=torch.tensor([[r_in, c_in]] * self.n_pumps, device=dev),
            pump_out=None,                                # lifts off-domain
            pump_q_m3s=torch.full((b, self.n_pumps), self.pump_rated, device=dev),
            pump_sump_m=torch.full((self.n_pumps,), SUMP_M, device=dev),
            # Hysteresis OFF for the controlled arm: the deadband IS the
            # incumbent policy, and leaving it in would gate the learned one
            # behind it. `baseline_action` puts it back in action space so the
            # two arms run through identical dynamics.
            pump_on_m=torch.zeros(self.n_pumps, device=dev),
            pump_off_m=torch.full((self.n_pumps,), -1.0, device=dev),
        )
        assert self.solver is not None
        self.solver.structures = st

    # -- forcing ------------------------------------------------------------- #
    def _river_stage(self, frac: float) -> Tensor:
        """River water level above its bed: a single-peaked rise and fall."""
        d = (frac - self._peak_at) / 0.30
        return 0.4 + self._rise * torch.exp(-0.5 * d * d)

    def _apply_river(self, frac: float) -> None:
        """Hold the river band at its stage, the way a boundary condition does."""
        assert self.solver is not None and self.h is not None
        z = self.solver.z
        lvl = (z[:, :, :1].amin(dim=1, keepdim=True)
               + self._river_stage(frac).view(-1, 1, 1))
        want = (lvl - z).clamp(min=0.0)
        m = self.river.unsqueeze(0).expand_as(self.h)
        self.h = torch.where(m, torch.maximum(self.h, want), self.h)

    # -- observation --------------------------------------------------------- #
    def _z_norm(self) -> Tensor:
        assert self.solver is not None
        z = self.solver.z
        lo = z.amin(dim=(-2, -1), keepdim=True)
        hi = z.amax(dim=(-2, -1), keepdim=True)
        zn = (z - lo) / (hi - lo).clamp(min=1e-6)
        assert self.h is not None
        return zn.expand_as(self.h)

    def _observe(self) -> Tensor:
        assert self.solver is not None and self.h is not None
        zn = self._z_norm()
        b = self.h.shape[0]
        mask = self.protect.to(self.h.dtype).expand(b, -1, -1)
        tf = torch.full_like(self.h[:, :1, :1], self.t / max(self.horizon, 1))
        tf = tf.expand(b, self.h_cells, self.w)
        return torch.stack([self.h, zn, mask, tf], dim=1)

    def set_state(self, state: Tensor) -> None:
        """Seed from an externally supplied observation (e.g. Perceive output).

        Resizes the episode's own state to match, which is not optional here.
        `BatchedFieldEnv` and `FireEnv` keep everything an episode needs in the
        observation tensor, so a caller can assign `env.state` and step. This
        env cannot: the per-episode forcing, the flux, the spent budget and the
        solver's batched bed all live outside it. `HybridPlannerTrainer.
        _pathwise_grad_from` assigns `env.state` DIRECTLY and then steps, so
        probing with `real_cost_episodes` different from `batch` crashed with
        "expanded size of the tensor (4) must match the existing size (16)".
        """
        state = state.to(self.device)
        b = state.shape[0]
        if self.h is None or b != self.h.shape[0]:
            self._resize(b)
        self.state = state
        self.h = state[:, 0].contiguous()
        self.t = 0

    def _resize(self, b: int) -> None:
        """Re-shape the episode state for a different batch, forcing and all."""
        assert self.solver is not None
        z = self.solver.z
        if z.shape[0] != b:
            self.solver.z = (z[:1].expand(b, -1, -1).clone() if z.shape[0] == 1
                             else z[:b].clone() if z.shape[0] > b
                             else torch.cat([z, z[-1:].expand(b - z.shape[0], -1, -1)]))
        self.solver._pump_running = None
        self.qx, self.qy = self.solver.zeros_flux(b)
        self._spent = torch.zeros(b, device=self.device)
        for name in ("_rain_ms", "_rise", "_peak_at"):
            v = getattr(self, name, None)
            if v is None:
                continue
            setattr(self, name, v[:b].clone() if v.shape[0] >= b
                    else torch.cat([v, v[-1:].expand(b - v.shape[0])]))
        st = self.solver.structures
        if st is not None:
            for name in ("gate_open", "pump_q_m3s"):
                v = getattr(st, name, None)
                if v is None:
                    continue
                setattr(st, name, v[:b].clone() if v.shape[0] >= b
                        else v[-1:].expand(b, -1).clone())

    @property
    def state(self) -> Tensor | None:
        return self._state

    @state.setter
    def state(self, value: Tensor | None) -> None:
        """Assigning the observation brings the rest of the episode with it.

        `HybridPlannerTrainer._pathwise_grad_from` does `env.state = state0`
        and then steps, on an env it may never have reset -- which is fine for
        `BatchedFieldEnv`, where the observation IS the episode. Here the
        forcing, the flux, the budget and the solver's bed all live outside
        it, so this has to stand in for a reset: lazily make one at the right
        batch, then adopt the depths it was handed.

        Three separate failures came from not doing this, each looking like a
        different bug: a shape mismatch at probe time, a loss with no
        `grad_fn`, and `call reset() before step()`.
        """
        if value is None:
            self._state = None
            return
        b = value.shape[0]
        if self.solver is None:
            # Deterministic, because a probe that reseeds the forcing every
            # time it is called measures the weather, not the model.
            self.reset(b, generator=torch.Generator().manual_seed(0))
        elif self.h is not None and b != self.h.shape[0]:
            self._resize(b)
        self.h = value[:, 0].contiguous()
        self._state = value

    # -- the incumbent ------------------------------------------------------- #
    @torch.no_grad()
    def baseline_action(self, state: Tensor, on_m: float = 0.30,
                        off_m: float = 0.05) -> Tensor:
        """The rule-based controller, expressed as an action.

        This is `FloodSolver._structure_demand`'s own deadband: run the pump
        flat out when the sump is deeper than `on_m`, stop below `off_m`, hold
        otherwise. It has no model of the river and no notion of a budget,
        which is exactly why it is worth beating -- and why beating it is not
        automatic, because a local level controller is a strong heuristic when
        the pump is the only way out.
        """
        h = state[:, 0]
        b = h.shape[0]
        avail = h.view(b, -1)[:, self._sump_flat()].clamp(min=0.0)
        if not hasattr(self, "_base_on") or self._base_on.shape[0] != b:
            self._base_on = torch.zeros(b, device=h.device)
        on = torch.where(avail > on_m, torch.ones_like(self._base_on),
                         torch.where(avail < off_m, torch.zeros_like(self._base_on),
                                     self._base_on))
        self._base_on = on
        gate = torch.ones(b, 1, device=h.device)          # the flap self-seals
        pump = on.view(b, 1).expand(b, self.n_pumps)
        return torch.cat([gate, pump], dim=1) * 2.0 - 1.0  # to [-1, 1]

    @torch.no_grad()
    def sealed_aware_action(self, state: Tensor) -> Tensor:
        """Pump only while the gate cannot drain. The honest opponent.

        `baseline_action` is the incumbent -- a level controller that knows
        nothing about the river. This one is the obvious next thought: leave
        the gate open, and spend the pump only during the window when the
        river outside is above the water inside, because that is the only time
        gravity is unavailable. It reads nothing the policy cannot read; the
        river's depth is in the observation's first channel.

        Measured at grid 32 / dx 120, 32 episodes, against a budget of six
        pump-hours:

            do nothing      -61.71    0.00 pump-h
            gate only       -45.16    0.00
            rule-based      -43.45    0.45
            sealed-aware    -21.67    3.11
            pump flat out   -20.39    6.00

        It takes 96% of flat-out's benefit for half the budget, which is why
        it and not "do nothing" is the number a learned policy has to beat. A
        policy that only matches this has learned the obvious and should be
        reported as such.
        """
        b = state.shape[0]
        h = state[:, 0]
        river_h = h[:, :, :2].amax(dim=(1, 2))
        sump_h = h.view(b, -1)[:, self._sump_flat()]
        sealed = (RIVER_BED_M + river_h) > (SUMP_M + sump_h)
        a = torch.full((b, self.action_dim), -1.0, device=h.device)
        a[:, 0] = 1.0                                   # the flap seals itself
        a[:, 1:] = torch.where(sealed, 1.0, -1.0).view(b, 1)
        return a

    def _sump_flat(self) -> int:
        if self._sump_rc is not None:
            r, c = self._sump_rc
            return r * self.w + c
        assert self.solver is not None and self.solver.structures is not None
        p = self.solver.structures.pump_in[0]
        return int(p[0].item()) * self.w + int(p[1].item())

    # -- dynamics ------------------------------------------------------------ #
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
        """action: (B, 1+P) in [-1, 1] -> gate opening and pump capacity fractions."""
        if self.solver is None or self.h is None:
            raise RuntimeError("call reset() before step()")
        st = self.solver.structures
        assert st is not None
        u = (action.clamp(-1.0, 1.0) + 1.0) / 2.0                  # -> [0,1]
        st.gate_open = u[:, :1]
        frac = u[:, 1:1 + self.n_pumps]
        st.pump_q_m3s = frac * self.pump_rated

        # Roll the control interval forward. `rain` is a per-episode source on
        # the whole box; the river band is re-held each interval so the stage
        # follows its hydrograph rather than draining away.
        rain = self._rain_ms.view(-1, 1, 1)
        if self.surrogate is not None:
            # One learned jump for the whole interval, differentiable end to
            # end. The flux state is not carried: the surrogate was fitted on
            # depth alone, so pretending to track `qx`/`qy` through it would
            # be bookkeeping with no model behind it.
            zn = self._z_norm()
            self.h = self.surrogate.step(
                self.h, zn, self.protect.to(self.h.dtype).expand_as(self.h),
                self.river.to(self.h.dtype).expand_as(self.h),
                u[:, 0], frac.mean(dim=1) if self.n_pumps
                else torch.zeros_like(u[:, 0]),
                self._rain_ms * 3.6e6)
        else:
            t_end = (self.t + 1) * self.control_s
            t_now = self.t * self.control_s
            while t_now < t_end:
                dt = min(self.solver.adaptive_dt(self.h), t_end - t_now)
                if dt <= 0:
                    break
                self.h, self.qx, self.qy = self.solver.step(
                    self.h, self.qx, self.qy, dt, rain=rain)
                t_now += dt
        self.t += 1
        self._apply_river(self.t / max(self.horizon, 1))
        reward = self._reward()                 # still joined to this action

        # Cost: pump-hours at rated capacity, which is energy to within a
        # constant. Accumulated so a Sauté-style budget observation can read it.
        hours = self.control_s / 3600.0
        spent = frac.mean(dim=1) * hours

        # Carry the values forward, drop the history.
        self.h = self.h.detach()
        self.qx, self.qy = self.qx.detach(), self.qy.detach()
        if not self.differentiable:
            # BOTH, not just the reward.
            #
            # `HybridPlannerTrainer` builds its pathwise branch from the
            # Lagrangian, `-(reward - lambda * cost)`. Detaching only the
            # reward leaves `cost` carrying a graph -- here it is the pump
            # fraction, straight from the action -- so the pathwise branch
            # becomes the CONSTRAINT TERM ALONE: a gradient that knows only
            # "pump less" and nothing about flooding. Worse, that term is
            # low-variance, so the variance-minimising mixture prefers it and
            # drives alpha toward 1. Measured before this line existed:
            # alpha walking 0.50 -> 0.55 with the pathwise branch carrying
            # only the constraint, and the training return going NaN.
            #
            # An environment that says it is not differentiable has to mean
            # it for every quantity the objective is built from.
            reward = reward.detach()
            spent = spent.detach()

        assert self._spent is not None
        self._spent = self._spent + spent.detach()

        self.state = self._observe()
        return self.state, reward, spent, self.t >= self.horizon

    def _reward(self) -> Tensor:
        """Minus the wet fraction of the protected area, depth-weighted.

        Wetness and depth both, because they answer different questions: a
        hectare under a centimetre is not a flood, and the portal's own
        threshold is 10 cm. Area alone lets a policy trade a large shallow
        nuisance for a small deep catastrophe, which is the wrong trade and
        the one `km2` has been warned about throughout this project.
        """
        assert self.h is not None
        b = self.h.shape[0]
        inside = self.protect.view(-1)
        h_in = self.h.view(b, -1)[:, inside]
        wet = (h_in > self.task.wet_m).to(h_in.dtype).mean(dim=1)
        deep = h_in.clamp(min=0.0).mean(dim=1)
        return -self.task.reward_scale * (wet + self.task.depth_weight * deep)

    # -- reporting ----------------------------------------------------------- #
    @torch.no_grad()
    def episode_stats(self) -> dict[str, Tensor]:
        """What a reader of the portal would ask for, not what the loss used."""
        assert self.h is not None and self._spent is not None
        b = self.h.shape[0]
        inside = self.protect.view(-1)
        h_in = self.h.view(b, -1)[:, inside]
        cell_km2 = (self.dx * self.dx) / 1e6
        return {
            "wet_km2": (h_in > self.task.wet_m).sum(dim=1) * cell_km2,
            "deepest_m": h_in.amax(dim=1),
            "pump_hours": self._spent.clone(),
        }


class FloodControlEnv(gym.Env):
    """Single-instance Gymnasium view of the polder, for anything outside this repo.

    `FloodOperateEnv` speaks the batched, all-torch protocol the planner and
    `baselines/safe_rl.py` consume, which is the right shape for a trainer that
    wants a gradient and sixteen worlds at once. It is the wrong shape for
    everybody else: an algorithm from outside this repository expects
    `gymnasium.make`, one episode at a time, numpy in and numpy out.

    This is that view, and it mirrors `PDEControlEnv` deliberately -- same
    `info["cost"]` convention, which is what the safe-RL baselines read and
    what Safety-Gymnasium and OmniSafe emit, so an OmniSafe install can be
    dropped in unchanged.

    Cost is pump-hours at the station's rated capacity. Reward is minus the
    depth-weighted wet fraction of the protected ground. Both are documented
    on `FloodOperateTask`; neither is in money.

    What it does NOT make true: nothing here has yet beaten a two-line level
    deadband (`FloodOperateEnv.baseline_action`), so this is an environment
    with a published API and not a benchmark with a published result. The
    distinction is in the module docstring and it matters.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid: int = 32,
        dx: float = 120.0,
        horizon: int = 12,
        control_minutes: float = 30.0,
        budget_pump_hours: float = 2.0,
        n_terrains: int = 64,
        surrogate: object | None = None,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        bank = make_polder_bank(n=n_terrains, grid=grid, dx=dx,
                                generator=torch.Generator().manual_seed(
                                    0 if seed is None else seed),
                                device=self.device)
        self._core = FloodOperateEnv(
            bank, task=FloodOperateTask(cost_limit=budget_pump_hours),
            horizon=horizon, control_minutes=control_minutes, dx=dx,
            surrogate=surrogate, device=self.device)
        self.protected_ha = float(bank["protected_ha"])

        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.episode_cost = 0.0

    # -- gymnasium API ------------------------------------------------------ #
    def reset(self, *, seed: int | None = None,
              options: "dict[str, Any] | None" = None
              ) -> "tuple[np.ndarray, dict[str, Any]]":
        # `super().reset(seed=seed)` seeds `self.np_random`, which is what
        # `action_space.sample()` draws from and what Gymnasium's own
        # `check_env` verifies. Without it the env looks seeded -- episodes
        # reproduce, because the torch generator below is doing that work --
        # while the action space stays on a global stream, so a random policy
        # is not reproducible from the seed the caller gave.
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action: "np.ndarray"
             ) -> "tuple[np.ndarray, float, bool, bool, dict[str, Any]]":
        act = torch.as_tensor(action, dtype=torch.float32,
                              device=self.device).view(1, -1).clamp(-1.0, 1.0)
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {
            "cost": float(cost[0]),
            "episode_cost": self.episode_cost,
            "cost_limit": self._core.task.cost_limit,
        }
        if done:
            # What a reader of the portal would ask for, not what the loss
            # used. Only at the end, because they are episode quantities.
            st = self._core.episode_stats()
            info.update(wet_km2=float(st["wet_km2"][0]),
                        deepest_m=float(st["deepest_m"][0]),
                        pump_hours=float(st["pump_hours"][0]))
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        # `terminated` stays False: the episode ends on the control horizon,
        # which is a time limit and not an absorbing state. Reporting it as
        # termination would tell a bootstrapping critic the value beyond the
        # horizon is zero, which is false -- the water is still there.
        return obs, float(reward[0]), False, bool(done), info

    # -- accessors the trainers and the portal use -------------------------- #
    @property
    def task(self) -> FloodOperateTask:
        return self._core.task

    @property
    def core(self) -> "FloodOperateEnv":
        return self._core

    def baseline_action(self, obs: "np.ndarray") -> "np.ndarray":
        """The level-deadband incumbent, for anyone benchmarking against it."""
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.baseline_action(t)[0].cpu().numpy()


def register_envs() -> None:
    """Make `gymnasium.make("FloodOperate-v0")` work.

    Idempotent, because importing this module twice in one process is normal
    and a duplicate registration is an error rather than a no-op.
    """
    from gymnasium.envs.registration import register, registry

    for env_id, kwargs in (
        # The trainable preset: 3.84 km of domain at 120 m, which preserves
        # the policy ordering of the 60 m grid at a fifth of the cost.
        ("FloodOperate-v0", {}),
        # Same physical domain, finer cells. Five times slower; this is the
        # one to report final numbers on.
        ("FloodOperateFine-v0", {"grid": 64, "dx": 60.0}),
    ):
        if env_id in registry:
            continue
        register(id=env_id, entry_point="pspe.envs.flood_env:FloodControlEnv",
                 kwargs=dict(kwargs), max_episode_steps=None)
