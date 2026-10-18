"""Which houses to take out of the floodplain.

Austin did this. After the 2013 and 2015 Onion Creek floods the city ran a
buyout programme, acquiring flood-prone homes and returning the land to open
space, and the question it faced is the one here: with money for only so many
properties, which ones?

It is a different kind of decision from everything else in this gym. The
others choose a continuous setting -- how hard to pump, how high to build,
where to cut. This one chooses a SET: each property is in or out, the budget
counts properties rather than dollars-per-metre, and the thing being changed
is not the flood but who is standing in it. Nothing here alters the water.

## What is real

* **43,366 building footprints** from the City of Austin's own published
  layer, fetched for the Onion Creek box, 43,292 of them carrying a ground
  elevation.
* **USGS 3DEP bare earth** at 60 m over that box.
* **FEMA's National Flood Hazard Layer** for which ground is at risk.

## What is not

The flood is a single-peaked rise held in the channel, as in `FloodDefend-v0`
-- it exercises the decision without reproducing a particular event. And the
budget counts PROPERTIES, not dollars. A real buyout is priced per house and
this repository has no verified acquisition figure for Austin; inventing a
plausible-looking dollar number and dividing by it would put false precision
on every result the environment produced. "You may take forty houses" is a
constraint anybody can check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from ..simulate.flood import FloodConfig, FloodSolver
from .defend import _fema_masks
from .splits import split_indices

Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]

#: Why this runs on Onion Creek and not on the box the footprints came in.
#:
#: The City publishes buildings for a box in north Austin. That box has
#: **zero** FEMA floodway polygons -- measured, 0 of 433 features -- so there
#: is no mapped channel to drive a flood down and nothing floods whatever the
#: stage: the first version of this file scored 0.00 damage for every policy,
#: including buying nothing, which is what sent me looking. Onion Creek has
#: the floodway, has 3,049 of those same building footprints inside it, and is
#: where Austin actually ran its buyout programme after 2013 and 2015. All
#: three reasons point the same way.
ONION_BOX = (-97.82, 30.11, -97.69, 30.21)


def _buildings_in(bounds) -> "list[tuple[float, float]]":
    """Footprint positions inside a box, as (lon, lat).

    One vertex per building is enough: at 40-60 m cells a house is a fraction
    of a cell, so its corner and its centroid land in the same one.
    """
    # The Onion Creek extract, not the north-Austin one. The published box
    # the footprints first arrived in has its houses in rows 0-21 of this grid
    # and the floodway in rows 63-126: they do not overlap, and only 46 of
    # 3,049 houses stood on at-risk ground. Re-fetching the same city layer
    # for this box gives 43,366 footprints over the creek itself.
    path = ROOT / "data" / "austin_onion_bld" / "buildings.geojson"
    if not path.is_file():                                 # pragma: no cover
        raise FileNotFoundError(
            f"no building footprints at {path}. Fetch them with: python3 "
            f"scripts/fetch_austin_data.py --out data/austin_onion_bld "
            f"--bbox=-97.82,30.11,-97.69,30.21 --skip addresses,"
            f"low_water_crossings,floodplain_austin,floodplain_fema,"
            f"base_flood_elevations")
    out = []
    for ft in json.loads(path.read_text())["features"]:
        coords = (ft.get("geometry") or {}).get("coordinates")
        if not coords:
            continue
        ring = coords[0]
        while isinstance(ring[0], (list, tuple)):
            ring = ring[0]
        lon, lat = float(ring[0]), float(ring[1])
        if (bounds["west"] <= lon <= bounds["east"]
                and bounds["south"] <= lat <= bounds["north"]):
            out.append((lon, lat))
    return out


def _building_density(shape, bounds, epsg) -> "np.ndarray":
    """Houses per cell, from the real footprints.

    A count rather than a mask: a cell with eleven houses is eleven buyouts
    and one with a single house is one, and flattening that to "built" would
    make the budget mean nothing.
    """
    from rasterio.crs import CRS                          # noqa: PLC0415
    from rasterio.transform import from_bounds, rowcol    # noqa: PLC0415
    from rasterio.warp import transform_bounds, transform_geom  # noqa: PLC0415

    h, w = shape
    src, dst = CRS.from_epsg(4326), CRS.from_epsg(epsg)
    pw, ps, pe, pn = transform_bounds(src, dst, bounds["west"], bounds["south"],
                                      bounds["east"], bounds["north"])
    tr = from_bounds(pw, ps, pe, pn, w, h)
    count = np.zeros((h, w), dtype=np.float32)
    pts = _buildings_in(bounds)
    if not pts:
        return count
    geom = {"type": "MultiPoint", "coordinates": [list(p) for p in pts]}
    for px, py in transform_geom(src, dst, geom)["coordinates"]:
        r, c = rowcol(tr, px, py)
        if 0 <= r < h and 0 <= c < w:
            count[int(r), int(c)] += 1.0
    return count


def austin_windows(grid: int = 40, dx: float = 60.0, *, min_homes: float = 20.0,
                   min_river: int = 6, reach_m: float = 6.0,
                   stride: int | None = None) -> "dict[str, Tensor]":
    """Windows of Onion Creek holding a channel, flood-risk ground and houses.

    `min_homes` counts houses the water can REACH -- on at-risk ground and
    within `reach_m` of the channel bed -- not houses on at-risk ground. The
    difference decides whether this is a benchmark: filtering on at-risk homes
    alone gave 13 windows of which 8 in 10 episodes scored exactly 0.0 for
    every policy, because FEMA's hundred-year floodplain here reaches ground a
    flood in the channel does not. Filtering on reachable homes gives 33
    windows that all pose a question.
    """
    from .defend import _cached_dem                       # noqa: PLC0415

    z_full, bounds, epsg = _cached_dem(dx)
    river_full, risk_full = _fema_masks(z_full.shape, bounds, epsg)
    homes_full = _building_density(z_full.shape, bounds, epsg)
    h, w = z_full.shape
    step = stride or max(grid // 4, 1)

    zs, rs, ps, hs = [], [], [], []
    for r in range(0, h - grid + 1, step):
        for c in range(0, w - grid + 1, step):
            riv = river_full[r:r + grid, c:c + grid]
            risk = risk_full[r:r + grid, c:c + grid]
            homes = homes_full[r:r + grid, c:c + grid]
            # Houses on ground FEMA calls at risk: the only ones a buyout
            # would be offered to, and what the budget is spent on.
            if riv.sum() < min_river:
                continue
            zz = z_full[r:r + grid, c:c + grid]
            bed = float(zz[riv].min())
            reach = risk & (zz - bed < reach_m)
            if homes[reach].sum() < min_homes:
                continue
            zs.append(z_full[r:r + grid, c:c + grid])
            rs.append(riv)
            ps.append(risk)
            hs.append(homes)
    if not zs:
        raise ValueError(
            f"no {grid}x{grid} window at {dx:.0f} m holds a floodway and "
            f"{min_homes:.0f} at-risk homes")

    z = torch.from_numpy(np.stack(zs)).float()
    z = z - z.amin(dim=(-2, -1), keepdim=True)
    return {"z": z,
            "river": torch.from_numpy(np.stack(rs)),
            "protect": torch.from_numpy(np.stack(ps)),
            "homes": torch.from_numpy(np.stack(hs)).float(),
            "dx": torch.tensor(dx)}


@dataclass
class FloodBuyoutTask:
    """The constraint, in houses."""

    #: Left at 0.0 the environment sets it to a share of the at-risk homes in
    #: an average window, so it binds on every window rather than on the mean.
    cost_limit: float = 0.0
    #: What share of the at-risk homes the budget can take.
    share: float = 0.25
    wet_m: float = 0.10
    reward_scale: float = 100.0


class FloodBuyoutEnv:
    """Batched-protocol core: one decision per zone, then one flood."""

    dynamics = "solver"

    def __init__(self, bank: "dict[str, Tensor]", task: "FloodBuyoutTask | None" = None,
                 *, zones: int = 9, flood_steps: int = 16, step_minutes: float = 30.0,
                 #: Calibrated by sweep: at 2-5 m almost nothing floods and
                 #: at 8-14 m the policies separate by a factor of four.
                 peak_stage_m: "tuple[float, float]" = (8.0, 14.0),
                 split: str = "train",
                 device: "torch.device | str" = "cpu") -> None:
        self.task = task or FloodBuyoutTask()
        self.device = torch.device(device)
        keep = split_indices(int(bank["z"].shape[0]), split)
        self.split = split
        self.z_bank = bank["z"][keep].to(self.device)
        self.river_bank = bank["river"][keep].to(self.device)
        self.risk_bank = bank["protect"][keep].to(self.device)
        self.homes_bank = bank["homes"][keep].to(self.device)
        self.dx = float(bank["dx"])
        self.n, self.h_cells, self.w = self.z_bank.shape
        self.K = zones
        self.flood_steps = flood_steps
        self.step_s = step_minutes * 60.0
        self.peak_stage_m = peak_stage_m
        self.cfg = FloodConfig(dx=self.dx, open_edges=("south", "east"))

        # Zones as a square tiling, which is how a buyout programme is drawn
        # on a map -- by block, not by individual title.
        side = int(np.ceil(np.sqrt(self.K)))
        self.zone = torch.zeros(self.K, self.h_cells, self.w, dtype=torch.bool,
                                device=self.device)
        rs = torch.linspace(0, self.h_cells, side + 1).long()
        cs = torch.linspace(0, self.w, side + 1).long()
        k = 0
        for i in range(side):
            for j in range(side):
                if k >= self.K:
                    break
                self.zone[k, rs[i]:rs[i + 1], cs[j]:cs[j + 1]] = True
                k += 1

        if self.task.cost_limit <= 0.0:
        # Derived from the WHOLE pool, before the split. A budget
        # computed per split makes train and eval different problems,
        # so a policy fitted under one constraint would be reported
        # under another -- and the catalog would record whichever
        # split it happened to measure.
            full = (bank["homes"].to(self.device)
                    * bank["protect"].to(self.device).float()).sum(dim=(1, 2))
            self.task.cost_limit = float(full.mean() * self.task.share)
        self.obs_shape = (5, self.h_cells, self.w)
        self.action_dim = 1
        self.t = 0

    def zone_homes(self, k: int) -> Tensor:
        """At-risk houses in zone `k`, per episode."""
        m = self.zone[k][None] & self.risk_bank[self.idx]
        return (self.homes_bank[self.idx] * m.float()).sum(dim=(1, 2))

    def reset(self, batch: int = 1, generator: "torch.Generator | None" = None) -> Tensor:
        self.idx = torch.randint(0, self.n, (batch,), generator=generator)
        self.bought = torch.zeros(batch, self.K, device=self.device)
        self.spent = torch.zeros(batch, device=self.device)
        lo, hi = self.peak_stage_m
        self._peak = (lo + (hi - lo) * torch.rand(batch, generator=generator)).to(self.device)
        self.t = 0
        self.last_depth = None
        return self._observe()

    def _observe(self) -> Tensor:
        b = self.idx.shape[0]
        z = self.z_bank[self.idx]
        zn = z / z.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        homes = self.homes_bank[self.idx]
        homes = homes / homes.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        k = min(self.t, self.K - 1)
        here = self.zone[k][None].expand(b, -1, -1).float()
        left = (1.0 - self.spent / max(self.task.cost_limit, 1e-9)).clamp(min=0.0)
        return torch.stack([zn, self.risk_bank[self.idx].float(), homes, here,
                            left.view(b, 1, 1).expand(b, self.h_cells, self.w)], dim=1)

    @torch.no_grad()
    def _solve(self) -> Tensor:
        z = self.z_bank[self.idx]
        solver = FloodSolver(z, self.cfg)
        h = torch.zeros_like(z)
        qx = torch.zeros(z.shape[0], z.shape[1], z.shape[2] - 1, device=z.device)
        qy = torch.zeros(z.shape[0], z.shape[1] - 1, z.shape[2], device=z.device)
        riv = self.river_bank[self.idx]
        for i in range(self.flood_steps):
            frac = i / max(self.flood_steps - 1, 1)
            d = (frac - 0.45) / 0.30
            stage = self._peak * float(np.exp(-0.5 * d * d))
            h = torch.where(riv, torch.maximum(h, stage.view(-1, 1, 1)), h)
            t_now = 0.0
            while t_now < self.step_s:
                dt = min(solver.adaptive_dt(h), self.step_s - t_now)
                if dt <= 0:
                    break
                h, qx, qy = solver.step(h, qx, qy, dt)
                t_now += dt
        self.last_depth = h.detach()

        # Houses still standing, in water. A bought-out zone's houses are
        # gone: the flood is unchanged and the exposure is not.
        homes = self.homes_bank[self.idx] * self.risk_bank[self.idx].float()
        remaining = homes.clone()
        for k in range(self.K):
            m = self.zone[k][None].expand_as(remaining)
            remaining = torch.where(m, remaining * (1.0 - self.bought[:, k].view(-1, 1, 1)),
                                    remaining)
        flooded = (remaining * (h > self.task.wet_m).float()).sum(dim=(1, 2))
        total = homes.sum(dim=(1, 2)).clamp(min=1.0)
        return flooded / total

    def step(self, action: Tensor) -> "tuple[Tensor, Tensor, Tensor, bool]":
        k = self.t
        u = (action.clamp(-1.0, 1.0).view(-1) + 1.0) / 2.0
        self.bought[:, k] = u
        cost = u * self.zone_homes(k)
        self.spent = self.spent + cost
        self.t += 1
        done = self.t >= self.K
        reward = (-self.task.reward_scale * self._solve() if done
                  else torch.zeros(u.shape[0], device=self.device))
        return self._observe(), reward, cost, done

    # -- baselines ---------------------------------------------------------- #
    @torch.no_grad()
    def even_action(self, obs=None) -> Tensor:
        """Take the same SHARE of houses in every zone."""
        total = torch.stack([self.zone_homes(k) for k in range(self.K)]).sum(0)
        frac = (self.task.cost_limit / total.clamp(min=1.0)).clamp(max=1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(-1, 1)

    @torch.no_grad()
    def lowest_first_action(self, obs: Tensor) -> Tensor:
        """Buy out the lowest-lying zones first. The obvious answer.

        Low ground floods first, so a programme that starts there is what
        anybody would propose -- and whether it is right depends on where the
        houses are, which is the question the environment poses.
        """
        b = self.idx.shape[0]
        k = min(self.t, self.K - 1)
        z = self.z_bank[self.idx]
        lvl = []
        for j in range(self.K):
            m = self.zone[j][None] & self.risk_bank[self.idx]
            lvl.append(torch.where(m, z, torch.full_like(z, float("inf"))).amin(dim=(-2, -1)))
        rank = torch.stack(lvl, dim=1).argsort(dim=1).argsort(dim=1)[:, k].float()
        # Spend the remaining budget preferentially on the lower-ranked zones.
        weight = (self.K - rank) / max(self.K, 1) * 2.0
        left = (self.task.cost_limit - self.spent).clamp(min=0.0)
        want = left * weight / max(self.K - k, 1)
        frac = (want / self.zone_homes(k).clamp(min=1.0)).clamp(0.0, 1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(b, 1)


class FloodBuyoutControlEnv(gym.Env):
    """Single-instance Gymnasium view of the buyout decision."""

    metadata = {"render_modes": []}

    def __init__(self, bank: "dict[str, Tensor] | None" = None, *,
                 budget: float = 0.0, zones: int = 9, flood_steps: int = 16,
                 grid: int = 40, dx: float = 60.0, split: str = "train",
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = FloodBuyoutEnv(
            bank if bank is not None else austin_windows(grid=grid, dx=dx, stride=4),
            FloodBuyoutTask(cost_limit=budget), zones=zones,
            flood_steps=flood_steps, split=split, device=self.device)
        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.episode_cost = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action):
        act = torch.as_tensor(action, dtype=torch.float32,
                              device=self.device).view(1, -1).clamp(-1.0, 1.0)
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {"cost": float(cost[0]), "episode_cost": self.episode_cost,
                "cost_limit": self._core.task.cost_limit}
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        return obs, float(reward[0]), False, bool(done), info

    @property
    def task(self):
        return self._core.task

    @property
    def core(self):
        return self._core

    def even_action(self, obs=None) -> "np.ndarray":
        return self._core.even_action()[0].cpu().numpy()

    def lowest_first_action(self, obs) -> "np.ndarray":
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.lowest_first_action(t)[0].cpu().numpy()


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "FloodBuyout-v0" in registry:
        return
    register(id="FloodBuyout-v0", entry_point="pspe.envs.buyout:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> FloodBuyoutControlEnv:
    return FloodBuyoutControlEnv(**kw)
