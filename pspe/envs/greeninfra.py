"""Where to put the stormwater infrastructure, before the storm.

Every other flood environment here is handed its water: a stage held in the
channel, rising and falling on a curve this project chose. This one is handed
a **storm**. Rain falls on the whole domain, at 1 km and hourly, from NOAA's
Analysis of Record for Calibration -- the real field that fell on Onion Creek
on 31 October 2013, or on the Memorial Day weekend of 2015, or on five other
days that actually flooded Austin. The flooding is pluvial: it comes from the
sky onto the ground you are standing on, not from a river that rose somewhere
upstream.

The decision is where to spend a stormwater retrofit budget -- bioretention,
permeable pavement, detention -- across K sub-catchments, chosen before
anybody knows which storm is coming. It is scored on the **low ground**: the
bottom quarter of the window by elevation, where runoff collects. Scoring the
whole window instead makes the placement decision vanish, for a reason that
is correct physics and is recorded in `_solve`.

## What is real

* **The rain.** NOAA AORC v1.1, hourly, ~1 km, from the public S3 Zarr. Seven
  storms, each a real three-day window bracketing a real Austin flood. The
  wettest, 29-31 October 2015, put 245 mm over the box and 331 mm on its
  wettest cell.
* **The impervious cover.** NLCD 2021 land cover at 208 m, converted to an
  impervious fraction through the published class ranges. This is what decides
  where treatment can do anything: you cannot retrofit a forest.
* **The ground.** USGS 3DEP bare earth, the same tiles every other flood
  environment here uses.

## What is not

The treatment model is lumped: a treated cell absorbs `capture` of the rain
that falls on it, up to its impervious fraction. Real bioretention has a
storage volume that fills, a drawdown rate, and an underdrain; none of that is
here, and a cell that has absorbed all day keeps absorbing. That makes this
environment optimistic about sustained storms and roughly right about short
ones.

The budget is **treated impervious area**, not dollars. Green-infrastructure
unit costs vary by an order of magnitude with the technique, and this
repository has no verified rate for Austin -- unlike the levee environment,
whose USACE figures carry a derivation. Inventing a dollar rate would put
false precision on every result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..simulate.flood import FloodConfig, FloodSolver
from .defend import _cached_dem
from .splits import split_indices

Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]
AORC_DIR = ROOT / "data" / "aorc"
NLCD_PATH = ROOT / "data" / "nlcd" / "nlcd2021_austin.tif"

#: Impervious fraction by NLCD developed class, from the published ranges in
#: the NLCD legend: developed open <20%, low 20-49%, medium 50-79%, high
#: 80-100%. Midpoints, because the class raster is what is available and a
#: midpoint is the honest reading of a band. Everything undeveloped is 0.
NLCD_IMPERVIOUS = {21: 0.10, 22: 0.35, 23: 0.65, 24: 0.90}


def load_storms() -> "list[dict]":
    """Every fetched storm, as hourly mm over the box."""
    if not AORC_DIR.is_dir() or not any(
            f for f in AORC_DIR.glob("aorc_*.npz")
            if not f.name.startswith("aorc_heat_")):
        raise FileNotFoundError(
            f"no AORC storms in {AORC_DIR}. Fetch them with: python3 "
            f"scripts/fetch_aorc_storms.py --out data/aorc  (needs s3fs, "
            f"xarray and zarr; the TACC Vista `dsc` environment has all three)")
    out = []
    # `aorc_heat_*.npz` live in the same directory and hold temperature, not
    # rain. Globbing `aorc_*.npz` swallowed them and failed on the first
    # missing key, which is a better failure than silently treating degrees
    # Celsius as millimetres -- but it is still a failure, so they are
    # excluded by name.
    for f in sorted(AORC_DIR.glob("aorc_*.npz")):
        if f.name.startswith("aorc_heat_"):
            continue
        with np.load(f, allow_pickle=False) as d:
            out.append({"name": str(d["name"]), "precip": d["precip_mm_h"],
                        "lat": d["lat"], "lon": d["lon"], "file": f.name})
    return out


def _impervious(shape, bounds, epsg) -> np.ndarray:
    """Impervious fraction on the DEM grid, from NLCD land-cover classes."""
    import rasterio                                    # noqa: PLC0415
    from rasterio.crs import CRS                       # noqa: PLC0415
    from rasterio.transform import from_bounds         # noqa: PLC0415
    from rasterio.warp import Resampling, reproject, transform_bounds  # noqa: PLC0415

    if not NLCD_PATH.is_file():
        raise FileNotFoundError(
            f"no land cover at {NLCD_PATH}. It is a 160x160 crop of NLCD 2021 "
            f"over the Austin box; see `docs` or re-cut it from the MRLC "
            f"GeoServer WMS.")
    h, w = shape
    dst = CRS.from_epsg(epsg)
    pw, ps, pe, pn = transform_bounds(CRS.from_epsg(4326), dst, bounds["west"],
                                      bounds["south"], bounds["east"],
                                      bounds["north"])
    tr = from_bounds(pw, ps, pe, pn, w, h)
    with rasterio.open(NLCD_PATH) as src:
        classes = np.zeros((h, w), dtype="float32")
        # NEAREST, not average: these are class codes, and the mean of
        # "deciduous forest" and "developed high" is a class that does not
        # exist. The conversion to a fraction happens after.
        reproject(source=rasterio.band(src, 1), destination=classes,
                  dst_transform=tr, dst_crs=dst, resampling=Resampling.nearest)
    imp = np.zeros_like(classes)
    for code, frac in NLCD_IMPERVIOUS.items():
        imp[np.round(classes) == code] = frac
    return imp


def onion_catchments(grid: int = 80, dx: float = 60.0, *,
                     min_impervious: float = 0.04, stride: int | None = None
                     ) -> "dict[str, Tensor]":
    """Windows of Onion Creek that have something to retrofit.

    80 cells, which is 4.8 km, and the size is a measurement rather than a
    default. At 40 cells no placement rule separates from any other: even,
    paved-first and uphill-first all score -0.089 against doing nothing, to
    three decimals, because over 2.4 km the runoff from any catchment reaches
    the same low ground. At 4.8 km there is enough distance between a
    catchment and the receptor for the choice to matter, and it does.

    A window with no impervious cover poses no question -- there is nothing to
    treat and every policy scores the same -- so it is dropped rather than
    left to dilute the measurement.
    """
    z_full, bounds, epsg = _cached_dem(dx)
    imp_full = _impervious(z_full.shape, bounds, epsg)
    h, w = z_full.shape
    step = stride or max(grid // 4, 1)

    zs, ims = [], []
    for r in range(0, h - grid + 1, step):
        for c in range(0, w - grid + 1, step):
            imp = imp_full[r:r + grid, c:c + grid]
            if imp.mean() < min_impervious:
                continue
            zs.append(z_full[r:r + grid, c:c + grid])
            ims.append(imp)
    if not zs:
        raise ValueError(f"no {grid}x{grid} window has {min_impervious:.0%} "
                         f"impervious cover")
    z = torch.from_numpy(np.stack(zs)).float()
    z = z - z.amin(dim=(-2, -1), keepdim=True)
    return {"z": z, "impervious": torch.from_numpy(np.stack(ims)).float(),
            "dx": torch.tensor(dx)}


@dataclass
class FloodGreenInfraTask:
    """The constraint, in treated impervious area."""

    #: Left at 0.0 the environment sets it: the impervious area of an average
    #: window times `share`. Derived from the ground, like every other budget
    #: here, and from the WHOLE pool so the splits pose the same problem.
    cost_limit: float = 0.0
    #: What fraction of the impervious cover the budget can retrofit. A
    #: quarter is ambitious for a real programme and small enough that where
    #: it goes still matters.
    share: float = 0.25
    #: Fraction of the rain a fully treated impervious cell absorbs.
    capture: float = 0.8
    wet_m: float = 0.15
    reward_scale: float = 100.0
    #: The share of each window, by elevation, that counts as the receptor.
    #: A tenth, set by measurement: at a quarter no placement rule separates
    #: from any other, and at a tenth spreading the budget evenly is
    #: resolvably better than putting it uphill.
    receptor_quantile: float = 0.10


class FloodGreenInfraEnv:
    """Batched core: one decision per sub-catchment, then one real storm."""

    dynamics = "solver"

    def __init__(self, bank: "dict[str, Tensor]", task: "FloodGreenInfraTask | None" = None,
                 *, catchments: int = 9, hours: int = 24,
                 split: str = "train", storm_split: str = "train",
                 device: "torch.device | str" = "cpu") -> None:
        self.task = task or FloodGreenInfraTask()
        self.device = torch.device(device)
        keep = split_indices(int(bank["z"].shape[0]), split)
        self.split = split
        self.z_bank = bank["z"][keep].to(self.device)
        self.imp_bank = bank["impervious"][keep].to(self.device)
        self.dx = float(bank["dx"])
        self.n, self.h_cells, self.w = self.z_bank.shape
        self.K = catchments
        self.hours = hours
        self.cfg = FloodConfig(dx=self.dx, open_edges=("south", "east"))

        # Storms are held out SEPARATELY from windows. A policy that has seen
        # every storm on four windows has still seen every storm, and the
        # question this environment asks -- what do you build before you know
        # what is coming -- is only honest if some storms are new.
        storms = load_storms()
        pick = split_indices(len(storms), storm_split)
        self.storms = [storms[i] for i in pick]
        self.storm_split = storm_split
        self.storm_names = [s["name"] for s in self.storms]

        # Sub-catchments as a square tiling. Real catchments would be traced
        # from flow accumulation; a tiling is what a retrofit programme is
        # actually drawn on, by block.
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
            # From the full pool, not the split: train and eval must pose the
            # same problem.
            full = bank["impervious"].to(self.device).sum(dim=(1, 2))
            self.task.cost_limit = float(full.mean() * self.task.share
                                         * self.dx * self.dx)
        self.obs_shape = (5, self.h_cells, self.w)
        self.action_dim = 1
        self.t = 0

    # -- episode ------------------------------------------------------------ #
    def zone_impervious_m2(self, k: int) -> Tensor:
        """Impervious area in catchment `k`, per episode, in square metres."""
        m = self.zone[k][None].expand(self.idx.shape[0], -1, -1)
        return (self.imp_bank[self.idx] * m.float()).sum(dim=(1, 2)) * self.dx * self.dx

    def reset(self, batch: int = 1, generator: "torch.Generator | None" = None) -> Tensor:
        self.idx = torch.randint(0, self.n, (batch,), generator=generator)
        self.storm_idx = torch.randint(0, len(self.storms), (batch,), generator=generator)
        self.treated = torch.zeros(batch, self.K, device=self.device)
        self.spent = torch.zeros(batch, device=self.device)
        self.t = 0
        self.last_depth = None
        return self._observe()

    def _observe(self) -> Tensor:
        b = self.idx.shape[0]
        z = self.z_bank[self.idx]
        zn = z / z.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        imp = self.imp_bank[self.idx]
        k = min(self.t, self.K - 1)
        here = self.zone[k][None].expand(b, -1, -1).float()
        done_so_far = torch.zeros_like(imp)
        for j in range(self.K):
            m = self.zone[j][None].expand(b, -1, -1)
            done_so_far = torch.where(m, self.treated[:, j].view(-1, 1, 1)
                                      .expand_as(done_so_far), done_so_far)
        left = (1.0 - self.spent / max(self.task.cost_limit, 1e-9)).clamp(min=0.0)
        return torch.stack([zn, imp, here, done_so_far,
                            left.view(b, 1, 1).expand(b, self.h_cells, self.w)], dim=1)

    def hyetograph(self) -> Tensor:
        """(B, hours) mm per hour, averaged over each episode's own storm.

        The AORC field is 1 km and the solver runs at `dx`; rather than
        regridding a 36x36 rain field onto a 40x40 window whose footprint it
        only partly covers, the storm is reduced to its box-mean hyetograph and
        applied uniformly. The intensity and the timing are the real storm's;
        the spatial structure of the rain is not used. That is a real
        limitation and is stated in the catalog.
        """
        out = []
        for s in self.storm_idx.tolist():
            series = self._window(self.storms[s]["precip"].mean(axis=(1, 2)))
            out.append(series)
        return torch.from_numpy(np.stack(out)).float().to(self.device)

    def _window(self, series: "np.ndarray") -> "np.ndarray":
        """The wettest `hours` of a storm, not the first.

        Each AORC pull is a three-day window BRACKETING the event, so the rain
        is in the middle of it. Taking the first 24 hours simulated the day
        before it rained -- which is why the first run of this environment
        found 0.95% of the window flooded and every policy within 0.01 of
        every other.
        """
        n = self.hours
        if len(series) <= n:
            return np.pad(series, (0, n - len(series))).astype("float32")
        totals = np.convolve(series, np.ones(n), mode="valid")
        start = int(totals.argmax())
        return series[start:start + n].astype("float32")

    def treated_field(self) -> Tensor:
        """(B, H, W) fraction of each cell's rain that is captured."""
        b = self.idx.shape[0]
        frac = torch.zeros(b, self.h_cells, self.w, device=self.device)
        for k in range(self.K):
            m = self.zone[k][None].expand(b, -1, -1)
            frac = torch.where(m, self.treated[:, k].view(-1, 1, 1).expand_as(frac), frac)
        # Only impervious ground can be retrofitted, and only that share of the
        # rain is captured.
        return (frac * self.imp_bank[self.idx] * self.task.capture).clamp(0.0, 1.0)

    @torch.no_grad()
    def _solve(self) -> Tensor:
        z = self.z_bank[self.idx]
        solver = FloodSolver(z, self.cfg)
        h = torch.zeros_like(z)
        qx = torch.zeros(z.shape[0], z.shape[1], z.shape[2] - 1, device=z.device)
        qy = torch.zeros(z.shape[0], z.shape[1] - 1, z.shape[2], device=z.device)
        hye = self.hyetograph()                       # (B, hours) mm/h
        capture = self.treated_field()                # (B, H, W)
        for hour in range(self.hours):
            mm_h = hye[:, hour].view(-1, 1, 1)
            rain = mm_h / 1000.0 / 3600.0 * (1.0 - capture)      # m/s
            now = 0.0
            while now < 3600.0:
                dt = min(solver.adaptive_dt(h), 3600.0 - now)
                if dt <= 0:
                    break
                h, qx, qy = solver.step(h, qx, qy, dt, rain=rain)
                now += dt
        self.last_depth = h.detach()
        # Scored on the LOW GROUND, not the whole window.
        #
        # Measured both ways. Over the whole window, where you retrofit makes
        # no difference at all -- even, paved-first and uphill-first all come
        # in at -0.089 against doing nothing, to three decimals. That is
        # correct physics for a volume-limited problem: capturing a quarter of
        # the impervious rain removes the same volume wherever it is captured,
        # so only HOW MUCH matters and the decision this environment poses
        # would be a decision about nothing.
        #
        # Water collects downhill, and so does damage. Scoring the bottom
        # quarter of the window gives the choice somewhere to matter: uphill
        # treatment stops water before it arrives, and the ordering separates.
        low = self._receptor()
        wet = ((h > self.task.wet_m) & low).float().sum(dim=(1, 2))
        return wet / low.float().sum(dim=(1, 2)).clamp(min=1.0)

    def _receptor(self) -> Tensor:
        """The lowest quarter of each window: where the water ends up."""
        z = self.z_bank[self.idx]
        q = torch.quantile(z.flatten(1), self.task.receptor_quantile,
                           dim=1).view(-1, 1, 1)
        return z <= q

    def step(self, action: Tensor) -> "tuple[Tensor, Tensor, Tensor, bool]":
        k = self.t
        u = (action.clamp(-1.0, 1.0).view(-1) + 1.0) / 2.0
        self.treated[:, k] = u
        cost = u * self.zone_impervious_m2(k)
        self.spent = self.spent + cost
        self.t += 1
        done = self.t >= self.K
        reward = (-self.task.reward_scale * self._solve() if done
                  else torch.zeros(u.shape[0], device=self.device))
        return self._observe(), reward, cost, done

    # -- baselines ---------------------------------------------------------- #
    @torch.no_grad()
    def even_action(self, obs=None) -> Tensor:
        """Retrofit the same share of impervious cover in every catchment."""
        total = torch.stack([self.zone_impervious_m2(k) for k in range(self.K)]).sum(0)
        frac = (self.task.cost_limit / total.clamp(min=1.0)).clamp(max=1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(-1, 1)

    @torch.no_grad()
    def most_paved_action(self, obs: Tensor) -> Tensor:
        """Spend where the pavement is. The obvious answer.

        Impervious cover is what generates runoff, so treating the most paved
        catchments first is what anybody proposes. Whether it is right depends
        on where that runoff goes, which is the question.
        """
        return self._rank_action(lambda j: self.zone_impervious_m2(j))

    @torch.no_grad()
    def highest_first_action(self, obs: Tensor) -> Tensor:
        """Spend uphill, so the water is stopped before it collects."""
        z = self.z_bank[self.idx]
        def score(j):
            m = self.zone[j][None].expand(z.shape[0], -1, -1)
            return torch.where(m, z, torch.zeros_like(z)).sum(dim=(1, 2))
        return self._rank_action(score)

    def _rank_action(self, score_fn) -> Tensor:
        """Give the remaining budget to this catchment in proportion to rank."""
        b = self.idx.shape[0]
        k = min(self.t, self.K - 1)
        scores = torch.stack([score_fn(j) for j in range(self.K)], dim=1)   # (B,K)
        rank = scores.argsort(dim=1, descending=True).argsort(dim=1)[:, k].float()
        weight = (self.K - rank) / max(self.K, 1) * 2.0
        left = (self.task.cost_limit - self.spent).clamp(min=0.0)
        # `weight` reaches 2.0, so a rank-weighted share of what is left can
        # exceed what is left once it is spent over the remaining catchments.
        # Capping at the remainder is the difference between a baseline and a
        # policy that was allowed to overspend: before this, uphill-first
        # cleared the budget in 7 episodes of 12 and paved-first in 10.
        want = torch.minimum(left * weight / max(self.K - k, 1), left)
        frac = (want / self.zone_impervious_m2(k).clamp(min=1.0)).clamp(0.0, 1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(b, 1)


class FloodGreenInfraControlEnv(gym.Env):
    """Single-instance Gymnasium view of the stormwater retrofit decision."""

    metadata = {"render_modes": []}

    def __init__(self, bank: "dict[str, Tensor] | None" = None, *,
                 budget: float = 0.0, catchments: int = 9, hours: int = 24,
                 grid: int = 80, dx: float = 60.0, split: str = "train",
                 storm_split: "str | None" = None,
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = FloodGreenInfraEnv(
            bank if bank is not None else onion_catchments(grid=grid, dx=dx,
                                                           stride=8),
            FloodGreenInfraTask(cost_limit=budget), catchments=catchments,
            hours=hours, split=split,
            storm_split=split if storm_split is None else storm_split,
            device=self.device)
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
                "cost_limit": self._core.task.cost_limit,
                "storm": self._core.storm_names[int(self._core.storm_idx[0])]}
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

    def most_paved_action(self, obs) -> "np.ndarray":
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.most_paved_action(t)[0].cpu().numpy()

    def highest_first_action(self, obs) -> "np.ndarray":
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.highest_first_action(t)[0].cpu().numpy()


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "FloodGreenInfra-v0" in registry:
        return
    register(id="FloodGreenInfra-v0",
             entry_point="pspe.envs.greeninfra:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> FloodGreenInfraControlEnv:
    return FloodGreenInfraControlEnv(**kw)
