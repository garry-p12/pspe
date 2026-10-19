"""Which blocks to cool, before the heat wave.

Heat kills more people in Texas than floods do, and it does it without a
hydrograph: the air is simply too hot for too long, and who suffers depends on
where they live and whether anything shades or reflects. This environment
poses the planning decision -- spend a weatherisation and cooling budget
across K blocks, chosen before anybody knows which heat wave is coming.

## What is real

* **The heat.** NOAA AORC v1.1, hourly, ~1 km, from the public S3 Zarr. Six
  real Austin heat waves, each a seven-day window: August 2023 peaked at
  42.2 C, July 2018 at 43.7 C, and the 2011 drought week sat above 35 C for
  41% of its hours.
* **Where the people are.** 43,366 City of Austin building footprints, as a
  population proxy -- see the caveat below, because it is a proxy and not a
  census.
* **What can be treated.** NLCD 2021 land cover, converted to an impervious
  fraction. A cool roof needs a roof and a street tree needs a street.

**The two obvious rules disagree here, and that is the point.** Treating the
hottest blocks and treating the most built-up blocks pick almost entirely
different ground: over the study box their top cells overlap by 12%, with a
correlation of +0.39 between heat and building density. An environment where
the candidate heuristics agree has no decision in it.

## What is assumed, and it is one thing

Cooling per unit of treatment is a **literature coefficient, not a model**.
`cooling_c` is the air-temperature reduction at full treatment of a cell's
impervious area; published neighbourhood-scale estimates run from about 0.3 C
for cool roofs alone to 1-2 C where tree canopy is added, so 1.5 C is the
optimistic end of a mixed programme. Nothing here simulates a surface energy
balance. Every result scales with that number, and a reader should treat the
ORDERING of policies as the finding and the magnitudes as conditional on it.

That is the only invented quantity. Three things the original specification
asked for were dropped rather than invented, because the data to support them
is not available here:

* **Indoor overheating** needs a thermal model per building. ResStock or a
  calibrated RC model would supply one; this repository has neither, and
  coefficients chosen to look plausible would put false precision on every
  episode. The reward is outdoor exposure.
* **Households without reliable air conditioning** cannot be located. The
  American Community Survey does not publish air conditioning at tract level
  in Texas, and RECS is regional. The reward weights by people, not by people
  without air conditioning.
* **Oldest housing stock first**, one of the two proposed bars, needs a year
  built. The City of Austin building-footprint layer carries OBJECTID,
  heights and elevations and no construction date -- checked against the live
  service, not assumed. The bar is building density instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .buyout import _building_density
from .defend import _cached_dem
from .greeninfra import AORC_DIR, _impervious
from .splits import split_indices

Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]


def load_heatwaves() -> "list[dict]":
    """Every fetched heat wave, as hourly degrees C over the box."""
    files = sorted(AORC_DIR.glob("aorc_heat_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no AORC heat waves in {AORC_DIR}. Fetch them with: python3 "
            f"scripts/fetch_aorc_storms.py --out data/aorc --heat")
    out = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            out.append({"name": str(d["name"]), "temp_c": d["temp_c"],
                        "lat": d["lat"], "lon": d["lon"]})
    return out


def _heat_on_grid(temp: "np.ndarray", lat, lon, shape, bounds) -> "np.ndarray":
    """A 1 km AORC field, nearest-neighboured onto the DEM grid.

    Nearest and not bilinear: the field is already smooth at 1 km and the
    environment's decision is which 60 m cells to treat, so interpolating
    would invent sub-kilometre structure that AORC does not resolve and that
    a policy could then appear to exploit.
    """
    h, w = shape
    ys = np.linspace(bounds["north"], bounds["south"], h)
    xs = np.linspace(bounds["west"], bounds["east"], w)
    iy = np.clip(np.searchsorted(lat, ys), 0, len(lat) - 1)
    ix = np.clip(np.searchsorted(lon, xs), 0, len(lon) - 1)
    return temp[:, iy][:, :, ix]            # (T, h, w)


def austin_blocks(grid: int = 80, dx: float = 60.0, *,
                  min_homes: float = 150.0, stride: int | None = None
                  ) -> "dict[str, Tensor]":
    """Windows of Austin with people in them and something to treat."""
    z, bounds, epsg = _cached_dem(dx)
    homes = _building_density(z.shape, bounds, epsg)
    imp = _impervious(z.shape, bounds, epsg)
    h, w = z.shape
    step = stride or max(grid // 6, 1)

    hs, ims, idxs = [], [], []
    for r in range(0, h - grid + 1, step):
        for c in range(0, w - grid + 1, step):
            win_h = homes[r:r + grid, c:c + grid]
            if win_h.sum() < min_homes:
                continue
            hs.append(win_h)
            ims.append(imp[r:r + grid, c:c + grid])
            idxs.append((r, c))
    if not hs:
        raise ValueError(f"no {grid}x{grid} window holds {min_homes:.0f} buildings")
    return {"homes": torch.from_numpy(np.stack(hs)).float(),
            "impervious": torch.from_numpy(np.stack(ims)).float(),
            "origin": torch.tensor(idxs), "dx": torch.tensor(dx),
            "bounds": bounds, "shape": z.shape}


@dataclass
class HeatRetrofitTask:
    """The constraint, in treated area, and the one assumption."""

    cost_limit: float = 0.0
    #: Share of the impervious area the budget can treat.
    share: float = 0.25
    #: Air-temperature reduction at full treatment of a cell's impervious
    #: area, in degrees C. A LITERATURE COEFFICIENT, not a model: published
    #: neighbourhood-scale estimates run 0.3 C for cool roofs alone to 1-2 C
    #: with canopy. Every magnitude here scales with it; the ordering of
    #: policies is the finding.
    cooling_c: float = 1.5
    #: The temperature above which an hour counts as dangerous.
    threshold_c: float = 35.0
    reward_scale: float = 100.0


class HeatRetrofitEnv:
    """Batched core: one decision per block, then the heat wave is counted.

    There is no PDE here and that is deliberate -- the reward is a count over
    a real temperature field, so an episode costs milliseconds rather than
    seconds, and the pool can be large.
    """

    dynamics = "observed"

    def __init__(self, bank: "dict", task: "HeatRetrofitTask | None" = None, *,
                 blocks: int = 9, split: str = "train",
                 wave_split: str = "train",
                 device: "torch.device | str" = "cpu") -> None:
        self.task = task or HeatRetrofitTask()
        self.device = torch.device(device)
        keep = split_indices(int(bank["homes"].shape[0]), split)
        self.split = split
        self.homes = bank["homes"][keep].to(self.device)
        self.imp = bank["impervious"][keep].to(self.device)
        self.origin = bank["origin"][keep]
        self.dx = float(bank["dx"])
        self.n, self.h_cells, self.w = self.homes.shape
        self.K = blocks

        waves = load_heatwaves()
        pick = split_indices(len(waves), wave_split)
        self.waves = [waves[i] for i in pick]
        self.wave_split = wave_split
        self.wave_names = [w["name"] for w in self.waves]
        # Each wave, once, on the full DEM grid; windows index into it.
        self._fields = [
            torch.from_numpy(_heat_on_grid(w["temp_c"], w["lat"], w["lon"],
                                           bank["shape"], bank["bounds"]))
            .float().to(self.device) for w in self.waves]

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
            full = bank["impervious"].to(self.device).sum(dim=(1, 2))
            self.task.cost_limit = float(full.mean() * self.task.share
                                         * self.dx * self.dx)
        self.obs_shape = (5, self.h_cells, self.w)
        self.action_dim = 1
        self.t = 0

    # -- the heat each window sees ------------------------------------------ #
    def window_heat(self) -> Tensor:
        """(B, T, H, W) hourly degrees C for each episode's window and wave."""
        out = []
        for b in range(self.idx.shape[0]):
            r, c = self.origin[self.idx[b]].tolist()
            f = self._fields[int(self.wave_idx[b])]
            out.append(f[:, r:r + self.h_cells, c:c + self.w])
        return torch.stack(out)

    def zone_impervious_m2(self, k: int) -> Tensor:
        m = self.zone[k][None].expand(self.idx.shape[0], -1, -1)
        return (self.imp[self.idx] * m.float()).sum(dim=(1, 2)) * self.dx * self.dx

    def reset(self, batch: int = 1, generator: "torch.Generator | None" = None) -> Tensor:
        self.idx = torch.randint(0, self.n, (batch,), generator=generator)
        self.wave_idx = torch.randint(0, len(self.waves), (batch,), generator=generator)
        self.treated = torch.zeros(batch, self.K, device=self.device)
        self.spent = torch.zeros(batch, device=self.device)
        self.t = 0
        self._heat = self.window_heat()
        self.last_exposure = None
        return self._observe()

    def _observe(self) -> Tensor:
        b = self.idx.shape[0]
        homes = self.homes[self.idx]
        homes = homes / homes.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        hot = (self._heat > self.task.threshold_c).float().mean(dim=1)
        k = min(self.t, self.K - 1)
        here = self.zone[k][None].expand(b, -1, -1).float()
        done = torch.zeros_like(homes)
        for j in range(self.K):
            m = self.zone[j][None].expand(b, -1, -1)
            done = torch.where(m, self.treated[:, j].view(-1, 1, 1).expand_as(done), done)
        left = (1.0 - self.spent / max(self.task.cost_limit, 1e-9)).clamp(min=0.0)
        return torch.stack([hot, homes, self.imp[self.idx], here,
                            left.view(b, 1, 1).expand(b, self.h_cells, self.w)], dim=1)

    def cooling_field(self) -> Tensor:
        """(B, H, W) degrees C removed from each cell by what has been built."""
        b = self.idx.shape[0]
        frac = torch.zeros(b, self.h_cells, self.w, device=self.device)
        for k in range(self.K):
            m = self.zone[k][None].expand(b, -1, -1)
            frac = torch.where(m, self.treated[:, k].view(-1, 1, 1).expand_as(frac), frac)
        return frac * self.imp[self.idx] * self.task.cooling_c

    @torch.no_grad()
    def _score(self) -> Tensor:
        """Population-weighted share of hours spent above the threshold."""
        cooled = self._heat - self.cooling_field()[:, None]
        hot = (cooled > self.task.threshold_c).float().mean(dim=1)   # (B,H,W)
        pop = self.homes[self.idx]
        self.last_exposure = hot.detach()
        return ((hot * pop).sum(dim=(1, 2))
                / pop.sum(dim=(1, 2)).clamp(min=1.0))

    def step(self, action: Tensor) -> "tuple[Tensor, Tensor, Tensor, bool]":
        k = self.t
        u = (action.clamp(-1.0, 1.0).view(-1) + 1.0) / 2.0
        self.treated[:, k] = u
        cost = u * self.zone_impervious_m2(k)
        self.spent = self.spent + cost
        self.t += 1
        done = self.t >= self.K
        reward = (-self.task.reward_scale * self._score() if done
                  else torch.zeros(u.shape[0], device=self.device))
        return self._observe(), reward, cost, done

    # -- baselines ---------------------------------------------------------- #
    @torch.no_grad()
    def even_action(self, obs=None) -> Tensor:
        """Treat the same share of impervious area in every block."""
        total = torch.stack([self.zone_impervious_m2(k) for k in range(self.K)]).sum(0)
        frac = (self.task.cost_limit / total.clamp(min=1.0)).clamp(max=1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(-1, 1)

    @torch.no_grad()
    def hottest_action(self, obs: Tensor) -> Tensor:
        """Treat the hottest blocks first. One of the two obvious answers."""
        hot = (self._heat > self.task.threshold_c).float().mean(dim=1)
        return self._rank(lambda j: self._zone_sum(hot, j))

    @torch.no_grad()
    def most_people_action(self, obs: Tensor) -> Tensor:
        """Treat the most built-up blocks first. The other obvious answer."""
        return self._rank(lambda j: self._zone_sum(self.homes[self.idx], j))

    @torch.no_grad()
    def exposure_first_action(self, obs: Tensor) -> Tensor:
        """Treat where heat and people coincide. The obvious synthesis.

        It is the rule anybody writes once told the first two disagree, and it
        is still not obviously right: cooling helps most where the temperature
        sits just ABOVE the threshold, since a block already far above it
        stays above after 1.5 C comes off.
        """
        hot = (self._heat > self.task.threshold_c).float().mean(dim=1)
        return self._rank(lambda j: self._zone_sum(hot * self.homes[self.idx], j))

    def _zone_sum(self, field: Tensor, j: int) -> Tensor:
        m = self.zone[j][None].expand(field.shape[0], -1, -1)
        return (field * m.float()).sum(dim=(1, 2))

    def _rank(self, score_fn) -> Tensor:
        b = self.idx.shape[0]
        k = min(self.t, self.K - 1)
        scores = torch.stack([score_fn(j) for j in range(self.K)], dim=1)
        rank = scores.argsort(dim=1, descending=True).argsort(dim=1)[:, k].float()
        weight = (self.K - rank) / max(self.K, 1) * 2.0
        left = (self.task.cost_limit - self.spent).clamp(min=0.0)
        want = torch.minimum(left * weight / max(self.K - k, 1), left)
        frac = (want / self.zone_impervious_m2(k).clamp(min=1.0)).clamp(0.0, 1.0)
        return (2.0 * frac * (1.0 - 1e-6) - 1.0).view(b, 1)


class HeatRetrofitControlEnv(gym.Env):
    """Single-instance Gymnasium view of the cooling-programme decision."""

    metadata = {"render_modes": []}

    def __init__(self, bank: "dict | None" = None, *, budget: float = 0.0,
                 blocks: int = 9, grid: int = 80, dx: float = 60.0,
                 split: str = "train", wave_split: "str | None" = None,
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = HeatRetrofitEnv(
            bank if bank is not None else austin_blocks(grid=grid, dx=dx),
            HeatRetrofitTask(cost_limit=budget), blocks=blocks, split=split,
            wave_split=split if wave_split is None else wave_split,
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
                "heatwave": self._core.wave_names[int(self._core.wave_idx[0])]}
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        return obs, float(reward[0]), False, bool(done), info

    @property
    def task(self):
        return self._core.task

    @property
    def core(self):
        return self._core

    def _np(self, t):
        return t[0].cpu().numpy()

    def even_action(self, obs=None):
        return self._np(self._core.even_action())

    def hottest_action(self, obs):
        return self._np(self._core.hottest_action(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]))

    def most_people_action(self, obs):
        return self._np(self._core.most_people_action(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]))

    def exposure_first_action(self, obs):
        return self._np(self._core.exposure_first_action(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]))


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "HeatRetrofit-v0" in registry:
        return
    register(id="HeatRetrofit-v0", entry_point="pspe.envs.heat:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> HeatRetrofitControlEnv:
    return HeatRetrofitControlEnv(**kw)
