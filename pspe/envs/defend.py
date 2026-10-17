"""Where to spend the money, before you know which flood is coming.

The other flood environment in this repository operates a polder: twelve
settings on a gate and a pump while a flood passes, on generated terrain. This
one is the decision that comes years earlier and is the one the portal exists
to support -- **how to spend a fixed capital budget raising defences along a
real creek, chosen before anybody knows what will arrive.**

Three things make it a different problem rather than the same one rearranged:

* **The terrain is real.** USGS 3DEP bare earth over the Onion Creek study
  box, with the channel and the at-risk ground taken from FEMA's own National
  Flood Hazard Layer -- `FLOODWAY` for the creek, zones A/AE/AO for the
  hundred-year floodplain -- rather than from a threshold this file invented.
  1,718 ha of real floodplain, not a generated dish.
* **The budget is money.** US dollars at the USACE National Levee Safety
  Program rate, escalated to 4Q25 and Texas-adjusted, through the embankment's
  real trapezoidal section -- so the cost of a metre of height RISES with
  height and a 3 m levee is twice a 2 m one rather than 1.5 times.
* **The flood is unknown when you choose.** The peak stage is drawn per
  episode and is NOT in the observation. A policy that could see it would be
  solving a different and much easier problem, and the gap between the two is
  the thing this environment exists to measure.

## What it does not do

The levee is hypothetical. Onion Creek has no continuous embankment and this
environment does not claim one is planned; it asks what a budget spent on one
would buy. The damage is depth-weighted flooded area of the FEMA floodplain,
not dollars of property loss -- converting the two needs a depth-damage curve
this repository does not have and should not guess.

The hydrograph is a single-peaked rise and fall at the channel, not a routed
flood from a gauge record. That is the same forcing `FloodOperateEnv` uses and
the same limitation: it exercises the decision, it does not reproduce an event.
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

Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# what things cost
# --------------------------------------------------------------------------- #
#: Compacted embankment fill in place, 2025 US dollars per cubic metre,
#: Austin-adjusted. USACE National Levee Safety Program, *Summary of Costs
#: Associated with Levee-related Activities*, Nov 2023 ($3.80-$40.80/CY at Mar
#: 2023), escalated CWCCIS FC11 1144.19 -> 1214.81 and multiplied by the 0.88
#: Texas state factor. The portal quotes the same number from
#: `portal/src/lib/rates.ts`, which carries the full derivation and the
#: cross-check against USACE Galveston's Hunting Bayou bid tabs.
EARTH_FILL_USD_M3 = 15.2
#: The brochure's own section: 10 ft crest, 1V:3H batters.
CREST_WIDTH_M = 3.048
SIDE_SLOPE = 3.0


def earth_levee_usd(length_m: "Tensor | float", height_m: "Tensor | float"):
    """What a run of embankment costs, from its real trapezoidal section.

    NOT a rate per metre per metre. A trapezoid's area is
    `crest * H + slope * H^2`, so cost per metre of height rises with height.
    Quoting a flat rate understates a tall wall and overstates a short one,
    which is the error the portal shipped before `rates.ts` was written.
    """
    m3_per_m = CREST_WIDTH_M * height_m + SIDE_SLOPE * height_m * height_m
    return length_m * m3_per_m * EARTH_FILL_USD_M3


# --------------------------------------------------------------------------- #
# real ground
# --------------------------------------------------------------------------- #
#: The Onion Creek study box, as `data/austin_onion/SUMMARY.json` defines it.
STUDY_BOX = (-97.82, 30.11, -97.69, 30.21)


def _cached_dem(dx: float) -> "tuple[np.ndarray, dict, int]":
    """The study box's terrain, from the cache if it is there.

    `build_domain` caches by a key over the box, resolution and source, so a
    hit costs a file read. A miss reaches USGS 3DEP, which is the right
    behaviour for a portal and the wrong one inside an environment's
    constructor -- so a miss raises rather than quietly spending a minute on
    the network while a training run waits.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from pspe.simulate.real.anywhere import _dem_key, _dem_load   # noqa: PLC0415

    w, s, e, n = STUDY_BOX
    # "auto" is what the portal passes and therefore what is in the cache; it
    # resolves to 3DEP inside the United States and the Domain records which
    # model it actually got, so the provenance below is read rather than
    # assumed. "3dep" is tried too, for a cache filled by a direct call.
    dom = None
    for source in ("auto", "3dep"):
        dom = _dem_load(_dem_key(w, s, e, n, dx, 700, source), "onion_creek")
        if dom is not None:
            break
    if dom is None:
        raise FileNotFoundError(
            f"no cached terrain for the Onion Creek box at {dx:.0f} m. "
            f"Fetch it once with: python3 -c \"from pspe.simulate.real.anywhere "
            f"import build_domain; build_domain(*{STUDY_BOX}, dx={dx}, "
            f"source='auto')\"")
    if dom.source != "3dep":
        raise ValueError(
            f"the cached terrain for this box is {dom.source!r}, not USGS 3DEP "
            f"bare earth. Copernicus GLO-30 is a SURFACE model: it reads metres "
            f"high over trees and buildings, so a levee height measured against "
            f"it would be measured against the canopy.")
    return dom.z.astype(np.float32), dom.bounds, dom.epsg


def _fema_masks(shape, bounds, epsg) -> "tuple[np.ndarray, np.ndarray]":
    """Channel and hundred-year floodplain, from FEMA rather than a threshold.

    Deriving a channel by thresholding elevations is easy and wrong: it finds
    every low spot, including the ones water never reaches. FEMA's National
    Flood Hazard Layer already says which ground is the floodway and which is
    the one-per-cent-annual-chance floodplain, and it is the layer the city
    regulates against.
    """
    import rasterio.features as rf                       # noqa: PLC0415
    from rasterio.crs import CRS                         # noqa: PLC0415
    from rasterio.transform import from_bounds           # noqa: PLC0415
    from rasterio.warp import transform_bounds, transform_geom  # noqa: PLC0415

    h, w = shape
    src, dst = CRS.from_epsg(4326), CRS.from_epsg(epsg)
    pw, ps, pe, pn = transform_bounds(src, dst, bounds["west"], bounds["south"],
                                      bounds["east"], bounds["north"])
    tr = from_bounds(pw, ps, pe, pn, w, h)

    path = ROOT / "data" / "austin_onion" / "floodplain_fema.geojson"
    feats = json.loads(path.read_text())["features"]

    def burn(pred) -> np.ndarray:
        geoms = [transform_geom(src, dst, f["geometry"]) for f in feats
                 if pred(f["properties"])]
        if not geoms:
            return np.zeros((h, w), dtype=bool)
        return rf.rasterize([(g, 1) for g in geoms], out_shape=(h, w),
                            transform=tr, dtype="uint8").astype(bool)

    river = burn(lambda p: p.get("FLOODWAY") == "Floodway")
    plain = burn(lambda p: str(p.get("FLOOD_ZONE")) in ("AE", "A", "AO"))
    return river, plain & ~river


def onion_creek_windows(
    grid: int = 48, dx: float = 60.0, *, min_protect: float = 0.06,
    min_river: int = 8, stride: int | None = None,
) -> "dict[str, Tensor]":
    """Cut the study box into square windows that contain a decision.

    A window is kept only if it holds some channel and enough floodplain for
    a levee to protect anything -- otherwise the bank would be a bank across
    dry hills, and an environment full of those would report that defences do
    not help.

    Unlike `make_polder_bank`, the masks here are PER WINDOW: real ground does
    not put the creek in the same place twice, and this environment is built
    around that from the start rather than retrofitted. That is also why this
    is a separate environment and not a terrain swap in `FloodOperateEnv`,
    whose structure builder takes the sump from batch element 0 and whose
    river boundary reads column 0 as the channel -- both true of a generated
    polder and neither true here.
    """
    z_full, bounds, epsg = _cached_dem(dx)
    river_full, protect_full = _fema_masks(z_full.shape, bounds, epsg)
    h, w = z_full.shape
    step = stride or max(grid // 3, 1)

    zs, rs, ps = [], [], []
    for r in range(0, h - grid + 1, step):
        for c in range(0, w - grid + 1, step):
            riv = river_full[r:r + grid, c:c + grid]
            pro = protect_full[r:r + grid, c:c + grid]
            if riv.sum() < min_river or pro.mean() < min_protect:
                continue
            zs.append(z_full[r:r + grid, c:c + grid])
            rs.append(riv)
            ps.append(pro)
    if not zs:
        raise ValueError(
            f"no {grid}x{grid} window at {dx:.0f} m holds both a floodway and "
            f"{min_protect:.0%} floodplain; loosen min_protect or shrink grid")

    z = torch.from_numpy(np.stack(zs)).float()
    # Each window is referred to its own lowest point, so a policy reads relief
    # rather than absolute altitude and windows are comparable to each other.
    z = z - z.amin(dim=(-2, -1), keepdim=True)
    return {
        "z": z,
        "river": torch.from_numpy(np.stack(rs)),
        "protect": torch.from_numpy(np.stack(ps)),
        "dx": torch.tensor(dx),
    }


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
@dataclass
class FloodDefendTask:
    """The constraint, in money.

    `cost_limit` is US dollars of construction for the whole programme. Three
    million is about 1.3 km of 2 m embankment at the USACE rate, which is the
    order of a real neighbourhood-scale project and small enough against this
    box's 1,718 ha that the allocation has to be chosen rather than sprayed.
    """

    #: Left at 0.0 the environment SETS it, from the ground: the cost of a
    #: uniform `design_height_m` embankment along the mean bank line. That is
    #: the only way to be sure it binds. A fixed figure cannot: these windows
    #: hold about a kilometre of bank, so the most that can physically be
    #: spent at `max_height_m` is a few hundred thousand dollars, and the
    #: $3,000,000 this field first carried could not be spent at all. A budget
    #: no policy can exhaust is not a constraint, and the project has already
    #: published one of those (`run_evac_budgets`: "the budget does not bind,
    #: across a 2.6x range").
    cost_limit: float = 0.0
    #: What the budget buys if it is spread evenly. Below `max_height_m`, so
    #: concentrating it somewhere is always possible and always costs
    #: somewhere else.
    design_height_m: float = 1.6
    #: Depth at which protected ground counts as flooded, matching the rest of
    #: this project.
    wet_m: float = 0.10
    #: Returns in percent of the protected area, so they are O(1).
    reward_scale: float = 100.0
    #: The tallest embankment a segment may be given.
    max_height_m: float = 3.0


class FloodDefendEnv:
    """Batched core. `reset(batch, generator)` / `step(action)`, as the rest.

    One step per segment: the episode is K decisions about where the money
    goes, the budget accrues as they are taken, and the flood is solved once
    at the end. Reward is zero until then -- it is a design problem, and
    pretending a part-built levee has a value would be inventing a number.
    """

    dynamics = "solver"

    def __init__(self, bank: "dict[str, Tensor]", task: "FloodDefendTask | None" = None,
                 *, segments: int = 8, flood_steps: int = 24,
                 step_minutes: float = 30.0,
                 #: Peak DEPTH in the channel, metres. Spanning the height
                 #: a budget can build means some floods are smaller than the
                 #: defences were sized for and some are larger, which is the
                 #: uncertainty the decision is taken under.
                 peak_stage_m: "tuple[float, float]" = (2.0, 5.0),
                 flood_quantile: float = 0.35,
                 setback: int = 2,
                 device: "torch.device | str" = "cpu") -> None:
        self.task = task or FloodDefendTask()
        self.device = torch.device(device)
        self.z_bank = bank["z"].to(self.device)
        self.river_bank = bank["river"].to(self.device)
        self.protect_bank = bank["protect"].to(self.device)
        self.dx = float(bank["dx"])
        self.n, self.h_cells, self.w = self.z_bank.shape
        self.K = segments
        self.flood_steps = flood_steps
        self.step_s = step_minutes * 60.0
        self.peak_stage_m = peak_stage_m
        self.cfg = FloodConfig(dx=self.dx, open_edges=("south", "east"))

        # Where a levee can go: protected ground within `setback` cells of the
        # channel. A real embankment sits back from the bank rather than in
        # the water, and FEMA's floodway is narrow -- 197 cells in the whole
        # study box -- so a one-cell dilation leaves almost nothing to build
        # on.
        riv = self.river_bank.float().unsqueeze(1)
        kernel = 2 * setback + 1
        near = F.max_pool2d(riv, kernel, stride=1, padding=setback)[:, 0] > 0
        self.bankline = near & self.protect_bank                    # (N,H,W)

        # K segments ALONG the creek, not column strips.
        #
        # Strips were the first attempt and they do not work on real ground:
        # Onion Creek runs diagonally across these windows, so eight column
        # strips gave segment lengths of [0, 0, 0, 0, 960, 60, 0, 0] metres --
        # six of the eight decisions were about nothing. Projecting each
        # window's bank cells onto their own principal axis and splitting by
        # rank gives K segments of equal length that follow the channel
        # wherever it happens to go.
        seg = torch.zeros(self.n, self.K, self.h_cells, self.w,
                          dtype=torch.bool, device=self.device)
        keep = []
        for i in range(self.n):
            rc = self.bankline[i].nonzero(as_tuple=False)
            # A window with fewer bank cells than segments cannot pose the
            # question: some of its decisions would be about nothing, and an
            # episode drawn there would score whatever the terrain does. They
            # are dropped rather than left to produce a silent no-op.
            if rc.shape[0] < self.K:
                continue
            centred = rc.float() - rc.float().mean(dim=0, keepdim=True)
            # Principal direction of the bank: the way the creek runs here.
            axis = torch.linalg.svd(centred, full_matrices=False)[2][0]
            idx = rc[(centred @ axis).argsort()]
            # `tensor_split`, not `chunk`: chunk sizes its pieces by ceiling
            # division and can return FEWER than K of them, which left the
            # last segment of some windows empty -- a decision about nothing,
            # the bug this whole block exists to prevent.
            for k, part in enumerate(torch.tensor_split(idx, self.K)):
                if part.numel():
                    seg[i, k, part[:, 0], part[:, 1]] = True
            keep.append(i)
        if not keep:
            raise ValueError(
                f"no window holds {self.K} segments of bank; lower `segments`, "
                f"raise `setback`, or cut larger windows")
        sel = torch.tensor(keep, device=self.device)
        self.z_bank = self.z_bank[sel]
        self.river_bank = self.river_bank[sel]
        self.protect_bank = self.protect_bank[sel]
        self.bankline = self.bankline[sel]
        self.seg_mask = seg[sel]
        self.n = len(keep)
        self.dropped = int(sel.shape[0] != seg.shape[0]) and (seg.shape[0] - len(keep))

        # How big a flood has to be, HERE, to be worth deciding about.
        #
        # The first version drew the peak stage from a fixed 1.5-5.0 m and most
        # windows never flooded at all: three consecutive single-episode draws
        # scored exactly 0.0 whatever was built, because this is real ground
        # and a floodplain can sit ten metres above its channel. An
        # environment whose episodes are mostly dead cannot rank anything --
        # it is the defect `FloodEvacuate-v0` has and the reason nothing
        # should be trained on that one.
        #
        # So the flood is referred to each window's own relief: `stage_ref` is
        # the height above the channel bed that would reach the
        # `flood_quantile` of protected ground if water simply filled to a
        # level. `task.py` has warned about exactly this since the polder bank
        # was built -- calibrating a testbed means checking the RETURN
        # separates, not only that the cost does.
        bed = torch.where(self.river_bank, self.z_bank,
                          torch.full_like(self.z_bank, float("inf"))).amin(dim=(-2, -1))
        above = self.z_bank - bed.view(-1, 1, 1)
        ref = []
        for i in range(self.n):
            vals = above[i][self.protect_bank[i]]
            ref.append(vals.quantile(flood_quantile) if vals.numel()
                       else torch.tensor(1.0))
        self.stage_ref = torch.stack(ref).clamp(min=0.25)             # (N,)

        if self.task.cost_limit <= 0.0:
            # Mean bank length over the windows that survived, at the design
            # height. Stored back on the task so everything downstream -- the
            # gym wrapper, `safe_rl`, the portal -- reads one number.
            lengths = self.seg_mask.sum(dim=(1, 2, 3)).float() * self.dx
            self.task.cost_limit = float(
                earth_levee_usd(lengths.mean(), self.task.design_height_m))

        self.obs_shape = (5, self.h_cells, self.w)
        self.action_dim = 1          # one segment per step, height only
        self._reset_state()

    # -- episode ------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self.t = 0
        self.idx = None
        self.heights = None
        self.spent = None
        self.last_depth = None

    def reset(self, batch: int = 1, generator: "torch.Generator | None" = None) -> Tensor:
        g = generator
        self.idx = torch.randint(0, self.n, (batch,), generator=g)
        self.heights = torch.zeros(batch, self.K, device=self.device)
        self.spent = torch.zeros(batch, device=self.device)
        lo, hi = self.peak_stage_m
        # Drawn now, revealed never: the policy designs without knowing it.
        rel = lo + (hi - lo) * torch.rand(batch, generator=g)
        self._peak = rel.to(self.device) * self.stage_ref[self.idx]
        self.t = 0
        return self._observe()

    def segment_length_m(self, k: int) -> Tensor:
        """Metres of bank line in segment `k`, per episode in the batch."""
        cells = self.seg_mask[self.idx, k].sum(dim=(1, 2))
        return cells.float() * self.dx

    def _observe(self) -> Tensor:
        b = self.idx.shape[0]
        z = self.z_bank[self.idx]
        zn = z / z.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        prot = self.protect_bank[self.idx].float()
        riv = self.river_bank[self.idx].float()
        k = min(self.t, self.K - 1)
        here = self.seg_mask[self.idx, k].float()
        left = (1.0 - (self.spent / self.task.cost_limit)).clamp(min=0.0)
        budget = left.view(b, 1, 1).expand(b, self.h_cells, self.w)
        return torch.stack([zn, prot, riv, here, budget], dim=1)

    def raised_bed(self) -> Tensor:
        """The terrain as built: the chosen height added along each segment."""
        z = self.z_bank[self.idx].clone()
        for k in range(self.K):
            m = self.seg_mask[self.idx, k]
            z = z + m.float() * self.heights[:, k].view(-1, 1, 1)
        return z

    @torch.no_grad()
    def _solve(self) -> Tensor:
        """Run the flood over the built terrain. Returns wet fraction of protect."""
        z = self.raised_bed()
        solver = FloodSolver(z, self.cfg)
        h = torch.zeros_like(z)
        qx = torch.zeros(z.shape[0], z.shape[1], z.shape[2] - 1, device=z.device)
        qy = torch.zeros(z.shape[0], z.shape[1] - 1, z.shape[2], device=z.device)
        riv = self.river_bank[self.idx]

        for i in range(self.flood_steps):
            frac = i / max(self.flood_steps - 1, 1)
            d = (frac - 0.45) / 0.30
            stage = self._peak.to(z.device) * float(np.exp(-0.5 * d * d))
            # Depth IN the channel, held per cell, so the water surface
            # follows the bed downstream.
            #
            # The first version held one global level, `min(z over channel) +
            # stage`, which is what the generated polder does because its
            # river band is flat. A real creek falls through a window, and the
            # FEMA floodplain reaches ground BELOW the lowest channel cell --
            # measured here, the 20th percentile of protected ground sits 5.8 m
            # under it. A single level therefore either left the channel dry
            # or drowned the whole window before any levee could matter, which
            # is why the first calibration sweep found no stage at which
            # building helped.
            h = torch.where(riv, torch.maximum(h, stage.view(-1, 1, 1)), h)
            t_now, t_end = 0.0, self.step_s
            while t_now < t_end:
                dt = min(solver.adaptive_dt(h), t_end - t_now)
                if dt <= 0:
                    break
                h, qx, qy = solver.step(h, qx, qy, dt)
                t_now += dt

        # Kept so a viewer can render the flood that was actually solved,
        # rather than an illustration of one.
        self.last_depth = h.detach()
        prot = self.protect_bank[self.idx]
        wet = ((h > self.task.wet_m) & prot).sum(dim=(1, 2)).float()
        total = prot.sum(dim=(1, 2)).float().clamp(min=1.0)
        return wet / total

    def step(self, action: Tensor) -> "tuple[Tensor, Tensor, Tensor, bool]":
        """One segment's height. Cost in dollars; reward only at the end."""
        k = self.t
        u = (action.clamp(-1.0, 1.0).view(-1) + 1.0) / 2.0        # -> [0,1]
        height = u * self.task.max_height_m
        self.heights[:, k] = height
        cost = earth_levee_usd(self.segment_length_m(k), height)
        self.spent = self.spent + cost

        self.t += 1
        done = self.t >= self.K
        if done:
            reward = -self.task.reward_scale * self._solve()
        else:
            reward = torch.zeros(u.shape[0], device=self.device)
        return self._observe(), reward, cost, done

    # -- baselines ---------------------------------------------------------- #
    @torch.no_grad()
    def uniform_action(self) -> Tensor:
        """Spread the budget evenly over every segment that has a bank.

        Solved rather than guessed: total cost is
        `sum_k L_k (c H + s H^2) r` at a common height H, which is a quadratic
        in H with one positive root at the budget.
        """
        L = torch.stack([self.segment_length_m(k) for k in range(self.K)]).sum(0)
        L = L.clamp(min=1.0)
        a = SIDE_SLOPE * L * EARTH_FILL_USD_M3
        b = CREST_WIDTH_M * L * EARTH_FILL_USD_M3
        # Aim one part per million inside the budget. The episode's cost is a
        # sum of eight floats and the limit is one, so a height solved to spend
        # the budget EXACTLY came out a few cents over in half the episodes and
        # read as a budget violation. Same class of defect as the firebreak
        # baselines' sqrt/square round trip, and the same remedy: six orders of
        # magnitude below the precision anything here is reported at.
        budget = self.task.cost_limit * (1.0 - 1e-6)
        H = (-b + (b * b + 4 * a * budget).sqrt()) / (2 * a)
        H = H.clamp(max=self.task.max_height_m)
        return (2.0 * (H / self.task.max_height_m) - 1.0).view(-1, 1)

    @torch.no_grad()
    def lowest_first_action(self, state: Tensor) -> Tensor:
        """Build where the bank is lowest, which is where water crosses first.

        The obvious engineering heuristic and the one a learned policy has to
        beat. It spends the whole budget on the segments with the lowest mean
        bank elevation, tallest first, until the money runs out.
        """
        b = state.shape[0]
        k = min(self.t, self.K - 1)
        z = self.z_bank[self.idx]
        order = []
        for j in range(self.K):
            m = self.seg_mask[self.idx, j]
            lvl = torch.where(m, z, torch.full_like(z, float("inf"))).amin(dim=(-2, -1))
            order.append(lvl)
        lvl = torch.stack(order, dim=1)                              # (B,K)
        rank = lvl.argsort(dim=1).argsort(dim=1)                     # 0 = lowest
        # Split what is left over the segments not yet built, weighted towards
        # the low ones: the lowest gets the largest share.
        remaining = (self.K - k)
        weight = (self.K - rank[:, k].float()) / max(self.K, 1)
        share = (self.task.cost_limit - self.spent).clamp(min=0.0) * weight \
            / max(remaining, 1) * 2.0
        L = self.segment_length_m(k).clamp(min=1.0)
        a = SIDE_SLOPE * L * EARTH_FILL_USD_M3
        c = CREST_WIDTH_M * L * EARTH_FILL_USD_M3
        H = (-c + (c * c + 4 * a * share).sqrt()) / (2 * a)
        H = H.clamp(min=0.0, max=self.task.max_height_m)
        return (2.0 * (H / self.task.max_height_m) - 1.0).view(b, 1)


# --------------------------------------------------------------------------- #
# the gymnasium view
# --------------------------------------------------------------------------- #
class FloodDefendControlEnv(gym.Env):
    """Single-instance Gymnasium view of the capital-allocation decision.

    Eight steps, one per segment of bank, then the flood is solved once and
    the whole reward lands. Cost is dollars on `info["cost"]` per step, and
    `cost_limit` is the programme budget for the episode -- the convention
    `safe_rl.py`, Safety-Gymnasium and OmniSafe all read.

    The two baselines are `uniform_action`, which spreads the budget to a
    common height, and `lowest_first_action`, which puts it where the bank is
    lowest. The second is the obvious engineering answer and the first beats
    it, which is the opening a learned policy has.
    """

    metadata = {"render_modes": []}

    def __init__(self, bank: "dict[str, Tensor] | None" = None, *,
                 budget: float = 0.0, segments: int = 8, flood_steps: int = 16,
                 grid: int = 48, dx: float = 60.0,
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = FloodDefendEnv(
            bank if bank is not None else onion_creek_windows(grid=grid, dx=dx),
            FloodDefendTask(cost_limit=budget), segments=segments,
            flood_steps=flood_steps, device=self.device)
        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.episode_cost = 0.0

    def reset(self, *, seed: "int | None" = None,
              options: "dict | None" = None) -> "tuple[np.ndarray, dict]":
        # Seeds `self.np_random`, which `check_env` verifies. It does NOT seed
        # `action_space.sample()` -- that generator is reached through
        # `action_space.seed()`. See `FloodControlEnv.reset`.
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        state = self._core.reset(batch=1, generator=self._generator)
        self.episode_cost = 0.0
        return state[0].detach().cpu().numpy().astype(np.float32), {"cost": 0.0}

    def step(self, action: "np.ndarray"
             ) -> "tuple[np.ndarray, float, bool, bool, dict]":
        act = torch.as_tensor(action, dtype=torch.float32,
                              device=self.device).view(1, -1).clamp(-1.0, 1.0)
        with torch.no_grad():
            state, reward, cost, done = self._core.step(act)
        self.episode_cost += float(cost[0])
        info = {"cost": float(cost[0]), "episode_cost": self.episode_cost,
                "cost_limit": self._core.task.cost_limit}
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        # The programme ends when every segment has been decided. That is a
        # time limit, not an absorbing state.
        return obs, float(reward[0]), False, bool(done), info

    @property
    def task(self) -> FloodDefendTask:
        return self._core.task

    @property
    def core(self) -> "FloodDefendEnv":
        return self._core

    def uniform_action(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """Spread the budget to one height along the whole bank."""
        return self._core.uniform_action()[0].cpu().numpy()

    def lowest_first_action(self, obs: "np.ndarray") -> "np.ndarray":
        """Spend where the bank is lowest. The obvious engineering answer."""
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.lowest_first_action(t)[0].cpu().numpy()


def register_envs() -> None:
    """Make `gymnasium.make("FloodDefend-v0")` work. Idempotent."""
    from gymnasium.envs.registration import register, registry

    if "FloodDefend-v0" in registry:
        return
    register(id="FloodDefend-v0", entry_point="pspe.envs.defend:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> FloodDefendControlEnv:
    """Entry point for `gymnasium.make`, on the real Onion Creek windows."""
    return FloodDefendControlEnv(**kw)
