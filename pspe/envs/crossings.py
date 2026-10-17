"""When to put the barricade down, and when to take it up again.

In Texas the flood that kills you is usually ankle deep and across a road.
Austin instruments twenty-two barricade arms at eleven low-water crossings for
exactly this, and the duty decision is the one here: the water is rising
somewhere upstream, the crossing is still dry, and closing it strands everyone
who uses that road.

It is the same shape of problem as `FloodEvacuate-v0` and better founded in
one specific way. There, the harm is a model's estimate of who was exposed and
nobody has ever evacuated on our instruction. Here, **whether the crossing is
under water is a depth the solver computes** -- a physical fact about a place,
at a cell whose coordinates come from the city's own asset register. What the
environment still cannot know is how many cars would have driven in, so it
does not pretend to: the harm is intervals of road open while under water, and
not a casualty count derived from traffic nobody counted.

## What is real

* **Eleven crossings**, by name and coordinate, from the City of Austin's
  `INLANDWATERS_flood_low_water_crossings` layer.
* **USGS 3DEP bare earth at 30 m** around each one, fetched per site.
* **The depth at the crossing**, from the same local-inertial solver as every
  other flood environment here, precomputed per site and per flood size into a
  library -- the way `baselines/evac` precomputes routing -- because solving
  eleven domains inside a step loop makes an environment nobody runs twice.

## What is not

The hydrograph is a single-peaked rise held in the channel, not a routed event
from a gauge. And there is no traffic model, by choice; see above.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..simulate.flood import FloodConfig, FloodSolver

Tensor = torch.Tensor
ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "data" / "evac" / "crossing_depths.npz"

#: Half-width of the terrain box fetched around each crossing, in degrees.
HALF_DEG = 0.011
#: How much deeper than the "turn around" mark the water must be before the
#: barricade goes down, when the allowance will not cover the whole flood.
#: Set by sweep in `scripts/score_gym.py`'s own protocol.
RATION_MULTIPLE = 2.0

#: Flood sizes in the library: peak depth held in the channel, metres.
STAGES = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
#: Half-hour intervals in an episode. Long enough that the flood is an
#: EVENT inside it rather than the whole of it: at 24 slots with a wide pulse
#: the crossing was under water almost the entire episode, every rule closed
#: the road for all 24 intervals, and there was no timing decision left to
#: make -- only a budget to bust.
SLOTS = 36


def crossing_sites() -> "list[tuple[str, float, float]]":
    """The distinct crossings, averaging the arms that share a road."""
    path = ROOT / "data" / "austin_crossings" / "low_water_crossings.geojson"
    if not path.is_file():
        raise FileNotFoundError(
            f"no crossings at {path}. Fetch with: python3 "
            f"scripts/fetch_austin_data.py --out data/austin_crossings "
            f"--bbox=-97.95,30.05,-97.55,30.45 --skip buildings,addresses,"
            f"floodplain_austin,base_flood_elevations")
    sites: "dict[str, list[tuple[float, float]]]" = {}
    for f in json.loads(path.read_text())["features"]:
        lon, lat = f["geometry"]["coordinates"]
        name = re.sub(r"\s*(Master|Slave).*$", "",
                      str(f["properties"].get("CROSSING_DESCRIPTION"))).strip()
        sites.setdefault(name, []).append((lat, lon))
    return [(n, sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v))
            for n, v in sorted(sites.items())]


def build_library(force: bool = False) -> "dict[str, np.ndarray]":
    """Solve each crossing at each flood size once, and keep the depth curve.

    The result is `(sites, stages, slots)` of depth in metres at the crossing
    cell. Eleven domains at 30 m for twenty-four intervals is a few minutes of
    solver; doing it inside an environment step would make this unusable, and
    doing it approximately would throw away the one thing this environment has
    that the evacuation one does not.
    """
    if LIBRARY.is_file() and not force:
        with np.load(LIBRARY, allow_pickle=False) as d:
            return {"depth": d["depth"], "names": d["names"]}

    from ..simulate.real.anywhere import build_domain     # noqa: PLC0415

    sites = crossing_sites()
    depth = np.zeros((len(sites), len(STAGES), SLOTS), dtype=np.float32)
    for si, (name, lat, lon) in enumerate(sites):
        dom = build_domain(lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG,
                           lat + HALF_DEG, dx=30.0, name=name, source="auto")
        z = torch.from_numpy(dom.z.astype(np.float32))[None]
        z = z - z.amin()
        h_cells, w = z.shape[1], z.shape[2]
        # The crossing is where the road DIPS, which is the low point near the
        # published coordinate -- not the coordinate's own cell. Reading the
        # centre cell exactly left 8 of 11 crossings bone dry at every flood
        # size, because a point fixed to five decimal places lands wherever it
        # lands inside a 30 m cell and the asset register records the arm, not
        # the invert. The minimum within 120 m is the dip.
        r0, c0 = h_cells // 2, w // 2
        rad = max(int(round(120.0 / float(dom.dx))), 1)
        patch = z[0, max(r0 - rad, 0):r0 + rad + 1, max(c0 - rad, 0):c0 + rad + 1]
        off = int(patch.argmin())
        rc = (max(r0 - rad, 0) + off // patch.shape[1],
              max(c0 - rad, 0) + off % patch.shape[1])
        # The channel: the lowest cells, which at a low-water crossing is the
        # creek the road dips through.
        thresh = torch.quantile(z.flatten(), 0.04)
        river = z[0] <= thresh
        cfg = FloodConfig(dx=float(dom.dx), open_edges=("south", "east"))
        for gi, peak in enumerate(STAGES):
            solver = FloodSolver(z.clone(), cfg)
            h = torch.zeros_like(z)
            qx = torch.zeros(1, h_cells, w - 1)
            qy = torch.zeros(1, h_cells - 1, w)
            for t in range(SLOTS):
                frac = t / max(SLOTS - 1, 1)
                # A narrow pulse: the water arrives, covers the road for a
                # few hours, and goes. That window is the decision.
                d = (frac - 0.45) / 0.10
                stage = peak * float(np.exp(-0.5 * d * d))
                h[0] = torch.where(river, torch.maximum(h[0], torch.tensor(stage)), h[0])
                now = 0.0
                while now < 1800.0:
                    dt = min(solver.adaptive_dt(h), 1800.0 - now)
                    if dt <= 0:
                        break
                    h, qx, qy = solver.step(h, qx, qy, dt)
                    now += dt
                depth[si, gi, t] = float(h[0, rc[0], rc[1]])
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    names = np.array([s[0] for s in sites])
    np.savez_compressed(LIBRARY, depth=depth, names=names)
    return {"depth": depth, "names": names}


@dataclass
class FloodCrossingTask:
    """The constraint, in half-hours of closed road."""

    #: Intervals the barricade may be down. Left at 0.0 the environment sets
    #: it from the library: the median number of intervals a crossing is
    #: actually under water, so a perfectly timed closure just fits and
    #: closing all day does not.
    cost_limit: float = 0.0
    #: The depth at which driving the crossing is the thing that kills people.
    #: Six inches, which is the figure every "Turn Around Don't Drown" notice
    #: in Texas carries.
    wet_m: float = 0.15
    reward_scale: float = 100.0


class FloodCrossingEnv(gym.Env):
    """One crossing, one flood, twenty-four decisions.

    Reward is minus the share of intervals the road was OPEN while under
    water. Cost is the intervals it was closed. The flood is drawn per episode
    and the policy sees the depth so far but not what is coming, which is the
    whole difficulty: the barricade has to go down before the water arrives.
    """

    metadata = {"render_modes": []}

    def __init__(self, task: "FloodCrossingTask | None" = None, *,
                 budget: float = 0.0, device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.task = task or FloodCrossingTask()
        lib = build_library()
        depth = lib["depth"]
        # Only the crossings that actually go under. Six of the eleven stay
        # dry at every flood size in the library -- their boxes hold no creek
        # the solver can fill -- and an episode drawn at one of those has no
        # decision in it: never closing is optimal and nothing distinguishes
        # any policy. Keeping them would make six episodes in eleven dead,
        # which is the defect `FloodEvacuate-v0` already has.
        live = [i for i in range(depth.shape[0])
                if depth[i].max() > (task.wet_m if task else 0.15)]
        if not live:
            raise ValueError("no crossing in the library ever goes under water")
        self.depth = torch.from_numpy(depth[live])             # (S, G, T)
        self.names = [str(lib["names"][i]) for i in live]
        self.dry_names = [str(lib["names"][i]) for i in range(depth.shape[0])
                          if i not in live]
        if self.task.cost_limit <= 0.0:
            wet = (self.depth > self.task.wet_m).sum(dim=-1).float()
            self.task.cost_limit = float(wet.median())
        if budget > 0:
            self.task.cost_limit = budget
        self.n_sites, self.n_stages, self.slots = self.depth.shape
        self.device = torch.device(device)
        # Depth so far, depth now, how far through, how much budget is left.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.episode_cost = 0.0

    # -- episode ------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed if seed is not None else 0)
        self.site = int(rng.integers(self.n_sites))
        self.stage = int(rng.integers(self.n_stages))
        self.curve = self.depth[self.site, self.stage]
        self.t = 0
        self.episode_cost = 0.0
        self._exposed = 0.0
        return self._obs(), {"cost": 0.0, "site": self.names[self.site]}

    def _obs(self) -> "np.ndarray":
        now = float(self.curve[min(self.t, self.slots - 1)])
        seen = float(self.curve[:max(self.t, 1)].max())
        left = max(0.0, 1.0 - self.episode_cost / max(self.task.cost_limit, 1e-9))
        return np.array([now, seen, self.t / self.slots, left], dtype=np.float32)

    def step(self, action):
        closed = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) > 0.0
        now = float(self.curve[min(self.t, self.slots - 1)])
        wet = now > self.task.wet_m
        if wet and not closed:
            self._exposed += 1.0
        cost = 1.0 if closed else 0.0
        self.episode_cost += cost
        self.t += 1
        done = self.t >= self.slots
        reward = (-self.task.reward_scale * self._exposed / self.slots) if done else 0.0
        info = {"cost": cost, "episode_cost": self.episode_cost,
                "cost_limit": self.task.cost_limit, "wet": wet}
        return self._obs(), float(reward), False, bool(done), info

    # -- baselines ---------------------------------------------------------- #
    def never_close_action(self, obs=None) -> "np.ndarray":
        return np.array([-1.0], dtype=np.float32)

    def always_close_action(self, obs=None) -> "np.ndarray":
        return np.array([1.0], dtype=np.float32)

    def react_action(self, obs: "np.ndarray") -> "np.ndarray":
        """Close once the water is over the line. What a crew on site does.

        It is always one interval late, by construction -- the crossing has to
        be under water before anybody sees that it is -- which is exactly the
        gap a forecast is supposed to close, and the reason this is the rule
        worth beating rather than a strawman.
        """
        return np.array([1.0 if obs[0] > self.task.wet_m else -1.0], dtype=np.float32)

    def budgeted_react_action(self, obs: "np.ndarray") -> "np.ndarray":
        """Close while the road is under water, and stop when the allowance
        runs out. The incumbent.

        The two words that matter are "and stop". `react_action` closes
        whenever the crossing is wet and clears the budget in 4 episodes of
        12, because the water stays up longer than the allowance covers.
        Checking the allowance before putting the barricade down is one line,
        and it is the difference between a rule that is feasible and one that
        is not -- which is the whole point of the cost column.

        It is still a greedy rule: it covers the FIRST wet intervals rather
        than the worst ones, and under a harm model that weights every wet
        interval equally there is nothing better to prefer. A harm model with
        depth in it would leave room for a policy to do better, and that is a
        change to this environment rather than to its baseline.
        """
        left = float(obs[3])
        wet = float(obs[0]) > self.task.wet_m
        return np.array([1.0 if (wet and left > 1e-9) else -1.0], dtype=np.float32)

    def ration_action(self, obs: "np.ndarray",
                      multiple: "float | None" = None) -> "np.ndarray":
        """Close only while the water is deepest. The incumbent.

        The budget is a little under the time the crossing spends under water,
        so the road cannot be closed for the whole flood and the question is
        WHICH wet intervals to cover. Closing greedily from the first wet
        interval -- which is what a crew on site does, and what
        `react_action` does -- runs the allowance out before the peak and
        clears the budget in only 4 episodes of 12.

        This closes above a higher mark instead, rationing the closure to the
        deepest part. The multiple is set by sweep, not by taste: see
        `RATION_MULTIPLE`.
        """
        m = RATION_MULTIPLE if multiple is None else multiple
        return np.array([1.0 if float(obs[0]) > self.task.wet_m * m else -1.0],
                        dtype=np.float32)

    def anticipate_action(self, obs: "np.ndarray", lead: float = 0.5) -> "np.ndarray":
        """Close when the water is within `lead` of the line and still rising."""
        now, seen = float(obs[0]), float(obs[1])
        rising = now >= seen - 1e-6
        near = now > self.task.wet_m * lead
        return np.array([1.0 if (near and rising) or now > self.task.wet_m else -1.0],
                        dtype=np.float32)


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "FloodCrossing-v0" in registry:
        return
    register(id="FloodCrossing-v0", entry_point="pspe.envs.crossings:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> FloodCrossingEnv:
    return FloodCrossingEnv(**kw)
