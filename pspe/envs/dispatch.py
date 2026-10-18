"""Which fire to fight, when several are burning at once.

`Firebreak-v0` asks where to cut a break on one fire. This asks the question a
duty officer actually faces in a bad week: four fires are running, there are
only so many crews, and sending them to one is not sending them to another.

The decision is between fires rather than within one, which is a different
problem and a harder one -- the within-fire question has a tidy answer
(`greedy_action`, put the crews where it burns) and this one does not, because
the fire that is burning hardest is not always the fire where a crew-day buys
the most.

## What it is built on

The same spread model and the same held-out NDWS records as `Firebreak-v0`:
about 1,689 real fire days, from which each episode draws `fires` at random to
stand as simultaneous incidents. Nothing new is assumed -- the treatment model
is the one already fitted and already caveated, and the counterfactual is just
as unobservable here as there. What is new is only the allocation.

Crews sent to a fire are spread uniformly over it, because this environment is
about the split between fires and folding the within-fire placement back in
would confound the two questions. `Firebreak-v0` is where placement is asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .fire_env import FireEnv, FireTask

Tensor = torch.Tensor


@dataclass
class FireDispatchTask:
    """The constraint, in crew-days spread over the whole week."""

    #: Fraction of ONE patch that can be treated per step, summed over the
    #: episode and over every fire. Four fires sharing what one fire had in
    #: `Firebreak-v0` -- which is the point: the crews do not multiply.
    cost_limit: float = 0.03
    reward_scale: float = 100.0


class FireDispatchEnv:
    """Batched-protocol core. One episode holds `fires` simultaneous incidents."""

    dynamics = "surrogate"

    def __init__(self, surrogate, data: "dict[str, np.ndarray]",
                 task: "FireDispatchTask | None" = None, *,
                 fires: int = 4, horizon: int = 3, grid: int = 64,
                 patches: int = 4, split: str = "train",
                 device: "torch.device | str" = "cpu") -> None:
        self.task = task or FireDispatchTask()
        self.device = torch.device(device)
        self.split = split
        self.fires = fires
        self.horizon = horizon
        # One `FireEnv` whose BATCH is this episode's set of fires. The fires
        # advance independently, which is what simultaneous incidents do.
        self._fire = FireEnv(surrogate, data,
                             FireTask(cost_limit=task.cost_limit if task else 0.03,
                                      reward_scale=1.0),
                             grid=grid, patches=patches, horizon=horizon,
                             split=split, device=self.device)
        self.grid, self.patches = grid, patches
        self.obs_shape = (self.fires + 1, grid, grid)
        self.action_dim = fires
        self.t = 0

    def reset(self, batch: int = 1, generator: "torch.Generator | None" = None) -> Tensor:
        if batch != 1:
            raise ValueError("FireDispatchEnv runs one episode at a time; the "
                             "batch dimension is already spent on the fires")
        self._state = self._fire.reset(batch=self.fires, generator=generator)
        self.t = 0
        self.spent = 0.0
        return self._observe()

    def _observe(self) -> Tensor:
        """Every fire's front, stacked, plus how much of the week is left.

        A dispatcher sees all the incidents at once, so the observation is all
        of them -- one channel per fire -- rather than one fire at a time.
        """
        fronts = self._state[:, 0]                              # (F, H, W)
        left = torch.full((1, self.grid, self.grid),
                          1.0 - self.t / max(self.horizon, 1), device=self.device)
        return torch.cat([fronts, left], dim=0)[None]           # (1, F+1, H, W)

    def step(self, action: Tensor) -> "tuple[Tensor, Tensor, Tensor, bool]":
        """action: (1, F) in [-1, 1] -> the day's crews, split over the fires."""
        u = (action.clamp(-1.0, 1.0).view(-1) + 1.0) / 2.0      # -> [0,1]
        total = u.sum().clamp(min=1e-6)
        share = u / total                                        # where crews go
        # How much of the day's allowance is used at all. Sending nobody is a
        # real choice and costs nothing, so the magnitude is kept rather than
        # normalised away.
        effort = u.max().clamp(max=1.0)
        day = self.task.cost_limit / max(self.horizon, 1) * effort
        per_fire = share * day * self.fires                      # spread over each

        k = self._fire.action_dim
        intensity = per_fire.view(self.fires, 1).expand(self.fires, k)
        act = FireEnv.action_for(intensity.clamp(0.0, 1.0) * (1.0 - 1e-6))
        with torch.no_grad():
            self._state, reward, cost, done = self._fire.step(act)
        self.t += 1
        burned = -reward.mean() * self.task.reward_scale / max(self._fire.task.reward_scale, 1.0)
        spent = cost.sum().view(1) / self.fires
        self.spent += float(spent)
        return self._observe(), (-burned).view(1), spent, bool(done)

    # -- baselines ---------------------------------------------------------- #
    @torch.no_grad()
    def even_action(self, obs: "Tensor | None" = None) -> Tensor:
        """Split the crews equally. The dispatcher with no information."""
        return torch.ones(1, self.fires, device=self.device)

    @torch.no_grad()
    def biggest_fire_action(self, obs: Tensor) -> Tensor:
        """Everything to whichever fire is burning most. The obvious answer.

        It is the instinct -- the big one is the emergency -- and it is the
        rule a learned dispatcher has to beat, because the fire that is
        largest is not necessarily the one where a crew-day removes the most
        burned ground.
        """
        fronts = self._state[:, 0]
        size = fronts.flatten(1).sum(dim=1)
        a = torch.full((1, self.fires), -1.0, device=self.device)
        a[0, int(size.argmax())] = 1.0
        return a

    @torch.no_grad()
    def most_people_action(self, obs: Tensor) -> Tensor:
        """Everything to the fire with the most people in front of it."""
        pop = self._state[:, self._fire.pop].flatten(1).sum(dim=1)
        fronts = self._state[:, 0].flatten(1).sum(dim=1)
        score = pop * (fronts > 0).float()
        a = torch.full((1, self.fires), -1.0, device=self.device)
        a[0, int(score.argmax())] = 1.0
        return a


class FireDispatchControlEnv(gym.Env):
    """Single-instance Gymnasium view of the dispatch decision."""

    metadata = {"render_modes": []}

    def __init__(self, surrogate=None, data=None, *, budget: float = 0.03,
                 fires: int = 4, horizon: int = 3, split: str = "train",
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        if surrogate is None or data is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from service.fire import _model, _records            # noqa: PLC0415
            surrogate, data = _model(), _records()
        self._core = FireDispatchEnv(surrogate, data, FireDispatchTask(cost_limit=budget),
                                     fires=fires, horizon=horizon, split=split,
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

    def _np(self, t: Tensor) -> "np.ndarray":
        return t[0].detach().cpu().numpy().astype(np.float32)

    def even_action(self, obs=None):
        return self._np(self._core.even_action())

    def biggest_fire_action(self, obs):
        return self._np(self._core.biggest_fire_action(obs))

    def most_people_action(self, obs):
        return self._np(self._core.most_people_action(obs))


def register_envs() -> None:
    from gymnasium.envs.registration import register, registry

    if "FireDispatch-v0" in registry:
        return
    register(id="FireDispatch-v0", entry_point="pspe.envs.dispatch:_make_default",
             max_episode_steps=None)


def _make_default(**kw) -> FireDispatchControlEnv:
    return FireDispatchControlEnv(**kw)
