"""Ordering an evacuation while the roads close: the decision this site has.

`FloodOperateEnv` is a polder, and there is no polder at Onion Creek. The
operational decision that genuinely exists there — on infrastructure that
genuinely exists there — is **when to tell which neighbourhoods to leave**,
against a road network that degrades as the water rises.

The portal already computes the hard part of it. `pspe.plan.routes` carries
time-dependent routing over the solver's snapshots, and `latest_departure`
answers *"the last moment someone can leave and still get through"* — whose
own docstring makes the case better than this one could:

    "you have until 14:45" is a decision, "this road floods at some point"
    is not.

`eval/run_evac_policy.py` already evaluates that decision for ONE traveller
under ensemble uncertainty, with five hand-built arms and no training. What
is missing, and what this is, is the sequential multi-zone version: several
neighbourhoods, a population in each, an order that can be given at any
interval, and a cost for giving it too early.

## Why it is a decision and not an arithmetic problem

If the policy knew the deadline it would wait until just before it. It does
not. It sees the flood **so far** — the same depths the portal's Forecast tab
shows — and has to infer how bad this one is going to be. That is the
`filtered` arm's insight from `run_evac_policy.py` generalised: depths you can
see now tell you which member of the forecast distribution you are living in.

Ordering early is not free. An evacuation that turns out to be unnecessary
costs disruption, credibility, and the next warning's compliance. So the cost
is people-intervals under order, and the budget is how much disruption the
operator is willing to spend.

## Why this is fast enough to train in

Routing inside the step loop would make an episode cost seconds. It does not
have to: whether a zone can still get out, leaving at interval t under
member m, does not depend on the policy. `build_egress_table` precomputes
that once per ensemble with the real graph and the real schedules, and the
environment is then a lookup. Real roads, real floods, real routing — and a
step that costs microseconds.

## What this cannot tell you

The same thing every planning environment here says. No record contains the
counterfactual: nobody logged what Onion Creek would have done under a
different evacuation order. The ensemble spans a forcing multiplier and
nothing else — not timing, not spatial pattern, not the solver's own error —
so it is a narrower uncertainty than an operator faces, and the numbers are a
mechanism result rather than a forecast of lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..plan import routes

Tensor = torch.Tensor


@dataclass
class EvacuateTask:
    """Reward weights and the one hard constraint.

    `cost_limit` is in **people-intervals of disruption**, normalised by the
    total population, so 1.0 means "you may have the whole town under an
    evacuation order for one control interval". An order given early costs
    for every interval it stands.
    """

    cost_limit: float = 2.0
    #: Weight on someone who did not get out, relative to one interval of
    #: disruption for one person. Stranding is much worse than inconvenience,
    #: and saying how much worse is a policy choice, not a measurement.
    stranded_weight: float = 50.0
    reward_scale: float = 1.0


@dataclass
class EgressTable:
    """Can zone k still get out, leaving at interval t, under member m?

    `ok[m, k, t]` is the answer and `minutes[m, k, t]` is how long it takes,
    both from real time-dependent routing on the real graph. `deadline[m, k]`
    is the last interval with a route — the quantity `latest_departure`
    reports and the one an operator is actually given.
    """

    ok: np.ndarray            # (M, K, T) bool
    minutes: np.ndarray       # (M, K, T) float, inf where there is no route
    #: Water at the zone itself, per slot. The harm is being THERE while it
    #: floods, not being ordered at an awkward moment.
    exposure: np.ndarray      # (M, K, T) float, metres
    #: What an operator can see at the time, per member per slot: mean water
    #: across the network, which rises smoothly from the first hour and is
    #: proportional to how big this flood is going to be.
    #:
    #: It must LEAD the closures or there is no decision. Defaulting it to the
    #: fraction of routes already cut gave a signal that moves only once
    #: egress is gone: every trigger threshold then fired at the same slot,
    #: too late to matter, and all three scored identically to never ordering
    #: while paying 10.88 of 12 disruption. "By the time you can see it on the
    #: roads it is already too late" is true, and it is the reason the gauge
    #: and not the road network is what a warning watches.
    signal: np.ndarray        # (M, T) float, normalised
    deadline: np.ndarray      # (M, K) int, -1 if never reachable
    population: np.ndarray    # (K,)
    zone_nodes: list[int]
    safe_node: int
    interval_min: float
    #: Where each zone sits on a coarse raster, from its real lon/lat. The
    #: observation is a FIELD rather than a vector of zones, because
    #: neighbouring neighbourhoods flood together and a convolutional policy
    #: can use that. It is also what the rest of this repository's policies
    #: and critics expect: `GaussianFieldPolicy` strides by two three times,
    #: so a (4, 1, 8) observation fails with "kernel size can't be greater
    #: than actual input size" before training starts.
    zone_rc: list[tuple[int, int]] = field(default_factory=list)
    grid: int = 8
    members: int = field(init=False)
    zones: int = field(init=False)
    slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.members, self.zones, self.slots = self.ok.shape


def rasterise_zones(graph: "routes.Graph", zone_nodes: list[int],
                    grid: int = 8) -> list[tuple[int, int]]:
    """Put each zone in a cell of a `grid` x `grid` raster by its real position.

    Spread over the zones' own bounding box rather than the domain's, so the
    cells are used even when the zones cluster. Collisions are walked to the
    nearest free cell: two neighbourhoods in one cell would have their states
    summed, and a policy would see one zone where there are two.
    """
    pts = [graph.nodes[i] for i in zone_nodes]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    w, e = min(lons), max(lons)
    s_, n = min(lats), max(lats)
    taken: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for lon, lat in pts:
        c = int(round((lon - w) / max(e - w, 1e-9) * (grid - 1)))
        r = int(round((n - lat) / max(n - s_, 1e-9) * (grid - 1)))
        if (r, c) in taken:
            best, bd = None, 1e9
            for rr in range(grid):
                for cc in range(grid):
                    if (rr, cc) in taken:
                        continue
                    d = (rr - r) ** 2 + (cc - c) ** 2
                    if d < bd:
                        best, bd = (rr, cc), d
            r, c = best if best else (r, c)
        taken.add((r, c))
        out.append((r, c))
    return out


def build_egress_table(
    graph: "routes.Graph",
    schedules: "list[routes.Schedule]",
    zone_nodes: list[int],
    safe_node: int,
    population: np.ndarray,
    *,
    limit_m: float = 0.25,
    slots: int = 12,
    interval_min: float = 30.0,
    max_travel_min: float = 90.0,
    lead_slots: int = 3,
) -> EgressTable:
    """Route every (member, zone, departure slot) once, so the env never has to.

    This is the only expensive thing in the environment and it is paid once.

    `max_travel_min` is not a detail. `fastest_at` allows WAITING, and our
    runs include the recession, so a crossing shut at two hours can be open at
    six -- which means a route always exists if you are willing to sit in the
    flood until it drains. Scored on feasibility alone, every zone escapes at
    every slot under every member and no arm separates from any other: the
    first run of this env returned exactly 0.00 for all five.

    Someone parked on a roadside for six hours waiting for water to recede
    has not evacuated; they are sheltering in a flood zone, which is the
    outcome the order exists to prevent. So getting out means ARRIVING, within
    a time a person would actually drive.
    """
    M, K, T = len(schedules), len(zone_nodes), slots
    ok = np.zeros((M, K, T), dtype=bool)
    minutes = np.full((M, K, T), np.inf, dtype=np.float32)
    for m, sched in enumerate(schedules):
        for k, src in enumerate(zone_nodes):
            for t in range(T):
                r = routes.fastest_at(graph, src, safe_node, sched, limit_m,
                                      depart_min=t * interval_min)
                mins = float(getattr(r, "minutes", np.inf)) if r is not None else np.inf
                if np.isfinite(mins):
                    minutes[m, k, t] = mins
                    ok[m, k, t] = mins <= max_travel_min
    # Water at each zone, taken as the deepest on the roads meeting it.
    exposure = np.zeros((M, K, T), dtype=np.float32)
    for m, sched in enumerate(schedules):
        for k, node in enumerate(zone_nodes):
            inc = [ei for ei, _ in graph.adj.get(node, [])]
            for t in range(T):
                exposure[m, k, t] = max((sched.at(ei, t * interval_min)
                                         for ei in inc), default=0.0)

    deadline = np.full((M, K), -1, dtype=np.int64)
    for m in range(M):
        for k in range(K):
            w = np.flatnonzero(ok[m, k])
            if w.size:
                deadline[m, k] = int(w.max())
    # The signal LEADS local conditions, because a gauge upstream does.
    #
    # Without the lead there is no decision to make. Measured on this network:
    # egress closes at slot 1 or 2 of 12, while a signal taken from local
    # water is still 0.00 at slot 0 and only separates the members by slot 2 --
    # so every policy reduces to "order at t=0 or never", and the environment
    # is a coin flip rather than a timing problem.
    #
    # That is not an artefact, it is this creek: Onion Creek rises and drains
    # inside a day, which is the same hazard timescale that makes Sentinel-1
    # useless here. The lead has to come from somewhere upstream of the
    # crossings, which is exactly what `Onion Ck at Twin Creeks Rd` is and
    # what the portal's own forecast provides.
    mag = np.array([[float(np.mean(sched.depths[t])) for t in range(T)]
                    for sched in schedules], dtype=np.float32)
    mag = mag / max(float(mag.max()), 1e-6)
    sig = np.stack([mag[:, min(t + lead_slots, T - 1)] for t in range(T)], axis=1)

    return EgressTable(zone_rc=rasterise_zones(graph, list(zone_nodes)),
                       ok=ok, minutes=minutes, exposure=exposure, signal=sig,
                       deadline=deadline,
                       population=np.asarray(population, dtype=np.float32),
                       zone_nodes=list(zone_nodes), safe_node=safe_node,
                       interval_min=interval_min)


class FloodEvacuateEnv:
    """BatchedFieldEnv-compatible: reset / step / obs_shape / action_dim.

    One `step` is one control interval. The action is a per-zone order
    intensity; the observation is what an operator can see by then.

    Observation, per zone, as a `(C, K)` field so the same convolutional
    policy class can read it:

        0  fraction of this zone's people still here
        1  the flood signal observed so far, normalised  (what is knowable)
        2  how far into the event we are
        3  whether this zone is already under order

    What it deliberately does NOT contain is the deadline. That is the thing
    being inferred, and handing it over turns the problem into arithmetic.
    """

    dynamics = "truth"

    def __init__(
        self,
        table: EgressTable,
        task: EvacuateTask | None = None,
        *,
        signal: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.table = table
        self.task = task or EvacuateTask()
        self.device = torch.device(device)
        self.horizon = table.slots
        self.K = table.zones
        self.pop = torch.as_tensor(table.population, dtype=torch.float32,
                                   device=self.device)
        self.pop_total = float(self.pop.sum())
        self.wet_m = 0.10        # the depth the rest of this project calls wet
        # What the operator can see, which must lead the closures. See the
        # note on `EgressTable.signal`.
        if signal is None:
            signal = table.signal
        self.signal = torch.as_tensor(np.asarray(signal, dtype=np.float32),
                                      device=self.device)
        self.grid = int(getattr(table, "grid", 8))
        rc = list(getattr(table, "zone_rc", []) or
                  [(i // self.grid, i % self.grid) for i in range(self.K)])
        self._rows = torch.tensor([r for r, _ in rc], dtype=torch.long,
                                  device=self.device)
        self._cols = torch.tensor([c for _, c in rc], dtype=torch.long,
                                  device=self.device)
        self.state: Tensor | None = None
        self.t = 0

    # -- interface ----------------------------------------------------------- #
    @property
    def action_dim(self) -> int:
        return self.K

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (4, self.grid, self.grid)

    # -- episode ------------------------------------------------------------- #
    def reset(self, batch: int = 1, generator: torch.Generator | None = None,
              member: Tensor | None = None) -> Tensor:
        if member is None:
            member = torch.randint(0, self.table.members, (batch,),
                                   generator=generator)
        self._member = member.to(self.device)
        b = batch
        self.remaining = self.pop[None].expand(b, self.K).clone()
        self.under_order = torch.zeros(b, self.K, device=self.device)
        self.exposed = torch.zeros(b, self.K, device=self.device)
        self.t = 0
        self.state = self._observe()
        return self.state

    def set_state(self, state: Tensor) -> None:
        self.state = state.to(self.device)
        self.t = 0

    def _observe(self) -> Tensor:
        """The zones scattered onto their raster, as a (B, 4, G, G) field.

        A field and not a vector of zones, for two reasons. Neighbourhoods
        that are near each other flood together, so geography is signal a
        convolutional encoder can use. And everything else in this repository
        -- `GaussianFieldPolicy`, `FieldCritic` -- expects a field: handed
        (4, 1, 8) they fail before the first update with "kernel size can't
        be greater than actual input size", because the encoder strides by
        two three times.
        """
        b = self.remaining.shape[0]
        g = self.grid
        frac = self.remaining / self.pop[None].clamp(min=1e-6)
        sig = self.signal[self._member, min(self.t, self.horizon - 1)].view(b, 1)
        tf = torch.full((b, 1), self.t / max(self.horizon, 1), device=self.device)

        out = torch.zeros(b, 4, g, g, device=self.device)
        rows = self._rows
        cols = self._cols
        out[:, 0, rows, cols] = frac
        out[:, 3, rows, cols] = self.under_order
        # The signal and the clock are global, so they fill the field rather
        # than sitting on the zones: a conv kernel that never sees a zone cell
        # would otherwise read neither.
        out[:, 1] = sig.view(b, 1, 1)
        out[:, 2] = tf.view(b, 1, 1)
        return out

    # -- the incumbents ------------------------------------------------------ #
    @torch.no_grad()
    def order_now_action(self, state: Tensor) -> Tensor:
        """Evacuate everyone immediately. Safe, maximally disruptive."""
        return torch.ones(state.shape[0], self.K, device=state.device)

    @torch.no_grad()
    def never_action(self, state: Tensor) -> Tensor:
        return torch.full((state.shape[0], self.K), -1.0, device=state.device)

    @torch.no_grad()
    def trigger_action(self, state: Tensor, threshold: float = 0.35) -> Tensor:
        """Order when the observed signal crosses a threshold. The incumbent.

        This is what a warning system does: watch one number, act when it
        passes a level. It has no model of how much worse this flood is going
        to get, which is exactly the thing a policy could learn — and exactly
        why it is the arm worth beating rather than a strawman.
        """
        sig = state[:, 1].flatten(1)[:, 0]        # the signal fills the field
        on = (sig >= threshold).float() * 2.0 - 1.0
        return on.view(-1, 1).expand(-1, self.K).contiguous()

    # -- dynamics ------------------------------------------------------------ #
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool]:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        u = (action.clamp(-1.0, 1.0) + 1.0) / 2.0            # -> [0, 1]
        b = u.shape[0]

        # Who was told to go, and whether the roads still let them.
        #
        # People ordered onto a road that is cut do NOT drive into it: they
        # stay. That is the correction that makes this a timing problem at
        # all. Scoring an order given at a bad moment as an instant casualty
        # made "never order" cost nothing and a mistimed trigger strand
        # everyone -- 0.0% against 100.0%, with the optimum at doing nothing.
        idx = self._member.cpu().numpy()
        slot = min(self.t, self.horizon - 1)
        ok = torch.as_tensor(self.table.ok[idx, :, slot], device=self.device)
        water = torch.as_tensor(self.table.exposure[idx, :, slot],
                                device=self.device)

        leaving = self.remaining * u * ok.float()        # only those who can
        self.remaining = self.remaining - leaving
        # An order, once given, stands: disruption accrues every interval.
        self.under_order = torch.maximum(self.under_order, u)

        # The harm: people still in the zone while the zone is under water.
        # Accrued per interval, so being late costs more the longer it lasts.
        self.exposed = self.exposed + self.remaining * (water > self.wet_m).float()

        # Cost: people-intervals of disruption, normalised by the town.
        cost = (self.under_order * self.pop[None]).sum(dim=1) / max(self.pop_total, 1e-6)

        self.t += 1
        done = self.t >= self.horizon

        hurt = self.exposed.sum(dim=1) / max(self.pop_total, 1e-6)
        reward = -self.task.reward_scale * self.task.stranded_weight * hurt
        self.state = self._observe()
        return self.state, reward, cost, done

    # -- reporting ----------------------------------------------------------- #
    @torch.no_grad()
    def episode_stats(self) -> dict[str, Tensor]:
        return {
            "exposed_person_intervals": self.exposed.sum(dim=1),
            "exposed_frac": self.exposed.sum(dim=1) / max(self.pop_total, 1e-6),
            "still_there": self.remaining.sum(dim=1) / max(self.pop_total, 1e-6),
            "under_order_frac": (self.under_order * self.pop[None]).sum(dim=1)
            / max(self.pop_total, 1e-6),
        }


class EvacuateControlEnv(gym.Env):
    """Single-instance Gymnasium view of the evacuation decision.

    `FloodEvacuateEnv` speaks the batched all-torch protocol this repository's
    planner and `baselines/safe_rl.py` consume. That is the right shape for a
    trainer that wants sixteen worlds at once and the wrong shape for anything
    outside: an outside algorithm expects `gymnasium.make`, one episode at a
    time, numpy in and numpy out.

    This is that view, mirroring `FloodControlEnv` deliberately -- same
    `info["cost"]` convention, which is what the safe-RL baselines read and
    what Safety-Gymnasium and OmniSafe emit, so an OmniSafe install drops in
    unchanged.

    **Reward** is minus the people-intervals spent under water, scaled.
    **Cost** is people-intervals under evacuation order, normalised by the
    town -- the disruption a city pays and the credibility it spends. Neither
    is in money.

    What this does NOT make true: nothing has beaten a one-number trigger here
    (`FloodEvacuateEnv.trigger_action`), and that family reaches only three
    distinct outcomes across its whole range, so the achievable gap is small
    and discontinuous. This is an environment with a published API, not a
    benchmark with a published result.

    The egress table is the expensive part and is built once by the caller --
    `baselines.evac.table.load()` caches about eight minutes of real
    time-dependent routing -- so constructing this is cheap and an episode
    costs microseconds while the geometry stays real.
    """

    metadata = {"render_modes": []}

    def __init__(self, table: EgressTable, *,
                 budget: float = 32.0,
                 reward_scale: float = 1.0,
                 device: "torch.device | str" = "cpu",
                 seed: "int | None" = None) -> None:
        super().__init__()
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self._core = FloodEvacuateEnv(
            table, EvacuateTask(cost_limit=budget, reward_scale=reward_scale),
            device=self.device)
        c, h, w = self._core.obs_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(c, h, w),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self._core.action_dim,),
                                       dtype=np.float32)
        self.zones = int(table.zones)
        self.population = [float(p) for p in table.population]
        self.episode_cost = 0.0

    # -- gymnasium API ------------------------------------------------------ #
    def reset(self, *, seed: "int | None" = None,
              options: "dict | None" = None) -> "tuple[np.ndarray, dict]":
        # Seeds `self.np_random`, which is what Gymnasium's `check_env`
        # verifies. It does NOT seed `action_space.sample()` -- the action
        # space keeps its own generator, seeded by `action_space.seed(n)` -- so
        # reproducing a random policy here takes both seeds. See the note in
        # `FloodControlEnv.reset`.
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
        info = {
            "cost": float(cost[0]),
            "episode_cost": self.episode_cost,
            "cost_limit": self._core.task.cost_limit,
        }
        if done:
            st = self._core.episode_stats()
            for k, v in st.items():
                info[k] = float(v[0]) if hasattr(v, "shape") else float(v)
        obs = state[0].detach().cpu().numpy().astype(np.float32)
        # `terminated` stays False: the episode ends on the control horizon,
        # a time limit rather than an absorbing state. Calling it termination
        # tells a bootstrapping critic the value beyond the horizon is zero,
        # which is false -- the people are still there.
        return obs, float(reward[0]), False, bool(done), info

    # -- what a benchmark needs to compare against -------------------------- #
    @property
    def task(self) -> EvacuateTask:
        return self._core.task

    @property
    def core(self) -> "FloodEvacuateEnv":
        return self._core

    def never_action(self, obs: "np.ndarray | None" = None) -> "np.ndarray":
        """Order nobody out. The best FEASIBLE policy known on this environment.

        Exposed because that is true and not because it is useful. Measured
        over 12 episodes against the 32-town-interval budget:

            never order              -4862.7    0.00   feasible 12/12
            trigger at 0.90          -4862.7    7.00   feasible 10/12
            trigger at 0.70          -4862.7   19.75   feasible  7/12
            random                   -2168.5   44.63   feasible  0/12
            trigger at 0.55          -1242.3   38.25   feasible  2/12
            order everyone at once   -1242.3   48.00   feasible  0/12

        The trigger family has one degree of freedom on one global signal, and
        on this ensemble it expresses exactly two behaviours: at 0.70 and above
        it scores never-ordering to the decimal because it never fires, and at
        0.55 and below it scores ordering-everyone-immediately to the decimal.
        The useful behaviour costs 38 to 48 against a limit of 32.

        So nothing known here is both feasible and better than doing nothing,
        and a learned policy has somewhere real to go: the action is per zone
        and per slot, eight degrees of freedom against the trigger's one.
        """
        return np.full(self._core.K, -1.0, dtype=np.float32)

    def trigger_action(self, obs: "np.ndarray",
                       threshold: float = 0.55) -> "np.ndarray":
        """The incumbent: order everyone when the signal crosses `threshold`.

        0.55 is the best member of the family at a budget of 32 from the sweep
        in `baselines/run_evac_frontier.py`, and the word that used to sit here
        -- "feasible" -- was wrong. Measured per episode on 2026-10-16 at this
        threshold, the cost is bimodal by ensemble member:

            member 0  cost  0.00      the signal never crosses 0.55
            member 1  cost  0.00
            member 2  cost 39.00      every one of these busts the 32 limit
            member 3  cost 47.00
            member 4  cost 48.00
            member 5  cost 48.00

        The mean is 28.22, under the limit, and **no episode spends anything
        near it**. The trigger is feasible in 96 of 256 episodes and those are
        exactly the ones where it does nothing. A mean over a bimodal cost
        distribution is not an episode-level feasibility claim -- the same
        mistake as reporting `cost_mean_over_run` beside a final policy's
        return, and `run_evac_frontier` corrected that one in its own docstring
        while its baseline cost axis kept doing it.

        No threshold in 0.30-0.99 is both feasible per episode and best on
        return: below 0.60 the return is the best available (-1630) and the
        limit is cleared in at most 5 of 32 episodes; at and above 0.70 the
        limit is usually cleared and the return is 3x worse (-4991).
        """
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        return self._core.trigger_action(t, threshold)[0].cpu().numpy()


def register_envs() -> None:
    """Make `gymnasium.make("FloodEvacuate-v0")` work.

    The egress table is eight minutes of real routing, so it is NOT built here
    -- the entry point loads the cached one through `baselines.evac.table`,
    which builds it on a miss and reuses it thereafter. An environment whose
    constructor silently costs eight minutes is not one anybody will call
    twice.

    Idempotent: importing this module twice in one process is normal and a
    duplicate registration raises rather than passing.
    """
    from gymnasium.envs.registration import register, registry

    if "FloodEvacuate-v0" in registry:
        return
    register(id="FloodEvacuate-v0",
             entry_point="pspe.envs.evacuate:_make_default",
             max_episode_steps=None)


def _make_default(budget: float = 32.0, **kw) -> EvacuateControlEnv:
    """Entry point for `gymnasium.make`, with the Onion Creek egress table."""
    from baselines.evac import table as _t

    return EvacuateControlEnv(_t.load(), budget=budget, **kw)
