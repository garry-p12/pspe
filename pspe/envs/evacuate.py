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

import numpy as np
import torch

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
    members: int = field(init=False)
    zones: int = field(init=False)
    slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.members, self.zones, self.slots = self.ok.shape


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

    return EgressTable(ok=ok, minutes=minutes, exposure=exposure, signal=sig,
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
        self.state: Tensor | None = None
        self.t = 0

    # -- interface ----------------------------------------------------------- #
    @property
    def action_dim(self) -> int:
        return self.K

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (4, 1, self.K)

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
        b = self.remaining.shape[0]
        frac = self.remaining / self.pop[None].clamp(min=1e-6)
        sig = self.signal[self._member, min(self.t, self.horizon - 1)]
        sig = sig.view(b, 1).expand(b, self.K)
        tf = torch.full((b, self.K), self.t / max(self.horizon, 1),
                        device=self.device)
        return torch.stack([frac, sig, tf, self.under_order], dim=1).unsqueeze(2)

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
        sig = state[:, 1, 0, 0]
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
