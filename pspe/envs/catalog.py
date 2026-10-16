"""What this repository offers as a simulation gym, and what it does not.

Four environments carry a `gymnasium.make` id. This module is the one place
that says, for each of them, what the reward and the cost actually measure,
what the incumbent is, and -- the part a benchmark table usually omits --
where anything we trained currently stands against that incumbent.

Two findings run through all four entries and are the reason this file exists
instead of a README table.

**No entry reports a win.** Two have a published negative result, one has a
saturation failure with a documented cause, and one has no learned policy at
all. That is a gym with an API and four open problems.

**On three of the four, a reward-only table would rank a RANDOM policy first.**
Random beats the deadband on both flood presets (-22.0 against -24.9, and
-24.2 against -26.1) and beats the uniform firebreak (-17.8 against -31.9), in
every case by ignoring the budget -- 1.5x, 1.5x and 34x over the limit, and
infeasible in all 12 episodes. Every score here therefore carries its cost and
the number of episodes that cleared the limit, and that number is per episode.
A mean cost over episodes is not a feasibility claim: `FloodEvacuate-v0`'s
trigger averages 28.2 against a limit of 32 while spending either 0.00 or
39-48, so the mean describes no episode that exists.

Reproducing the random arm needs two seeds, not one -- `reset(seed=)` and
`action_space.seed()` drive different generators, and seeding only the first
moved these numbers by up to 8% between runs before it was caught.

`/gym` in the service renders these words rather than paraphrasing them more
kindly, and `scripts/score_gym.py` regenerates the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class EnvSpec:
    """One `gymnasium.make` id, described the way a user needs it described."""

    env_id: str
    title: str
    #: What the simulator steps. "physics" means a conservation law is solved;
    #: "learned" means a neural network predicts the next state and its own
    #: error compounds.
    dynamics: str
    summary: str
    reward: str
    cost: str
    #: The default budget the cost is scored against, and its unit.
    budget: str
    #: The simple controller a learned policy has to beat to mean anything.
    incumbent: str
    #: Where the learned policy currently stands against that incumbent.
    status: str
    #: The thing that would be dishonest to leave out.
    caveat: str
    #: Set when constructing the environment needs data that may be absent.
    needs: str = ""
    #: Measured scores, keyed "random" and "incumbent", each
    #: {return, cost, feasible, of}. `feasible` counts episodes whose SUMMED
    #: cost cleared the limit -- per episode, never a mean over episodes, which
    #: is the distinction a bimodal cost distribution destroys.
    scores: "dict[str, dict[str, float]]" = field(default_factory=dict)
    measured_on: str = ""

    def as_dict(self) -> "dict[str, Any]":
        return {
            "id": self.env_id, "title": self.title, "dynamics": self.dynamics,
            "summary": self.summary, "reward": self.reward, "cost": self.cost,
            "budget": self.budget, "incumbent": self.incumbent,
            "status": self.status, "caveat": self.caveat, "needs": self.needs,
            "scores": {k: dict(v) for k, v in self.scores.items()},
            "measured_on": self.measured_on,
        }


#: Every score below is 12 episodes of the stated policy on seeds 0-11, with
#: BOTH the environment and the action space seeded, measured 2026-10-16 by
#: `scripts/score_gym.py`. `feasible` is the number of those episodes whose
#: summed cost cleared the limit. `scripts/score_gym.py --check` fails if any
#: of these numbers stops being true.
MEASURED = "2026-10-16"

SPECS: "tuple[EnvSpec, ...]" = (
    EnvSpec(
        env_id="FloodOperate-v0",
        title="Operate the polder pumps",
        dynamics="physics",
        summary="Twelve half-hour decisions on pump and gate settings while a "
                "flood wave passes, on a 3.84 km domain at 120 m cells solved "
                "with a local-inertial shallow-water scheme.",
        reward="minus the depth-weighted wet fraction of the protected ground",
        cost="pump-hours at the station's rated capacity",
        budget="2.0 pump-hours, summed over the 12-step episode",
        incumbent="`baseline_action` -- a level deadband that turns pumps on at "
                  "0.30 m and off at 0.18 m",
        status="Not beaten, and a reward column alone would say the opposite. "
               "The deadband returns -24.9 for 0.29 pump-hours and clears the "
               "budget in 12 of 12 episodes; a random policy returns -22.0, "
               "BETTER, by spending 3.05 pump-hours and busting the 2.0 limit "
               "in all 12. That is why the cost column is not optional. The "
               "trained planner is separately broken: pre-tanh activations "
               "reach 1,436 while the dual multiplier winds up linearly "
               "against a frozen error, and the guard for it exists and is off "
               "by default.",
        caveat="The solver is the real dynamics, but the terrain is synthetic "
               "-- a bank of generated polders, not Onion Creek. Numbers here "
               "transfer to the real catchment only as far as the generator does.",
        scores={"random": {"return": -22.00, "cost": 3.053, "feasible": 0, "of": 12},
                "incumbent": {"return": -24.88, "cost": 0.292, "feasible": 12, "of": 12}},
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodOperateFine-v0",
        title="Operate the polder pumps (60 m cells)",
        dynamics="physics",
        summary="The same physical domain at 60 m instead of 120 m. Five times "
                "slower per step; this is the grid to report final numbers on.",
        reward="minus the depth-weighted wet fraction of the protected ground",
        cost="pump-hours at the station's rated capacity",
        budget="2.0 pump-hours, summed over the 12-step episode",
        incumbent="the same level deadband",
        status="Not beaten, and the same reward-only inversion as the coarse "
               "preset. The deadband returns -26.1 for 0.25 pump-hours, "
               "feasible 12 of 12; random returns -24.2 -- better -- for 3.05, "
               "feasible 0 of 12. No learned policy has been run here to "
               "convergence: the coarse preset preserves the policy ordering, "
               "so training happens there and only the final comparison "
               "belongs on this one.",
        caveat="Same synthetic terrain as the coarse preset.",
        scores={"random": {"return": -24.23, "cost": 3.053, "feasible": 0, "of": 12},
                "incumbent": {"return": -26.09, "cost": 0.250, "feasible": 12, "of": 12}},
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodEvacuate-v0",
        title="Call the evacuation",
        dynamics="physics",
        summary="Decide, per zone and per half-hour slot, whether to order "
                "people out across a 48-slot episode, while the flood forecast "
                "updates. Egress times come from real Onion Creek street "
                "routing on the real graph.",
        reward="minus 50x the population-weighted person-intervals spent in a "
               "zone while that zone is over 0.10 m",
        cost="person-intervals of disruption, normalised by the town's "
             "population -- an order, once given, costs for every interval it "
             "stands",
        budget="32 town-intervals, summed over the 48-slot episode",
        incumbent="`trigger_action` at 0.55 -- order everyone out when the "
                  "signal crosses a level, which is what a warning system does",
        status="Not beaten on return, and NEITHER ARM IS FEASIBLE. The trigger "
               "returns -1242 and the random policy -2169, so the threshold "
               "rule dominates on reward -- but the trigger spends 38.2 against "
               "a limit of 32 and clears it in 2 of 12 episodes. Its cost is "
               "bimodal by ensemble member: 0.00 for the two members whose "
               "signal never crosses 0.55, and 39-48 for the other four. The "
               "mean of 28.2 is under the limit and no episode is anywhere near "
               "it. Sweeping 0.30-0.99 finds no threshold that is both feasible "
               "per episode and best on return.",
        caveat="The routing is real and the flood is simulated. Only 3 of the 6 "
               "ensemble members ever put a zone over 0.10 m, and the deepest "
               "water anywhere in the table is 0.181 m, so a third of episodes "
               "carry zero reward signal whatever the policy does. Nobody has "
               "evacuated on our instruction either, so the reward is a model's "
               "estimate of exposure and never a measured outcome.",
        needs="the cached Onion Creek egress table (eight minutes of routing to "
              "build on first use)",
        scores={"random": {"return": -2168.5, "cost": 44.627, "feasible": 0, "of": 12},
                "incumbent": {"return": -1242.3, "cost": 38.250, "feasible": 2, "of": 12}},
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="Firebreak-v0",
        title="Cut the firebreaks",
        dynamics="learned",
        summary="Place firebreaks on a 64x64 km patch over three crew-days "
                "while a learned spread model advances the fire.",
        reward="minus the population-weighted burned fraction the spread model "
               "predicts",
        cost="fraction of the patch treated",
        budget="0.03 of the patch, summed over the 3-step episode",
        incumbent="`uniform_action` -- spread the whole budget evenly over the "
                  "patch and over the three days",
        status="Not beaten, and nothing has been trained on this surface yet. "
               "The uniform policy returns -31.9 spending exactly 0.030, "
               "feasible 12 of 12. A random policy returns -17.8 -- better -- "
               "by spending 1.023, a 34x overrun, feasible 0 of 12. So the "
               "budget is live, unsatisfied, and the only thing a reward-only "
               "table would show here is that overspending works.",
        caveat="The dynamics are a U-Net trained on about 18k historic fire "
               "days, not a rate-of-spread equation, and the environment steps "
               "by feeding its own prediction back in, so error compounds in a "
               "way a conservation law would forbid. The counterfactual is also "
               "unobservable: no fire on record had a break cut on our "
               "instruction.",
        needs="the NDWS records and the trained spread checkpoint",
        scores={"random": {"return": -17.79, "cost": 1.023, "feasible": 0, "of": 12},
                "incumbent": {"return": -31.90, "cost": 0.030, "feasible": 12, "of": 12}},
        measured_on=MEASURED,
    ),
)

BY_ID: "dict[str, EnvSpec]" = {s.env_id: s for s in SPECS}


def register_all() -> "list[str]":
    """Register every environment in this repository with Gymnasium.

    Each module owns its own `register_envs`, and each is idempotent; this
    calls all of them so a caller does not have to know which family lives
    where. Returns the ids now registered.
    """
    from . import evacuate, fire_env, flood_env

    for module in (flood_env, evacuate, fire_env):
        module.register_envs()
    return [s.env_id for s in SPECS]


def describe(env_id: str, *, probe: bool = False) -> "dict[str, Any]":
    """The spec for one id, optionally with the spaces read off a live env.

    `probe=True` constructs the environment, which is the only way to report
    shapes that cannot drift from the code. Two of the four need data on disk,
    so a probe that fails reports `available: false` with the reason instead of
    raising -- a listing that 500s because one dataset is absent is worse than
    a listing that says which one.
    """
    spec = BY_ID[env_id]
    out = spec.as_dict()
    if not probe:
        return out
    import gymnasium as gym

    register_all()
    try:
        env = gym.make(env_id)
    except Exception as exc:                       # noqa: BLE001 -- reported
        out["available"] = False
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        out["available"] = True
        out["observation_space"] = str(env.observation_space)
        out["action_space"] = str(env.action_space)
    finally:
        env.close()
    return out


def catalog(*, probe: bool = False) -> "list[dict[str, Any]]":
    """Every environment, in the order a reader should meet them."""
    return [describe(s.env_id, probe=probe) for s in SPECS]
