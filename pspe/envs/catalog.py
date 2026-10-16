"""What this repository offers as a simulation gym, and what it does not.

Four environments carry a `gymnasium.make` id. This module is the one place
that says, for each of them, what the reward and the cost actually measure,
which policies have been run on it, and where anything learned currently
stands. `/gym` in the service renders these words rather than paraphrasing
them more kindly, and `scripts/score_gym.py --check` fails if any number here
stops being true.

Each entry carries a `ladder`: every policy measured on that environment, with
its return, its spend, and the number of episodes whose summed cost cleared the
limit. The ladder, not the summary, is the artefact -- four findings came out
of building it and none of them is visible in a table of returns.

**A weak incumbent invents a result.** Three of the four environments named a
baseline that a reward-only table ranked BELOW a random policy, which reads as
"random beats the heuristic". It was an artefact of the baseline. On both flood
presets, holding the pumps at the one constant fraction the budget exactly
affords beats both the named deadband and random, while staying feasible in
12 of 12 episodes: -19.90 against the deadband's -24.88 and random's -22.00.
The deadband loses because it underspends by 7x, not because flood control is
hard. After adding that constant the inversion survives on ONE of four
environments, not three.

**The strongest known policy is often infeasible.** `sealed_aware_action`
documents itself as "the number a learned policy has to beat" at a budget of
six pump-hours. At this environment's budget of two it spends 3.42 and clears
the limit in 2 of 12 episodes. It is the best policy on reward and not a bar
anything has to clear.

**On `FloodEvacuate-v0`, nothing known is both feasible and better than doing
nothing.** The trigger family collapses to two policies: at thresholds of 0.70
and above it scores exactly what never ordering scores, to one decimal, because
it never fires; at 0.55 and below it scores exactly what ordering everyone
immediately scores. The useful one costs 38-48 against a limit of 32.

**`Firebreak-v0`'s default budget makes the task nearly inert.** From treating
nothing (-32.31) to the best feasible placement found (-31.08) is a span of
1.23, under 4% of the return, while treating the whole patch reaches -0.85. At
3% of a patch it barely matters where the break goes, so a reported margin much
larger than 4% at this budget is more likely a bug than a result.

One mechanical note, because it cost two runs: reproducing a random policy
needs two seeds. `reset(seed=)` and `action_space.seed()` drive different
generators, and seeding only the first moved these numbers by up to 8% between
runs.
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
    #: The budget as a number, for anything that has to compute against it --
    #: the portal turns a spend into "all of it" or "34x over" and cannot do
    #: that from the prose in `budget`. `scripts/score_gym.py --check` asserts
    #: it equals the live environment's `task.cost_limit`, so it cannot drift.
    cost_limit: float = 0.0
    #: Measured scores, keyed "random" and "incumbent", each
    #: {return, cost, feasible, of}. `feasible` counts episodes whose SUMMED
    #: cost cleared the limit -- per episode, never a mean over episodes, which
    #: is the distinction a bimodal cost distribution destroys.
    scores: "dict[str, dict[str, float]]" = field(default_factory=dict)
    #: Every policy measured on this environment, weakest return first, as
    #: {key, policy, return, cost, feasible, of, note}. `key` is what
    #: `scripts/score_gym.py` matches on, so every row here is re-measured by
    #: `--ladder --check` rather than merely written down. `note` is empty unless the
    #: row needs a word of warning -- that it is infeasible, or that it is
    #: another policy in disguise.
    ladder: "tuple[dict[str, Any], ...]" = ()
    measured_on: str = ""

    # -- the same facts, for somebody who does not work on this ------------- #
    #
    # The fields above are written for the paper: they say "episode",
    # "incumbent", "feasible", "pump-hours at rated capacity". The portal has
    # readers who want to know what the thing decides and whether it worked,
    # and that is a different register, not a different set of facts. Keeping
    # both here rather than paraphrasing in the page means there is still one
    # place to correct when a number moves.
    #
    # The rule for these: no term that needs this repository to understand. No
    # "policy", "return", "cost", "episode", "budget-edge", "feasible".
    plain_title: str = ""
    plain_decision: str = ""      # what you decide, and how often
    plain_score: str = ""         # what counts as a bad outcome
    plain_budget: str = ""        # what you are allowed to spend
    plain_status: str = ""        # one sentence on where this stands
    plain_caveat: str = ""        # the thing it would be unfair not to say

    def as_dict(self) -> "dict[str, Any]":
        return {
            "id": self.env_id, "title": self.title, "dynamics": self.dynamics,
            "summary": self.summary, "reward": self.reward, "cost": self.cost,
            "budget": self.budget, "incumbent": self.incumbent,
            "status": self.status, "caveat": self.caveat, "needs": self.needs,
            "cost_limit": self.cost_limit,
            "scores": {k: dict(v) for k, v in self.scores.items()},
            "ladder": [dict(row) for row in self.ladder],
            "measured_on": self.measured_on,
            "plain": {
                "title": self.plain_title, "decision": self.plain_decision,
                "score": self.plain_score, "budget": self.plain_budget,
                "status": self.plain_status, "caveat": self.plain_caveat,
                "dynamics": ("really simulated" if self.dynamics == "physics"
                             else "guessed by a model"),
            },
        }


#: Every score and ladder row below is 12 episodes of the stated policy on
#: seeds 0-11, with BOTH the environment and the action space seeded, measured
#: 2026-10-16 by `scripts/score_gym.py`. `feasible` counts the episodes whose
#: SUMMED cost cleared the limit -- per episode, never a mean across them.
MEASURED = "2026-10-16"

SPECS: "tuple[EnvSpec, ...]" = (
    EnvSpec(
        env_id="FloodOperate-v0",
        cost_limit=2.0,
        plain_title='Run the pumps',
        plain_decision='Every half hour for six hours, decide how far to open the drain gate and how hard to run the pump, while a flood goes past.',
        plain_score='how much of the protected ground ends up under water',
        plain_budget='two hours of pumping',
        plain_status='Nothing we have trained beats steady pumping -- a setting you pick once and leave alone.',
        plain_caveat='The water is properly simulated. The landscape is invented, so these numbers are about the problem, not about Onion Creek.',
        title="Operate the polder pumps",
        dynamics="physics",
        summary="Twelve half-hour decisions on a gate and a pump while a flood "
                "wave passes, on a 3.84 km domain at 120 m cells solved with a "
                "local-inertial shallow-water scheme.",
        reward="minus the depth-weighted wet fraction of the protected ground",
        cost="pump-hours at the station's rated capacity. The gate is free, "
             "which matters: opening it is worth 30.9 of return at no cost, so "
             "most of what looks like control here is gravity",
        budget="2.0 pump-hours, summed over the 12-step episode",
        incumbent="`budget_edge_action` -- gate open, pumps held at the one "
                  "constant fraction the budget exactly affords (-19.90 at "
                  "2.000, feasible 12/12). It replaces the level deadband "
                  "`baseline_action` as the bar, because the deadband "
                  "underspends by 7x and gives up 5.0 of return doing it.",
        status="Not beaten. The trained planner saturates: pre-tanh activations "
               "reach 1,436 while the dual multiplier winds up linearly against "
               "a frozen error, and the guard for it exists and is off by "
               "default. Worth knowing before reading any learned number here: "
               "a policy scoring between -22 and -20 has beaten random's return "
               "without matching a two-parameter constant.",
        caveat="The solver is the real dynamics, but the terrain is synthetic "
               "-- a bank of generated polders, not Onion Creek. Numbers here "
               "transfer to the real catchment only as far as the generator does.",
        scores={"random": {"return": -22.00, "cost": 3.053, "feasible": 0, "of": 12},
                "incumbent": {"return": -19.90, "cost": 2.000, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'do nothing', "policy": 'do nothing', "return": -56.9, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'gate_only', "plain": 'open the gate, never pump', "policy": 'gate open, pumps off', "return": -26.05, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'the gate is free -- cost prices pumps only -- and opening it is worth 30.9 of return for nothing'},
        {"key": 'deadband', "plain": 'pump when the sump gets deep', "policy": 'deadband (`baseline_action`)', "return": -24.88, "cost": 0.292, "feasible": 12, "of": 12,
         "note": "the environment's named incumbent; it holds gate-open/pumps-off on most steps and spends 0.29 of a 2.0 budget"},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -22.0, "cost": 3.053, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'budget_edge', "plain": 'steady pumping', "policy": 'budget edge (`budget_edge_action`)', "return": -19.9, "cost": 2.0, "feasible": 12, "of": 12,
         "note": "the best FEASIBLE policy known; return is monotone in the pump fraction across the whole feasible range, so the budget edge is this family's optimum"},
        {"key": 'sealed_aware', "plain": 'pump only when the river blocks the gate', "policy": 'sealed-aware (`sealed_aware_action`)', "return": -13.93, "cost": 3.417, "feasible": 2, "of": 12,
         "note": 'INFEASIBLE here; its own docstring reports it against a 6.0 budget'},
        {"key": 'flat_out', "plain": 'pump flat out', "policy": 'pump flat out', "return": -13.59, "cost": 6.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 3x the budget'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodOperateFine-v0",
        cost_limit=2.0,
        plain_title='Run the pumps, in finer detail',
        plain_decision='The same six hours on a map with four times as many squares. Slower to run, and the one to quote final numbers from.',
        plain_score='how much of the protected ground ends up under water',
        plain_budget='two hours of pumping',
        plain_status='Nothing has been trained here yet. Steady pumping is still ahead of everything that has.',
        plain_caveat='Same invented landscape as the coarser version.',
        title="Operate the polder pumps (60 m cells)",
        dynamics="physics",
        summary="The same physical domain at 60 m instead of 120 m. Five times "
                "slower per step; this is the grid to report final numbers on.",
        reward="minus the depth-weighted wet fraction of the protected ground",
        cost="pump-hours at the station's rated capacity; the gate is free",
        budget="2.0 pump-hours, summed over the 12-step episode",
        incumbent="`budget_edge_action` (-23.30 at 2.000, feasible 12/12), for "
                  "the same reason as the coarse preset",
        status="Not beaten, and no learned policy has been run here to "
               "convergence: the coarse preset preserves the policy ordering, "
               "so training happens there and only the final comparison belongs "
               "on this one. The whole ladder shifts about 3 points worse than "
               "the coarse grid and keeps its order.",
        caveat="Same synthetic terrain as the coarse preset.",
        scores={"random": {"return": -24.23, "cost": 3.053, "feasible": 0, "of": 12},
                "incumbent": {"return": -23.30, "cost": 2.000, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'do nothing', "policy": 'do nothing', "return": -58.53, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'gate_only', "plain": 'open the gate, never pump', "policy": 'gate open, pumps off', "return": -27.05, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'deadband', "plain": 'pump when the sump gets deep', "policy": 'deadband (`baseline_action`)', "return": -26.09, "cost": 0.25, "feasible": 12, "of": 12,
         "note": 'the named incumbent, spending 0.25 of a 2.0 budget'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -24.23, "cost": 3.053, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'budget_edge', "plain": 'steady pumping', "policy": 'budget edge (`budget_edge_action`)', "return": -23.3, "cost": 2.0, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known'},
        {"key": 'sealed_aware', "plain": 'pump only when the river blocks the gate', "policy": 'sealed-aware (`sealed_aware_action`)', "return": -17.89, "cost": 3.833, "feasible": 2, "of": 12,
         "note": 'INFEASIBLE'},
        {"key": 'flat_out', "plain": 'pump flat out', "policy": 'pump flat out', "return": -17.03, "cost": 6.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 3x the budget'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodEvacuate-v0",
        cost_limit=32.0,
        plain_title='Call the evacuation',
        plain_decision='Every half hour for a day, decide which of eight neighbourhoods to tell to leave. Driving times come from the real Austin street map.',
        plain_score='how many people are still in a street while it floods, and for how long',
        plain_budget='enough disruption to move the whole town for 32 half-hours',
        plain_status='Every approach that stays within the budget does nothing at all, and the one that gets people out costs more than it is allowed.',
        plain_caveat='The streets are real and the flood is simulated. In half the forecasts nothing floods, the deepest water anywhere is ankle-high, and nobody has ever evacuated on our say-so -- so the harm is an estimate, never something that happened.',
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
        incumbent="doing nothing, at -4862.7 for 0.000, is the best FEASIBLE "
                  "policy known. The named `trigger_action` at 0.55 scores far "
                  "better (-1242.3) and spends 38.25, clearing the limit in 2 "
                  "of 12 episodes.",
        status="NOTHING KNOWN IS BOTH FEASIBLE AND BETTER THAN DOING NOTHING, "
               "which is the honest status and not a placeholder. The trigger "
               "family collapses to two policies: at 0.70 and above it scores "
               "exactly what never ordering scores, because it never fires; at "
               "0.55 and below it scores exactly what ordering everyone "
               "immediately scores, and costs 38-48. A learned policy has "
               "somewhere real to go here -- a per-zone, per-slot order is 8 "
               "degrees of freedom against the trigger's one -- and nothing has "
               "gone there yet.",
        caveat="The routing is real and the flood is simulated. Only 3 of the 6 "
               "ensemble members ever put a zone over 0.10 m, and the deepest "
               "water anywhere in the table is 0.181 m, so a third of episodes "
               "carry zero reward signal whatever the policy does. Nobody has "
               "evacuated on our instruction either, so the reward is a model's "
               "estimate of exposure and never a measured outcome.",
        needs="the cached Onion Creek egress table (eight minutes of routing to "
              "build on first use)",
        # `incumbent` is doing nothing, because that is the best feasible
        # policy known here. The trigger's own numbers are in the ladder.
        scores={"random": {"return": -2168.5, "cost": 44.627, "feasible": 0, "of": 12},
                "incumbent": {"return": -4862.65, "cost": 0.000, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'never', "plain": 'never tell anyone to leave', "policy": 'never order', "return": -4862.7, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known on this environment, which is the finding'},
        {"key": 'trigger_90', "plain": 'warn only at the very last moment', "policy": 'trigger at 0.90', "return": -4862.7, "cost": 7.0, "feasible": 10, "of": 12,
         "note": 'scores never-ordering exactly: the signal never crosses 0.90'},
        {"key": 'trigger_70', "plain": 'warn very late', "policy": 'trigger at 0.70', "return": -4862.7, "cost": 19.75, "feasible": 7, "of": 12,
         "note": 'also never fires, so also never-ordering'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -2168.5, "cost": 44.627, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'trigger_55', "plain": 'warn when the river rises', "policy": 'trigger at 0.55 (`trigger_action`)', "return": -1242.3, "cost": 38.25, "feasible": 2, "of": 12,
         "note": 'the named incumbent. Scores order-everyone-at-once exactly, and its cost is bimodal by ensemble member -- 0.00 or 39-48, never near 32 -- so its 28.2 mean describes no episode that exists'},
        {"key": 'always', "plain": 'tell everyone to leave straight away', "policy": 'order everyone at once', "return": -1242.3, "cost": 48.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 1.5x the budget; the trigger at 0.55 is this policy'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="Firebreak-v0",
        cost_limit=0.03,
        plain_title='Cut firebreaks',
        plain_decision='Three days to send crews to clear strips of fuel across a 64 km square, while the fire spreads.',
        plain_score='how much populated ground burns',
        plain_budget='crews enough to clear 3% of the area',
        plain_status='Nothing has been trained here, and the crews are too few to matter much: clearing nothing and clearing the best strip we found differ by under 4%.',
        plain_caveat='The fire is guessed by a model trained on past fires, not worked out from physics, and each day it is fed its own guess from the day before. No real fire ever had a break cut where we said, so nobody can know what would really have happened.',
        title="Cut the firebreaks",
        dynamics="learned",
        summary="Place firebreaks on a 64x64 km patch over three crew-days "
                "while a learned spread model advances the fire.",
        reward="minus the population-weighted burned fraction the spread model "
               "predicts",
        cost="fraction of the patch treated",
        budget="0.03 of the patch, summed over the 3-step episode",
        incumbent="`greedy_action` -- put the day's share on the patch that is "
                  "alight most (-31.08 at 0.0300, feasible 12/12). It beats "
                  "`uniform_action` by 0.82 and treating nothing by 1.23.",
        status="Not beaten, nothing has been trained on this surface yet, and "
               "the budget makes the task nearly inert: the whole feasible span "
               "from treating nothing to the best placement found is 1.23, "
               "under 4 percent of the return. A margin much larger than that "
               "at this budget is more likely a bug than a result. The regime "
               "where placement matters is a bigger budget -- the "
               "uniform-to-greedy gap widens to 2.06 at a 6 percent budget and "
               "5.16 at 24 percent. Random still returns -17.79, better than "
               "anything feasible, by spending 34x the limit.",
        caveat="The dynamics are a U-Net trained on about 18k historic fire "
               "days, not a rate-of-spread equation, and the environment steps "
               "by feeding its own prediction back in, so error compounds in a "
               "way a conservation law would forbid. The counterfactual is also "
               "unobservable: no fire on record had a break cut on our "
               "instruction.",
        needs="the NDWS records and the trained spread checkpoint",
        scores={"random": {"return": -17.79, "cost": 1.023, "feasible": 0, "of": 12},
                "incumbent": {"return": -31.08, "cost": 0.030, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'clear nothing', "policy": 'no treatment', "return": -32.31, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'uniform', "plain": 'spread the crews evenly', "policy": 'uniform over patches and days (`uniform_action`)', "return": -31.9, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'first shipped spending `cost_limit` on EVERY step, which is the horizon times the budget'},
        {"key": 'greedy', "plain": 'send the crews where it is burning', "policy": 'greedy onto the hottest patch (`greedy_action`)', "return": -31.08, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known, and only 1.23 better than treating nothing'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -17.79, "cost": 1.0231, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 34x the budget'},
        {"key": 'flat_out', "plain": 'clear the whole area', "policy": 'treat the whole patch flat out', "return": -0.85, "cost": 3.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 100x the budget; it shows the model does respond, which is the only thing it shows'},
        ),
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
