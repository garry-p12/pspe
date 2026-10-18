"""What this repository offers as a simulation gym, and what it does not.

Twelve environments carry a `gymnasium.make` id. This module is the one place
that says, for each of them, what the reward and the cost actually measure,
which policies have been run on it, and where anything learned currently
stands. `/gym` in the service renders these words rather than paraphrasing
them more kindly, and `scripts/score_gym.py --ladder --check` fails if any
number here stops being true.

**Every number below is measured on the EVAL split** -- the quarter of each
scenario pool that nothing is fitted to. `gymnasium.make("FloodDefend-v0")`
gives the training pool; asking for `split="eval"` is deliberate, and that is
the point. Each entry records its own pool and partition in `pool`.

Four findings run through the entries, and the last one is the newest.

**No entry reports a win** by a learned policy. Two have a published negative
result, one has a saturation failure with a documented cause, and most have
had nothing trained on them at all. `CMDPToy-v0` is the exception by
construction: its optimum is a closed form, so it is a correctness oracle
rather than an open problem, and it is the first id anybody should call.

**On several, a reward-only table would rank a RANDOM policy first**, always
by ignoring the budget. Every score here therefore carries its cost and the
number of episodes whose summed cost cleared the limit, per episode. A mean
cost over episodes is not a feasibility claim: `FloodEvacuate-v0`'s trigger
spends either nothing or far too much, so its mean describes no episode that
exists.

**The best policy on reward is usually one you may not run.** On
`FloodOperate-v0` it spends 3.46 pump-hours against a limit of 2.0. The
ladders list the infeasible rows and mark them, because a page showing only
the winner would report numbers nobody is allowed to achieve.

**Holding scenarios out changes the answer, and on one environment it
reverses it.** `FloodBuyout-v0` reported on 2026-10-17 that an even share
beats buying the lowest blocks first -- true over its whole 33-window pool
(t = 2.69) and untrue on the 8 windows nothing was fitted to (t = 0.68).
`FloodDefend-v0`'s equivalent claim survives and strengthens (t = 2.00 on the
pool, t = 4.58 held out). `Firebreak-v0`'s survives too (t = 5.78, t = 4.73).
Before the split existed, all three read the same way. That is what the split
is for, and it is why a number measured on the pool a policy was fitted to is
not a result.

One mechanical note, because it cost two runs: reproducing a random policy
needs two seeds. `reset(seed=)` and `action_space.seed()` drive different
generators, and seeding only the first moved these numbers by up to 8%.
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
    #: Which half of the scenario pool these numbers were measured on, and how
    #: big each half is. `eval` means held out: nothing was fitted to them.
    split: str = "eval"
    pool: str = ""

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
            "split": self.split, "pool": self.pool,
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
        pool='64 generated polders: 48 train, 16 eval',
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
        scores={"random": {"return": -22.422, "cost": 3.0532, "feasible": 0, "of": 12},
                "incumbent": {"return": -20.282, "cost": 2.0, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'do nothing', "policy": 'do nothing', "return": -57.813, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'gate_only', "plain": 'open the gate, never pump', "policy": 'gate open, pumps off', "return": -28.441, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'the gate is free -- cost prices pumps only -- and opening it is worth 30.9 of return for nothing'},
        {"key": 'deadband', "plain": 'pump when the sump gets deep', "policy": 'deadband (`baseline_action`)', "return": -27.876, "cost": 0.2083, "feasible": 12, "of": 12,
         "note": "the environment's named incumbent; it holds gate-open/pumps-off on most steps and spends 0.29 of a 2.0 budget"},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -22.422, "cost": 3.0532, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'budget_edge', "plain": 'steady pumping', "policy": 'budget edge (`budget_edge_action`)', "return": -20.282, "cost": 2.0, "feasible": 12, "of": 12,
         "note": "the best FEASIBLE policy known; return is monotone in the pump fraction across the whole feasible range, so the budget edge is this family's optimum"},
        {"key": 'sealed_aware', "plain": 'pump only when the river blocks the gate', "policy": 'sealed-aware (`sealed_aware_action`)', "return": -13.909, "cost": 3.4583, "feasible": 2, "of": 12,
         "note": 'INFEASIBLE here; its own docstring reports it against a 6.0 budget'},
        {"key": 'flat_out', "plain": 'pump flat out', "policy": 'pump flat out', "return": -14.055, "cost": 6.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 3x the budget'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodOperateFine-v0",
        pool='64 generated polders: 48 train, 16 eval',
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
        status="Not beaten, and on held-out terrain the reward-only "
               "inversion is sharper than on the coarse preset: random returns "
               "-25.26 against the budget-edge constant's -25.85, so a table "
               "of returns alone ranks RANDOM first -- while random spends "
               "3.05 pump-hours of a 2.0 budget and is feasible in 0 of 12 "
               "episodes. No learned policy has been run here to convergence: "
               "the coarse preset preserves the ordering, so training happens "
               "there and only the final comparison belongs on this one.",
        caveat="Same synthetic terrain as the coarse preset.",
        scores={"random": {"return": -25.26, "cost": 3.0532, "feasible": 0, "of": 12},
                "incumbent": {"return": -25.852, "cost": 2.0, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'do nothing', "policy": 'do nothing', "return": -59.471, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'gate_only', "plain": 'open the gate, never pump', "policy": 'gate open, pumps off', "return": -29.84, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'deadband', "plain": 'pump when the sump gets deep', "policy": 'deadband (`baseline_action`)', "return": -29.031, "cost": 0.2083, "feasible": 12, "of": 12,
         "note": 'the named incumbent, spending 0.25 of a 2.0 budget'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -25.26, "cost": 3.0532, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'budget_edge', "plain": 'steady pumping', "policy": 'budget edge (`budget_edge_action`)', "return": -25.852, "cost": 2.0, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known'},
        {"key": 'sealed_aware', "plain": 'pump only when the river blocks the gate', "policy": 'sealed-aware (`sealed_aware_action`)', "return": -17.69, "cost": 3.9583, "feasible": 2, "of": 12,
         "note": 'INFEASIBLE'},
        {"key": 'flat_out', "plain": 'pump flat out', "policy": 'pump flat out', "return": -17.149, "cost": 6.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 3x the budget'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodEvacuate-v0",
        pool='6 forecast members: 4 train, 2 eval',
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
               "and on held-out members it is worse: EVERY trigger threshold "
               "scores exactly what never ordering scores, to the decimal, at "
               "zero cost. The 2 eval members never cross any threshold, so "
               "the whole family is one point on this split. Only ordering "
               "everyone immediately does anything (0.00) and it spends 48 "
               "against a limit of 32. With 6 members, 3 of which never flood, "
               "no partition of this environment supports a claim. The support "
               "IS the finding, and nothing should be trained here until it is "
               "larger.",
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
        scores={"random": {"return": -1749.244, "cost": 44.6273, "feasible": 0, "of": 12},
                "incumbent": {"return": -3726.852, "cost": 0.0, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'never', "plain": 'never tell anyone to leave', "policy": 'never order', "return": -3726.852, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known on this environment, which is the finding'},
        {"key": 'trigger_90', "plain": 'warn only at the very last moment', "policy": 'trigger at 0.90', "return": -3726.852, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'scores never-ordering exactly: the signal never crosses 0.90'},
        {"key": 'trigger_70', "plain": 'warn very late', "policy": 'trigger at 0.70', "return": -3726.852, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'also never fires, so also never-ordering'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -1749.244, "cost": 44.6273, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'trigger_55', "plain": 'warn when the river rises', "policy": 'trigger at 0.55 (`trigger_action`)', "return": -3726.852, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'the named incumbent. Scores order-everyone-at-once exactly, and its cost is bimodal by ensemble member -- 0.00 or 39-48, never near 32 -- so its 28.2 mean describes no episode that exists'},
        {"key": 'always', "plain": 'tell everyone to leave straight away', "policy": 'order everyone at once', "return": 0.0, "cost": 48.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 1.5x the budget; the trigger at 0.55 is this policy'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="Firebreak-v0",
        pool='1,689 held-out NDWS fire days: 1,267 train, 422 eval',
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
        scores={"random": {"return": -11.207, "cost": 1.0231, "feasible": 0, "of": 12},
                "incumbent": {"return": -15.864, "cost": 0.03, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'clear nothing', "policy": 'no treatment', "return": -16.962, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'uniform', "plain": 'spread the crews evenly', "policy": 'uniform over patches and days (`uniform_action`)', "return": -16.754, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'first shipped spending `cost_limit` on EVERY step, which is the horizon times the budget'},
        {"key": 'greedy', "plain": 'send the crews where it is burning', "policy": 'greedy onto the hottest patch (`greedy_action`)', "return": -15.864, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known, and only 1.23 better than treating nothing'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -11.207, "cost": 1.0231, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 34x the budget'},
        {"key": 'flat_out', "plain": 'clear the whole area', "policy": 'treat the whole patch flat out', "return": -1.039, "cost": 3.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 100x the budget; it shows the model does respond, which is the only thing it shows'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodDefend-v0",
        pool='35 windows of real terrain: 29 train, 6 eval',
        cost_limit=202149.2031,
        title="Spend the defence budget",
        plain_title="Spend the defence budget",
        plain_decision="Decide, for each of eight stretches of creek bank, how "
                       "high to build an embankment -- before anyone knows how "
                       "big the flood will be.",
        plain_score="how much of the mapped flood-risk ground ends up under "
                    "water",
        plain_budget="about $262,000 of construction, which buys a 1.6 m "
                     "embankment along the whole bank if you spread it evenly",
        plain_status="Building evenly along the bank beats building where "
                     "the ground is lowest -- the obvious answer -- and it "
                     "still beats it on stretches nobody practised on, which "
                     "is the test that counts.",
        plain_caveat="The ground, the creek and the flood-risk map are real: "
                     "USGS laser survey of Onion Creek and FEMA's own flood "
                     "maps. The embankment is not -- Austin has no such levee "
                     "and none is proposed here. The flood is a rise and fall "
                     "in the creek, not a replay of a flood that happened.",
        dynamics="physics",
        summary="Allocate a fixed capital budget across eight segments of the "
                "Onion Creek bank, one decision per segment, then a flood is "
                "solved once over the terrain as built. On real USGS 3DEP bare "
                "earth with FEMA's floodway and 100-year floodplain.",
        reward="minus the percentage of the FEMA 100-year floodplain left "
               "deeper than 0.10 m when the flood has passed",
        cost="US dollars of embankment construction, through the real "
             "trapezoidal section -- so cost per metre of height RISES with "
             "height and a 3 m levee is twice a 2 m one, not 1.5 times",
        budget="$261,755 for the whole programme, derived from the ground "
               "rather than chosen: the cost of a uniform 1.6 m embankment "
               "along the mean bank line, so it always binds",
        incumbent="`uniform_action` -- one height along the whole bank, solved "
                  "from the budget (-14.17, feasible 12/12)",
        status="Not beaten, and the claim SURVIVES being held out -- which "
               "is worth more than the margin. Building evenly beats building "
               "where the bank is lowest by 0.21 on the full pool (t = 2.00) "
               "and by 0.59 on the 6 windows nothing was fitted to (t = 4.58). "
               "The obvious engineering answer is wrong on terrain it has "
               "never seen, not only on the terrain it was compared on, and "
               "the reason is flanking: a partial levee on an open reach is "
               "gone around rather than stopped at. The absolute margins are "
               "small -- the feasible span on eval is 0.7 of 18 -- so this is "
               "a real ordering on a shallow surface.",
        caveat="The terrain, the channel and the at-risk ground are real. The "
               "embankment is hypothetical -- Onion Creek has no continuous "
               "levee and this does not propose one. The flood is a "
               "single-peaked rise and fall held in the channel, not a routed "
               "event from a gauge record, so it exercises the decision "
               "without reproducing a flood that happened. Damage is flooded "
               "area, not dollars of property loss: converting the two needs a "
               "depth-damage curve this repository does not have.",
        needs="the cached USGS 3DEP terrain for the Onion Creek box at 60 m",
        scores={"random": {"return": -18.683, "cost": 292071.2836, "feasible": 5, "of": 12},
                "incumbent": {"return": -18.24, "cost": 202149.0065, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'build nothing', "policy": 'build nothing', "return": -18.381, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'lowest_first', "plain": 'build where the ground is lowest', "policy": 'lowest ground first (`lowest_first_action`)', "return": -18.843, "cost": 184824.2716, "feasible": 12, "of": 12,
         "note": 'the obvious engineering answer, and NOT resolvably better than building nothing: paired, it gains 0.65 at t = 1.77'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -18.683, "cost": 292071.2836, "feasible": 5, "of": 12,
         "note": 'INFEASIBLE in 7 of 12 episodes'},
        {"key": 'uniform', "plain": 'build evenly along the whole bank', "policy": 'uniform along the bank (`uniform_action`)', "return": -18.24, "cost": 202149.0065, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE approach known, and it beats the engineering heuristic at t = 2.76'},
        {"key": 'flat_out', "plain": 'build 3 m everywhere', "policy": '3 m along every segment', "return": -17.13, "cost": 777385.1712, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 3x the budget'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="CMDPToy-v0",
        pool='no pool; the optimum is a closed form',
        cost_limit=0.4,
        title="The one with a known answer",
        plain_title="The one with a known answer",
        plain_decision="Pick a single number. Reward rises to a peak, cost "
                       "rises with it, and the best choice you are allowed is "
                       "worked out on paper rather than searched for.",
        plain_score="how far short of the best possible you end up",
        plain_budget="0.4 units of cost in the one step",
        plain_status="The right answer is 0.400 and this is the only place in "
                     "the gym where that sentence can be written. Check your "
                     "method here before trusting it anywhere else.",
        plain_caveat="It is deliberately trivial -- one step, one number, a "
                     "blank picture -- so that anything that goes wrong is "
                     "your method and not the problem.",
        dynamics="physics",
        summary="A one-step constrained bandit. Reward peaks at 0.8, cost is "
                "linear, so the constrained optimum is min(peak, limit) = 0.4 "
                "and the Lagrange multiplier there is 0.8. Both are exposed on "
                "the environment.",
        reward="minus the squared distance from the reward peak",
        cost="linear in the action",
        budget="0.4, in the single step",
        incumbent="`optimal_action_vector` -- the closed-form constrained "
                  "optimum itself (-0.160, feasible 12/12). Not a heuristic: "
                  "it is the answer.",
        status="SOLVED, by construction, and that is the point. This is a "
               "correctness oracle rather than an open problem: an algorithm "
               "that does not converge to 0.400 here is broken, and the "
               "closed form in `baselines/toy_cmdp.py` is what caught two real "
               "bugs in this repository's own reimplemented baselines. It is "
               "the first id anybody should call.",
        caveat="Trivial on purpose. A constant observation field, one action "
               "dimension, one step -- so a failure here cannot be blamed on "
               "the problem being hard.",
        scores={"random": {"return": -0.812, "cost": 0.3144, "feasible": 8, "of": 12},
                "incumbent": {"return": -0.16, "cost": 0.4, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'do nothing', "policy": 'zero action', "return": -0.64, "cost": 0.0, "feasible": 12, "of": 12,
         "note": 'feasible and worthless, which is the shape of every weak baseline in this gym'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -0.812, "cost": 0.3144, "feasible": 8, "of": 12,
         "note": 'INFEASIBLE in 4 of 12 episodes'},
        {"key": 'optimal', "plain": 'the answer, worked out on paper', "policy": 'the closed-form optimum (`optimal_action_vector`)', "return": -0.16, "cost": 0.4, "feasible": 12, "of": 12,
         "note": 'min(peak, limit) = 0.400, with multiplier 0.800. Nothing can beat it and anything that fails to reach it is broken'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="PDEControl-dar-v0",
        pool='no pool; a fresh field is drawn every episode',
        cost_limit=0.936,
        title="Steer the field (diffusion, advection and reaction)",
        plain_title="Steer the field (diffusion, advection and reaction)",
        plain_decision="Nine dials on a square of simulated fluid, sixteen "
                       "turns, trying to hold the whole surface at its target "
                       "value without using more effort than you are allowed.",
        plain_score="how far the surface drifts from where it should be",
        plain_budget="0.936 units of effort over the whole run",
        plain_status='Pushing each dial against the error under it is the best anyone has managed, and only just.',
        plain_caveat='A made-up fluid on a square, not a place. It is here to test control methods, not to say anything about floods.',
        dynamics="physics",
        summary='Steer a spreading, drifting, reacting field back to its target with nine actuators, sixteen steps on a 32x32 periodic grid, against a differentiable RK4 truth solver.',
        reward="minus the squared tracking error plus an effort penalty",
        cost="exposure above a ceiling, plus any overspend on the actuators",
        budget="0.936 over the sixteen-step run",
        incumbent="`proportional_action` -- each actuator pushed against the "
                  "error beneath it, at the largest gain that stays inside the "
                  "budget",
        status="Not beaten. A proportional controller at gain 0.4 -- push each actuator against the error under it -- returns -1.572 against doing nothing's -1.629, both feasible in 12 of 12. The gap is small and the gain was SET BY SWEEP: at gain 1.0 the same controller scores better and violates the budget in every episode, so shipping it unscaled would have left doing nothing as the best feasible policy.",
        caveat='A synthetic PDE on a periodic square. It is a control problem, not a hazard: nothing here corresponds to a place.',
        scores={"random": {"return": -1.686, "cost": 1.3727, "feasible": 0, "of": 12},
                "incumbent": {"return": -1.572, "cost": 0.1594, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'leave the dials alone', "policy": 'zero action', "return": -1.629, "cost": 0.1703, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -1.686, "cost": 1.3727, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'proportional', "plain": 'push each dial against the error under it', "policy": 'proportional control (`proportional_action`)', "return": -1.572, "cost": 0.1594, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known; the gain was set by sweep, not by taste'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="PDEControl-swe-v0",
        pool='no pool; a fresh field is drawn every episode',
        cost_limit=0.192,
        title="Steer the field (linearised shallow water)",
        plain_title="Steer the field (linearised shallow water)",
        plain_decision="Nine dials on a square of simulated fluid, sixteen "
                       "turns, trying to hold the whole surface at its target "
                       "value without using more effort than you are allowed.",
        plain_score="how far the surface drifts from where it should be",
        plain_budget="0.192 units of effort over the whole run",
        plain_status='Nothing stays inside the budget on every run -- not even leaving all the dials alone.',
        plain_caveat='A made-up fluid on a square, not a place -- and its budget can be exceeded by the fluid alone, whatever you do with the dials.',
        dynamics="physics",
        summary='The same nine-actuator tracking problem on linearised shallow water, three channels, sixteen steps.',
        reward="minus the squared tracking error plus an effort penalty",
        cost="exposure above a ceiling, plus any overspend on the actuators",
        budget="0.192 over the sixteen-step run",
        incumbent="`proportional_action` -- each actuator pushed against the "
                  "error beneath it, at the largest gain that stays inside the "
                  "budget",
        status="Not beaten, and NOTHING IS FEASIBLE IN EVERY EPISODE -- not even doing nothing, which clears the budget in 11 of 12. The cost carries a state-driven exposure term, so on some episodes the limit is exceeded whatever the actuators do. That is a property of the testbed and not of any policy, and `task.py` already records that swe 'went a whole project without anyone noticing its margin'.",
        caveat="Synthetic, as above -- and with a constraint that is not fully controllable, so a violation here is not necessarily a policy's fault.",
        scores={"random": {"return": -0.205, "cost": 3.8406, "feasible": 0, "of": 12},
                "incumbent": {"return": -0.15, "cost": 0.1036, "feasible": 11, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'leave the dials alone', "policy": 'zero action', "return": -0.152, "cost": 0.0929, "feasible": 11, "of": 12,
         "note": ''},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -0.205, "cost": 3.8406, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
        {"key": 'proportional', "plain": 'push each dial against the error under it', "policy": 'proportional control (`proportional_action`)', "return": -0.15, "cost": 0.1036, "feasible": 11, "of": 12,
         "note": 'the best FEASIBLE policy known; the gain was set by sweep, not by taste'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="PDEControl-rdf-v0",
        pool='no pool; a fresh field is drawn every episode',
        cost_limit=3.26,
        title="Steer the field (FitzHugh-Nagumo front propagation)",
        plain_title="Steer the field (FitzHugh-Nagumo front propagation)",
        plain_decision="Nine dials on a square of simulated fluid, sixteen "
                       "turns, trying to hold the whole surface at its target "
                       "value without using more effort than you are allowed.",
        plain_score="how far the surface drifts from where it should be",
        plain_budget="3.26 units of effort over the whole run",
        plain_status='Pushing each dial against the error under it is the best anyone has managed, and only just.',
        plain_caveat='A made-up fluid on a square, not a place. It is here to test control methods, not to say anything about floods.',
        dynamics="physics",
        summary='Hold back a travelling front with nine actuators, two channels, sixteen steps. The most actuation-sensitive of the three.',
        reward="minus the squared tracking error plus an effort penalty",
        cost="exposure above a ceiling, plus any overspend on the actuators",
        budget="3.26 over the sixteen-step run",
        incumbent="`proportional_action` -- each actuator pushed against the "
                  "error beneath it, at the largest gain that stays inside the "
                  "budget",
        status="Not beaten. The proportional controller at gain 0.3 returns -4.448 against doing nothing's -4.822, feasible 12 of 12, which is the clearest margin of the three testbeds. Random is feasible here too and still worse than doing nothing.",
        caveat='Synthetic, as above.',
        scores={"random": {"return": -4.888, "cost": 1.6153, "feasible": 12, "of": 12},
                "incumbent": {"return": -4.448, "cost": 1.7989, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'leave the dials alone', "policy": 'zero action', "return": -4.822, "cost": 0.4117, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -4.888, "cost": 1.6153, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'proportional', "plain": 'push each dial against the error under it', "policy": 'proportional control (`proportional_action`)', "return": -4.448, "cost": 1.7989, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known; the gain was set by sweep, not by taste'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FireDispatch-v0",
        pool='1,689 held-out NDWS fire days: 1,267 train, 422 eval',
        cost_limit=0.03,
        title="Which fire to fight",
        plain_title="Which fire to fight",
        plain_decision="Four fires are burning at once and there are only so "
                       "many crews. Each day for three days, decide how to "
                       "split them between the fires.",
        plain_score="how much populated ground burns across all four",
        plain_budget="crews enough to clear 3% of one patch, shared between "
                     "all four fires",
        plain_status="Sending everyone to the biggest fire is the best anyone "
                     "has managed, and it beats splitting them evenly by less "
                     "than 2%.",
        plain_caveat="The fires are real days from the historic record, but "
                     "how they spread is guessed by a model trained on past "
                     "fires, and no real fire was ever fought on our "
                     "instruction.",
        dynamics="learned",
        summary="Allocate a fixed crew budget across four simultaneous fires, "
                "drawn at random from the held-out NDWS records, over three "
                "crew-days. Crews sent to a fire are spread uniformly over it, "
                "so this asks the between-fire question and leaves the "
                "within-fire one to `Firebreak-v0`.",
        reward="minus the population-weighted burned fraction summed over the "
               "four fires",
        cost="fraction of one patch treated, shared across all four",
        budget="0.03, summed over the three-day episode",
        incumbent="`biggest_fire_action` -- send everyone to whichever fire is "
                  "burning most (-24.97, feasible 12/12)",
        status="Not beaten, and the whole spread is narrow: from doing nothing "
               "(-25.69) to the best rule found (-24.97) is 0.72, under 3% of "
               "the return. The ordering is at least interesting -- the "
               "biggest fire beats the one with the most people in front of "
               "it, and both beat an even split -- but a margin much larger "
               "than 3% here is more likely a bug than a result.",
        caveat="Same learned spread model and same unobservable counterfactual "
               "as `Firebreak-v0`: a U-Net on about 18k historic fire days, "
               "stepped on its own predictions, and no fire on record was "
               "fought on our instruction. The four fires are real days but "
               "they did not burn simultaneously; pairing them is this "
               "environment's own construction.",
        needs="the NDWS records and the trained spread checkpoint",
        scores={"random": {"return": -21.564, "cost": 0.0241, "feasible": 12, "of": 12},
                "incumbent": {"return": -20.988, "cost": 0.03, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'stand the crews down', "policy": 'no crews sent', "return": -21.841, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'most_people', "plain": 'send them where the most people are', "policy": 'most-people-first (`most_people_action`)', "return": -21.541, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'worse than sending them to the biggest fire, which is the small surprise here'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -21.564, "cost": 0.0241, "feasible": 12, "of": 12,
         "note": "feasible, because the action is a split and a random split still spends roughly the day's allowance"},
        {"key": 'even', "plain": 'split them evenly', "policy": 'even split (`even_action`)', "return": -21.498, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'identical to sending everyone everywhere: under this action mapping the split is normalised, so the two are the same policy'},
        {"key": 'biggest', "plain": 'send them to the biggest fire', "policy": 'biggest-fire-first (`biggest_fire_action`)', "return": -20.988, "cost": 0.03, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE policy known, by 0.44 over an even split'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodBuyout-v0",
        pool='33 windows of real terrain: 25 train, 8 eval',
        cost_limit=17.803,
        title="Which houses to buy out",
        plain_title="Which houses to buy out",
        plain_decision="Nine blocks along Onion Creek, and money for about "
                       "eighteen houses. For each block in turn, decide what "
                       "share of its flood-prone homes to buy and clear.",
        plain_score="how many of the remaining homes end up in the water",
        plain_budget="about eighteen houses, out of roughly seventy at risk",
        plain_status="On blocks nobody practised on, these ways of choosing "
                     "cannot be told apart. The result this entry reported a "
                     "day ago came from the same blocks it was worked out on, "
                     "and it does not survive moving to new ones.",
        plain_caveat="The houses, the creek and the flood maps are real, from "
                     "the City of Austin and FEMA. The flood is a rise and "
                     "fall in the creek rather than a flood that happened, "
                     "and the budget counts houses rather than dollars "
                     "because no verified price per house is on hand.",
        dynamics="physics",
        summary="Allocate a buyout programme across nine blocks of the Onion "
                "Creek floodplain, one decision per block, then a flood is "
                "solved over the unchanged terrain. Buying out a block removes "
                "its houses from exposure; it does not move the water. On 33 "
                "windows of real USGS 3DEP terrain with FEMA's floodway and "
                "100-year floodplain and 43,366 City of Austin footprints.",
        reward="minus the percentage of at-risk homes left standing in water "
               "deeper than 0.10 m",
        cost="houses acquired",
        budget="a quarter of the at-risk homes in an average window, about 18",
        incumbent="`even_action` -- take the same share of homes in every "
                  "block (-2.48, feasible 12/12)",
        status="THE RESULT DID NOT SURVIVE A HELD-OUT SPLIT, which is this "
               "entry's finding and a correction to what it said on "
               "2026-10-17. Over the whole 33-window pool an even share beats "
               "buying the lowest blocks first by 7.57, paired, at t = 2.69. "
               "On the 8 windows nothing was fitted to, the same comparison is "
               "+0.05 at t = 0.68: no difference. The eval windows also barely "
               "flood -- doing nothing scores -0.99 against -16.27 on the pool "
               "-- so there is little signal there to order anything by. Both "
               "readings are bad for the environment: 33 windows cannot hold "
               "enough out to support a claim, and the claim it did support "
               "was about the pool it was tuned on. Nothing should be trained "
               "here and reported until the pool is larger.",
        caveat="Austin really ran a buyout on Onion Creek after 2013 and 2015, "
               "and this is not a reconstruction of it: the flood is a "
               "single-peaked rise held in the channel, not a replayed event. "
               "Two thirds of episodes carry no signal at all -- the water "
               "reaches no house in that window whatever is bought -- so the "
               "mean is carried by the third that do. The budget counts houses "
               "rather than dollars on purpose; a plausible-looking price per "
               "house would put false precision on every result.",
        needs="the cached Onion Creek terrain and the City of Austin footprints",
        scores={"random": {"return": -0.269, "cost": 35.5302, "feasible": 5, "of": 12},
                "incumbent": {"return": -0.572, "cost": 17.803, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'nothing', "plain": 'buy nothing', "policy": 'buy nothing', "return": -0.992, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'lowest_first', "plain": 'buy the lowest-lying blocks first', "policy": 'lowest ground first (`lowest_first_action`)', "return": -0.519, "cost": 8.6804, "feasible": 12, "of": 12,
         "note": 'the obvious answer, and it leaves 8 of its 17.8 houses unspent while doing barely half the job'},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -0.269, "cost": 35.5302, "feasible": 5, "of": 12,
         "note": 'INFEASIBLE in 5 of 12 episodes'},
        {"key": 'even', "plain": 'buy the same share everywhere', "policy": 'even share (`even_action`)', "return": -0.572, "cost": 17.803, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE approach known, for exactly the budget'},
        {"key": 'flat_out', "plain": 'buy every flood-prone house', "policy": 'clear the floodplain', "return": 0.0, "cost": 58.9167, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE, 2.5x the budget; nothing floods because nothing is left'},
        ),
        measured_on=MEASURED,
    ),
    EnvSpec(
        env_id="FloodCrossing-v0",
        pool='5 crossings, split by site: 4 train, 1 eval',
        cost_limit=29.0,
        title="When to put the barricade down",
        plain_title="When to close the road",
        plain_decision="Water is rising at a low crossing. Every half hour "
                       "for eighteen hours, decide whether the barricade is "
                       "down -- knowing that closing the road strands "
                       "everyone who needs it.",
        plain_score="how long the road stays open while it is under water",
        plain_budget="29 half-hours of closure, against a flood that covers "
                     "the road for about 30",
        plain_status="Closing while the road is under water AND stopping when "
                     "the allowance runs out is the best anyone has managed. "
                     "Closing whenever it is wet, which is what a crew on "
                     "site does, runs out of allowance and only stays within "
                     "it in 4 runs of 12.",
        plain_caveat="The crossings, their positions and the ground around "
                     "them are real. Whether the road is under water is "
                     "worked out by the simulator, not guessed -- but how "
                     "many cars would have driven in is not modelled at all, "
                     "so this counts time, not casualties.",
        dynamics="physics",
        summary="Eighteen hours of half-hourly open/close decisions at one of "
                "five real Austin low-water crossings, with the depth at the "
                "crossing precomputed by the same local-inertial solver for "
                "six flood sizes.",
        reward="minus the percentage of intervals the road was OPEN while "
               "under 0.15 m of water -- six inches, the figure every Texas "
               "'Turn Around Don't Drown' notice carries",
        cost="half-hours with the barricade down",
        budget="29 intervals of 36, the median time a crossing is actually "
               "under water -- so the whole flood cannot be covered and which "
               "intervals to cover is the decision",
        incumbent="`budgeted_react_action` -- close while the road is under "
                  "water and stop when the allowance runs out (-3.24, "
                  "feasible 12/12)",
        status="Not beaten, and the two words that separate the bar from the "
               "obvious rule are 'and stop'. On the held-out crossing, closing "
               "whenever it is wet scores a perfect 0.00 and clears the budget "
               "in 0 episodes of 12; checking the allowance first costs 7.18 "
               "of return and makes it feasible in all 12. Deciding at random "
               "is feasible and scores -43.29, so the band for a learned "
               "policy is wide. Read these knowing the split: ONE held-out "
               "crossing of five. The ordering is the same on the full pool "
               "(t = 30.45 against never closing, and infinite on eval), but "
               "the eval sample is a single site.",
        caveat="Five crossings, not eleven: the other six stay dry at every "
               "flood size in the library, because the terrain box fetched "
               "around them holds no creek the solver can fill, and an episode "
               "there would have no decision in it. The hydrograph is a "
               "single-peaked rise held in the channel rather than a routed "
               "event. And every wet interval counts the same, because no "
               "traffic count for these roads is on hand -- a harm model with "
               "depth or traffic in it would leave more room for a policy to "
               "be clever than this one does.",
        needs="the precomputed crossing depth library (about eight minutes of "
              "solver on first use) and the City of Austin crossing layer",
        scores={"random": {"return": -43.287, "cost": 18.3333, "feasible": 12, "of": 12},
                "incumbent": {"return": -7.176, "cost": 29.0, "feasible": 12, "of": 12}},
        ladder=(
        {"key": 'never', "plain": 'never close the road', "policy": 'never close', "return": -87.731, "cost": 0.0, "feasible": 12, "of": 12,
         "note": ''},
        {"key": 'random', "plain": 'decide at random', "policy": 'random', "return": -43.287, "cost": 18.3333, "feasible": 12, "of": 12,
         "note": 'feasible, and for a long while the best feasible thing known here'},
        {"key": 'budgeted', "plain": 'close while it is wet, and stop when the allowance runs out', "policy": 'budgeted closure (`budgeted_react_action`)', "return": -7.176, "cost": 29.0, "feasible": 12, "of": 12,
         "note": 'the best FEASIBLE rule known; it differs from the one below by a single check'},
        {"key": 'ration', "plain": 'close only when the water is deepest', "policy": 'rationed closure (`ration_action`)', "return": -2.778, "cost": 30.5833, "feasible": 1, "of": 12,
         "note": 'INFEASIBLE in 5 of 12'},
        {"key": 'react', "plain": 'close whenever it is wet', "policy": 'react on site (`react_action`)', "return": 0.0, "cost": 31.5833, "feasible": 0, "of": 12,
         "note": 'a perfect score and INFEASIBLE in 8 of 12 -- what a crew on site does, and it runs the allowance out'},
        {"key": 'always', "plain": 'keep it closed all day', "policy": 'always closed', "return": 0.0, "cost": 36.0, "feasible": 0, "of": 12,
         "note": 'INFEASIBLE in every episode'},
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
    from . import (buyout, crossings, defend, dispatch, evacuate, fire_env,
                   flood_env, pde_env, toy)

    for module in (flood_env, evacuate, fire_env, defend, buyout, dispatch,
                   crossings, pde_env, toy):
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
