"""The gym, served: what you can `gymnasium.make`, and where it stands.

Three routes, and the reason they exist is that a benchmark page is usually
the most flattering document a project produces. This one is built so it
cannot be:

* `GET /gym/environments` lists the four ids with the reward, the cost, the
  budget, the incumbent, and the measured scores -- each score carrying the
  number of episodes whose summed cost cleared the limit, never a mean over
  episodes.
* `GET /gym/environment/{id}` adds the live observation and action spaces, or
  says which data is missing.
* `POST /gym/rollout` runs one episode of a named policy and returns the
  per-step reward and cost. `random` is offered because on `Firebreak-v0` it
  still has the best return of anything run, and a visitor should be able to
  watch it get there by spending 34x the budget.

Everything prose-shaped is read from `pspe.envs.catalog`, which is also what
`scripts/score_gym.py --check` tests. There is no second copy of these claims
to fall out of date.

Named `gym_routes` and not `gym`: `app.py` runs as a script with this
directory on `sys.path`, so a module called `gym.py` here would shadow the
legacy `gym` package for anything that imports it -- stable-baselines3 and
shimmy both still do -- and a dependency would silently receive a FastAPI
router where it expected an RL library.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pspe.envs import BY_ID, SPECS, catalog, register_all      # noqa: E402

router = APIRouter(prefix="/gym", tags=["gym"])

#: The strongest FEASIBLE policy known on each environment. `score_gym.py`
#: holds the same mapping and `tests/test_gym_surface.py` asserts the two
#: agree, so this cannot drift into naming a baseline that no longer exists.
INCUMBENT = {
    "FloodOperate-v0": "budget_edge_action",
    "FloodOperateFine-v0": "budget_edge_action",
    "FloodEvacuate-v0": "jit_action",
    "Firebreak-v0": "greedy_action",
    "FloodDefend-v0": "uniform_action",
    'CMDPToy-v0': 'optimal_action_vector',
    'PDEControl-dar-v0': 'proportional_action',
    'PDEControl-swe-v0': 'proportional_action',
    'PDEControl-rdf-v0': 'proportional_action',
    'FireDispatch-v0': 'biggest_fire_action',
    'FloodBuyout-v0': 'even_action',
    'FloodCrossing-v0': 'budgeted_react_action',
    'FloodGreenInfra-v0': 'most_paved_action',
    'HeatRetrofit-v0': 'exposure_first_action',
}

#: One episode of `FloodEvacuate-v0` is 48 steps of real routing lookups and
#: one of `FloodOperateFine-v0` is twelve 64x64 solver steps. Both are fine;
#: this cap only stops a malformed request running forever.
MAX_STEPS = 256


# --------------------------------------------------------------------------- #
# listing
# --------------------------------------------------------------------------- #
@router.get("/environments")
def environments(probe: bool = False) -> "dict[str, Any]":
    """Every environment, with the honest status attached to each.

    `probe=true` constructs each one to report its real spaces, which costs a
    few seconds and needs the egress table and the fire checkpoint on disk. An
    environment whose data is absent comes back `available: false` with the
    reason rather than failing the whole listing.
    """
    register_all()
    rows = catalog(probe=probe)
    for row in rows:
        row["incumbent_method"] = INCUMBENT[row["id"]]
    return {
        "environments": rows,
        # The page shows these two; the technical pair below is for anything
        # reading this API rather than the portal.
        "plain_headline": (
            "Nothing we have trained beats a simple rule on any of these "
            "fourteen. And on most of them the approach with the best score "
            "is one you could not afford to use."
        ),
        "plain_how_to_read": (
            "Each row is a way of deciding. Damage is measured against doing "
            "nothing, so 100% is what happens if nobody acts and lower is "
            "better. The approaches are split by whether you could actually "
            "pay for them."
        ),
        # The sentences a visitor needs before reading any number below.
        "headline": (
            "No environment here has been beaten by a learned policy, and the "
            "bar on each is the strongest policy that stays inside the budget "
            "-- not the baseline the environment originally named. Three of "
            "those named baselines were weak enough that a table of returns "
            "ranked a random policy above them, which reads as a result and is "
            "an artefact. FloodEvacuate-v0 was the worst case until "
            "2026-10-27: nothing known was both feasible and better than "
            "doing nothing, because its six-member ensemble sampled the band "
            "where evacuating is affordable AND useful only twice and the "
            "eval split drew neither of those members. On the sixteen-member "
            "ensemble a per-zone rule beats doing nothing by 3995 inside the "
            "budget."
        ),
        "ladder_note": (
            "Each environment's ladder lists every policy measured on it, the "
            "infeasible ones included, because the best policy on reward is "
            "usually infeasible: on FloodOperate-v0 it spends 3.4 pump-hours "
            "against a limit of 2.0."
        ),
        "feasibility_note": (
            "`feasible` counts episodes whose SUMMED cost cleared the limit. "
            "Per episode, never averaged across them: FloodEvacuate-v0's "
            "trigger at 0.55 averages 44.8 against a limit of 32 while "
            "spending either nothing or the lot, so the mean describes no "
            "episode that exists. Its incumbent is now `jit_action`, which "
            "is inside the budget in all twelve."
        ),
        "measured_on": SPECS[0].measured_on,
        "reproduce": "python3 scripts/score_gym.py --check",
    }


@router.get("/environment/{env_id}")
def environment(env_id: str) -> "dict[str, Any]":
    """One environment, probed, so the spaces come from the code and not prose."""
    if env_id not in BY_ID:
        raise HTTPException(404, f"no environment {env_id!r}; "
                                 f"have {', '.join(BY_ID)}")
    from pspe.envs.catalog import describe

    row = describe(env_id, probe=True)
    row["incumbent_method"] = INCUMBENT[env_id]
    return row


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
class RolloutRequest(BaseModel):
    env_id: str = Field(..., description="one of the ids from /gym/environments")
    policy: Literal["incumbent", "random", "nothing"] = Field(
        "incumbent",
        description="`incumbent` is the strongest feasible policy known on "
                    "that environment, `random` samples the action space, and "
                    "`nothing` holds the low end of every action. The low end "
                    "and not the midpoint: every one of these environments maps "
                    "an action to an effort by `(a + 1) / 2`, so action 0 means "
                    "half throttle. Asking for `zero` here used to spend 0.75 "
                    "of a 0.03 firebreak budget while claiming to do nothing.")
    seed: int = Field(0, ge=0, le=2**31 - 1)


class StepRow(BaseModel):
    step: int
    reward: float
    cost: float
    episode_cost: float


class RolloutResponse(BaseModel):
    env_id: str
    policy: str
    seed: int
    steps: list[StepRow]
    total_reward: float
    total_cost: float
    cost_limit: float
    feasible: bool
    #: What this one episode does and does not establish.
    caveat: str


@router.post("/rollout", response_model=RolloutResponse)
def rollout(req: RolloutRequest) -> RolloutResponse:
    """One episode, step by step, so the cost can be watched accumulating.

    The budget is spent over the episode, so a per-step cost that looks small
    is not the quantity under constraint. This returns both: `cost` per step
    and `episode_cost` as it accrues, with `feasible` read off the total.
    """
    if req.env_id not in BY_ID:
        raise HTTPException(404, f"no environment {req.env_id!r}; "
                                 f"have {', '.join(BY_ID)}")
    import gymnasium as gym

    register_all()
    try:
        env = gym.make(req.env_id)
    except Exception as exc:                                   # noqa: BLE001
        spec = BY_ID[req.env_id]
        raise HTTPException(
            503, f"{req.env_id} needs {spec.needs or 'data'} which is not "
                 f"present: {type(exc).__name__}: {exc}") from exc

    inner = env.unwrapped
    try:
        # Both seeds: `reset(seed=)` drives the environment's stream and
        # `action_space.seed()` the one `sample()` draws from. Seeding one
        # leaves a random rollout unreproducible.
        env.action_space.seed(req.seed)
        obs, _ = env.reset(seed=req.seed)

        name = INCUMBENT[req.env_id]
        if req.policy == "incumbent":
            policy = getattr(inner, name)
        elif req.policy == "random":
            policy = lambda _o: env.action_space.sample()       # noqa: E731
        else:
            low = np.full(env.action_space.shape, -1.0, dtype=np.float32)
            policy = lambda _o: low                             # noqa: E731

        rows: "list[StepRow]" = []
        total_r = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total_r += float(reward)
            rows.append(StepRow(step=len(rows) + 1, reward=float(reward),
                                cost=float(info["cost"]),
                                episode_cost=float(info["episode_cost"])))
            if terminated or truncated or len(rows) >= MAX_STEPS:
                break

        limit = float(info["cost_limit"])
        total_c = rows[-1].episode_cost
        spec = BY_ID[req.env_id]
        return RolloutResponse(
            env_id=req.env_id, policy=req.policy, seed=req.seed, steps=rows,
            total_reward=total_r, total_cost=total_c, cost_limit=limit,
            feasible=total_c <= limit,
            caveat=spec.caveat)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# play
# --------------------------------------------------------------------------- #
"""Driving a real environment by hand, one decision at a time.

The point of this surface is that it is **not a toy**. A step of
`FloodOperate-v0` is 0.04 s of the same local-inertial solver the research
runs, so the water a visitor watches spread is the water the benchmark
measured -- not an illustration of it, and not a simplified model whose
numbers would have to be disclaimed. The scene the portal draws is a
downsampled view of the environment's own state.

What a visitor chooses is deliberately coarse: four pump settings rather than
a continuous dial. The underlying action space is continuous and a policy may
use all of it; these are the settings an operator would actually be offered,
and a control that cannot be worked by hand teaches nobody anything.
"""

import uuid

#: Live sessions. Small and bounded: this is a demonstration surface, not a
#: service anyone should build on, and an unbounded dict of live environments
#: is a memory leak with a polite name.
_SESSIONS: "dict[str, dict[str, Any]]" = {}
MAX_SESSIONS = 24
SCENE_N = 16                      # the scene is drawn at 16x16, whatever the grid


def _control_spec(env_id: str) -> "list[dict[str, Any]]":
    """The choices a person is offered, and the action each one means.

    Every option carries the exact action vector it sends, so the page has no
    opinion about what a choice means -- it posts back a label and the service
    looks it up here.
    """
    if env_id.startswith("FloodOperate"):
        return [
            {"name": "Gate", "options": [
                {"label": "closed", "action": [-1.0]},
                {"label": "half", "action": [0.0]},
                {"label": "open", "action": [1.0]}],
             "default": 2},
            {"name": "Pump", "options": [
                {"label": "off", "action": [-1.0]},
                {"label": "quarter", "action": [-0.5]},
                {"label": "half", "action": [0.0]},
                {"label": "full", "action": [1.0]}],
             "default": 0},
        ]
    if env_id == "FloodDefend-v0":
        return [{"name": "Build here", "options": [
            {"label": "nothing", "action": [-1.0]},
            {"label": "1 m", "action": [-1.0 + 2.0 / 3.0]},
            {"label": "2 m", "action": [-1.0 + 4.0 / 3.0]},
            {"label": "3 m", "action": [1.0]}], "default": 0}]
    if env_id == "FloodEvacuate-v0":
        # Two options, because the trigger family collapses to exactly these
        # two behaviours on this ensemble -- which is the environment's own
        # finding and not a simplification made for the page.
        return [{"name": "Order out", "options": [
            {"label": "nobody", "action": [-1.0] * 8},
            {"label": "everybody", "action": [1.0] * 8}], "default": 0}]
    if env_id == "Firebreak-v0":
        return [{"name": "Crews", "options": [
            {"label": "stand down", "action": None, "policy": "nothing"},
            {"label": "spread evenly", "action": None, "policy": "uniform_action"},
            {"label": "where it burns", "action": None, "policy": "greedy_action"}],
            "default": 1}]
    if env_id == "FireDispatch-v0":
        return [{"name": "Crews", "options": [
            {"label": "stand down", "action": None, "policy": "nothing"},
            {"label": "split evenly", "action": None, "policy": "even_action"},
            {"label": "the biggest fire", "action": None, "policy": "biggest_fire_action"},
            {"label": "where the people are", "action": None,
             "policy": "most_people_action"}], "default": 2}]
    if env_id == "HeatRetrofit-v0":
        return [{"name": "Cool this block", "options": [
            {"label": "none", "action": [-1.0]},
            {"label": "a quarter", "action": [-0.5]},
            {"label": "half", "action": [0.0]},
            {"label": "all of it", "action": [1.0]}], "default": 0}]
    if env_id == "FloodGreenInfra-v0":
        return [{"name": "Retrofit here", "options": [
            {"label": "none", "action": [-1.0]},
            {"label": "a quarter", "action": [-0.5]},
            {"label": "half", "action": [0.0]},
            {"label": "all of it", "action": [1.0]}], "default": 0}]
    if env_id == "FloodBuyout-v0":
        return [{"name": "Buy here", "options": [
            {"label": "none", "action": [-1.0]},
            {"label": "a quarter", "action": [-0.5]},
            {"label": "half", "action": [0.0]},
            {"label": "all of them", "action": [1.0]}], "default": 0}]
    if env_id == "FloodCrossing-v0":
        return [{"name": "Barricade", "options": [
            {"label": "up, road open", "action": [-1.0]},
            {"label": "down, road closed", "action": [1.0]}], "default": 0}]
    if env_id == "CMDPToy-v0":
        # The action IS the number, so the options are the number. 0.40 is
        # the right answer and it is offered plainly -- hiding it would make a
        # guessing game out of the one environment whose point is that the
        # answer is known.
        return [{"name": "Dial", "options": [
            {"label": "0.00", "action": [0.0]},
            {"label": "0.20", "action": [0.2]},
            {"label": "0.40 (the answer)", "action": [0.4]},
            {"label": "0.80", "action": [0.8]}], "default": 0}]
    if env_id.startswith("PDEControl-"):
        return [{"name": "Dials", "options": [
            {"label": "leave alone", "action": None, "policy": "zero_action"},
            {"label": "gentle", "action": None, "policy": "proportional:0.3"},
            {"label": "firm", "action": None, "policy": "proportional:0.6"},
            {"label": "hard", "action": None, "policy": "proportional:1.0"}],
            "default": 1}]
    return []


def _scene(env_id: str, inner, obs: "np.ndarray") -> "dict[str, Any]":
    """The environment's own state, downsampled to something drawable.

    `ground` is relief in 0..1, `water` is the quantity that spreads -- depth
    in metres for a flood, burning fraction for a fire -- `protect` is what is
    being defended and `marker` is the channel, the bank being built, or the
    ignition. Each is reduced to `SCENE_N` by taking a block MAXIMUM for water
    and a mean for the rest: a flood one cell wide is still a flood, and
    averaging it away would draw a drier map than the solver computed.
    """
    def down(a: "np.ndarray", how: str = "mean") -> "list[list[float]]":
        n = SCENE_N
        a = np.asarray(a, dtype="float32")
        h, w = a.shape
        if h < n or w < n:
            # Smaller than the scene: repeat, do not reshape. `FloodEvacuate-v0`
            # is an 8x8 field and the block-reshape below would have raised on
            # it -- which it did, as a 500 on that environment's first frame.
            a = np.repeat(np.repeat(a, int(np.ceil(n / h)), axis=0),
                          int(np.ceil(n / w)), axis=1)
            h, w = a.shape
        rs, cs = max(h // n, 1), max(w // n, 1)
        a = a[:rs * n, :cs * n].reshape(n, rs, n, cs)
        out = a.max(axis=(1, 3)) if how == "max" else a.mean(axis=(1, 3))
        return [[round(float(v), 4) for v in row] for row in out]

    if env_id.startswith("FloodOperate"):
        core = inner.core
        return {
            "ground": down(obs[1]), "water": down(obs[0], "max"),
            "protect": down(obs[2]), "marker": down(core.river.cpu().numpy().astype("float32")),
            "legend": {"ground": "protected ground", "water": "water",
                       "marker": "river"},
            "water_unit": "m", "water_scale": 2.0,
        }
    if env_id == "FloodDefend-v0":
        core = inner.core
        i = core.idx[0].item()
        built = core.seg_mask[i].any(dim=0).cpu().numpy().astype("float32")
        depth = (core.last_depth[0].cpu().numpy() if core.last_depth is not None
                 else np.zeros_like(obs[0]))
        return {
            "ground": down(obs[0]), "water": down(depth, "max"),
            "protect": down(obs[1]), "marker": down(built),
            "legend": {"ground": "ground at risk", "water": "water",
                       "marker": "where a levee can go"},
            "water_unit": "m", "water_scale": 4.0,
        }
    if env_id == "FloodEvacuate-v0":
        # Eight zones, not a map. The egress table is per zone and per slot, so
        # there is no grid to draw -- each zone becomes a band, which is an
        # honest picture of an eight-element decision and not a pretend
        # aerial view.
        core = inner.core
        k = core.K
        pop = core.pop.cpu().numpy().astype("float32")
        pop = pop / max(float(pop.max()), 1e-6)
        order = core.under_order[0].cpu().numpy().astype("float32")
        # Blue is water everywhere else on this page, so blue is water here
        # too: the depth the egress table records at each zone this slot. The
        # first version drew "people still in the zone" in the water colour,
        # which made a full town read as a flood.
        slot = min(core.t, core.horizon - 1)
        member = int(core._member[0].item())
        depth = core.table.exposure[member, :, slot].astype("float32")
        band = lambda v: np.repeat(np.asarray(v, dtype="float32")[:, None], k, axis=1)
        return {
            "ground": down(band(pop)), "water": down(band(depth), "max"),
            "protect": down(band((pop > 0).astype("float32"))),
            "marker": down(band(order)),
            "legend": {"ground": "how many live there",
                       "water": "water in the zone",
                       "marker": "told to leave"},
            # The deepest water anywhere in this egress table is 0.181 m. Drawn
            # on the flood environments' metre scale it is two pixels and
            # invisible, so the column is scaled to what this environment
            # actually reaches -- and the page prints the scale, because an
            # unlabelled rescale is how an ankle-deep flood comes to look like
            # a catastrophe.
            "water_unit": "m", "water_scale": 0.2,
        }
    if env_id == "Firebreak-v0":
        core = inner.core
        return {
            "ground": down(obs[core.ndvi]), "water": down(obs[0], "max"),
            "protect": down(obs[core.pop]), "marker": down(obs[0], "max"),
            "legend": {"ground": "vegetation", "water": "fire",
                       "marker": "burning"},
            "water_unit": "", "water_scale": 1.0,
        }
    if env_id == "FireDispatch-v0":
        # Four fires, laid out as four quadrants, because a dispatcher sees
        # all the incidents at once and that is the whole decision.
        fires = obs[:-1]
        n = fires.shape[-1]
        half = n // 2
        import numpy as _np
        tile = _np.zeros((n, n), dtype="float32")
        for i, f in enumerate(fires[:4]):
            r, c = (i // 2) * half, (i % 2) * half
            tile[r:r + half, c:c + half] = f[::2, ::2][:half, :half]
        pop = inner.core._fire.pop
        people = inner.core._state[:, pop].mean(dim=0).cpu().numpy()
        return {
            "ground": down(people), "water": down(tile, "max"),
            "protect": down((people > people.mean()).astype("float32")),
            "marker": down(tile, "max"),
            "legend": {"ground": "people", "water": "fire, four at once",
                       "marker": "burning"},
            "water_unit": "", "water_scale": 1.0,
        }
    if env_id == "FloodBuyout-v0":
        core = inner.core
        depth = (core.last_depth[0].cpu().numpy() if core.last_depth is not None
                 else np.zeros_like(obs[0]))
        return {
            "ground": down(obs[0]), "water": down(depth, "max"),
            "protect": down(obs[1]), "marker": down(obs[2]),
            "legend": {"ground": "ground", "water": "water",
                       "marker": "houses"},
            "water_unit": "m", "water_scale": 4.0,
        }
    if env_id == "HeatRetrofit-v0":
        core = inner.core
        hot = (core.last_exposure[0].cpu().numpy()
               if core.last_exposure is not None else obs[0])
        return {
            "ground": down(obs[1]), "water": down(hot, "max"),
            "protect": down(obs[2]), "marker": down(obs[3]),
            "legend": {"ground": "where people are",
                       "water": "hours over 35 C", "marker": "cooled"},
            "water_unit": "", "water_scale": 1.0,
        }
    if env_id == "FloodGreenInfra-v0":
        core = inner.core
        depth = (core.last_depth[0].cpu().numpy() if core.last_depth is not None
                 else np.zeros_like(obs[0]))
        return {
            "ground": down(obs[0]), "water": down(depth, "max"),
            "protect": down(obs[1]), "marker": down(obs[3]),
            "legend": {"ground": "ground", "water": "water",
                       "marker": "retrofitted"},
            "water_unit": "m", "water_scale": 1.0,
        }
    if env_id == "FloodCrossing-v0":
        # The real ground around this crossing, from the cached 3DEP tile the
        # depth library was solved on. Schematic would have been easier and
        # would have been the only drawing on this page that stands for
        # nothing.
        from pspe.envs.crossings import HALF_DEG, crossing_sites  # noqa: PLC0415
        from pspe.simulate.real.anywhere import build_domain      # noqa: PLC0415

        name = inner.names[inner.site]
        lat, lon = next((la, lo) for n, la, lo in crossing_sites() if n == name)
        dom = build_domain(lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG,
                           lat + HALF_DEG, dx=30.0, name=name, source="auto")
        z = dom.z.astype("float32")
        z = z - z.min()
        h, w = z.shape
        half = min(h, w) // 3
        r0, c0 = h // 2 - half, w // 2 - half
        patch = z[r0:r0 + 2 * half, c0:c0 + 2 * half]
        now = float(inner.curve[min(inner.t, inner.slots - 1)])
        # Water sits where the ground is below the crossing's depth, which is
        # what "the road is under N metres" means on this terrain.
        low = patch.min()
        water = np.clip((low + now) - patch, 0.0, None)
        road = (patch <= low + 0.5).astype("float32")
        return {
            "ground": down(patch), "water": down(water, "max"),
            "protect": down(road), "marker": down(road),
            "legend": {"ground": "ground around the crossing",
                       "water": "water", "marker": "the dip in the road"},
            "water_unit": "m", "water_scale": 2.0,
        }
    if env_id == "CMDPToy-v0":
        # One number against the known answer. There is no field to draw --
        # the observation is a constant -- so the drawing is the comparison:
        # your dial on the left, the optimum on the right.
        import numpy as _np
        n = 16
        grid = _np.zeros((n, n), dtype="float32")
        chosen = float(getattr(inner, "_last_action", 0.0))
        grid[:, : n // 2] = max(chosen, 0.0)
        grid[:, n // 2:] = inner.optimal_action
        mark = _np.zeros((n, n), dtype="float32")
        mark[:, n // 2:] = 1.0
        return {
            "ground": down(grid), "water": down(_np.zeros((n, n), "float32")),
            "protect": down(_np.zeros((n, n), "float32")), "marker": down(mark),
            "legend": {"ground": "the dial you set", "water": "",
                       "marker": "the right answer"},
            "water_unit": "", "water_scale": 1.0,
        }
    if env_id.startswith("PDEControl-"):
        task = inner.task
        field = obs[task.tracked_channel]
        over = np.clip(field - task.u_max, 0.0, None)
        basis = inner._basis.basis.sum(dim=0).cpu().numpy()
        return {
            "ground": down(field - field.min()), "water": down(over, "max"),
            "protect": down((basis > 0.3 * basis.max()).astype("float32")),
            "marker": down((basis > 0.3 * basis.max()).astype("float32")),
            "legend": {"ground": "the field", "water": "above the ceiling",
                       "marker": "where the dials reach"},
            "water_unit": "", "water_scale": float(max(over.max(), 0.2)),
        }
    return {}


def _stats(env_id: str, inner, sess) -> "list[dict[str, Any]]":
    """The three numbers along the top: where you are, what you have spent,
    and how it is going."""
    spent, limit = sess["spent"], float(inner.task.cost_limit)
    horizon = sess["horizon"]
    unit = {"FloodOperate-v0": "pump-hours", "FloodOperateFine-v0": "pump-hours",
            "FloodEvacuate-v0": "town-intervals",
            "Firebreak-v0": "of the patch", "FloodDefend-v0": "dollars",
            "FireDispatch-v0": "of a patch", "FloodBuyout-v0": "houses",
            "FloodCrossing-v0": "closed half-hours",
            "CMDPToy-v0": "units",
            "FloodGreenInfra-v0": "m2 retrofitted",
            "HeatRetrofit-v0": "m2 cooled"}.get(env_id, "units of effort")
    spent_text = (f"${spent:,.0f} of ${limit:,.0f}" if unit == "dollars"
                  else f"{spent:,.3g} of {limit:,.3g} {unit}")
    step_word = {"FloodOperate-v0": "Half hour", "FloodOperateFine-v0": "Half hour",
                 "FloodEvacuate-v0": "Half hour", "Firebreak-v0": "Crew day",
                 "FloodDefend-v0": "Stretch", "FireDispatch-v0": "Crew day",
                 "FloodBuyout-v0": "Block", "FloodCrossing-v0": "Half hour",
                 "CMDPToy-v0": "Step",
                 "FloodGreenInfra-v0": "Catchment",
                 "HeatRetrofit-v0": "Block"}.get(env_id, "Turn")
    return [
        {"label": "time", "value": f"{step_word} {min(sess['t'] + 1, horizon)} of {horizon}"},
        {"label": "spent", "value": spent_text,
         "fraction": round(min(spent / max(limit, 1e-9), 1.5), 4),
         "over": spent > limit},
        {"label": "damage", "value": f"{sess['damage']}%"},
    ]


def _damage_pct(env_id: str, total_return: float) -> int:
    """This run's return on the same scale the ladder uses.

    Both scales come from `EnvSpec.harm_pct`, which is the point of calling it
    rather than repeating the arithmetic: the denominator is doing nothing's
    return, so 100 means "no better than not acting" and a live run is placed
    against the rungs like for like -- with the one caveat the page states,
    that the rungs are twelve episodes and this is one.
    """
    return BY_ID[env_id].harm_pct(total_return)


def _evict() -> None:
    while len(_SESSIONS) >= MAX_SESSIONS:
        old = min(_SESSIONS, key=lambda k: _SESSIONS[k]["touched"])
        try:
            _SESSIONS[old]["env"].close()
        except Exception:                                  # noqa: BLE001
            pass
        _SESSIONS.pop(old, None)


class PlayStart(BaseModel):
    env_id: str
    seed: int = Field(0, ge=0, le=2**31 - 1)


class PlayStep(BaseModel):
    session: str
    #: One label per control group, in the order `/gym/play/start` listed them.
    choices: "list[str]"
    #: Run the remaining steps with the same choices.
    to_end: bool = False


def _session_payload(sess, *, done: bool = False) -> "dict[str, Any]":
    env_id, inner, obs = sess["env_id"], sess["env"].unwrapped, sess["obs"]
    out = {
        "session": sess["id"], "env_id": env_id, "t": sess["t"],
        "horizon": sess["horizon"], "done": done,
        "scene": _scene(env_id, inner, obs),
        "stats": _stats(env_id, inner, sess),
        "cost_limit": float(inner.task.cost_limit),
        "spent": sess["spent"], "over_budget": sess["spent"] > float(inner.task.cost_limit),
    }
    if done:
        out["result"] = {
            "damage": sess["damage"],
            "return": round(sess["return"], 3),
            "spent": sess["spent"],
            "feasible": sess["spent"] <= float(inner.task.cost_limit),
            "note": "One episode, against ladder rows that are each twelve.",
        }
    return out


@router.post("/play/start")
def play_start(req: PlayStart) -> "dict[str, Any]":
    """Begin a hand-driven episode of the real environment."""
    if req.env_id not in BY_ID:
        raise HTTPException(404, f"no environment {req.env_id!r}")
    controls = _control_spec(req.env_id)
    if not controls:
        raise HTTPException(400, f"{req.env_id} has no hand controls")
    import gymnasium as gym

    register_all()
    _evict()
    try:
        env = gym.make(req.env_id)
    except Exception as exc:                               # noqa: BLE001
        spec = BY_ID[req.env_id]
        raise HTTPException(503, f"{req.env_id} needs {spec.needs or 'data'}: "
                                 f"{type(exc).__name__}") from exc
    env.action_space.seed(req.seed)
    obs, _ = env.reset(seed=req.seed)
    inner = env.unwrapped
    horizon = (getattr(inner, "slots", None)
               or getattr(getattr(inner, "core", inner), "horizon", None)
               or getattr(getattr(inner, "core", inner), "K", None)
               or getattr(inner, "_core", inner) and
               getattr(getattr(inner, "_core", inner), "horizon", None)
               or 12)
    sess = {"id": uuid.uuid4().hex[:16], "env": env, "env_id": req.env_id,
            "obs": obs, "t": 0, "spent": 0.0, "return": 0.0, "damage": 0,
            "horizon": int(horizon), "touched": time.time()}
    _SESSIONS[sess["id"]] = sess
    out = _session_payload(sess)
    out["controls"] = controls
    out["plain"] = BY_ID[req.env_id].as_dict()["plain"]
    return out


@router.post("/play/step")
def play_step(req: PlayStep) -> "dict[str, Any]":
    """Take one decision -- or the rest of them -- and return the real state."""
    sess = _SESSIONS.get(req.session)
    if sess is None:
        raise HTTPException(404, "that run has expired; start another")
    sess["touched"] = time.time()
    env, inner = sess["env"], sess["env"].unwrapped
    groups = _control_spec(sess["env_id"])

    parts, policy = [], None
    for group, label in zip(groups, req.choices):
        opt = next((o for o in group["options"] if o["label"] == label), None)
        if opt is None:
            raise HTTPException(400, f"{label!r} is not an option for {group['name']}")
        if opt.get("policy"):
            policy = opt["policy"]
        else:
            parts.extend(opt["action"])

    def action_now() -> "np.ndarray":
        """Whatever the chosen options mean, as an action.

        A policy option names a method on the environment, optionally with one
        parameter after a colon -- `proportional:0.6` is the proportional
        controller at gain 0.6. The page never computes an action itself; it
        posts labels and this resolves them, so the two cannot disagree about
        what "firm" means.
        """
        if policy is None:
            return np.asarray(parts, dtype=np.float32)
        if policy == "nothing":
            return np.full(env.action_space.shape, -1.0, dtype=np.float32)
        name, _, arg = policy.partition(":")
        fn = getattr(inner, name, None) or getattr(inner, name + "_action", None)
        if fn is None:
            raise HTTPException(500, f"{sess['env_id']} has no {name!r}")
        out = fn(sess["obs"], float(arg)) if arg else fn(sess["obs"])
        return np.asarray(out, dtype=np.float32)

    done = False
    for _ in range(sess["horizon"] if req.to_end else 1):
        obs, reward, terminated, truncated, info = env.step(action_now())
        sess["obs"] = obs
        sess["t"] += 1
        sess["spent"] += float(info["cost"])
        sess["return"] += float(reward)
        sess["damage"] = _damage_pct(sess["env_id"], sess["return"])
        if terminated or truncated:
            done = True
            break
    return _session_payload(sess, done=done)


@router.post("/play/stop")
def play_stop(session: str) -> "dict[str, str]":
    sess = _SESSIONS.pop(session, None)
    if sess is not None:
        try:
            sess["env"].close()
        except Exception:                                  # noqa: BLE001
            pass
    return {"status": "stopped"}
