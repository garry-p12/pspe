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
    "FloodEvacuate-v0": "never_action",
    "Firebreak-v0": "greedy_action",
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
            "Nothing we have trained beats a simple rule on any of these four. "
            "And on most of them the approach with the best score is one you "
            "could not afford to use."
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
            "an artefact. On FloodEvacuate-v0, nothing known is both feasible "
            "and better than doing nothing."
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
            "trigger averages 28.2 against a limit of 32 while actually "
            "spending either 0.00 or 39-48, so the mean describes no episode "
            "that exists."
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
