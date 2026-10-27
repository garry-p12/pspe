"""Train and evaluate on different scenarios, or you have measured nothing.

Every environment in this gym that draws episodes from a finite pool drew
them from the WHOLE pool, at training time and at reporting time alike. A
policy fitted to `FloodDefend-v0` saw all 35 windows and was then scored on
all 35 windows, and nothing anywhere would have noticed a policy that had
simply memorised them. On a 33-window environment that is not a hypothetical
failure mode; it is the expected one.

This module is the fix: one deterministic partition per pool, held fixed
forever, so a number reported on `eval` is a number about scenarios the
policy was never fitted to.

## The rules

**The partition never moves.** It is a fixed permutation under `SPLIT_SEED`,
not a random draw per run and not a function of anything a caller passes. A
split that moves is worse than no split, because two runs reporting "eval"
would not be reporting the same thing.

**`train` is the default.** `gymnasium.make("FloodDefend-v0")` gives the
training pool, because the common case is fitting something; asking for
`split="eval"` is a deliberate act and that is the point. The catalog's
numbers are measured on `eval`.

**Where a split cannot be meaningful, it says so.** `FloodEvacuate-v0` has
six ensemble members, three of which never flood anything. A quarter of six
is one. Partitioning it is arithmetic, not a generalisation test, and
`SPLIT_NOTES` records that rather than letting the word "eval" imply more
than it can carry.
"""

from __future__ import annotations

import numpy as np

#: Fixed for the life of the project. Changing it invalidates every number
#: ever reported on a split, which is why it is a constant and not an
#: argument.
SPLIT_SEED = 20261018

#: Share of each pool held out. A quarter: large enough that an eval figure
#: is not one lucky scenario, small enough to leave a usable training set on
#: the thin pools.
EVAL_FRACTION = 0.25

SPLITS = ("train", "eval", "all")

#: Pools where the word "eval" would promise more than the support can give.
SPLIT_NOTES = {
    "FloodEvacuate-v0":
        "6 ensemble members, 3 of which never flood anything. A quarter of "
        "six is one member, so this is a partition and not a generalisation "
        "test; the environment's support is the finding, not its split.",
    "FloodCrossing-v0":
        "5 crossings, split BY SITE so an eval episode is a crossing the "
        "policy has never seen -- which is the generalisation that matters "
        "here -- but it is one site held out of five.",
}


def split_indices(n: int, split: str = "train", *,
                  eval_fraction: float = EVAL_FRACTION,
                  seed: int = SPLIT_SEED,
                  strata: "np.ndarray | None" = None) -> "np.ndarray":
    """The indices of `n` scenarios belonging to `split`.

    Deterministic: the same `n` always yields the same partition, so two runs
    reporting an eval number are reporting it on the same scenarios.

    A pool too small to split without emptying one side comes back whole, for
    both splits, rather than silently handing back one scenario -- and the
    caller is expected to say so. `SPLIT_NOTES` is where this gym says it.

    `strata` is a per-scenario severity key. With it the partition walks the
    severity ORDER and takes every k-th scenario for eval, so both sides span
    the range; without it the partition is the random permutation it has
    always been. This exists because a random partition of a thin pool can
    hand one side a degenerate slice and nothing notices. On
    `FloodEvacuate-v0` the six forecast members have signal maxima from 0.3312
    to 1.0, the random split put {0, 4} in eval, and the two eval members span
    0.3312 and 0.8392 with NOTHING in between -- so eval returns came out
    near-binary (-3726.85 under the budget, 0.00 over it) and twelve seeds
    carried two distinct outcomes. A stratified split cannot fix a pool of six
    being a pool of six, but it can stop the partition from concentrating the
    severity on one side.

    Stratifying is opt-in per pool rather than global, and deliberately so: it
    changes which scenarios are held out, which invalidates every number
    previously recorded on that pool's splits. A pool whose random partition
    already spans its range gains nothing and would pay that cost for free.

    What this does NOT fix is worth stating, because it was built in the belief
    that it would. The three pools in this gym with degenerate eval support are
    degenerate for three different reasons, and severity stratification
    addresses none of them: `FloodEvacuate-v0` has six members so any quarter
    is two members whatever order they are in; `FloodBuyout-v0` has
    thirty-three windows and a perfectly good partition, but its incumbent
    scores 0.0 on eleven of twelve eval scenarios because there is no harm
    left to prevent; and `FloodCrossing-v0`'s returns are quantised to three
    levels. Thin support, uninformative scenarios and a coarse reward are not
    the same defect. The measure that separates them is per-scenario
    DISCRIMINATION -- the gap between doing nothing and the best known policy
    -- which `scripts/gym_discrimination.py` reports.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if split == "all":
        return np.arange(n)
    n_eval = int(round(n * eval_fraction))
    if n < 2 or n_eval < 1 or n - n_eval < 1:
        # Nothing to partition. Returning the whole pool for both sides is
        # honest; returning an empty eval set would make every eval number
        # undefined, and returning one scenario would make it noise wearing
        # the word "held out".
        return np.arange(n)
    if strata is not None:
        key = np.asarray(strata, dtype=np.float64).reshape(-1)
        if key.shape[0] != n:
            raise ValueError(
                f"strata has {key.shape[0]} entries for a pool of {n}")
        # Ties broken by the same permutation the unstratified path uses, so
        # equal strata give a RANDOM partition rather than index order. It is
        # not the same partition as the unstratified path -- that one takes
        # the first `n_eval` of a permutation and this one walks the sorted
        # order in steps -- so stratifying is a change of partition even when
        # the strata carry no information.
        jitter = np.random.default_rng(seed).permutation(n)
        order = np.lexsort((jitter, key))
        step = max(1, int(round(n / n_eval)))
        picked = np.sort(order[step // 2::step][:n_eval])
        if split == "eval":
            return picked
        return np.sort(np.setdiff1d(np.arange(n), picked))
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[n_eval:] if split == "train" else order[:n_eval])


def describe_split(n: int, split: str, env_id: str = "") -> str:
    """One line saying what this split actually is, for a docstring or a page."""
    idx = split_indices(n, split)
    note = SPLIT_NOTES.get(env_id, "")
    body = (f"{len(idx)} of {n} scenarios" if split != "all"
            else f"all {n} scenarios")
    if len(idx) == n and split != "all":
        body += " (pool too small to partition; train and eval are the same)"
    return f"{body}{'. ' + note if note else ''}"
