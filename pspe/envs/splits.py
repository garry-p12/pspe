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
                  seed: int = SPLIT_SEED) -> "np.ndarray":
    """The indices of `n` scenarios belonging to `split`.

    Deterministic: the same `n` always yields the same partition, so two runs
    reporting an eval number are reporting it on the same scenarios.

    A pool too small to split without emptying one side comes back whole, for
    both splits, rather than silently handing back one scenario -- and the
    caller is expected to say so. `SPLIT_NOTES` is where this gym says it.
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
