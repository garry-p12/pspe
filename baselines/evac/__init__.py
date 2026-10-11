"""The Onion Creek evacuation experiments, and the data they stand on.

An evacuation decision needs three things this package keeps separate:

* **the flood** -- an ensemble over forcing multiplier, each member solved by
  `FloodSolver` on the real DEM and sampled onto the real road graph. Built by
  `prepare.py`, saved as `schedules.npy` (M, T, E) of edge depths in metres.
* **the network** -- the Onion Creek road graph as a GeoJSON FeatureCollection,
  saved as `roads.json`. Also written by `prepare.py`.
* **the egress table** -- for every (member, zone, departure slot), whether a
  route out still exists, from real time-dependent routing. Expensive (~8
  minutes), deterministic, and cached by `table.py`.

The experiments then read the cache and cost seconds instead of hours.

Data lives in `$PSPE_EVAC_DATA`, defaulting to `data/evac/` in the repository.
The artefacts are not committed -- `schedules.npy` is 7.6 MB of solver output
and regenerating it is the honest way to reproduce a result that depends on
it. `prepare.py` writes both inputs; every script here fails with the command
to run rather than a stack trace when they are absent.
"""
