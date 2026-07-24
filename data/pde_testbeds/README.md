# PDE testbed data

Generated locally, never downloaded:

```bash
python scripts/generate_data.py testbed=dar     # diffusion-advection-reaction
python scripts/generate_data.py testbed=swe     # shallow-water-like transport
python scripts/generate_data.py testbed=rdf     # reaction-diffusion front
```

Each writes `<testbed>_<grid>.npz` — the resolution is in the filename so a
32×32 ablation run cannot silently overwrite (or be loaded in place of) a 64×64
dataset. Trainers call `ensure_dataset(testbed, grid=...)`, which generates the
missing resolution rather than failing.

Contents:

| array | shape | meaning |
|---|---|---|
| `states` | `(N, T+1, C, H, W)` | solver trajectory |
| `controls` | `(N, T, 1, H, W)` | intervention field applied at each step |

Trajectories are generated **with the control channel excited** (random
actuator amplitudes, re-drawn every `control_hold` steps). A surrogate fitted
to uncontrolled data cannot answer the planner's counterfactual, so this is not
optional.

Channel counts: `dar` 1 (`u`), `swe` 3 (`h, u, v`), `rdf` 2 (`u, v`).

`.npz` files are gitignored; regenerate rather than commit them. At the default
size (256 trajectories, 24 steps, 64x64) each file is roughly 60-180 MB.
