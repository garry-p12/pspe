"""Cache the spatial panel behind figA3, and measure the inflation behind §5.8.

Reproduces the §5.3 setting exactly -- the same occlusion generator, the same
driver normalisation the checkpoint was trained under, and the same three ways
of filling the hidden cells -- so the appendix figure never needs the dataset
or a GPU, and the number it quotes can be re-derived.

    python scripts/build_fire_panel.py                 # panel + inflation table
    python scripts/build_fire_panel.py --no-measure    # panel only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_ndws_partial import Perceive, cloud_mask  # noqa: E402
from pspe.simulate.real import (  # noqa: E402
    NDWSConfig, driver_stats, load_ndws, normalize_drivers,
)

GRID, SEED = 64, 0
CKPT = ROOT / "runs" / "ndws_partial_local" / "r0.15_seed_0" / "perceive.pt"
OUT = ROOT / "runs" / "ndws_partial_local" / "panel.npz"


def estimate(model, prev, drivers, hide) -> torch.Tensor:
    visible = prev * (~hide)
    with torch.no_grad():
        return torch.cat([
            torch.sigmoid(model(torch.cat([
                visible[b:b + 128][:, None],
                (~hide[b:b + 128])[:, None].float(),
                drivers[b:b + 128],
            ], 1)))[:, 0]
            for b in range(0, len(prev), 128)
        ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.6, help="occlusion for the panel")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--no-measure", action="store_true")
    args = ap.parse_args()

    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, grid=GRID))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, grid=GRID))
    val["drivers"] = normalize_drivers(val["drivers"], driver_stats(train))

    prev = torch.as_tensor(np.where(val["states"][:, 0] >= 0, val["states"][:, 0], 0.0),
                           dtype=torch.float32)[:, 0]
    drivers = torch.as_tensor(val["drivers"], dtype=torch.float32)
    model = Perceive(2 + drivers.shape[1])
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # Same generator seed as the experiment, so the hidden region is the one the
    # reported decision numbers were measured under.
    hide = cloud_mask(len(prev), GRID, args.rate, torch.Generator().manual_seed(SEED + 900))
    est = estimate(model, prev, drivers, hide)
    visible = prev * (~hide)

    # A patch where the story is legible: a real front, with the cloud actually
    # over a good part of it -- the case the estimator exists for.
    fire = prev > 0.5
    burning = fire.flatten(1).sum(1)
    covered = (fire & hide).flatten(1).sum(1)
    frac = covered / burning.clamp(min=1)
    cand = torch.nonzero((burning > 45) & (burning < 400) & (frac > 0.45) & (frac < 0.8))[:, 0]
    # The MEDIAN case by hidden-cell recall. Taking the best would flatter the
    # estimator, and the point of the figure does not need flattery.
    recov = sorted((float(((est[i] > 0.5) & fire[i] & hide[i]).sum() / covered[i]), int(i))
                   for i in cand)
    recall, i = recov[len(recov) // 2]

    mean_vis = visible[i].sum() / (~hide[i]).sum().clamp(min=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        truth=prev[i].numpy(), hide=hide[i].numpy(), visible=visible[i].numpy(),
        est=est[i].numpy(), persist=torch.where(hide[i], mean_vis, visible[i]).numpy(),
        rate=np.float32(args.rate), index=np.int32(i), recall=np.float32(recall),
        burning=np.int32(burning[i]), covered=np.int32(covered[i]),
    )
    over = float(((est[i] > 0.5) & hide[i]).sum() / covered[i].clamp(min=1))
    print(f"wrote {OUT.relative_to(ROOT)}  sample {i} of {len(cand)} candidates: "
          f"{int(burning[i])} burning, {int(covered[i])} hidden, "
          f"recall {recall:.2f}, area x{over:.1f}", flush=True)

    if args.no_measure:
        return

    # Section 5.8: the estimator's area inflation, and the fact that the SOFT
    # mass -- which is what the planner consumes -- inflates as the thresholded
    # area tightens. The decision loss tracks the soft column, not the hard one.
    print(f"\n{'occlusion':>9} {'patches':>8} {'hard ratio':>11} {'soft ratio':>11}")
    for rate in (0.15, 0.35, 0.6):
        h = cloud_mask(len(prev), GRID, rate, torch.Generator().manual_seed(SEED + 900))
        e = estimate(model, prev, drivers, h)
        true_a = ((prev > 0.5) & h).flatten(1).sum(1).float()
        hard_a = ((e > 0.5) & h).flatten(1).sum(1).float()
        soft_a = (e * h).flatten(1).sum(1)
        keep = true_a > 5                      # patches with hidden fire to speak of
        print(f"{rate:>9} {int(keep.sum()):>8} "
              f"{(hard_a[keep] / true_a[keep]).median():>11.2f} "
              f"{(soft_a[keep] / true_a[keep]).median():>11.2f}", flush=True)


if __name__ == "__main__":
    main()
