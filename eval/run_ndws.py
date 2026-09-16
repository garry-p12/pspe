#!/usr/bin/env python
"""Phase R3: the surrogate on observed wildfire spread, not simulated dynamics.

    python scripts/download_ndws.py                    # once, needs a Kaggle token
    python eval/run_ndws.py --epochs 20 --seed 0

Predicts the day-t+1 fire mask from the day-t mask plus eleven explanatory
channels (terrain, weather, drought, vegetation, population). Every other
dynamics number in this repository is scored against a solver written in this
repository; this one is scored against what the fire actually did.

Metric choice matters more than usual here. Next-day fire is a rare-positive
problem — around 1% of cells ignite — so accuracy and MSE are both dominated by
correctly predicting "no fire" everywhere, which is why this reports
**AUC-PR** (the published metric for this dataset) alongside them. A model that
predicts all-zeros gets ~0.99 accuracy and an AUC-PR equal to the prevalence.
Both are printed so the comparison cannot be read the flattering way by
accident.

Unlabelled cells (-1 in the source) are excluded from both loss and metrics
rather than treated as no-fire.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.simulate.real import (  # noqa: E402
    NDWSConfig,
    driver_stats,
    fire_stats,
    load_ndws,
    normalize_drivers,
)
from pspe.utils import RunLogger, get_device, project_path, seed_everything, timer  # noqa: E402


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise definition.

    Written out rather than pulled from sklearn: it is a dozen lines, sklearn is
    not otherwise a dependency, and the exact convention (interpolated or not)
    matters when the number is compared against a published one.
    """
    order = np.argsort(-scores)
    labels = labels[order]
    cum_tp = np.cumsum(labels)
    precision = cum_tp / np.arange(1, labels.size + 1)
    total_positives = labels.sum()
    if total_positives == 0:
        return float("nan")
    return float((precision * labels).sum() / total_positives)


WIND_DIRECTION_CHANNEL = 1   # "th", the second entry of DRIVER_FEATURES, +1 for the mask


def _flip(x: torch.Tensor, horizontal: bool) -> torch.Tensor:
    return torch.flip(x, dims=[-1] if horizontal else [-2])


def batched(data: dict[str, np.ndarray], idx: np.ndarray, device,
            augment: bool = False) -> tuple[torch.Tensor, ...]:
    """Input = [day-t fire mask, drivers]; target = day-t+1 mask; plus validity."""
    states = torch.as_tensor(data["states"][idx], dtype=torch.float32, device=device)
    drivers = torch.as_tensor(data["drivers"][idx], dtype=torch.float32, device=device)
    valid = torch.as_tensor(data["valid"][idx], device=device)

    prev = states[:, 0]                       # (B, 1, H, W)
    target = states[:, 1, 0]                  # (B, H, W)
    target_valid = valid[:, 1, 0]
    # Unlabelled input cells are clamped to 0: the model has to be given some
    # number, and "no fire observed" is the honest reading for an input pixel.
    # Targets are never filled — they are masked out of the loss instead.
    prev = torch.where(prev >= 0, prev, torch.zeros_like(prev))
    x = torch.cat([prev, drivers], dim=1)

    if augment:
        # Flips only, and the wind direction channel is corrected under them.
        # A raw flip would pair a mirrored fire front with an unmirrored wind
        # vector — physically impossible examples, which is worse than no
        # augmentation. `th` is standardised, so mirroring is a sign flip in
        # the component the mirror reverses.
        for horizontal in (bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5)):
            if not horizontal:
                continue
            x = _flip(x, horizontal=True)
            target = _flip(target, horizontal=True)
            target_valid = _flip(target_valid, horizontal=True)
            x[:, WIND_DIRECTION_CHANNEL] = -x[:, WIND_DIRECTION_CHANNEL]
    return x, target, target_valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="directory of .tfrecord files")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="positive class weight. Default is sqrt(1/prevalence) "
                             "(~9 on NDWS), not 1/prevalence (~77): the latter buys "
                             "recall so aggressively the model predicts fire almost "
                             "everywhere, which costs precision and AUC-PR alike")
    parser.add_argument("--published-aucpr", type=float, default=None,
                        help="published AUC-PR to print alongside ours, from the "
                             "NDWS paper's table. Left unset rather than guessed.")
    parser.add_argument("--width", type=int, default=32,
                        help="base channel width; published baselines are wider")
    parser.add_argument("--models", nargs="+", default=["fno", "unet", "hybrid"],
                        choices=["fno", "unet", "hybrid"])
    parser.add_argument("--augment", action="store_true",
                        help="random flips. Wind direction is corrected under the "
                             "flip; without that correction the augmentation teaches "
                             "the model physically impossible wind/spread pairs")
    parser.add_argument("--test", action="store_true",
                        help="final numbers on the TEST split. Leave off while "
                             "iterating: choosing anything on test makes the "
                             "published comparison meaningless")
    parser.add_argument("--out", default="runs/ndws")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(root, use_tensorboard=False)

    kwargs = {"grid": args.grid}
    if args.root:
        kwargs["root"] = args.root
    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, **kwargs))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, **kwargs))

    # Standardise the driver channels using TRAIN statistics only, then apply
    # the same transform to eval. Computing them over both would leak.
    dstats = driver_stats(train)
    train["drivers"] = normalize_drivers(train["drivers"], dstats)
    val["drivers"] = normalize_drivers(val["drivers"], dstats)

    stats = fire_stats(train)
    print(f"train samples {stats['samples']:.0f} | labelled "
          f"{stats['labelled_fraction']:.3f} | fire prevalence "
          f"{stats['fire_prevalence']:.4f}")

    in_channels = 1 + train["drivers"].shape[1]
    n = train["states"].shape[0]
    # Class weight from the observed prevalence: without it the optimum is
    # "predict no fire", which the loss is entirely happy with.
    prevalence = max(stats["fire_prevalence"], 1e-6)
    weight = args.pos_weight if args.pos_weight is not None else (1.0 / prevalence) ** 0.5
    pos_weight = torch.tensor(max(weight, 1.0), device=device)
    print(f"pos_weight {float(pos_weight):.1f}")

    def evaluate(model, data) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for begin in range(0, data["states"].shape[0], args.batch):
                idx = np.arange(begin, min(begin + args.batch, data["states"].shape[0]))
                x, target, valid = batched(data, idx, device)
                probs = torch.sigmoid(model(x)[:, 0])
                scores.append(probs[valid].cpu().numpy())
                labels.append((target[valid] > 0.5).float().cpu().numpy())
        return np.concatenate(scores), np.concatenate(labels)

    def train_one(name: str) -> dict:
        seed_everything(args.seed)          # same init draw and batch order per arm
        model = make_fire_model(name, in_channels, grid=args.grid,
                                width=args.width).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = {"aucpr": -1.0, "state": None, "epoch": -1}

        with timer() as clock:
            for epoch in range(args.epochs):
                model.train()
                perm = np.random.permutation(n)
                losses = []
                for begin in range(0, n, args.batch):
                    idx = perm[begin:begin + args.batch]
                    x, target, valid = batched(train, idx, device, augment=args.augment)
                    logits = model(x)[:, 0]
                    loss_map = F.binary_cross_entropy_with_logits(
                        logits, (target > 0.5).float(),
                        pos_weight=pos_weight, reduction="none",
                    )
                    loss = (loss_map * valid).sum() / valid.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                sched.step()

                # Model selection on the VALIDATION split, never on test. The
                # last epoch is not the best epoch on a rare-event task, and
                # picking it by test score would invalidate the comparison the
                # whole exercise is for.
                val_scores, val_labels = evaluate(model, val)
                val_aucpr = average_precision(val_scores, val_labels)
                if val_aucpr > best["aucpr"]:
                    best = {"aucpr": val_aucpr, "epoch": epoch,
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                logger.log(epoch, **{f"{name}/train_loss": sum(losses) / len(losses),
                                     f"{name}/val_aucpr": val_aucpr})
                print(f"[{name}] epoch {epoch + 1}/{args.epochs} "
                      f"loss {losses[-1]:.4f} val AUC-PR {val_aucpr:.4f}")

        if best["state"] is not None:
            model.load_state_dict(best["state"])
        report = test if args.test else val
        scores, labels = evaluate(model, report)
        prevalence = float(labels.mean())
        row = {
            "model": name,
            "split": "test" if args.test else "val",
            "AUC-PR": round(average_precision(scores, labels), 4),
            # The floor a constant predictor gets for free.
            "AUC-PR floor": round(prevalence, 4),
            "best epoch (val)": best["epoch"] + 1,
            "params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "cells": int(labels.size),
            "wall (s)": round(clock.seconds, 1),
        }
        if hasattr(model, "spectral_contribution"):
            # Did the operator path earn its place? Zero means the model learned
            # to ignore it, which is itself the answer to the hybrid question.
            row["spectral gate"] = round(model.spectral_contribution(), 4)
        if args.published_aucpr is not None:
            row["published AUC-PR"] = args.published_aucpr
            row["vs published"] = round(row["AUC-PR"] / args.published_aucpr, 3)
        return row

    test = None
    if args.test:
        test = load_ndws(NDWSConfig(split="test", n_samples=args.n_eval, **kwargs))
        test["drivers"] = normalize_drivers(test["drivers"], dstats)

    rows = [train_one(name) for name in args.models]
    rows.sort(key=lambda r: -r["AUC-PR"])

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
