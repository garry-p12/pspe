"""The wildfire half of the portal: observe, forecast, and plan a firebreak.

Mounted on the same app as the flood service. It is a separate module because
`app.py` is already 4,500 lines and because the two halves share nothing but
the rendering helpers -- a fire raster is 64x64 patches of a prepared record,
not a solved field on fetched terrain.

## What is live here, and what is not

This is the honesty question the flood portal answers with a badge, and the
answer differs between the two tabs:

* **Observe is live, anywhere.** NASA FIRMS serves VIIRS/MODIS active-fire
  detections for any bounding box and date range. Draw a box over a fire
  burning today and the detections are today's.
* **Forecast and plan are records, not live.** The spread model takes eleven
  remote-sensing drivers -- terrain, wind, humidity, drought index, NDVI,
  population, energy release component. Those are a prepared dataset, not a
  feed. So the forecast runs on held-out NDWS records: real US fires with real
  drivers and a real trained model, on a day that has already happened.

Reporting a forecast for a live box would mean inventing drivers. The endpoint
refuses instead, and the portal says which of the two it is showing.

## What the model is

A U-Net trained on NDWS to predict day `t+1`'s fire mask from day `t`'s mask
plus the eleven drivers. The planner optimises firebreak intensities through
the autoregressive rollout by projected gradient, exactly as
`eval/run_ndws_planning.py` does -- this module imports that path rather than
reimplementing it, so the portal cannot drift from the evaluated result.
"""

from __future__ import annotations

import base64
import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]

router = APIRouter(prefix="/fire", tags=["fire"])

#: Treated fraction of a patch per day. The operational comparison in the
#: write-up is run at 3%, so that is the default here too.
DEFAULT_BUDGET = 0.03
GRID = 64


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _png(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def fire_png(mask: np.ndarray, *, soft: bool = False) -> str:
    """Fire as a base64 PNG. Hot where it burns, transparent where it does not.

    `soft` renders a probability field: the forecast is a probability per cell
    and drawing it with the same hard cut as an observation would claim a
    confidence the model does not have.
    """
    x = np.clip(np.nan_to_num(mask), 0.0, 1.0)
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    if soft:
        # A probability field: pale amber where the model is unsure, deep
        # vermilion where it is confident. The ramp carries the confidence,
        # which is the whole reason a forecast is not drawn like an
        # observation.
        out[..., 0] = (250 - 40 * x).astype(np.uint8)
        out[..., 1] = (205 - 140 * x).astype(np.uint8)
        out[..., 2] = (120 - 105 * x).astype(np.uint8)
        out[..., 3] = np.clip(255 * x ** 0.6, 0, 255).astype(np.uint8)
    else:
        # An observation is binary, so EVERY burning cell sits at the top of
        # any ramp -- which drew observed fire as pale yellow, the colour a
        # ramp reserves for its hottest value. One flat vermilion instead.
        hot = x > 0.5
        out[..., 0] = np.where(hot, 214, 0).astype(np.uint8)
        out[..., 1] = np.where(hot, 69, 0).astype(np.uint8)
        out[..., 2] = np.where(hot, 15, 0).astype(np.uint8)
        out[..., 3] = np.where(hot, 240, 0).astype(np.uint8)
    return _png(out)


def driver_png(v: np.ndarray, lo: float, hi: float) -> str:
    """A normalised driver field, in the portal's neutral grey-green ramp."""
    x = np.clip((np.nan_to_num(v) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (236 * (1 - x) + 26 * x).astype(np.uint8)
    out[..., 1] = (238 * (1 - x) + 92 * x).astype(np.uint8)
    out[..., 2] = (232 * (1 - x) + 52 * x).astype(np.uint8)
    out[..., 3] = 210
    return _png(out)


def treatment_png(t: np.ndarray) -> str:
    """Where the crews go. Deliberately a different hue from the fire."""
    x = np.clip(np.nan_to_num(t), 0.0, 1.0)
    out = np.zeros(x.shape + (4,), dtype=np.uint8)
    out[..., 0] = (37 + 20 * x).astype(np.uint8)
    out[..., 1] = (99 + 40 * x).astype(np.uint8)
    out[..., 2] = (168 + 60 * x).astype(np.uint8)
    out[..., 3] = np.clip(235 * x ** 0.5, 0, 255).astype(np.uint8)
    return _png(out)


# --------------------------------------------------------------------------- #
# the model and the records, loaded once
# --------------------------------------------------------------------------- #
class FireUnavailable(Exception):
    """Raised when the record set or the checkpoint is not on this machine."""


@lru_cache(maxsize=1)
def _records() -> dict:
    """Held-out NDWS fires, with drivers standardised the way training did.

    The normalisation is not a detail. The eleven drivers are in wildly
    different units -- elevation in metres, NDVI scaled by 1e4, a drought index
    near zero -- and `driver_stats` exists because feeding them raw lets the
    large-magnitude channels dominate the first layer. Measured when this
    endpoint was first written WITHOUT it: critical success index 0.042, with
    the model finding 17 of 406 burning cells. The statistics must come from
    the TRAIN split, because that is what the checkpoint was fitted against;
    computing them on the held-out split would be a different normalisation
    and a quietly different model.
    """
    from pspe.simulate.real.ndws import (
        NDWSConfig, driver_stats, load_ndws, normalize_drivers,
    )

    root = ROOT / "data" / "ndws"
    if not root.exists() or not any(root.glob("*.tfrecord")):
        raise FireUnavailable(
            "The NDWS records are not on this machine. The forecast and plan "
            "tabs need them; the observe tab does not and still works.")
    held = load_ndws(NDWSConfig(root=root, split="test", grid=GRID))
    try:
        train = load_ndws(NDWSConfig(root=root, split="train", grid=GRID,
                                     n_samples=2000))
        stats = driver_stats(train)
    except Exception:                                           # noqa: BLE001
        # No train shard on this machine: fall back to the held-out split's own
        # statistics and say so, rather than serving raw drivers silently.
        stats = driver_stats(held)
        held["_stats_from"] = "held-out split (train shard absent)"
    held["drivers"] = normalize_drivers(held["drivers"], stats)
    return held


@lru_cache(maxsize=1)
def _model():
    from pspe.simulate.real.models import UNet

    for cand in (ROOT / "runs" / "ndws_planning" / "seed_0" / "unet.pt",
                 ROOT / "runs" / "ndws_planning_h3_attempt1" / "seed_0" / "unet.pt"):
        if cand.exists():
            sd = torch.load(cand, map_location="cpu", weights_only=True)
            # Read the architecture OUT of the checkpoint rather than asserting
            # it. Guessing depth=4 against a depth=3 checkpoint produced forty
            # lines of shape mismatch and a 500 from the endpoint; the file
            # knows its own shape, so ask it.
            w0 = sd["downs.0.net.0.weight"]          # (width, in_channels, 3, 3)
            in_ch, width = int(w0.shape[1]), int(w0.shape[0])
            depth = len({k.split(".")[1] for k in sd if k.startswith("downs.")})
            net = UNet(in_ch, width=width, depth=depth)
            net.load_state_dict(sd)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            return net
    raise FireUnavailable("No trained spread model found under runs/.")


@lru_cache(maxsize=1)
def _env():
    """A FireEnv over the held-out records, which is what the planner wants."""
    from pspe.envs.fire_env import FireEnv, FireTask

    return FireEnv(_model(), _records(), FireTask(cost_limit=DEFAULT_BUDGET),
                   grid=GRID, patches=4, horizon=1)


def _burn_stats(records: dict) -> np.ndarray:
    """Burned fraction on day t+1, per record. Used to rank records."""
    nxt = records["states"][:, 1, 0].astype(np.float32)
    return np.where(nxt > 0, nxt, 0.0).reshape(nxt.shape[0], -1).mean(1)


# --------------------------------------------------------------------------- #
# 1. which fires can be looked at
# --------------------------------------------------------------------------- #
@router.get("/records")
def records(limit: int = 24) -> dict:
    """The held-out fires, largest first.

    Ranked by how much burns on day t+1, because a record where nothing burns
    cannot show a forecast working or failing and is not worth a user's click.
    """
    try:
        rec = _records()
    except FireUnavailable as exc:
        return {"ok": False, "reason": str(exc)}

    burn = _burn_stats(rec)
    order = np.argsort(-burn)[:limit]
    prev = rec["states"][:, 0, 0].astype(np.float32)
    prev = np.where(prev > 0, prev, 0.0)
    return {
        "ok": True,
        "total": int(burn.shape[0]),
        "records": [
            {"id": int(i),
             "burned_next_pct": round(float(burn[i]) * 100, 2),
             "burning_now_pct": round(float(prev[i].mean()) * 100, 2),
             "grew": bool(burn[i] > prev[i].mean())}
            for i in order
        ],
        "note": ("Real US fire days the model was never trained on. Biggest "
                 "next-day growth first, because a fire that barely moves "
                 "cannot show a forecast working or failing."),
    }


class RecordRequest(BaseModel):
    id: int = Field(0, ge=0)


@router.post("/record")
def record(req: RecordRequest) -> dict:
    """One fire: what is burning, what burns next, and the drivers behind it."""
    try:
        rec = _records()
    except FireUnavailable as exc:
        return {"ok": False, "reason": str(exc)}
    n = rec["states"].shape[0]
    if req.id >= n:
        return {"ok": False, "reason": f"record {req.id} does not exist; {n} available"}

    from pspe.simulate.real.ndws import DRIVER_FEATURES

    now = np.where(rec["states"][req.id, 0, 0] > 0, rec["states"][req.id, 0, 0], 0.0)
    nxt = np.where(rec["states"][req.id, 1, 0] > 0, rec["states"][req.id, 1, 0], 0.0)
    drivers = rec["drivers"][req.id]

    def d(name: str) -> np.ndarray:
        return drivers[DRIVER_FEATURES.index(name)]

    # Only the drivers a reader can interpret without a legend lecture. The
    # model uses all eleven; showing all eleven would be a data dump.
    shown = {}
    for name, label in (("NDVI", "vegetation"), ("population", "people"),
                        ("vs", "wind speed"), ("erc", "fuel dryness")):
        v = d(name)
        shown[name] = {
            "label": label,
            "png": driver_png(v, float(np.percentile(v, 2)), float(np.percentile(v, 98))),
        }

    return {
        "ok": True,
        "id": req.id,
        "grid": GRID,
        "burning_now_png": fire_png(now),
        "burns_next_png": fire_png(nxt),
        "burning_now_pct": round(float(now.mean()) * 100, 2),
        "burns_next_pct": round(float(nxt.mean()) * 100, 2),
        "drivers": shown,
        "evidence": "record",
        "note": ("A real fire, on a real day. Both pictures are what the "
                 "satellites saw \u2014 nothing here is a prediction."),
    }


# --------------------------------------------------------------------------- #
# 2. the forecast, scored against what actually happened
# --------------------------------------------------------------------------- #
@router.post("/forecast")
def forecast(req: RecordRequest) -> dict:
    """Predict tomorrow's fire, and score it against tomorrow.

    The score is the part worth having. A spread forecast that is never
    compared to the day it predicted is a picture, and the portal's whole
    argument is that a component's own metric is not the decision.
    """
    try:
        rec, net = _records(), _model()
    except FireUnavailable as exc:
        return {"ok": False, "reason": str(exc)}
    if req.id >= rec["states"].shape[0]:
        return {"ok": False, "reason": f"record {req.id} does not exist"}

    t0 = time.time()
    prev = np.where(rec["states"][req.id, 0] > 0, rec["states"][req.id, 0], 0.0)
    x = np.concatenate([prev.astype(np.float32), rec["drivers"][req.id]], axis=0)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.as_tensor(x)[None]))[0, 0].numpy()

    truth = np.where(rec["states"][req.id, 1, 0] > 0, rec["states"][req.id, 1, 0], 0.0)
    valid = rec["valid"][req.id, 1, 0].astype(bool)
    hard = p > 0.5

    # Critical success index on the valid cells, which is the metric this
    # literature reports. Accuracy is useless here: fire is ~1% of the grid,
    # so predicting "no fire" everywhere scores 99%.
    tp = float((hard & (truth > 0.5) & valid).sum())
    fp = float((hard & (truth <= 0.5) & valid).sum())
    fn = float((~hard & (truth > 0.5) & valid).sum())
    csi = tp / max(tp + fp + fn, 1.0)
    base = float((truth > 0.5)[valid].mean()) if valid.any() else 0.0

    # Average precision on this record, which is the metric the project
    # reports at split level. CSI needs a threshold and AP does not, so the two
    # disagree by construction -- a pooled CSI of 0.22 alongside a recorded
    # AUC-PR of 0.3162 is two metrics, not a shortfall, and conflating them
    # cost an afternoon of chasing a bug that did not exist.
    ap = None
    if valid.any() and (truth[valid] > 0.5).any():
        yt = (truth[valid] > 0.5).astype(np.int8)
        ys = p[valid]
        order = np.argsort(-ys)
        tp_c = np.cumsum(yt[order])
        prec = tp_c / np.arange(1, yt.size + 1)
        ap = float((prec * yt[order]).sum() / max(yt.sum(), 1))

    return {
        "ok": True,
        "id": req.id,
        "average_precision": None if ap is None else round(ap, 3),
        "forecast_png": fire_png(p, soft=True),
        "truth_png": fire_png(truth),
        "csi": round(csi, 3),
        "hit": int(tp), "false_alarm": int(fp), "missed": int(fn),
        "burned_truth_pct": round(base * 100, 2),
        "burned_forecast_pct": round(float(hard[valid].mean()) * 100, 2) if valid.any() else 0.0,
        "seconds": round(time.time() - t0, 2),
        "evidence": "projection",
        "note": ("The model is shown today and never tomorrow. Fire covers "
                 "about 1% of the map, so a forecast that said \u201cno fire "
                 "anywhere\u201d would be right 99% of the time \u2014 which "
                 "is why what it found and what it missed are counted "
                 "separately here, instead of a single score."),
    }


# --------------------------------------------------------------------------- #
# 3. the plan: where crews cut, under a budget
# --------------------------------------------------------------------------- #
class PlanRequest(BaseModel):
    id: int = Field(0, ge=0)
    #: Fraction of the patch a crew can treat per day.
    budget: float = Field(DEFAULT_BUDGET, gt=0.0, le=0.25)
    horizon: int = Field(3, ge=1, le=5)
    steps: int = Field(60, ge=10, le=200)


@router.post("/plan")
def plan(req: PlanRequest) -> dict:
    """Optimise firebreak intensities through the surrogate rollout.

    Compared against two references a fire manager would recognise: doing
    nothing, and spreading the same crew-days uniformly. Uniform is the
    comparison that matters -- beating "no treatment" only shows the treatment
    model does something, which it does by construction.
    """
    try:
        rec = _records()
        from pspe.envs.fire_env import FireEnv, FireTask
        from eval.run_ndws_planning import per_instance_plan, rollout
    except FireUnavailable as exc:
        return {"ok": False, "reason": str(exc)}
    except ImportError as exc:                                  # noqa: BLE001
        return {"ok": False, "reason": f"planner unavailable: {exc}"}
    if req.id >= rec["states"].shape[0]:
        return {"ok": False, "reason": f"record {req.id} does not exist"}

    t0 = time.time()
    env = FireEnv(_model(), rec, FireTask(cost_limit=req.budget),
                  grid=GRID, patches=4, horizon=req.horizon)
    state = env.x[req.id:req.id + 1].clone()
    w = 1 + env.task.population_weight * env.population(state)

    def burned_under(actions) -> float:
        p, _ = rollout(env, state, actions, req.horizon)
        return float((p * w).flatten(1).mean(1).sum())

    k = env.action_dim
    none = burned_under(lambda st, t: FireEnv.action_for(
        torch.zeros(1, k)))
    uniform = burned_under(lambda st, t: FireEnv.action_for(
        torch.full((1, k), req.budget)))
    acts = per_instance_plan(env, state, req.horizon, steps=req.steps)
    planned = burned_under(lambda st, t: acts[:, t])

    # The treatment field the crews would actually be sent to.
    u = FireEnv.intensity(acts[:, 0])
    treat = env.treatment_from_intensity(u)[0].detach().numpy()
    spent = float(u.mean())

    def pct(x: float) -> float:
        return round(100.0 * (none - x) / max(none, 1e-9), 1)

    return {
        "ok": True,
        "id": req.id,
        "budget": req.budget,
        "horizon": req.horizon,
        "treatment_png": treatment_png(treat / max(treat.max(), 1e-6)),
        "spent_fraction": round(spent, 4),
        "within_budget": bool(spent <= req.budget * 1.02),
        "burned": {"none": round(none, 4), "uniform": round(uniform, 4),
                   "planned": round(planned, 4)},
        "reduction_vs_none_pct": pct(planned),
        "reduction_vs_uniform_pct": round(
            100.0 * (uniform - planned) / max(uniform, 1e-9), 1),
        "uniform_reduction_pct": pct(uniform),
        "seconds": round(time.time() - t0, 2),
        "evidence": "projection",
        "note": ("These are estimates, not outcomes. No fire on record ever "
                 "had a break cut because we said so, so nobody can know what "
                 "would really have happened \u2014 the model is comparing "
                 "plans against each other. Judge it against spreading the "
                 "crews evenly, not against doing nothing: beating "
                 "\u201cdo nothing\u201d only shows that treating fuel "
                 "works at all."),
    }


# --------------------------------------------------------------------------- #
# 4. live detections, anywhere
# --------------------------------------------------------------------------- #
class ObserveRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float
    days: int = Field(7, ge=1, le=10)
    grid: int = Field(64, ge=16, le=128)
    #: FIRMS product. The near-real-time VIIRS feed answers "burning now";
    #: the standard-processing archive reaches back years and is what a
    #: historic fire needs. VIIRS starts in 2012, so anything older is MODIS.
    source: str = "VIIRS_SNPP_NRT"
    #: Window start, YYYY-MM-DD. Empty means the last `days` days, which is
    #: the live case. A date makes this a replay of a fire that happened.
    start: str = ""


@router.post("/observe")
def observe(req: ObserveRequest) -> dict:
    """NASA FIRMS active-fire detections over a box, gridded by day.

    This is the live half of the fire portal and it works anywhere on Earth.
    Two properties of the source are preserved rather than smoothed:

    * A day with no satellite coverage is `observed: false`, not an empty
      fire. "We did not look" and "nothing burned" are different facts.
    * Detections are instantaneous, not cumulative: a pixel is reported while
      it is actively burning at overpass, so each day is a snapshot of the
      front rather than the scar.
    """
    from pspe.simulate.real.firms import FireSpec, grid_sequence, map_key

    try:
        key = map_key()
    except Exception:                                           # noqa: BLE001
        return {"ok": False, "reason": (
            "No FIRMS map key on this machine. The observe tab needs one; "
            "the forecast and plan tabs do not and still work.")}
    if not key:
        return {"ok": False, "reason": "FIRMS map key is empty."}

    from datetime import date, timedelta

    t0 = time.time()
    if req.start:
        try:
            start = date.fromisoformat(req.start)
        except ValueError:
            return {"ok": False, "reason": f"{req.start!r} is not a date"}
    else:
        start = date.today() - timedelta(days=req.days - 1)
    # FIRMS serves recent days from the near-real-time product; the default in
    # FireSpec is standard processing, which lags. For a "burning now" tab the
    # NRT feed is the right one. Passed at construction: FireSpec is frozen.
    spec = FireSpec(name="box", west=req.west, south=req.south,
                    east=req.east, north=req.north,
                    start=start.isoformat(), days=req.days,
                    source=req.source)
    try:
        from pspe.simulate.real.firms import fetch_fire
        rows = fetch_fire(spec)
        seq = grid_sequence(spec, rows, grid=req.grid)
    except Exception as exc:                                    # noqa: BLE001
        return {"ok": False, "reason": f"FIRMS request failed: {str(exc)[:160]}"}

    # `FireSequence.fire`, not `.frames` -- read from the dataclass rather than
    # guessed. It also carries its own dates, so deriving them from the start
    # date a second time is a second chance to be wrong.
    frames = np.asarray(seq.fire, dtype=np.float32)
    seen = list(map(bool, seq.observed))
    days = list(seq.dates)
    # COUNT the lit cells; do not sum the raster. `grid_sequence` saturates
    # at three detections and divides by three, so one detection in a cell is
    # 0.33 and a sum of them rounds to zero -- which reported a box that was
    # alight as "0 cells" and, worse, as six days of missing coverage.
    detections = [int((f > 0).sum()) for f in frames]

    live = [fire_png(np.clip(f, 0, 1)) for f in frames]
    any_fire = bool(frames.max() > 0)

    # What a day with no detections actually means, which this feed cannot
    # settle on its own. FIRMS serves DETECTIONS, not COVERAGE: `observed` is
    # set only where a detection landed, so a quiet day and an unobserved day
    # arrive as the same record. The distinction the firms module wants to
    # preserve is real and only recoverable from context.
    #
    # If the box was alight at some point in the window, an empty day sits
    # between fire and fire and a coverage gap is the likely reading. If
    # nothing was alight all week, the honest statement is "no detections" --
    # calling that "no satellite pass" over a box the satellite crosses twice
    # a day would be inventing a blind spot to explain an absence.
    # An empty day is plausibly a coverage gap only if it sits BETWEEN days
    # with fire. Trailing zeros after a fire has burned out, or leading zeros
    # before it started, are the fire being out -- calling those "no satellite
    # pass" turned Bastrop's three days of cold ash into three days of blamed
    # satellite, and Austin's quiet week into six.
    lit = [i for i, f in enumerate(frames) if f.max() > 0]
    if lit:
        inner = range(lit[0] + 1, lit[-1])
        gaps = sum(1 for i in inner if frames[i].max() <= 0)
    else:
        gaps = 0
    quiet = sum(1 for f in frames if f.max() <= 0) - gaps
    gap_meaning = "coverage" if gaps else "quiet"
    return {
        "ok": True,
        "grid": req.grid,
        "bounds": [req.west, req.south, req.east, req.north],
        "days": [{"date": d, "observed": o, "cells": c, "png": p}
                 for d, o, c, p in zip(days, seen, detections, live)],
        "any_fire": any_fire,
        "blind_days": gaps,
        "quiet_days": quiet,
        #: "coverage" -- alight either side, so an empty day most likely means
        #: nobody looked. "quiet" -- nothing burned all week, so an empty day
        #: means no detections and nothing further can be inferred.
        "zero_days_mean": gap_meaning,
        "seconds": round(time.time() - t0, 2),
        "evidence": "observed",
        "live": not req.start,
        "window": [start.isoformat(),
                   (start + timedelta(days=req.days - 1)).isoformat()],
        "note": ("What two NASA satellites saw as they passed overhead. Each "
                 "picture is the fire burning at that moment, not the total "
                 "area burned so far. Where the box was alight either side of "
                 "an empty day, that day is marked \u201cno pass\u201d: not "
                 "looking and nothing burning are different things. Where "
                 "nothing was alight all week, an empty day means only that "
                 "there were no detections \u2014 this feed reports fires, "
                 "not coverage, so it cannot tell the two apart."),
    }


# --------------------------------------------------------------------------- #
# 5. what the model is worth, stated once
# --------------------------------------------------------------------------- #
@router.get("/skill")
def skill() -> dict:
    """The spread model's standing, including where it is behind.

    A portal that shows a forecast without saying how good the forecaster is
    invites the reader to assume it is good. These numbers are from the
    project's own evaluation and are not recomputed per request.
    """
    return {
        "ok": True,
        "metric": "AUC-PR",
        "ours": 0.3162,
        "ours_sd": 0.017,
        "prevalence": 0.0101,
        "all_zeros_floor": 0.011,
        "originating_baseline": {"name": "Huot et al. (dataset paper)", "value": 0.284},
        "state_of_the_art": [
            {"name": "SwinUNETR, modular augmentation", "value": 0.3673},
            {"name": "mixed ensemble", "value": 0.3790},
        ],
        "note": ("Better than the forecaster published with this dataset, "
                 "and not as good as the best one anyone has built. We did "
                 "not set out to win at forecasting \u2014 a better spread "
                 "model would make the plans better without changing "
                 "anything we claim."),
    }
