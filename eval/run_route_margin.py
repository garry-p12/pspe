"""Does the road out stay open? With a margin that can be checked.

The planner's existing guarantee is on a depth reduction: "this plan cuts flood
depth at the settlement by 21%, and by at least 20.8% nine times in ten". Right
structure, wrong scalar for the night of the flood, when the question is whether
the road is passable.

This runs the same machinery on the operational scalar. For a route r and a
levee plan p:

    m(p, r)  = deepest water anywhere on r, under a full solve of p
    m_hat    = the saturating surrogate's prediction of it, fitted to the
               solved library exactly as the depth planner fits its own
    q_hat    = split-conformal (1 - delta) quantile of |m - m_hat| over
               calibration plans for that route

and the route is DECLARED OPEN for a vehicle that drives through `limit` only
when `m_hat + q_hat <= limit`. The claim under test is the one that makes the
declaration worth anything: a route declared open is actually open at least
(1 - delta) of the time.

Three arms:

  no margin      declare on m_hat alone -- what a tool does when it believes
                 its own surrogate
  model error    the margin Proposition 3 rules out: the (1 - delta) quantile
                 of the surrogate's error over the DEPTH FIELD, pooled over
                 cells, which never looks at the route
  ours           the margin on the reported scalar, calibrated over held-out
                 plans for that route

Two things are reported for every arm, because either alone can be gamed. A
margin of +10 m would never be wrong about an open road, since it would declare
none open; so the table carries both the precision of the "open" call and the
number of genuinely open roads the arm refused to call open.

Run (district: a solved 37-plan library over a validated reference domain, all
cached, no network):

    python3 eval/run_route_margin.py --source library

Run (any town, solved on demand -- slower, and the forcing has to be one that
actually floods the network; see the degeneracy guard):

    python3 eval/run_route_margin.py --source service --place lismore --q 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.plan import margins, routes, surrogate  # noqa: E402

PLACES = {
    "lismore": {"lat": -28.8135, "lon": 153.2773, "name": "Lismore",
                "driver": "river"},
    "cedar-rapids": {"lat": 41.9779, "lon": -91.6656, "name": "Cedar Rapids",
                     "driver": "river"},
}


def sampler(field: np.ndarray, b: dict):
    """Point depth from a solved field, in the geographic frame of the box."""
    h, w = field.shape

    def at(lon: float, lat: float) -> float:
        fx = (lon - b["west"]) / (b["east"] - b["west"])
        fy = (b["north"] - lat) / (b["north"] - b["south"])
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return 0.0
        return float(field[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))])
    return at


def load_library(lib: Path, raster: str) -> tuple[np.ndarray, list[list[float]], dict, dict]:
    """The district's solved scenario library, the portal's own network and box.

    This is our solver on a domain whose reference solution the dataset authors
    validated against SAR, which is why it is the default: the flood is real
    (137 km of a 287 km network over 25 cm), the plans are solved, and nothing
    has to be fetched.
    """
    index = json.loads((lib / "index.json").read_text())
    peak = np.load(lib / raster)
    data = Path("portal/public/data")
    man = json.loads((data / "manifest.json").read_text())
    roads = json.loads((data / "roads.geojson").read_text())
    order = [o["raster_index"] for o in index["options"]]
    fields = [peak[i] for i in order]
    plans = [list(o["heights"]) for o in index["options"]]
    meta = {
        "name": index["scenario"].get("event", "district"),
        "dx_m": index["dx_m"],
        "window_hours": index["scenario"].get("window_hours"),
        "forcing": index["scenario"].get("forcing_note", ""),
        "settlement": man["levee_geometry"]["settlement"],
        "max_height_m": max(max(p) for p in plans) or 3.0,
    }
    return np.stack(fields), plans, man["bounds"], {"roads": roads, **meta}


def solve_service(args) -> tuple[np.ndarray, list[list[float]], dict, dict]:
    """Solve a town on demand, through the service's own plumbing.

    The service owns the domain builder, the forcing (including the
    stage-not-volume fix of defect 32) and the road fetch; a second copy of
    "what drives this flood" is how the portal and the paper come to disagree
    about the same town.
    """
    import torch
    from pspe.simulate.flood import FloodConfig, FloodSolver
    from service.app import (PlanRequest, build_domain, build_forcing,
                             fetch_roads, site_masks)

    place = PLACES[args.place]
    half = 0.055
    d_lon = half / math.cos(math.radians(place["lat"]))
    b = {"west": place["lon"] - d_lon, "east": place["lon"] + d_lon,
         "south": place["lat"] - half, "north": place["lat"] + half}
    dom = build_domain(b["west"], b["south"], b["east"], b["north"],
                       dx=args.dx, name=place["name"])
    z = np.asarray(dom.z, dtype=np.float32)
    print(f"  grid {z.shape} at {dom.dx:.0f} m, elevation "
          f"{np.nanmin(z):.1f}-{np.nanmax(z):.1f} m", flush=True)

    low = np.dstack(np.unravel_index(np.argsort(z.ravel()), z.shape))[0]
    sites, taken = [], []
    h_, w_ = z.shape
    for r, c in low:
        if any(abs(r - rr) < h_ // 10 and abs(c - cc) < w_ // 10 for rr, cc in taken):
            continue
        taken.append((r, c))
        sites.append({"lat": b["north"] - (r + 0.5) / h_ * (b["north"] - b["south"]),
                      "lon": b["west"] + (c + 0.5) / w_ * (b["east"] - b["west"]),
                      "width_m": 600.0})
        if len(sites) == args.sites:
            break

    req = PlanRequest(
        west=b["west"], south=b["south"], east=b["east"], north=b["north"],
        name=place["name"], dx=args.dx, rain_mm_h=args.rain_mm_h,
        storm_hours=args.storm_hours, run_hours=args.run_hours,
        driver=place["driver"], sites=sites, max_height_m=3.0,
        delta=args.delta, river_q_m3s=args.q,
    )
    plans = surrogate.enumerate_plans(args.sites, req.max_height_m)

    z0 = torch.as_tensor(dom.z, dtype=torch.float32)
    masks = site_masks(dom, req.sites)
    P = torch.tensor(plans, dtype=torch.float32)
    zb = z0[None] + (P[:, :, None, None] * masks[None]).sum(1)
    rows = zb.mean(dim=2).mean(dim=1)
    slope = max(abs(float((rows[0] - rows[-1]) / (zb.shape[1] * dom.dx))), 1e-5)
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=slope, dt_max=300.0)
    solver = FloodSolver(zb, cfg)
    h = torch.zeros(zb.shape[0], zb.shape[1], zb.shape[2])
    qx, qy = solver.zeros_flux(zb.shape[0])
    peak = h.clone()
    fx = build_forcing(dom, req)
    src, sea_mask, sea_level = fx["source"], fx["sea_mask"], fx["sea_level"]
    tt, dur, steps = 0.0, req.run_hours * 3600.0, 0
    while tt < dur:
        dt = min(solver.adaptive_dt(h), dur - tt)
        if dt <= 0:
            break
        on = tt < req.storm_hours * 3600.0
        r = (src if on else 0.0) if src is not None else 0.0
        h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=zb)
        if sea_mask is not None and on:
            want = torch.clamp(sea_level - zb, min=0.0)
            h = torch.where(sea_mask.expand_as(h), torch.maximum(h, want), h)
        peak = torch.maximum(peak, h)
        tt += dt
        steps += 1
    print(f"  solved {zb.shape[0]} plans in {steps} steps", flush=True)
    print(f"  driver: {fx['note'][:120]}", flush=True)

    cache = Path(args.out) / f"{args.place}_roads.json"
    if cache.exists():
        net = json.loads(cache.read_text())
    else:
        net = fetch_roads(b["west"], b["south"], b["east"], b["north"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(net))
    return (peak.numpy(), plans, b,
            {"roads": net, "name": place["name"], "dx_m": dom.dx,
             "forcing": fx["note"], "settlement": {"lat": place["lat"],
                                                   "lon": place["lon"]},
             "max_height_m": req.max_height_m})


def field_margin(fields: np.ndarray, plans: list[list[float]], max_h: float,
                 cal: list[int], delta: float, stride: int = 97,
                 wet_only: float = 0.0) -> float:
    """The margin that is measurable from the depth field alone.

    Proposition 3's object: fit the same saturating form per CELL from the
    single-measure runs, take the residual over cells and plans, and read its
    (1 - delta) quantile. It uses strictly more data than the route margin and
    never looks at the route, which is why it cannot be trusted with a
    route-level claim: the quantile of a pooled cell-wise error says nothing
    about the deepest cell on one particular road.

    `wet_only` restricts the pool to cells with at least that much water in the
    do-nothing field. Without it the pool is dominated by dry ground where the
    residual is identically zero, and the arm would be a strawman of our own
    construction rather than the best version of the margin our theory says
    cannot work. Both are reported.
    """
    base = fields[0]
    keep = base > wet_only if wet_only > 0 else None
    alpha: dict[int, np.ndarray] = {}
    for i, pl in enumerate(plans):
        act = [j for j, hh in enumerate(pl) if hh > 0]
        if len(act) == 1:
            alpha[act[0]] = base - fields[i]
    resid: list[float] = []
    for i in cal:
        parts = [alpha[j] * (hh / max_h)
                 for j, hh in enumerate(plans[i]) if hh > 0 and j in alpha]
        if len(parts) < 2:
            continue
        stack = np.stack(parts)
        lead = stack.max(axis=0)
        rest = stack.sum(axis=0) - lead
        pred = base - (lead + 100.0 * np.tanh(rest / 100.0))
        err = np.abs(fields[i] - pred)
        resid.extend((err[keep] if keep is not None else err.ravel())[::stride].tolist())
    if not resid:
        return 0.0
    q = margins.conformal_quantile(resid, delta)
    return max(0.0, q) if q is not None else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="library", choices=("library", "service"))
    ap.add_argument("--library", default="runs/scenarios_v2")
    ap.add_argument("--raster", default="peak_r1.npy")
    # service mode only
    ap.add_argument("--place", default="lismore", choices=sorted(PLACES))
    ap.add_argument("--dx", type=float, default=90.0)
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--rain-mm-h", type=float, default=50.0)
    ap.add_argument("--storm-hours", type=float, default=6.0)
    ap.add_argument("--run-hours", type=float, default=12.0)
    ap.add_argument("--q", type=float, default=None,
                    help="river inflow, m3/s; None lets the service estimate it")
    # the experiment
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--limit", type=float, default=routes.VEHICLE_DEPTH_M["car"])
    ap.add_argument("--routes", type=int, default=60,
                    help="random wet->dry pairs, on top of every trip out of town")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/route_margin")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "district" if args.source == "library" else args.place

    print(f"=== fields ({args.source}) ===", flush=True)
    if args.source == "library":
        fields, plans, b, meta = load_library(Path(args.library), args.raster)
    else:
        fields, plans, b, meta = solve_service(args)
    max_h = meta["max_height_m"]
    print(f"  {len(plans)} plans, grid {fields.shape[1:]} at {meta['dx_m']:.0f} m",
          flush=True)

    g = routes.build_graph(meta["roads"]["features"])
    print(f"  network: {len(g.nodes)} junctions, {len(g.edges)} edges, "
          f"{g.total_km:.0f} km", flush=True)

    print("=== sampling the network under every plan ===", flush=True)
    depths = np.array([routes.edge_depths(g, sampler(f, b)) for f in fields])
    base_d = depths[0].tolist()
    cut0 = routes.cut_km(g, base_d, args.limit)
    print(f"  do nothing: {cut0:.0f} km of {g.total_km:.0f} km over "
          f"{args.limit:.2f} m; deepest edge {depths[0].max():.2f} m", flush=True)

    # A flood that does not reach the roads cannot be used to study whether the
    # roads stay open. Saying so is the whole point: the first version of this
    # script reported 100% coverage for all three arms on a channel-confined
    # fresh, which is three arms agreeing that nothing was ever at risk.
    if cut0 < 0.5 and depths.max() < args.limit:
        raise SystemExit(
            f"DEGENERATE: no edge anywhere in this network reaches "
            f"{args.limit:.2f} m under any plan (deepest {depths.max():.2f} m). "
            f"There is no passability question here, and coverage measured on it "
            f"would be 100% for any margin including none. Use a forcing that "
            f"floods the network (--q for a river) or a domain where it does.")

    # ---- trips to score ------------------------------------------------
    # Every trip out of town, plus random wet-to-dry pairs. Enumerating from
    # the settlement is not a sample of anything, it is the operational
    # question; the random pairs are there so the population is not one origin.
    rng = random.Random(args.seed)
    st = meta["settlement"]
    src0 = routes.nearest_node(g, st["lon"], st["lat"])
    pairs = [(src0, n) for n in g.adj if n != src0]
    wetn = [n for n in g.adj
            if max((base_d[ei] for ei, _ in g.adj[n]), default=0.0) > args.limit * 0.5]
    dryn = [n for n in g.adj
            if max((base_d[ei] for ei, _ in g.adj[n]), default=0.0) <= 0.05]
    for _ in range(args.routes):
        if not wetn or not dryn:
            break
        s, t = rng.choice(wetn), rng.choice(dryn)
        if s != t:
            pairs.append((s, t))
    print(f"  {len(pairs)} candidate trips "
          f"({len(wetn)} junctions near water, {len(dryn)} on dry ground)",
          flush=True)

    need = max(2, int(round(1.0 / args.delta)) - 1)
    multi_all = [i for i, pl in enumerate(plans) if sum(1 for h in pl if h > 0) >= 2]
    if len(multi_all) < 2 * need:
        raise SystemExit(f"only {len(multi_all)} multi-measure plans; need "
                         f"{2 * need} to calibrate and test at delta={args.delta}")

    arms = {k: {"open_right": 0, "open_wrong": 0, "closed_but_open": 0,
                "closed_right": 0}
            for k in ("none", "model_error", "model_error_wet", "ours")}
    rows, bands, field_bands, wet_bands, cost_min = [], [], [], [], []
    trivial = 0

    for (s, t) in pairs:
        # The route is held FIXED across plans, and chosen on the do-nothing
        # field, because "does this road stay open" is a different question
        # from "is there any road": letting the route change per plan would let
        # a plan look good by sending the vehicle somewhere else.
        geom = routes.fastest(g, s, t, [0.0] * len(g.edges), args.limit)
        if not geom.ok or not geom.edges:
            continue
        m = depths[:, geom.edges].max(axis=1)

        # Trivial trips: the declaration is the same for every plan and every
        # arm. Counted, reported, and kept out of the coverage figures, because
        # a population of roads that no plan can put at risk would carry any
        # margin at all and prove nothing about this one.
        if (m <= args.limit).all() or (m > args.limit).all():
            trivial += 1
            continue

        alpha = [0.0] * len(plans[0])
        for i, pl in enumerate(plans):
            act = [j for j, hh in enumerate(pl) if hh > 0]
            if len(act) == 1:
                alpha[act[0]] = float(m[0] - m[i])
        red = [float(m[0] - v) for v in m]
        f = surrogate.fit(alpha, plans, red, max_h, args.delta)
        S = f["saturation_S"]

        def predict(i: int) -> float:
            parts = [alpha[j] * (hh / max_h) for j, hh in enumerate(plans[i]) if hh > 0]
            if not parts:
                return float(m[0])
            lead = max(parts)
            return float(m[0]) - (lead + S * math.tanh((sum(parts) - lead) / S))

        multi = list(multi_all)
        rng.shuffle(multi)
        cal, test = multi[:len(multi) // 2], multi[len(multi) // 2:]
        if len(cal) < need or not test:
            continue

        q_ours = margins.margin("residual", args.delta,
                                residuals=[abs(float(m[i]) - predict(i)) for i in cal])
        q_field = field_margin(fields, plans, max_h, cal, args.delta)
        # The fair version: pooled over flooded ground only, so the arm is not
        # being beaten by its own zeros.
        q_wet = field_margin(fields, plans, max_h, cal, args.delta,
                             stride=13, wet_only=0.05)

        for i in test:
            truth = bool(m[i] <= args.limit)
            p_hat = predict(i)
            for k, q in (("none", 0.0), ("model_error", q_field),
                         ("model_error_wet", q_wet), ("ours", q_ours)):
                said = p_hat + q <= args.limit
                key = ("open_right" if truth else "open_wrong") if said else \
                      ("closed_but_open" if truth else "closed_right")
                arms[k][key] += 1

        bands.append(q_ours)
        field_bands.append(q_field)
        wet_bands.append(q_wet)
        # What robustness costs: the quickest open route, against the quickest
        # route that is still open once every road is allowed to be the margin
        # deeper than predicted.
        nominal = routes.fastest(g, s, t, base_d, args.limit)
        robust = routes.fastest(g, s, t, [d + q_ours for d in base_d], args.limit)
        if nominal.ok and robust.ok:
            cost_min.append(robust.minutes - nominal.minutes)

        rows.append({
            "from": s, "to": t,
            "dry_minutes": round(geom.minutes, 1), "km": round(geom.km, 2),
            "base_max_depth_m": round(float(m[0]), 3),
            "best_plan_m": round(float(m.min()), 3),
            "worst_plan_m": round(float(m.max()), 3),
            "band_m": round(q_ours, 3), "field_band_m": round(q_field, 3),
            "fit_rmse_m": round(f["fit_rmse"], 4),
            "n_cal": len(cal), "n_test": len(test),
            "via": geom.names(g)[:4],
        })

    if not rows:
        raise SystemExit(
            f"DEGENERATE: all {trivial} trips were decided the same way under "
            f"every plan, so no arm can be distinguished from any other. "
            f"Nothing is reported.")

    def pct(a: dict) -> float:
        said = a["open_right"] + a["open_wrong"]
        return round(100 * a["open_right"] / said, 1) if said else float("nan")

    summary = {
        "source": args.source, "place": meta["name"], "dx_m": meta["dx_m"],
        "grid": list(fields.shape[1:]), "plans": len(plans),
        "delta": args.delta, "limit_m": args.limit,
        "network_km": round(g.total_km, 1),
        "cut_km_do_nothing": round(cut0, 1),
        "candidate_trips": len(pairs),
        "trips_decided_identically_under_every_plan": trivial,
        "trips_scored": len(rows),
        "median_band_m": round(float(np.median(bands)), 3),
        "median_field_band_m": round(float(np.median(field_bands)), 3),
        "median_field_band_wet_only_m": round(float(np.median(wet_bands)), 3),
        "arms": {k: {**v, "declared_open_and_was": pct(v)} for k, v in arms.items()},
        "robustness_cost_minutes_median": (round(float(np.median(cost_min)), 1)
                                           if cost_min else None),
        "forcing": meta.get("forcing", "")[:200],
        "seconds": round(time.time() - t0, 1),
    }
    (out / f"{tag}_routes.json").write_text(
        json.dumps({"summary": summary, "routes": rows}, indent=1))

    print(f"\n=== {len(rows)} trips where the answer depends on the plan ===")
    print(f"({trivial} of {len(pairs)} candidate trips were decided the same way "
          f"under every plan and are excluded)")
    print("\n| arm | called open | of those, open | open roads it would not call open |")
    print("|---|---|---|---|")
    for k, label in (("none", "no margin"),
                     ("model_error", "model-error margin, all cells"),
                     ("model_error_wet", "model-error margin, flooded cells"),
                     ("ours", f"ours, {100 * (1 - args.delta):.0f}% target")):
        a = arms[k]
        print(f"| {label} | {a['open_right'] + a['open_wrong']} | {pct(a)}% | "
              f"{a['closed_but_open']} |")
    print(f"\nmedian margin {summary['median_band_m']} m; field-error margin "
          f"{summary['median_field_band_m']} m all cells, "
          f"{summary['median_field_band_wet_only_m']} m flooded cells only")
    print(f"robustness costs {summary['robustness_cost_minutes_median']} min "
          f"on the median trip")
    print(f"\n{summary['seconds']}s -> {out}/{tag}_routes.json")


if __name__ == "__main__":
    main()
