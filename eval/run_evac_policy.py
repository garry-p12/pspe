"""When to leave, when the forecast might be wrong.

§5.14 gives a trip a route and a deadline against ONE solved flood. The deadline
is therefore conditional on the storm arriving as modelled, and an operator who
acts on it is trusting a single realisation. That is the gap this measures.

The decision is genuinely sequential and genuinely under partial observation,
which is what makes it the one place in this project where a policy earns its
keep rather than decorating a shortest path:

  * the network degrades while you drive, so a route is a schedule, not a path
    (§5.14 already handles this part);
  * the schedule depends on a forcing nobody knows exactly, so the schedule is
    a distribution over schedules;
  * and you learn about the forcing as the flood arrives -- depths you can see
    now tell you which member of that distribution you are living in.

Five arms, from the thing an operator does today to the thing this framework
argues for:

  leave-now      depart immediately on the ensemble-mean schedule
  static-peak    route on the mean member's PEAK field -- the standard flood-twin
                 output, and what §5.13 routes against
  dp-mean        time-dependent routing on the ensemble-mean schedule: §5.14,
                 believing the central forecast
  conformal      dp-mean with a conformal margin on the REPORTED SCALAR (deepest
                 water on the chosen route), calibrated across held-out members
  filtered       re-plan every slot against only those members still consistent
                 with the depths observed so far, then apply the margin to what
                 is left. Perceive feeding Plan, with no training

and `oracle`, which plans on the member that actually happens, as the ceiling.

Evaluation is leave-one-out over the ensemble: each member in turn is reality
and the others are the forecast. An arm SUCCEEDS on a trip if the traveller is
never on a road whose true depth exceeds what the vehicle can cross while they
are on it, and reaches the destination.

The ensemble is over a forcing MULTIPLIER, not a meteorological forecast
ensemble. It spans "how big is this flood" and nothing else -- not timing, not
spatial pattern, not the solver's own error. That is a narrower uncertainty than
an operator faces, and the numbers should be read as a mechanism result.

    python3 eval/run_evac_policy.py --dx 200 --members 8
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

from pspe.plan import margins, routes  # noqa: E402

PLACES = {
    "lismore-coraki": {
        "origin": {"lat": -28.8135, "lon": 153.2773, "name": "Lismore"},
        "destination": {"lat": -28.9956, "lon": 153.2906, "name": "Coraki"},
        "q_m3s": 3000.0, "driver": "river",
    },
}


def box_for(o: dict, d: dict) -> dict:
    span = math.hypot(
        (float(d["lon"]) - float(o["lon"])) * 111.320
        * math.cos(math.radians((float(o["lat"]) + float(d["lat"])) / 2)),
        (float(d["lat"]) - float(o["lat"])) * 110.540)
    pad = max(0.3 * span, 3.0)
    mid_lat = (float(o["lat"]) + float(d["lat"])) / 2
    mid_lon = (float(o["lon"]) + float(d["lon"])) / 2
    half_lat = max(span / 2 + pad, abs(float(o["lat"]) - float(d["lat"])) / 2 + pad) / 110.540
    half_lon = half_lat / max(math.cos(math.radians(mid_lat)), 0.2)
    return {"west": mid_lon - half_lon, "east": mid_lon + half_lon,
            "south": mid_lat - half_lat, "north": mid_lat + half_lat}


def solve_member(dom, box, mult: float, args, snap_h, flat, offs):
    """One ensemble member: this much water, on the same terrain."""
    import torch
    from pspe.simulate.flood import FloodConfig, FloodSolver
    from service.app import PlanRequest, build_forcing

    req = PlanRequest(
        west=box["west"], south=box["south"], east=box["east"], north=box["north"],
        name="evac", dx=dom.dx, rain_mm_h=args.rain_mm_h,
        storm_hours=args.storm_hours, run_hours=args.run_hours,
        driver=args.driver, river_q_m3s=args.q_m3s * mult, sites=[],
    )
    z = torch.as_tensor(dom.z, dtype=torch.float32)[None]
    fx = build_forcing(dom, req)
    cfg = FloodConfig(dx=dom.dx, open_edges=("south", "north", "east", "west"),
                      manning=0.035, bed_slope=1e-4, dt_max=300.0)
    solver = FloodSolver(z, cfg)
    h = torch.zeros_like(z)
    qx, qy = solver.zeros_flux(1)
    src, sea_mask, sea_level = fx["source"], fx["sea_mask"], fx["sea_level"]
    at = sorted(float(x) * 3600.0 for x in snap_h)
    rows, nxt, tt, dur = [], 0, 0.0, args.run_hours * 3600.0
    while tt < dur:
        dt = min(solver.adaptive_dt(h), dur - tt)
        if dt <= 0:
            break
        on = tt < args.storm_hours * 3600.0
        r = (src if on else 0.0) if src is not None else 0.0
        h, qx, qy = solver.step(h, qx, qy, dt, rain=r, z=z)
        if sea_mask is not None and on:
            lvl = torch.clamp(sea_level - z, min=0.0)
            h = torch.where(sea_mask.expand_as(h), torch.maximum(h, lvl), h)
        tt += dt
        while nxt < len(at) and tt >= at[nxt]:
            rows.append(routes.depths_from_field(h[0].numpy(), flat, offs))
            nxt += 1
    while nxt < len(at):
        rows.append(routes.depths_from_field(h[0].numpy(), flat, offs))
        nxt += 1
    return routes.Schedule(times=[x * 60.0 for x in snap_h], depths=rows), fx


def mean_schedule(scheds: list[routes.Schedule]) -> routes.Schedule:
    """The central forecast: elementwise mean over members, per edge per time."""
    arr = np.mean([np.asarray(s.depths) for s in scheds], axis=0)
    return routes.Schedule(times=list(scheds[0].times), depths=arr.tolist())


def inflate(sched: routes.Schedule, q: float) -> routes.Schedule:
    return routes.Schedule(times=list(sched.times),
                           depths=[[d + q for d in row] for row in sched.depths])


def walk(g, route, truth: routes.Schedule, limit: float, depart: float) -> dict:
    """Drive the plan against the member that actually happened.

    Failure is being on a road whose TRUE depth exceeds the limit while on it --
    not whether the plan looked safe when it was made.
    """
    if route is None or not route.ok:
        return {"arrived": False, "why": "no plan", "minutes": None,
                "worst_m": None}
    t, worst = depart, 0.0
    for ei in route.edges:
        t2 = t + g.edges[ei].minutes
        d = truth.worst_between(ei, t, t2)
        worst = max(worst, d)
        if d > limit:
            return {"arrived": False, "why": f"caught on {g.edges[ei].name or 'a road'}",
                    "minutes": round(t2 - depart, 1), "worst_m": round(d, 3)}
        t = t2
    return {"arrived": True, "why": "", "minutes": round(t - depart, 1),
            "worst_m": round(worst, 3)}


def route_scalar(g, route, sched: routes.Schedule, depart: float) -> float:
    """Deepest water on a route under a schedule: the scalar the margin covers.

    Also used to decide whether a trip is worth scoring at all, evaluated on the
    route the network gives with no water in it -- so trip selection depends on
    the ensemble's disagreement and not on any arm's choices.
    """
    if route is None or not route.ok:
        return float("nan")
    t, worst = depart, 0.0
    for ei in route.edges:
        t2 = t + g.edges[ei].minutes
        worst = max(worst, sched.worst_between(ei, t, t2))
        t = t2
    return worst


def belief_trace(g, s_, t_, truth: routes.Schedule,
                 others: list[routes.Schedule], args) -> list[dict]:
    """Per-slot belief state, and what departing at that slot would cost.

    The `filtered` arm walks forward and departs at the FIRST slot where a
    margin-feasible route exists. That rule is myopic by construction, and the
    results say so: it is caught by water 14 times out of 96 where the oracle is
    caught none. Departing later narrows the belief -- observed depths rule
    members out -- but roads close while you wait. That trade-off is a timing
    decision, and it is the one genuinely sequential choice in this project.

    None of this trajectory depends on the policy: which members survive at time
    t is a function of `truth` and t alone, and everything downstream is a
    function of the surviving set. So the trace is computed once per decision
    and every candidate policy is scored by scanning it, which is what makes
    nested cross-validation affordable at all.

    `arrives` is evaluated against `truth` and is therefore NOT a feature. It is
    the label, used for training on held-out members and for the hindsight bound.
    """
    trace = []
    n = max(len(others), 1)
    t_now = 0.0
    while t_now <= truth.horizon_min:
        keep = consistent(others, truth, t_now, args.obs_tol)
        sub = [others[i] for i in keep]
        fsub = mean_schedule(sub)
        r0 = routes.fastest_at(g, s_, t_, fsub, args.limit, t_now)
        vals = [route_scalar(g, r0, m, t_now) for m in sub] if r0.ok else []
        vals = [v for v in vals if not math.isnan(v)]
        qq = (margins.margin("residual", args.delta,
                             residuals=[abs(v - route_scalar(g, r0, fsub, t_now))
                                        for v in vals])
              if vals else 0.0)
        rr = routes.fastest_at(g, s_, t_, inflate(fsub, qq), args.limit, t_now)
        res = walk(g, rr, truth, args.limit, t_now) if rr.ok else None
        trace.append({
            # --- features, all computable without seeing `truth` ------------ #
            "t_frac": t_now / max(truth.horizon_min, 1e-9),
            "surv_frac": len(keep) / n,
            "spread": (float(np.std(vals)) if len(vals) > 1 else 0.0),
            "margin_m": qq,
            "feasible": bool(rr.ok),
            # --- label, from `truth`: never a feature ---------------------- #
            "arrives": bool(res and res["arrived"]),
            "minutes": (res["minutes"] if res and res["arrived"] else None),
            "t_now": t_now,
            # The route the feasibility and arrival above were evaluated on.
            # Held so the arms score the plan that was actually checked rather
            # than one re-derived from a different belief. Not serialised.
            "route": rr if rr.ok else None,
        })
        t_now += args.step_min
    return trace


# A policy is (survivor-fraction threshold, forced-go time fraction). Deliberately
# two parameters: 96 decisions under nested leave-one-out leaves ~72 training
# decisions per fold, which supports a threshold rule and does not support a
# network. Capacity is chosen from the sample size, not from ambition.
THETA = [i / 20 for i in range(21)]          # go once this fraction or fewer survive
TAU = [i / 20 for i in range(21)]            # ... or once this much time has passed


def apply_policy(trace: list[dict], theta: float, tau: float) -> dict | None:
    """First slot the policy departs at, or None if it never does.

    Waiting is only informative while members are still being ruled out, so the
    rule is: go when a margin-feasible route exists AND either the belief has
    narrowed to `theta` of the ensemble or `tau` of the window has elapsed. The
    second clause is what stops it waiting past the last feasible slot.
    """
    for slot in trace:
        if not slot["feasible"]:
            continue
        if slot["surv_frac"] <= theta or slot["t_frac"] >= tau:
            return slot
    return None


def best_timing(trace: list[dict]) -> dict | None:
    """The best slot in HINDSIGHT. An upper bound, not an arm.

    No causal policy can reach this: it reads `arrives`, which depends on the
    member that actually happened. It bounds what ANY timing rule could achieve
    on this belief trajectory, which separates "filtered departs at the wrong
    time" from "no departure time was any good".
    """
    ok = [s for s in trace if s["feasible"] and s["arrives"]]
    return min(ok, key=lambda s: s["t_now"]) if ok else None


def policy_score(recs: list[dict], theta: float, tau: float) -> float:
    """Fraction of decisions in `recs` on which this rule arrives."""
    got = 0
    for r in recs:
        sl = apply_policy(r["trace"], theta, tau)
        if sl is not None and sl["arrives"]:
            got += 1
    return got / max(len(recs), 1)


def fit_policy(train: list[dict]) -> tuple[float, float]:
    """Best (theta, tau) on `train` by exhaustive search over the grid.

    Ties go to the first maximiser, which with both grids ascending is the
    earliest-departing rule -- so when waiting buys nothing the fit reduces to
    the `filtered` baseline rather than to an arbitrary wait. The family
    therefore cannot score below `filtered` in-sample, which is the property
    that makes an improvement out-of-sample meaningful.
    """
    best, best_s = (0.0, 0.0), -1.0
    for th in THETA:
        for ta in TAU:
            v = policy_score(train, th, ta)
            if v > best_s:
                best, best_s = (th, ta), v
    return best


def nested_folds(traces: list[dict]) -> dict[int, list[dict]]:
    """Training set per outer fold: every trace whose truth member is not `k`.

    The outer evaluation holds out one ensemble member as reality. Fitting on
    all members would show the policy the member it is about to be scored
    against -- its schedule, the filtering that schedule induces, and the
    arrival labels that follow. This is the split that prevents it, and it is a
    function so a test can assert the exclusion rather than trust the comment.
    """
    members = sorted({r["member"] for r in traces})
    return {k: [r for r in traces if r["member"] != k] for k in members}


def consistent(scheds: list[routes.Schedule], truth: routes.Schedule,
               upto_min: float, tol: float) -> list[int]:
    """Which members agree with what has been seen up to now.

    This is the Perceive stage doing the only job it can do here: the depths
    already on the network are an observation, and they rule members out. No
    training, no filter parameters beyond a tolerance -- and the tolerance is
    stated rather than tuned.
    """
    keep = []
    for i, s in enumerate(scheds):
        err = 0.0
        n = 0
        for k, t in enumerate(s.times):
            if t > upto_min:
                break
            a = np.asarray(s.depths[k])
            b = np.asarray(truth.depths[k])
            err += float(np.abs(a - b).mean())
            n += 1
        if n == 0 or err / n <= tol:
            keep.append(i)
    return keep or list(range(len(scheds)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default="lismore-coraki", choices=sorted(PLACES))
    ap.add_argument("--dx", type=float, default=200.0)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--spread", type=float, default=0.6,
                    help="multiplier range: 1-spread .. 1+spread on discharge")
    ap.add_argument("--rain-mm-h", type=float, default=50.0)
    ap.add_argument("--storm-hours", type=float, default=6.0)
    ap.add_argument("--run-hours", type=float, default=12.0)
    ap.add_argument("--snapshots", type=int, default=13)
    ap.add_argument("--limit", type=float, default=routes.VEHICLE_DEPTH_M["car"])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--step-min", type=float, default=30.0)
    ap.add_argument("--trips", type=int, default=12)
    ap.add_argument("--obs-tol", type=float, default=0.02,
                    help="mean per-edge depth error, metres, for a member to "
                         "count as consistent with what has been observed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/evac_policy")
    args = ap.parse_args()

    t0 = time.time()
    place = PLACES[args.place]
    args.driver, args.q_m3s = place["driver"], place["q_m3s"]
    o, d = place["origin"], place["destination"]
    box = box_for(o, d)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from service.app import build_domain, fetch_roads

    print("=== terrain and network ===", flush=True)
    dom = build_domain(box["west"], box["south"], box["east"], box["north"],
                       dx=args.dx, name="evac")
    z = np.asarray(dom.z, dtype=np.float32)
    print(f"  grid {z.shape} at {dom.dx:.0f} m", flush=True)

    cache = out / "roads.json"
    net = (json.loads(cache.read_text()) if cache.exists()
           else fetch_roads(box["west"], box["south"], box["east"], box["north"]))
    if not cache.exists():
        cache.write_text(json.dumps(net))
    g = routes.build_graph(net.get("features", []))
    flat, offs = routes.edge_cells(g, box, z.shape)
    print(f"  {len(g.nodes)} junctions, {len(g.edges)} edges, "
          f"{g.total_km:.0f} km", flush=True)

    snap_h = [args.run_hours * i / (args.snapshots - 1) for i in range(args.snapshots)]
    mults = [1.0 - args.spread + 2 * args.spread * i / (args.members - 1)
             for i in range(args.members)]
    print(f"=== {args.members} members, discharge x{mults[0]:.2f}..{mults[-1]:.2f} "
          f"of {args.q_m3s:.0f} m3/s ===", flush=True)
    scheds = []
    for i, m in enumerate(mults):
        tm = time.time()
        sc, fx = solve_member(dom, box, m, args, snap_h, flat, offs)
        scheds.append(sc)
        print(f"  member {i} (x{m:.2f}): peak cut "
              f"{routes.cut_km(g, sc.peak(), args.limit):6.1f} km of "
              f"{g.total_km:.0f}, {time.time()-tm:.0f}s", flush=True)
    if fx["driver"] != args.driver:
        raise SystemExit(f"DEGENERATE: forcing fell back to {fx['driver']} "
                         f"({fx['note'][:120]}) -- the ensemble would be over a "
                         f"multiplier that does not reach the roads")

    spread_km = [routes.cut_km(g, s.peak(), args.limit) for s in scheds]
    if max(spread_km) - min(spread_km) < 1.0:
        raise SystemExit(
            f"DEGENERATE: every member cuts {min(spread_km):.1f}-"
            f"{max(spread_km):.1f} km, so the ensemble carries no disagreement "
            f"for a policy to be right or wrong about. Widen --spread.")

    # Trips the ensemble DISAGREES about.
    #
    # The first version of this picked destinations on ground the mean member
    # leaves dry, which selected away the entire question: no trip ever met
    # water, every arm scored 87.5% including the oracle, and six identical
    # rows looked like a result. A trip is in play only if the members straddle
    # the limit on it -- passable under some, cut under others -- which is the
    # same stratification section 5.13 applies and this did not inherit.
    rng = random.Random(args.seed)
    src = routes.nearest_node(g, float(o["lon"]), float(o["lat"]))
    dst0 = routes.nearest_node(g, float(d["lon"]), float(d["lat"]))
    cands = [dst0] + [n for n in g.adj if n not in (src, dst0)]
    rng.shuffle(cands[1:])

    pairs, skipped_unreachable, skipped_agreed = [], 0, 0
    for n in cands:
        if n == src or len(pairs) >= args.trips:
            continue
        geom = routes.fastest(g, src, n, [0.0] * len(g.edges), args.limit)
        if not geom.ok or not geom.edges:
            skipped_unreachable += 1
            continue
        vals = [route_scalar(g, geom, m, 0.0) for m in scheds]
        if not (min(vals) <= args.limit < max(vals)):
            skipped_agreed += 1
            continue
        pairs.append((src, n))
    print(f"=== {len(pairs)} trips the members disagree about "
          f"({skipped_agreed} agreed, {skipped_unreachable} unreachable) ===",
          flush=True)
    if not pairs:
        raise SystemExit(
            "DEGENERATE: no trip is passable under some members and cut under "
            "others, so every arm would score identically and the comparison "
            "would be of nothing. Widen --spread or raise --trips.")

    arms = ["leave-now", "static-peak", "dp-mean", "conformal", "filtered",
            "learned", "best-timing", "oracle"]
    tally = {a: {"arrived": 0, "caught": 0, "no_plan": 0, "minutes": [],
                 "waited": []} for a in arms}
    rows = []
    traces: list[dict] = []

    for k_true, truth in enumerate(scheds):
        others = [s for j, s in enumerate(scheds) if j != k_true]
        fc = mean_schedule(others)
        for (s_, t_) in pairs:
            if s_ == t_:
                continue
            # The margin: conformal quantile of the route scalar's error across
            # held-out members, which is the quantity the decision reads.
            base = routes.fastest_at(g, s_, t_, fc, args.limit, 0.0)
            resid = [abs(route_scalar(g, base, m, 0.0)
                         - route_scalar(g, base, fc, 0.0))
                     for m in others] if base.ok else []
            resid = [r for r in resid if not math.isnan(r)]
            q = margins.margin("residual", args.delta, residuals=resid) if resid else 0.0

            plans: dict[str, tuple] = {}
            plans["leave-now"] = (base if base.ok else None, 0.0)
            plans["static-peak"] = (
                routes.fastest(g, s_, t_, fc.peak(), args.limit), 0.0)
            dl, dlr = routes.latest_departure(g, s_, t_, fc, args.limit,
                                              step_min=args.step_min)
            plans["dp-mean"] = (base if base.ok else dlr, 0.0 if base.ok else (dl or 0.0))
            infl = inflate(fc, q)
            rb = routes.fastest_at(g, s_, t_, infl, args.limit, 0.0)
            plans["conformal"] = (rb if rb.ok else None, 0.0)
            ora = routes.fastest_at(g, s_, t_, truth, args.limit, 0.0)
            plans["oracle"] = (ora if ora.ok else None, 0.0)

            # filtered: step forward, keep members consistent with what has been
            # seen, re-plan on their mean plus the margin over what is left.
            dep, chosen, waited = 0.0, None, 0.0
            t_now = 0.0
            while t_now <= fc.horizon_min:
                keep = consistent(others, truth, t_now, args.obs_tol)
                sub = [others[i] for i in keep]
                fsub = mean_schedule(sub)
                r0 = routes.fastest_at(g, s_, t_, fsub, args.limit, t_now)
                rq = [abs(route_scalar(g, r0, m, t_now)
                          - route_scalar(g, r0, fsub, t_now)) for m in sub] if r0.ok else []
                rq = [x for x in rq if not math.isnan(x)]
                qq = margins.margin("residual", args.delta, residuals=rq) if rq else 0.0
                rr = routes.fastest_at(g, s_, t_, inflate(fsub, qq), args.limit, t_now)
                if rr.ok:
                    dep, chosen, waited = t_now, rr, t_now
                    break
                t_now += args.step_min
            plans["filtered"] = (chosen, dep)

            # The belief trajectory over every slot, for the learned arm and the
            # hindsight bound. Both are decided after the loop, because training
            # needs the other members' traces and cannot be done inline.
            tr = belief_trace(g, s_, t_, truth, others, args)
            traces.append({"member": k_true, "from": s_, "to": t_, "trace": tr})
            bt = best_timing(tr)
            plans["best-timing"] = ((bt["route"], bt["t_now"]) if bt
                                    else (None, 0.0))

            for a in arms:
                if a == "learned":
                    continue                     # scored after the policy is fit
                r, dp = plans[a]
                res = walk(g, r, truth, args.limit, dp)
                key = ("arrived" if res["arrived"]
                       else "no_plan" if res["why"] == "no plan" else "caught")
                tally[a][key] += 1
                if res["arrived"]:
                    tally[a]["minutes"].append(res["minutes"])
                    tally[a]["waited"].append(dp)
                rows.append({"member": k_true, "from": s_, "to": t_, "arm": a,
                             "depart_min": dp, **res, "margin_m": round(q, 3)})

    # ---------------------------------------------------------------------- #
    # The learned arm, with NESTED leave-one-out so nothing leaks.
    #
    # The outer evaluation already holds out one member as reality. Fitting the
    # policy on all members would let it see the member it is about to be scored
    # against -- the schedule, the filtering it induces, and the arrival labels
    # that follow from it. So for outer fold k the policy is fitted using only
    # traces whose truth member is NOT k, and applied to fold k unseen.
    #
    # With 8 members that is ~84 training decisions per fold for a two-parameter
    # rule. Thin, and the reason the policy family is a threshold rather than a
    # network: see THETA/TAU.
    by_member: dict[int, list[dict]] = {}
    for rec in traces:
        by_member.setdefault(rec["member"], []).append(rec)
    train_sets = nested_folds(traces)

    fitted: dict[int, tuple[float, float]] = {}
    for k in sorted(by_member):
        best = fit_policy(train_sets[k])
        fitted[k] = best
        for rec in by_member[k]:
            sl = apply_policy(rec["trace"], *best)
            r, dp = (sl["route"], sl["t_now"]) if sl else (None, 0.0)
            res = walk(g, r, scheds[k], args.limit, dp)
            key = ("arrived" if res["arrived"]
                   else "no_plan" if res["why"] == "no plan" else "caught")
            tally["learned"][key] += 1
            if res["arrived"]:
                tally["learned"]["minutes"].append(res["minutes"])
                tally["learned"]["waited"].append(dp)
            rows.append({"member": k, "from": rec["from"], "to": rec["to"],
                         "arm": "learned", "depart_min": dp, **res,
                         "theta": best[0], "tau": best[1]})

    # Why the learned arm and the hindsight bound coincide. If feasibility only
    # ever decays -- a flood that rises through the window closes roads and
    # reopens none -- then the earliest feasible departure is the best one and
    # no timing rule can help. That is a claim about the hazard, so it is
    # measured rather than asserted.
    mono = gap = n_dec = 0
    later_only = 0
    for rec in traces:
        tr = rec["trace"]
        feas = [i for i, sl in enumerate(tr) if sl["feasible"]]
        arr = [i for i, sl in enumerate(tr) if sl["feasible"] and sl["arrives"]]
        n_dec += 1
        if feas:
            # Monotone decay: once infeasible, never feasible again.
            last = feas[-1]
            if all(tr[i]["feasible"] for i in range(feas[0], last + 1)):
                mono += 1
            else:
                gap += 1
        if arr and feas and arr[0] > feas[0]:
            later_only += 1
    print(f"\n=== is waiting ever worth it? ({n_dec} decisions) ===", flush=True)
    print(f"  feasibility is a contiguous window: {mono}/{mono + gap}"
          f"  (reopens after closing: {gap})", flush=True)
    print(f"  decisions where a LATER departure arrives but the earliest "
          f"feasible one does not: {later_only}", flush=True)
    print("  zero means the earliest feasible departure is always optimal, so "
          "no timing policy can improve on it.", flush=True)

    print("\n=== learned timing policy, nested leave-one-out ===", flush=True)
    for k in sorted(fitted):
        th, ta = fitted[k]
        print(f"  fold {k}: go when <= {th:.2f} of members survive, "
              f"or after {ta:.2f} of the window", flush=True)
    uniq = set(fitted.values())
    print(f"  {len(uniq)} distinct rule(s) across {len(fitted)} folds"
          + ("  -- stable" if len(uniq) == 1 else "  -- fold-dependent, read with care"),
          flush=True)

    def pct(a: str) -> float:
        t = tally[a]
        n = t["arrived"] + t["caught"] + t["no_plan"]
        return round(100 * t["arrived"] / n, 1) if n else float("nan")

    summary = {
        "place": args.place, "dx_m": dom.dx, "grid": list(z.shape),
        "members": args.members, "spread": args.spread,
        "discharge_m3s": args.q_m3s, "limit_m": args.limit, "delta": args.delta,
        "trips": len(pairs), "network_km": round(g.total_km, 1),
        "peak_cut_km_range": [round(min(spread_km), 1), round(max(spread_km), 1)],
        "arms": {a: {"arrived_pct": pct(a), **{k: v for k, v in tally[a].items()
                                               if k not in ("minutes", "waited")},
                     "median_minutes": (round(float(np.median(tally[a]["minutes"])), 1)
                                        if tally[a]["minutes"] else None),
                     "median_wait_min": (round(float(np.median(tally[a]["waited"])), 1)
                                         if tally[a]["waited"] else None)}
                 for a in arms},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n=== {len(pairs)} trips x {args.members} held-out members "
          f"= {len(pairs) * args.members} decisions ===")
    print("| arm | got through | caught by water | no plan | median drive | median wait |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        t = tally[a]
        print(f"| {a} | **{pct(a)}%** | {t['caught']} | {t['no_plan']} | "
              f"{summary['arms'][a]['median_minutes']} min | "
              f"{summary['arms'][a]['median_wait_min']} min |")
    print(f"\npeak cut across members: {min(spread_km):.0f}-{max(spread_km):.0f} km "
          f"of {g.total_km:.0f} km")
    print(f"{summary['seconds']}s -> {out}/results.json")


if __name__ == "__main__":
    main()
