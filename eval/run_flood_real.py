"""Levee planning and the conformal margin on FloodCastBench terrain.

The real-terrain counterpart of `run_flood_span.py` / `run_flood_margin.py`,
which ran on a synthetic valley. Everything here comes from the archive: the
DEM, the initial depth field, the flood's magnitude and timing (from the
reference's mass budget), the settlement (chosen for how deeply it floods in the
reference) and the levee sites (the lowest point of each perimeter sector).

Stages:
    span    do-nothing against the best allocation -- the gate. `swe` was retired
            at ~7% and the synthetic flood testbed reached 100%; this asks what
            real topography gives.
    margin  the three recipes under rainfall uncertainty, with the solver as
            reality. The planner commits levee heights before the rainfall is
            known, which is the real design problem.

Run:
    python eval/run_flood_real.py --stage span  --root <extracted> --out runs/flood_real
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from pspe.plan.margins import MODES, margin
from pspe.simulate.flood import FloodConfig, FloodSolver
from pspe.simulate.real.floodcast_scenario import build_scenario


class RealReality:
    """Exposure-weighted peak depth over the settlement, for (plan, rain scale)."""

    def __init__(self, scn, args, device) -> None:
        self.s, self.args, self.device = scn, args, device
        rows = scn.z.mean(dim=1)
        slope = max(abs(float((rows[0] - rows[-1]) / (scn.z.shape[0] * scn.dx))), 1e-5)
        self.cfg = FloodConfig(dx=scn.dx, open_edges=tuple(args.open_edges),
                               manning=args.manning, bed_slope=slope, dt_max=300.0)
        self.slope = slope

    @property
    def k(self) -> int:
        return self.s.k

    @torch.no_grad()
    def depth(self, plans: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        s, args = self.s, self.args
        b = plans.shape[0]
        z = s.z[None] + (plans[:, :, None, None] * s.masks[None]).sum(1)
        solver = FloodSolver(z, self.cfg)
        h = s.h0[None].expand(b, *s.z.shape).clone()
        qx, qy = solver.zeros_flux(b)
        peak = h.clone()
        t, dur = 0.0, s.duration * args.duration_frac
        rain = (s.rain * scales)[:, None, None]
        while t < dur:
            dt = min(solver.adaptive_dt(h), dur - t)
            if dt <= 0:
                break
            h, qx, qy = solver.step(h, qx, qy, dt, rain=rain, z=z)
            peak = torch.maximum(peak, h)
            t += dt
        return (peak * s.exposure).flatten(1).sum(1)


class Surrogate(nn.Module):
    def __init__(self, k, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k + 1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, p, s):
        return self.net(torch.cat([p, s[:, None]], -1)).squeeze(-1)


def sample_scale(n, args, gen, device):
    return torch.exp(torch.randn(n, generator=gen, device=device) * args.rain_log_sigma)


def sample_plans(n, k, budget, max_h, gen, device):
    spend = torch.rand(n, 1, generator=gen, device=device) * budget
    w = torch.rand(n, k, generator=gen, device=device)
    return ((w / w.sum(1, keepdim=True)) * spend).clamp(max=max_h)


def batched(reality, plans, scales, batch, tag=""):
    """Solve in batches, announcing progress.

    The first run of the margin stage printed nothing for 91 minutes because
    this function was silent, so there was no way to tell a slow job from a
    stuck one until the wall clock decided.
    """
    out = []
    n = plans.shape[0]
    t0 = time.time()
    for i in range(0, n, batch):
        out.append(reality.depth(plans[i : i + batch], scales[i : i + batch]))
        done = min(i + batch, n)
        el = time.time() - t0
        print(f"    [{tag}] {done}/{n}  {el:.0f}s elapsed, "
              f"~{el / done * (n - done):.0f}s left", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--event", default="australia2022")
    ap.add_argument("--stage", choices=["span", "precond", "margin"], default="span")
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--start-hours", type=float, default=144.0)
    ap.add_argument("--end-hours", type=float, default=192.0)
    ap.add_argument("--duration-frac", type=float, default=1.0)
    ap.add_argument("--settle-radius", type=int, default=10)
    ap.add_argument("--settle-rc", type=int, nargs=2, default=None,
                    help="explicit settlement centre (row col), e.g. from the "
                         "controllability screen; default picks the deepest-"
                         "flooding block, which is NOT the best-defended one")
    ap.add_argument("--n-sites", type=int, default=6)
    ap.add_argument("--manning", type=float, default=0.035)
    ap.add_argument("--open-edges", nargs="*", default=[])
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--rain-log-sigma", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--f-fraction", type=float, default=0.3)
    ap.add_argument("--limit", type=float, default=-1.0,
                    help="depth limit; <0 places it midway between do-nothing "
                         "and the best achievable, which is where a constraint "
                         "actually binds")
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-cal", type=int, default=96)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--n-span", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--plan-samples", type=int, default=48)
    ap.add_argument("--plan-steps", type=int, default=250)
    ap.add_argument("--plan-lr", type=float, default=0.03)
    ap.add_argument("--w-violate", type=float, default=200.0)
    ap.add_argument("--w-spend", type=float, default=0.02)
    ap.add_argument("--arms", nargs="+", default=["residual", "none", "model_error", "episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/flood_real")
    args = ap.parse_args()

    device = torch.device(args.device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    scn = build_scenario(args.root, args.event, coarsen=args.coarsen,
                         start_hours=args.start_hours, end_hours=args.end_hours,
                         settle_radius=args.settle_radius, n_sites=args.n_sites,
                         settle_rc=tuple(args.settle_rc) if args.settle_rc else None,
                         device=device)
    reality = RealReality(scn, args, device)
    k = reality.k
    print(json.dumps(scn.meta, indent=2), flush=True)
    print(f"bed slope {reality.slope:.2e}  sites at elevations "
          f"{[round(e,1) for e in scn.site_elevations]}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    zero = torch.zeros(1, k, device=device)
    ones = torch.ones(1, device=device)

    if args.stage == "span":
        base = float(reality.depth(zero, ones))
        print(f"\ndo-nothing exposure-weighted peak depth {base:.5f} m", flush=True)
        print("\n=== single site at full height ===", flush=True)
        singles = []
        for i in range(k):
            p = torch.zeros(1, k, device=device); p[0, i] = args.max_height
            d = float(reality.depth(p, ones))
            singles.append({"site": i, "elev": scn.site_elevations[i], "depth": d,
                            "reduction_pct": 100 * (base - d) / base})
            print(f"  site {i} (perimeter elev {scn.site_elevations[i]:5.1f} m) "
                  f"depth {d:.5f}  reduction {100*(base-d)/base:+7.2f}%", flush=True)
        d_all = float(reality.depth(torch.full((1, k), args.max_height, device=device), ones))

        plan, spent, cur, trace = [0.0] * k, 0.0, base, []
        print(f"\n=== greedy under budget {args.budget} m ===", flush=True)
        while spent + args.step <= args.budget + 1e-9:
            best = None
            for i in range(k):
                if plan[i] + args.step > args.max_height:
                    continue
                c = list(plan); c[i] += args.step
                d = float(reality.depth(torch.tensor([c], device=device), ones))
                if best is None or d < best[0]:
                    best = (d, i)
            if best is None or best[0] >= cur - 1e-7:
                print("  no further improvement", flush=True); break
            cur, i = best; plan[i] += args.step; spent += args.step
            trace.append({"spent": spent, "site": i, "depth": cur,
                          "reduction_pct": 100 * (base - cur) / base})
            print(f"  +{args.step} at site {i} -> {cur:.5f} "
                  f"({100*(base-cur)/base:+7.2f}%) spent {spent:.1f}", flush=True)

        res = {"stage": "span", "do_nothing": base, "best_budgeted": cur,
               "best_plan": plan, "unconstrained": d_all,
               "span_pct_budgeted": 100 * (base - cur) / base,
               "span_pct_unconstrained": 100 * (base - d_all) / base,
               "singles": singles, "greedy": trace,
               "scenario": scn.meta, "config": vars(args)}
        (out / "span_real.json").write_text(json.dumps(res, indent=2))
        print(f"\n=== SPAN ON REAL TERRAIN: {res['span_pct_budgeted']:.2f}% budgeted, "
              f"{res['span_pct_unconstrained']:.2f}% unconstrained ===", flush=True)
        print("(synthetic valley gave 100%; swe was retired at ~7%)", flush=True)
        return

    if args.stage == "precond":
        # The gate, and nothing else. Proposition 2 needs the achievable span
        # and the spread of realised depth; it needs neither a trained surrogate
        # nor a planning loop. Running the full margin stage to discover the
        # margin does not fit would cost ~800 solves to learn what ~250 answer.
        cands = [torch.full((1, k), args.budget / k, device=device)]
        for j in range(k):
            c = torch.zeros(1, k, device=device)
            c[0, j] = min(args.budget, args.max_height)
            cands.append(c)
        q_span = sample_scale(args.n_span, args, gen, device)
        stack = torch.cat([p.expand(args.n_span, k) for p in [zero] + cands])
        sc = q_span.repeat(len(cands) + 1)
        dep = batched(reality, stack, sc, args.batch, "span").view(
            len(cands) + 1, args.n_span).mean(1)
        d_zero = float(dep[0])
        d_best = float(dep[1:].min())
        best_j = int(dep[1:].argmin())
        span = d_zero - d_best

        ref = torch.full((1, k), args.budget / (2 * k), device=device)
        q_ref = sample_scale(args.n_cal, args, gen, device)
        d_ref = batched(reality, ref.expand(args.n_cal, k), q_ref, args.batch, "sigma")
        sigma = float(d_ref.std())
        z_d = float(torch.distributions.Normal(0.0, 1.0).icdf(
            torch.tensor(1.0 - args.delta)))
        need = z_d * sigma          # bias needs a surrogate; reported separately
        head = args.f_fraction * span
        verdict = "PASS" if need < head else "FAIL"
        print(f"\n=== precondition at rain log-sigma {args.rain_log_sigma} ===", flush=True)
        print(f"  do-nothing      {d_zero:.5f} m", flush=True)
        print(f"  best allocation {d_best:.5f} m  (#{best_j}, 0=even spread)", flush=True)
        print(f"  span s          {span:.5f} m", flush=True)
        print(f"  sigma           {sigma:.5f} m", flush=True)
        print(f"  z*sigma         {need:.5f} m   (bias excluded: no surrogate here)", flush=True)
        print(f"  headroom f*s    {head:.5f} m   (f={args.f_fraction})", flush=True)
        print(f"  verdict         {verdict}   margin is "
              f"{100*need/max(span,1e-12):.0f}% of the span", flush=True)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "precond_real.json").write_text(json.dumps({
            "stage": "precond", "rain_log_sigma": args.rain_log_sigma,
            "do_nothing": d_zero, "best": d_best, "best_index": best_j,
            "span": span, "sigma": sigma, "z_delta": z_d,
            "required_margin_no_bias": need, "headroom": head,
            "margin_pct_of_span": 100 * need / max(span, 1e-12),
            "precondition": verdict, "scenario": scn.meta, "config": vars(args),
        }, indent=2))
        print(f"wrote {out/'precond_real.json'}", flush=True)
        return

    # ---- margin stage -----------------------------------------------------
    print("\n=== generating solver data ===", flush=True)
    ptr = sample_plans(args.n_train, k, args.budget, args.max_height, gen, device)
    str_ = sample_scale(args.n_train, args, gen, device)
    dtr = batched(reality, ptr, str_, args.batch, "train")
    print(f"  train depths {float(dtr.min()):.4f}..{float(dtr.max()):.4f} m", flush=True)

    model = Surrogate(k, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_val = max(8, args.n_train // 5)
    for _ in range(args.epochs):
        perm = torch.randperm(args.n_train - n_val, device=device)
        for i in range(0, args.n_train - n_val, args.mb):
            j = perm[i : i + args.mb]
            loss = (model(ptr[j], str_[j]) - dtr[j]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        eps = float((model(ptr[-n_val:], str_[-n_val:]) - dtr[-n_val:]).abs().mean())
    print(f"surrogate eps = {eps:.5f} m", flush=True)

    # span, over the best of the candidate allocations
    cands = [torch.full((1, k), args.budget / k, device=device)]
    for j in range(k):
        c = torch.zeros(1, k, device=device); c[0, j] = min(args.budget, args.max_height)
        cands.append(c)
    q_span = sample_scale(args.n_span, args, gen, device)
    stack = torch.cat([p.expand(args.n_span, k) for p in [zero] + cands])
    sc = q_span.repeat(len(cands) + 1)
    dep = batched(reality, stack, sc, args.batch).view(len(cands) + 1, args.n_span).mean(1)
    d_zero = float(dep[0]); d_best = float(dep[1:].min())
    span = d_zero - d_best
    limit = args.limit if args.limit > 0 else d_best + 0.5 * span
    print(f"do-nothing {d_zero:.5f}  best {d_best:.5f}  span {span:.5f}  limit {limit:.5f}",
          flush=True)

    ref = torch.full((1, k), args.budget / (2 * k), device=device)
    q_ref = sample_scale(args.n_cal, args, gen, device)
    d_ref = batched(reality, ref.expand(args.n_cal, k), q_ref, args.batch)
    sigma = float(d_ref.std())
    z_d = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - args.delta)))
    with torch.no_grad():
        g_ref = model(ref.expand(args.n_cal, k), q_ref)
    bias = float((d_ref - g_ref).mean())
    need, head = bias + z_d * sigma, args.f_fraction * span
    verdict = "PASS" if need < head else "FAIL"
    print(f"sigma {sigma:.5f}  eps {eps:.5f}  rho {sigma/max(eps,1e-12):.2f}", flush=True)
    print(f"precondition: need {need:.5f} vs headroom {head:.5f} -> {verdict} "
          f"({100*need/max(span,1e-12):.0f}% of span)", flush=True)

    def plan_for(qm):
        a = torch.full((1, k), args.budget / (2 * k), device=device, requires_grad=True)
        qs = sample_scale(args.plan_samples, args, gen, device)
        o = torch.optim.Adam([a], lr=args.plan_lr)
        tgt = limit - qm
        for _ in range(args.plan_steps):
            pred = model(a.expand(args.plan_samples, k), qs).mean()
            loss = args.w_violate * torch.relu(pred - tgt) + args.w_spend * a.clamp(min=0).sum()
            o.zero_grad(); loss.backward(); o.step()
            with torch.no_grad():
                a.clamp_(0.0, args.max_height)
                tot = a.sum()
                if float(tot) > args.budget:
                    a.mul_(args.budget / tot)
        return a.detach()

    a0 = plan_for(0.0)
    q_cal = sample_scale(args.n_cal, args, gen, device)
    d_cal = batched(reality, a0.expand(args.n_cal, k), q_cal, args.batch)
    with torch.no_grad():
        g_cal = model(a0.expand(args.n_cal, k), q_cal)
        g_hat = float(g_cal.mean())
    print(f"\nmargin-free plan {[round(float(x),3) for x in a0[0]]}  "
          f"realised {float(d_cal.mean()):.5f}  controlled {g_hat:.5f}", flush=True)

    results = {}
    for arm in args.arms:
        if arm == "none":
            qm = 0.0
        elif arm not in MODES:
            raise SystemExit(f"unknown arm {arm}")
        else:
            qm = margin(arm, args.delta,
                        model_err=[float(x) for x in (d_cal - g_cal)],
                        deviations=[float(x) for x in (d_cal - d_cal.mean())],
                        residuals=[float(x) for x in (d_cal - g_hat)],
                        bias=float((d_cal - g_cal).mean()))
        degenerate = qm >= limit
        a1 = plan_for(qm)
        q_ev = sample_scale(args.n_eval, args, gen, device)
        d_ev = batched(reality, a1.expand(args.n_eval, k), q_ev, args.batch)
        viol = float((d_ev > limit).float().mean())
        results[arm] = {"margin": qm, "plan": [float(x) for x in a1[0]],
                        "spend": float(a1.sum()), "violation_rate": viol,
                        "mean_depth": float(d_ev.mean()), "worst": float(d_ev.max()),
                        "covers": viol <= args.delta, "degenerate": bool(degenerate)}
        flag = "COVERS" if viol <= args.delta else "UNDER-COVERS"
        if degenerate:
            flag += " [DEGENERATE: margin >= limit]"
        print(f"  [{arm:12s}] q={qm:.5f} spend={float(a1.sum()):.3f} "
              f"violation={viol:.3f} (delta={args.delta})  {flag}", flush=True)

    res = {"stage": "margin", "epsilon": eps, "sigma": sigma,
           "rho": sigma / max(eps, 1e-12), "bias": bias, "span": span,
           "limit": limit, "required_margin": need, "headroom": head,
           "precondition": verdict, "do_nothing": d_zero, "best": d_best,
           "arms": results, "scenario": scn.meta, "config": vars(args),
           "note": ("Margin calibrated at the margin-free plan a0 and deployed at "
                    "a1=plan(q); exchangeability holds in the rainfall scale, not "
                    "across the plan shift. Forcing is uniform in space because "
                    "the archive ships no rainfall field.")}
    (out / "margin_real.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out/'margin_real.json'}", flush=True)


if __name__ == "__main__":
    main()
