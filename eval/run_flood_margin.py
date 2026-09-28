"""Phase 3: the conformal margin on flood, with a solver as reality.

The structure the paper's central claim needs, instantiated on a real hazard:

  decision      levee heights `a` at K sites, under a construction budget
  uncertainty   hydrograph magnitude Q, unknown when the levee is built
  surrogate     g(a, Q) -> exposure-weighted peak depth, learned from solves
  reality       the LISFLOOD-FP local-inertial solver (`pspe.simulate.flood`)
  constraint    realised depth D <= limit, at failure rate delta
  margin        tighten the limit to limit - q, q from `pspe.plan.margins`

This is the real levee-design problem: freeboard chosen under flood-frequency
uncertainty. It is also the setting where the three margin recipes separate. The
planner controls `E_Q[g(a, Q)]`, an *expectation*, while the limit is declared on
the *realised* depth of a single flood. That is exactly the mismatch §3.5 derived:
matched-pair scores cancel the per-instance scatter that actually breaches the
limit, so the default under-covers once the hydrograph spread exceeds the
surrogate's error.

The diagnostic is rho = sigma / epsilon, with sigma the spread of realised depth
across hydrographs and epsilon the surrogate's error. Coverage of the default
recipe should fail for rho >= 1 and hold below it. Both knobs are explicit here:
`--q-log-sigma` sets sigma, `--n-train`/`--hidden` set epsilon.

Unlike every planning result in Part 5, the reference model here is an accepted
solver rather than our own surrogate with noise added. It is still a model, not
the world (§6.2); it is route (b), taken deliberately.

Run:
    python3 eval/run_flood_margin.py --out runs/flood_margin --arms residual none
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from pspe.plan.margins import MODES, margin
from pspe.simulate.flood import (
    FloodConfig,
    FloodSolver,
    floodplain_terrain,
    levee_basis,
)

SITES = (0.42, 0.50, 0.60, 0.68, 0.78, 0.86)


# --------------------------------------------------------------------------- #
# Reality: the solver
# --------------------------------------------------------------------------- #
class FloodReality:
    """Exposure-weighted peak depth over the settlement, for (plan, hydrograph)."""

    def __init__(self, args, device: torch.device) -> None:
        self.args = args
        self.device = device
        z, exposure = floodplain_terrain(
            grid=args.grid, dx=args.dx, slope=args.slope,
            channel_depth=5.0, channel_width=5.0, berm_height=3.0,
            gaps=((0.42, 0.55), (0.60, 0.80), (0.78, 0.40)),
            town_centre=(0.62, 0.20), town_radius=0.060, town_drop=1.2,
            device=device,
        )
        self.z0 = z.to(device)
        self.exposure = exposure.to(device)
        self.masks = levee_basis(args.grid, dx=args.dx, sites=SITES).to(device)
        self.town = self.exposure > self.exposure.max() * 0.3
        self.cfg = FloodConfig(
            dx=args.dx, open_edges=("south",), manning=args.manning,
            bed_slope=args.slope,
        )
        # Geometry must be separated or the study is meaningless (rule 17).
        cells = self.town.nonzero()
        berm_min = int((self.z0[:, 25:45].argmax(dim=1) + 25).min())
        if int(cells[:, 1].max()) >= berm_min:
            raise SystemExit("geometry overlap: settlement reaches the berm")
        chan0 = int(round(0.5 * (args.grid - 1)))
        self.patch = (chan0 - 2, chan0 + 3)
        if self.patch[0] <= berm_min:
            raise SystemExit("inflow patch straddles the berm")

    @property
    def k(self) -> int:
        return self.masks.shape[0]

    @torch.no_grad()
    def depth(self, plans: torch.Tensor, inflows: torch.Tensor) -> torch.Tensor:
        """(B, K) plans and (B,) hydrographs -> (B,) exposure-weighted peak depth."""
        b = plans.shape[0]
        z = self.z0 + (plans[:, :, None, None] * self.masks[None]).sum(1)
        solver = FloodSolver(z, self.cfg)
        h = torch.zeros(b, self.args.grid, self.args.grid, device=self.device)
        qx, qy = solver.zeros_flux(b)
        peak = torch.zeros_like(h)
        lo, hi = self.patch
        t, dur = 0.0, self.args.days * 86400.0
        while t < dur:
            dt = min(solver.adaptive_dt(h), dur - t)
            frac = t / dur
            shape = min(frac / 0.33, max(0.0, (1.0 - frac) / 0.67))
            h, qx, qy = solver.step(h, qx, qy, dt, z=z)
            h[:, 0:3, lo:hi] = h[:, 0:3, lo:hi] + dt * (inflows[:, None, None] * shape)
            peak = torch.maximum(peak, h)
            t += dt
        return (peak * self.exposure).flatten(1).sum(1)


# --------------------------------------------------------------------------- #
# Surrogate
# --------------------------------------------------------------------------- #
class Surrogate(nn.Module):
    def __init__(self, k: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, plans: torch.Tensor, inflow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([plans, inflow[:, None]], dim=-1)
        return self.net(x).squeeze(-1)


def sample_plans(n: int, k: int, budget: float, max_h: float, gen, device) -> torch.Tensor:
    """Feasible plans: Dirichlet split of a random spend, capped per site."""
    spend = torch.rand(n, 1, generator=gen, device=device) * budget
    w = torch.rand(n, k, generator=gen, device=device)
    w = w / w.sum(1, keepdim=True)
    return (w * spend).clamp(max=max_h)


def sample_inflow(n: int, args, gen, device) -> torch.Tensor:
    """Log-normal hydrograph magnitude: flood-frequency uncertainty."""
    eps = torch.randn(n, generator=gen, device=device) * args.q_log_sigma
    return args.inflow * torch.exp(eps)


def generate(reality, n, args, gen, tag) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plans, inflows, depths = [], [], []
    done = 0
    while done < n:
        b = min(args.batch, n - done)
        p = sample_plans(b, reality.k, args.budget, args.max_height, gen, reality.device)
        q = sample_inflow(b, args, gen, reality.device)
        d = reality.depth(p, q)
        plans.append(p); inflows.append(q); depths.append(d)
        done += b
        print(f"  [{tag}] {done}/{n}", flush=True)
    return torch.cat(plans), torch.cat(inflows), torch.cat(depths)


def train_surrogate(plans, inflows, depths, args, device) -> tuple[Surrogate, float]:
    model = Surrogate(plans.shape[1], args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = plans.shape[0]
    n_val = max(8, n // 5)
    tr = slice(0, n - n_val)
    va = slice(n - n_val, n)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n - n_val, device=device)
        for i in range(0, n - n_val, args.mb):
            idx = perm[i : i + args.mb]
            loss = (model(plans[tr][idx], inflows[tr][idx]) - depths[tr][idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        err = (model(plans[va], inflows[va]) - depths[va]).abs().mean()
    return model, float(err)


# --------------------------------------------------------------------------- #
# Planning: minimise construction spend subject to the (margined) limit
# --------------------------------------------------------------------------- #
def plan_under_margin(model, reality, args, q_margin, gen) -> torch.Tensor:
    """Projected gradient on the differentiable surrogate.

    The planner controls E_Q[g(a, Q)] -- an expectation over hydrographs -- and
    must keep it below `limit - q_margin`, while spending as little as possible.
    """
    k = reality.k
    a = torch.full((1, k), args.budget / (2 * k), device=reality.device, requires_grad=True)
    qs = sample_inflow(args.plan_samples, args, gen, reality.device)
    opt = torch.optim.Adam([a], lr=args.plan_lr)
    target = args.limit - q_margin
    for _ in range(args.plan_steps):
        rep = a.expand(args.plan_samples, k)
        pred = model(rep, qs).mean()
        spend = a.clamp(min=0).sum()
        # Penalty form: heavy on violating the margined limit, light on spend.
        loss = args.w_violate * torch.relu(pred - target) + args.w_spend * spend
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a.clamp_(0.0, args.max_height)
            total = a.sum()
            if float(total) > args.budget:      # project onto the budget simplex
                a.mul_(args.budget / total)
    return a.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--dx", type=float, default=480.0)
    ap.add_argument("--slope", type=float, default=1.5e-4)
    ap.add_argument("--manning", type=float, default=0.03)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--inflow", type=float, default=1.4e-2)
    ap.add_argument("--q-log-sigma", type=float, default=0.25,
                    help="hydrograph log-spread: the sigma in rho = sigma/epsilon")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--max-height", type=float, default=3.0)
    ap.add_argument("--limit", type=float, default=0.06,
                    help="exposure-weighted peak depth limit, m")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--f-fraction", type=float, default=0.3,
                    help="fraction f of the achievable span s that may be spent "
                         "on safety; the precondition is b + z*sigma < f*s")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--n-span", type=int, default=48,
                    help="hydrographs per candidate in the span scan; the span is "
                         "an average, so it needs far fewer draws than calibration")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--plan-samples", type=int, default=64)
    ap.add_argument("--plan-steps", type=int, default=300)
    ap.add_argument("--plan-lr", type=float, default=0.02)
    ap.add_argument("--w-violate", type=float, default=200.0)
    ap.add_argument("--w-spend", type=float, default=0.02)
    ap.add_argument("--arms", nargs="+", default=["residual", "none", "model_error", "episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="report the precondition b + z*sigma < f*s and stop")
    ap.add_argument("--out", type=str, default="runs/flood_margin")
    args = ap.parse_args()

    device = torch.device(args.device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    reality = FloodReality(args, device)

    print("=== generating solver data ===", flush=True)
    ptr, itr, dtr = generate(reality, args.n_train, args, gen, "train")
    model, eps = train_surrogate(ptr, itr, dtr, args, device)
    print(f"surrogate mean abs error eps = {eps:.5f} m", flush=True)

    # sigma: spread of realised depth across hydrographs at a fixed reference plan.
    ref = torch.full((1, reality.k), args.budget / (2 * reality.k), device=device)
    q_ref = sample_inflow(args.n_cal, args, gen, device)
    d_ref = []
    for i in range(0, args.n_cal, args.batch):
        qq = q_ref[i : i + args.batch]
        d_ref.append(reality.depth(ref.expand(qq.shape[0], reality.k), qq))
    d_ref = torch.cat(d_ref)
    sigma = float(d_ref.std())
    rho = sigma / max(eps, 1e-12)
    print(f"sigma = {sigma:.5f} m   epsilon = {eps:.5f} m   rho = {rho:.3f}", flush=True)

    z_delta = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - args.delta)))
    with torch.no_grad():
        g_ref = model(ref.expand(args.n_cal, reality.k), q_ref)
    bias = float((d_ref - g_ref).mean())

    # The achievable span s, measured: doing nothing against the best plan the
    # budget allows. `s` is NOT `limit - realised`: the precondition asks whether
    # the margin fits inside the room the actuator actually buys, so it must be
    # compared against a fraction of the span (§5.5), not against the distance
    # from an arbitrary limit to an arbitrary reference plan.
    # `s` must be the BEST achievable, not an even spread of the budget. Two of
    # the six levee sites make flooding *worse* (§5.6: -5.09% and -2.06%, the
    # levee backwater effect), so spreading the budget evenly understates the
    # span and biases the verdict toward FAIL. Scan the single-site allocations
    # and the even spread, and take the best.
    zero = torch.zeros(1, reality.k, device=device)
    candidates = [torch.full((1, reality.k), args.budget / reality.k, device=device)]
    for j in range(reality.k):
        c = torch.zeros(1, reality.k, device=device)
        c[0, j] = min(args.budget, args.max_height)
        candidates.append(c)
    # Evaluate every candidate in ONE batched solve rather than one per
    # candidate. Measured on Vista: with six arms sharing a node, a sequential
    # scan of eight candidates overruns a 50-minute job, while the span only
    # needs enough hydrographs to average out -- far fewer than calibration does.
    n_span = min(args.n_span, args.n_cal)
    q_span = sample_inflow(n_span, args, gen, device)
    all_plans = [zero] + candidates
    plan_stack = torch.cat([pl.expand(n_span, reality.k) for pl in all_plans], dim=0)
    inflow_stack = q_span.repeat(len(all_plans))
    depths = []
    for i in range(0, plan_stack.shape[0], args.batch):
        depths.append(reality.depth(plan_stack[i : i + args.batch],
                                    inflow_stack[i : i + args.batch]))
    depths = torch.cat(depths).view(len(all_plans), n_span).mean(dim=1)
    d_zero_mean = float(depths[0])
    cand_depths = [float(x) for x in depths[1:]]
    best_idx = int(min(range(len(cand_depths)), key=lambda i: cand_depths[i]))
    d_full_mean = cand_depths[best_idx]
    print(f"  span scan: do-nothing {d_zero_mean:.5f}; best of "
          f"{len(candidates)} allocations {d_full_mean:.5f} "
          f"(#{best_idx}, 0=even spread)", flush=True)
    span = d_zero_mean - d_full_mean
    d_zero = torch.tensor([d_zero_mean]); d_full = torch.tensor([d_full_mean])
    headroom = args.f_fraction * span
    need = bias + z_delta * sigma
    print(f"\n=== precondition (Prop 2): b + z*sigma_agg < f*s ===", flush=True)
    print(f"  do-nothing depth   {float(d_zero.mean()):.5f} m", flush=True)
    print(f"  best-budget depth  {float(d_full.mean()):.5f} m", flush=True)
    print(f"  achievable span s  {span:.5f} m", flush=True)
    print(f"  bias b             {bias:+.5f} m", flush=True)
    print(f"  z_delta*sigma      {z_delta*sigma:.5f} m", flush=True)
    print(f"  required margin    {need:.5f} m", flush=True)
    print(f"  headroom f*s       {headroom:.5f} m  (f={args.f_fraction})", flush=True)
    verdict = "PASS" if need < headroom else "FAIL"
    print(f"  verdict: {verdict}   margin consumes {100*need/max(span,1e-12):.1f}% of the span", flush=True)
    summary = {
        "epsilon": eps, "sigma": sigma, "rho": rho, "bias": bias,
        "z_delta": z_delta, "required_margin": need, "headroom": headroom,
        "span": span, "f_fraction": args.f_fraction,
        "do_nothing_depth": d_zero_mean,
        "best_budget_depth": d_full_mean,
        "best_allocation_index": best_idx,
        "margin_pct_of_span": 100 * need / max(span, 1e-12),
        "precondition": verdict, "config": vars(args),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.calibrate_only:
        (out / "precondition.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {out/'precondition.json'}", flush=True)
        return

    print("\n=== calibration set at the margin-free plan ===", flush=True)
    a0 = plan_under_margin(model, reality, args, 0.0, gen)
    q_cal = sample_inflow(args.n_cal, args, gen, device)
    d_cal = []
    for i in range(0, args.n_cal, args.batch):
        qq = q_cal[i : i + args.batch]
        d_cal.append(reality.depth(a0.expand(qq.shape[0], reality.k), qq))
    d_cal = torch.cat(d_cal)
    with torch.no_grad():
        g_cal = model(a0.expand(args.n_cal, reality.k), q_cal)
        g_hat = float(g_cal.mean())      # the quantity the planner controls
    print(f"  margin-free plan {[round(float(x),3) for x in a0[0]]}", flush=True)
    print(f"  realised depth mean {float(d_cal.mean()):.5f}  controlled ghat {g_hat:.5f}", flush=True)

    results = {}
    for arm in args.arms:
        if arm == "none":
            q_m = 0.0
        elif arm not in MODES:
            raise SystemExit(f"unknown arm {arm}")
        else:
            q_m = margin(
                arm, args.delta,
                model_err=[float(x) for x in (d_cal - g_cal)],
                deviations=[float(x) for x in (d_cal - d_cal.mean())],
                residuals=[float(x) for x in (d_cal - g_hat)],
                bias=float((d_cal - g_cal).mean()),
            )
        # When the margin exceeds the limit the tightened target is negative and
        # no plan can meet it: the planner spends its whole budget and the
        # "coverage" it achieves is an artefact of infeasibility, not of the
        # recipe. The repo has seen this before as margin collapse (§3.5, 48x),
        # so detect and label it rather than reporting a misleading pass.
        infeasible = q_m >= args.limit
        if infeasible:
            print(f"  [{arm:12s}] q={q_m:.5f} >= limit {args.limit}: target is "
                  f"negative, plan is degenerate -- coverage is not attributable "
                  f"to the recipe", flush=True)
        a1 = plan_under_margin(model, reality, args, q_m, gen)
        q_ev = sample_inflow(args.n_eval, args, gen, device)
        d_ev = []
        for i in range(0, args.n_eval, args.batch):
            qq = q_ev[i : i + args.batch]
            d_ev.append(reality.depth(a1.expand(qq.shape[0], reality.k), qq))
        d_ev = torch.cat(d_ev)
        viol = float((d_ev > args.limit).float().mean())
        results[arm] = {
            "margin": q_m, "plan": [float(x) for x in a1[0]],
            "spend": float(a1.sum()), "violation_rate": viol,
            "mean_depth": float(d_ev.mean()), "worst_depth": float(d_ev.max()),
            "covers": viol <= args.delta,
            "margin_exceeds_limit": bool(infeasible),
        }
        flag = "COVERS" if viol <= args.delta else "UNDER-COVERS"
        if infeasible:
            flag += " [DEGENERATE]"
        print(f"  [{arm:12s}] q={q_m:.5f}  spend={float(a1.sum()):.3f}  "
              f"violation={viol:.3f} (delta={args.delta})  {flag}", flush=True)

    summary["arms"] = results
    summary["note"] = (
        "The margin is calibrated at the margin-free plan a0 and deployed at "
        "a1 = plan(q). Exchangeability therefore holds in the hydrograph, not "
        "across the plan shift from a0 to a1; any coverage gap attributable to "
        "that shift is a real effect and is reported, not corrected away."
    )
    (out / "margin.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'margin.json'}", flush=True)


if __name__ == "__main__":
    main()
