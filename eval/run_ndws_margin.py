#!/usr/bin/env python
"""The conformal margin on real fire dynamics, not on a synthetic PDE.

    python eval/run_ndws_margin.py --seed 0 --out runs/ndws_margin

Section 6 of the paper validated the *planner* on observed wildfire records but
enforced the crew budget by exact projection, so it never exercised the margin
the paper is about. This does.

The gap between model and reality is built, not assumed, from two models of the
same real fire spread at different quality:

    G   the planner's surrogate. Trained on fewer patches for fewer epochs, so
        it is wrong in the ordinary way a deployed model is wrong.
    F   stands in for reality. The stronger model, trained longer on more data
        with a different seed; its held-out AUC-PR against observed fire is
        reported, so the reader can see what "reality" is worth here.

This is sim2sim and does not resolve the counterfactual limitation: F is a
model, and no observational record contains a fire where a firebreak was cut on
our instruction. What it does give is a model/reality cost gap that nobody
tuned, over dynamics fitted to real fires rather than to a PDE we wrote.

The constraint is an EXPECTATION, as in any CMDP, and that is the whole point.
The objective is total burned area over the whole patch; the constraint is
burned area restricted to the most populated quarter of cells. These compete
for the same crews: the cells that dominate total burn are usually not the ones
people live in, so an objective-optimal plan leaves the settled quarter exposed
and meeting the constraint costs real objective performance.

An earlier version constrained population-WEIGHTED burn over all cells, which
does not compete: the weighting is monotone in the same field the objective
minimises, so the objective-optimal and constraint-optimal plans differed by
6e-5 and the multiplier had no lever. The limit calibration now checks for that
and says so rather than reporting a constraint that never binds.

A single multiplier, shared across fires and driven by the batch mean under G,
trades them off while each fire is planned at decision time. The crew budget
stays a hard projection, as before.

Violation is measured per evaluation batch, because the constrained quantity is
a mean. That is exactly the regime where conformalising matched-pair model
error undercovers: the across-fire scatter that makes batch means exceed the
limit is present in both terms of c_i - g_i and cancels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws import average_precision  # noqa: E402
from eval.run_ndws_planning import project_budget, train_unet  # noqa: E402
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.plan import margins  # noqa: E402
from pspe.plan.lagrangian import PIDLagrangian  # noqa: E402
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.simulate.real import (  # noqa: E402
    NDWSConfig, driver_stats, load_ndws, normalize_drivers,
)
from pspe.utils import get_device, project_path, seed_everything  # noqa: E402

# Order matters when a job can be truncated by a wall-clock limit: whatever is
# last is what gets lost. `residual` is the recipe under test, so it runs
# first, and the baselines follow. Stage 2's first attempt died at its 30-minute
# limit with 3 of 4 arms done and lost precisely the arm the experiment was for.
ARMS = ("residual", "none", "model_error", "episode")


class Perturbed(torch.nn.Module):
    """Stands in for reality: the planner's own model with a controlled discrepancy.

    Training a second, better model and hoping its gap from the first lands
    where a margin can work does not succeed on this task. Measured over four
    (budget, capacity) settings, the model-reality bias came out 2.0x to 4.9x
    the entire span the planner can move the constrained quantity over, so the
    constraint was unachievable and no margin could be assessed. More crew made
    it worse, not better: both the objective-optimal and constraint-optimal
    plans converge on the same floor, closing the gap between them.

    So the discrepancy is made a controlled variable instead. Two knobs, which
    separate the two things a margin has to cope with:

        shift    a logit offset. Reality burns more than the model says,
                 everywhere. This is the systematic bias b.
        scatter  a fixed, randomly initialised convolution of the input, so the
                 discrepancy is deterministic in the state and smooth in space.
                 Deterministic matters: the same fire must produce the same
                 error, or it is noise rather than model error, and the
                 matched-pair recipe would have nothing to cancel.

    This is a perturbed model, not an independently trained one, and the paper
    has to say so. What it buys is the ability to place b and eps where the
    claim can actually be tested, on dynamics still fitted to observed fires.
    """

    def __init__(self, base: torch.nn.Module, shift: float, scatter: float,
                 in_channels: int, seed: int = 0) -> None:
        super().__init__()
        self.base, self.shift, self.scatter = base, shift, scatter
        g = torch.Generator().manual_seed(seed + 9_000)
        self.disc = torch.nn.Conv2d(in_channels, 1, 7, padding=3)
        with torch.no_grad():
            for q in self.disc.parameters():
                q.copy_(torch.randn(q.shape, generator=g) * 0.25)
                q.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x) + self.shift
        if self.scatter:
            out = out + self.scatter * torch.tanh(self.disc(x))
        return out


def settled_mask(env: FireEnv, state, q: float = 0.75):
    """Cells in the top (1-q) of population density for each fire.

    The constrained region. Kept disjoint in spirit from where the bulk of the
    burn is, which is what makes the constraint cost something.
    """
    pop = env.population(state).flatten(1)
    thr = torch.quantile(pop, q, dim=1, keepdim=True)
    return (pop >= thr).float().view_as(env.population(state))


def plan(env: FireEnv, state, horizon, lam, steps, lr=0.05):
    """Per-instance firebreaks minimising total burn plus lam * settled burn.

    Budget stays a hard projection; the multiplier prices the constraint.
    """
    k = env.action_dim
    u = torch.full((state.shape[0], horizon, k), env.task.cost_limit, device=state.device)
    w = settled_mask(env, state)
    with torch.enable_grad():
        for _ in range(steps):
            u = u.detach().requires_grad_(True)
            env.set_state(state)
            burn = None
            for t in range(horizon):
                b = env.treatment_from_intensity(u[:, t])
                burn = env.predict(env.state, intensity=b)
                env.set_state(torch.cat([burn[:, None], state[:, 1:]], 1))
            total = burn.flatten(1).mean(1)
            settled = (burn * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1)
            loss = (total + lam * settled).sum()
            (g,) = torch.autograd.grad(loss, u)
            g = g / (g.flatten(1).norm(dim=1)[:, None, None] + 1e-12)
            u = project_budget(u.detach() - lr * g * (k ** 0.5), env.task.cost_limit)
    return u.detach()


@torch.no_grad()
def outcome(env: FireEnv, state, u, horizon):
    """(total burn, settled-quarter burn) per fire, under this env's model."""
    w = settled_mask(env, state)
    env.set_state(state)
    burn = state[:, 0]
    for t in range(horizon):
        b = env.treatment_from_intensity(u[:, t])
        burn = env.predict(env.state, intensity=b)
        env.set_state(torch.cat([burn[:, None], state[:, 1:]], 1))
    return (burn.flatten(1).mean(1),
            (burn * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.03)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--weak-epochs", type=int, default=4)
    ap.add_argument("--real-epochs", type=int, default=18)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--plan-steps", type=int, default=25)
    ap.add_argument("--plan-batch", type=int, default=32)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--probe-batches", type=int, default=4,
                    help="independent batches per probe. Each yields ONE "
                         "calibration residual, because the constrained "
                         "quantity is a batch mean.")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--cal-fires", type=int, default=256,
                    help="fires used to calibrate the limit")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="which recipes to run. A subset reuses the cached G, so "
                         "recovering one lost arm costs a quarter of the run.")
    ap.add_argument("--perturb-shift", type=float, default=0.0,
                    help="logit offset making reality burn more than the model says. "
                         "Sets the systematic bias b. 0 falls back to training a "
                         "second model, which this task cannot make feasible.")
    ap.add_argument("--perturb-scatter", type=float, default=0.0,
                    help="magnitude of a fixed input-dependent discrepancy, which "
                         "sets the per-instance model error eps and therefore "
                         "rho = sigma/eps.")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="report the limit, the controllable span, the model-reality "
                         "bias and the feasible headroom, then stop. Seconds instead "
                         "of an hour, and it is what decides whether a configuration "
                         "can work before any margin is fitted.")
    ap.add_argument("--limit-frac", type=float, default=0.35,
                    help="where the limit sits between the constraint-optimal and "
                         "objective-optimal plans. Also sets the headroom a margin "
                         "has to fit inside, so too small makes the task infeasible.")
    ap.add_argument("--cal-lambda", type=float, default=20.0,
                    help="multiplier standing in for a town-focused planner")
    ap.add_argument("--margin-delta", type=float, default=0.1)
    ap.add_argument("--kp", type=float, default=0.5)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/ndws_margin")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, grid=args.grid))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, grid=args.grid))
    dstats = driver_stats(train)
    train["drivers"] = normalize_drivers(train["drivers"], dstats)
    val["drivers"] = normalize_drivers(val["drivers"], dstats)

    from types import SimpleNamespace

    # Cache by everything that affects the weights. The perturbation knobs do
    # not, so a sweep over them retrains nothing: the whole point of making the
    # discrepancy a variable is being able to walk it across a threshold
    # cheaply.
    cache = project_path("runs/ndws_margin/_models"); cache.mkdir(parents=True, exist_ok=True)

    def fit(epochs, seed, width):
        key = cache / f"unet_n{args.n_train}_e{epochs}_w{width}_g{args.grid}_s{seed}.pt"
        if key.exists():
            m = make_fire_model("unet", 1 + train["drivers"].shape[1],
                                grid=args.grid, width=width).to(device)
            blob = torch.load(key, map_location=device)
            m.load_state_dict(blob["state"]); m.eval()
            print(f"  (cached {key.name}, val AUC-PR {blob['aucpr']:.4f})", flush=True)
            return m, float(blob["aucpr"])
        a = SimpleNamespace(grid=args.grid, epochs=epochs, batch=args.batch,
                            lr=args.lr, width=width, seed=seed)
        m, auc = train_unet(train, val, a, device)
        torch.save({"state": m.state_dict(), "aucpr": auc}, key)
        return m, auc

    print("training G (the planner's model) ...", flush=True)
    G, g_auc = fit(args.weak_epochs, args.seed, args.width)
    print(f"  G  val AUC-PR {g_auc:.4f}", flush=True)
    if args.perturb_shift or args.perturb_scatter:
        print(f"F = G perturbed (shift {args.perturb_shift}, "
              f"scatter {args.perturb_scatter}) ...", flush=True)
        F = Perturbed(G, args.perturb_shift, args.perturb_scatter,
                      1 + train["drivers"].shape[1], args.seed).to(device).eval()
        f_auc = float("nan")
    else:
        print("training F (stands in for reality) ...", flush=True)
        F, f_auc = fit(args.real_epochs, args.seed + 500, args.width + 16)
        print(f"  F  val AUC-PR {f_auc:.4f}", flush=True)

    task = FireTask(cost_limit=args.budget)
    envG = FireEnv(G, train, task, grid=args.grid, patches=args.patches,
                   horizon=args.horizon, device=device)
    envF = FireEnv(F, train, task, grid=args.grid, patches=args.patches,
                   horizon=args.horizon, device=device)
    evG = FireEnv(G, val, task, grid=args.grid, patches=args.patches,
                  horizon=args.horizon, device=device)
    evF = FireEnv(F, val, task, grid=args.grid, patches=args.patches,
                  horizon=args.horizon, device=device)

    gen = torch.Generator().manual_seed(args.seed)

    # --- calibrate the limit so it binds ----------------------------------- #
    # The constraint has to cost something, or every arm trivially satisfies it
    # and the experiment measures nothing. Two reference points, both under F:
    #
    #   pop_free  what the OBJECTIVE-optimal planner (lam = 0, minimising total
    #             burn) happens to leave on the populated area. It is free to
    #             sacrifice the town to spare a larger empty area.
    #   pop_best  what a planner that cares only about the town achieves
    #             (large lam). The best the constraint could ever ask for.
    #
    # The limit sits between them, so meeting it requires giving up a real share
    # of the objective. Setting it above pop_free instead would leave the
    # unconstrained planner already compliant, which is the failure mode this
    # comment exists to prevent.
    cal_idx = torch.randperm(envG.n, generator=gen)[:args.cal_fires]
    pop_free_all, pop_best_all = [], []
    for b in range(0, cal_idx.numel(), args.plan_batch):
        st = envG.reset(idx=cal_idx[b:b + args.plan_batch])
        u0 = plan(envG, st, args.horizon, 0.0, args.plan_steps)
        uB = plan(envG, st, args.horizon, args.cal_lambda, args.plan_steps)
        pop_free_all.append(outcome(envF, st, u0, args.horizon)[1])
        pop_best_all.append(outcome(envF, st, uB, args.horizon)[1])
    pop_free = float(torch.cat(pop_free_all).mean())
    pop_best = float(torch.cat(pop_best_all).mean())
    d = pop_best + args.limit_frac * (pop_free - pop_best)
    print(f"limit d = {d:.6f}   (objective-optimal leaves {pop_free:.6f}, "
          f"town-focused reaches {pop_best:.6f})", flush=True)
    # The largest margin the constraint can absorb before d_eff drops below
    # anything the planner can reach. A correctly calibrated margin that
    # exceeds this makes the constraint infeasible, which looks identical to
    # the margin failing and is not the same thing.
    headroom = d - pop_best
    print(f"  maximum feasible margin = {headroom:.6f} "
          f"({100 * headroom / d:.1f}% of the limit)", flush=True)
    span = (pop_free - pop_best) / max(abs(pop_free), 1e-12)
    print(f"  span between the two reference points: {span:.1%} of the limit", flush=True)
    if span < 0.05:
        print("  WARNING: the objective-optimal and constraint-optimal plans barely "
              "differ, so the constraint does not compete for crews and every arm "
              "will satisfy it trivially. Do not report this run.", flush=True)

    if args.calibrate_only:
        # The bias the dual cannot see: hold the G-estimate at d, ask F what it
        # actually delivers. If that exceeds the span, no margin can fit and the
        # configuration is unusable however well it is calibrated.
        # Averaged over several batches, not one. A single batch of 32 fires
        # gave estimates that differed by 8.6x between two machines for the
        # same nominal configuration, and the usable/infeasible verdict turns
        # on this number. The spread is reported so a marginal verdict is
        # visible as marginal rather than asserted.
        ests = []
        for _ in range(args.probe_batches * 2):
            st = envG.reset(idx=torch.randperm(envG.n, generator=gen)[:args.eval_batch])
            u_at_d = plan(envG, st, args.horizon, args.cal_lambda / 4, args.plan_steps)
            realised = float(outcome(envF, st, u_at_d, args.horizon)[1].mean())
            model_said = float(outcome(envG, st, u_at_d, args.horizon)[1].mean())
            ests.append(realised - model_said)
        bias_est = float(np.mean(ests))
        bias_sd = float(np.std(ests, ddof=1)) if len(ests) > 1 else 0.0
        span = pop_free - pop_best
        margin_of_error = 2 * bias_sd / max(len(ests), 1) ** 0.5
        verdict = ("usable" if bias_est < headroom
                   else "INFEASIBLE - a margin covering the bias cannot fit")
        if abs(bias_est - headroom) < margin_of_error:
            verdict += "  (MARGINAL: the estimate is within its own error of the threshold)"
        print(f"  span {span:.6f} ({100*span/d:.1f}% of d) | "
              f"bias {bias_est:.6f} +/- {bias_sd:.6f} ({100*bias_est/d:.1f}% of d) | "
              f"headroom {headroom:.6f} ({100*headroom/d:.1f}% of d)")
        print(f"  b / headroom = {bias_est/max(headroom,1e-12):.2f}   VERDICT: {verdict}")
        (root / "calibration.json").write_text(json.dumps(
            {"budget": args.budget, "weak_epochs": args.weak_epochs,
             "limit_frac": args.limit_frac, "d": d, "pop_best": pop_best,
             "pop_free": pop_free, "span": span, "bias": bias_est,
             "bias_sd": bias_sd, "n_bias_batches": len(ests),
             "headroom": headroom, "G_aucpr": g_auc, "F_aucpr": f_auc}, indent=2))
        return 0

    rows = []
    for arm in args.arms:
        seed_everything(args.seed)
        dual = PIDLagrangian(cost_limit=d, kp=args.kp, ki=args.ki, kd=args.kd, ema=0.7)
        model_err, devs, resid = [], [], []
        bias, d_eff, lam, ghat = 0.0, d, 0.0, None

        for it in range(args.iterations):
            idx = torch.randperm(envG.n, generator=gen)[:args.plan_batch]
            state = envG.reset(idx=idx)
            u = plan(envG, state, args.horizon, lam, args.plan_steps)
            _, gpop = outcome(envG, state, u, args.horizon)
            g_mean = float(gpop.mean())

            ghat = g_mean if ghat is None else 0.7 * ghat + 0.3 * g_mean

            if it % args.probe_every == 0:
                # The probe. The calibration unit must match the unit violation
                # is declared on. The constrained quantity here is an
                # EXPECTATION, violation is declared on a batch mean, so the
                # residuals have to be batch means too.
                #
                # Calibrating on per-fire residuals instead inflates the
                # quantile by sqrt(batch): measured, it consumed the entire
                # limit (margin 0.06403 against a limit of 0.064030), drove
                # d_eff to zero, made the constraint infeasible and left every
                # arm at 70-80% violation. That is this paper's own thesis
                # applied to its own experiment, and it is the second time the
                # same mistake has appeared in this project.
                batch_f, batch_g = [], []
                for _ in range(args.probe_batches):
                    pidx = torch.randperm(envG.n, generator=gen)[:args.eval_batch]
                    pstate = envG.reset(idx=pidx)
                    pu = plan(envG, pstate, args.horizon, lam, args.plan_steps)
                    gm = float(outcome(envG, pstate, pu, args.horizon)[1].mean())
                    fm = float(outcome(envF, pstate, pu, args.horizon)[1].mean())
                    batch_f.append(fm); batch_g.append(gm)
                    # Matched pair at the batch level: the fires drawn move fm
                    # and gm together, so this cancels the across-batch scatter.
                    model_err.append(fm - gm)
                    # Against the quantity the dual actually holds, which is a
                    # running estimate shared across batches, so the scatter
                    # survives. This is the distinction the paper is about.
                    resid.append(fm - ghat)
                f_mean = float(np.mean(batch_f))
                bias = 0.7 * bias + 0.3 * (f_mean - float(np.mean(batch_g)))
                devs += [x - f_mean for x in batch_f]
                if arm != "none":
                    d_eff = max(0.0, d - margins.margin(
                        arm, args.margin_delta, model_err=model_err,
                        deviations=devs, residuals=resid, bias=bias))
                    dual.cost_limit = d_eff
                lam = dual.update(f_mean)
            else:
                lam = dual.update(g_mean + bias)

        # --- held out: plan with the calibrated multiplier, score under F --- #
        held = torch.randperm(evG.n, generator=gen)[:min(evG.n, 640)]
        viol, batch_f, batch_total = 0, [], []
        for b in range(0, held.numel() - args.eval_batch + 1, args.eval_batch):
            sidx = held[b:b + args.eval_batch]
            state = evG.reset(idx=sidx)
            u = plan(evG, state, args.horizon, lam, args.plan_steps)
            tot, pop = outcome(evF, state, u, args.horizon)
            m = float(pop.mean())
            batch_f.append(m); batch_total.append(float(tot.mean()))
            viol += int(m > d)
        if arm != "none" and d - d_eff > headroom:
            print(f"  NOTE [{arm}]: margin {d - d_eff:.6f} exceeds the feasible "
                  f"headroom {headroom:.6f}; d_eff is below anything the planner "
                  f"can reach, so this arm is infeasible rather than miscalibrated. "
                  f"Raise --limit-frac to leave room.", flush=True)
        if arm != "none" and d - d_eff > 0.9 * d:
            print(f"  WARNING [{arm}]: the margin consumed {100*(d-d_eff)/d:.0f}% of "
                  f"the limit, so d_eff is ~0 and the constraint is infeasible. "
                  f"The calibration unit probably does not match the unit "
                  f"violation is declared on. Do not report this arm.", flush=True)
        rows.append({
            "arm": arm,
            "violating batches": round(viol / max(1, len(batch_f)), 4),
            "worst batch": round(max(batch_f), 5),
            "limit d": round(d, 5),
            "d_eff": round(d_eff, 5),
            "margin": round(d - d_eff, 5),
            "mean pop burn": round(float(np.mean(batch_f)), 5),
            "total burn (objective)": round(float(np.mean(batch_total)), 5),
            "lambda": round(lam, 3),
            "bias": round(bias, 5),
            "n batches": len(batch_f),
            "span": round(pop_free - pop_best, 6),
            "headroom": round(headroom, 6),
            "perturb shift": args.perturb_shift,
            "perturb scatter": args.perturb_scatter,
            "seed": args.seed,
        })
        print(f"[{arm:12s}] violating {rows[-1]['violating batches']:.3f}  "
              f"margin {rows[-1]['margin']:.5f}  "
              f"pop burn {rows[-1]['mean pop burn']:.5f}  "
              f"objective {rows[-1]['total burn (objective)']:.5f}", flush=True)

    table = markdown_table(rows)
    tag = "" if len(args.arms) == len(ARMS) else "_" + "+".join(args.arms)
    (root / f"results{tag}.md").write_text(
        f"# Conformal margin on real fire dynamics (seed {args.seed})\n\n"
        f"G val AUC-PR {g_auc:.4f}; F val AUC-PR {f_auc:.4f} "
        f"(nan when F is G perturbed: shift {args.perturb_shift}, "
        f"scatter {args.perturb_scatter}). "
        f"Target failure rate delta = {args.margin_delta}; violation is measured per "
        f"evaluation batch of {args.eval_batch} fires, because the constrained "
        f"quantity is an expectation.\n\n" + table + "\n")
    (root / f"results{tag}.json").write_text(json.dumps(
        {"rows": rows, "G_aucpr": g_auc, "F_aucpr": f_auc, "limit": d}, indent=2, default=float))
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
