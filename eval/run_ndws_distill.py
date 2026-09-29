#!/usr/bin/env python
"""Distil the decision-time planner into the amortised policy. §9.7's open test.

    python eval/run_ndws_distill.py --seed 0 --out runs/ndws_distill/s0

§9.7 concedes that our amortisation claim is the weakest thing in the paper:

    "a model-based planner CAN be amortised into a compact policy without
     performance loss, via MPO plus behaviour cloning on planner-generated data
     [Byravan et al., 2022] ... We trained a CNN from scratch under a budget
     dual for 200 iterations and did not try distillation from planner
     solutions -- the recipe the literature identifies as the one that works."

So the gap our paper reports between decision-time planning (36.5% burn
reduction) and the amortised policy (10.4%) may be a fact about *our training
recipe* rather than about amortisation. This runs the recipe we skipped.

The design keeps everything but the learning signal fixed:

  * the SAME network class as the amortised arm (`GaussianFieldPolicy`), same
    width, same call signature -- so capacity cannot explain a difference;
  * behaviour cloning on (state_t, planner action_t) pairs collected by rolling
    the per-instance planner out on TRAINING patches;
  * scored on HELD-OUT patches by the same `score_policy` every other arm uses.

Three outcomes and what each would mean:

  distilled ~= planner   -- our amortisation claim is wrong, and §9.7's warning
                            was right. The paper must be rewritten to claim only
                            that OUR from-scratch recipe fails.
  distilled ~= scratch   -- the gap is about amortisation in this setting, not
                            about the recipe, and the claim survives a test that
                            the literature says should have broken it.
  in between             -- report the number; the honest reading is that the
                            recipe recovers part of the gap.

Whichever it is, it is recorded. The prediction I would make from §5.1's own
diagnosis -- one policy must serve 1,024 distinct fires, and the planner's
solutions differ per fire -- is that distillation recovers SOME of the gap but
not all of it, because a single forward pass cannot reproduce 50 steps of
per-instance projected gradient descent on a fire it has not seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from eval.run_ndws_planning import (  # noqa: E402
    per_instance_plan, project_budget, score_policy, train_unet,
)
from pspe.envs.fire_env import FireEnv, FireTask  # noqa: E402
from pspe.plan import GaussianFieldPolicy  # noqa: E402
from pspe.simulate.real import (  # noqa: E402
    NDWSConfig, driver_stats, load_ndws, normalize_drivers,
)
from pspe.simulate.real.models import make_fire_model  # noqa: E402
from pspe.utils import get_device, project_path, seed_everything  # noqa: E402


@torch.no_grad()
def collect_demonstrations(env: FireEnv, idx: torch.Tensor, horizon: int,
                          plan_steps: int, batch: int = 32
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll the per-instance planner out and record (state_t, action_t) pairs.

    The planner solves each fire jointly over all `horizon` days, so the pairs
    are collected ALONG its own trajectory: the state the planner actually
    visits at step t, and what it does there. Cloning against states the planner
    never reaches would teach the policy nothing it needs at evaluation time,
    which is the standard behaviour-cloning distribution-shift trap.

    The state update mirrors `rollout` exactly -- advance channel 0 with the
    predicted ignition probability -- so the demonstrations lie on the same
    trajectory the scoring function will walk.
    """
    states, actions = [], []
    for begin in range(0, idx.numel(), batch):
        sel = idx[begin:begin + batch]
        state = env.reset(idx=sel)
        with torch.enable_grad():
            plan = per_instance_plan(env, state, horizon, steps=plan_steps)
        plan = plan.detach()
        st = state
        for t in range(horizon):
            a = plan[:, t]
            states.append(st.detach().cpu())
            actions.append(a.detach().cpu())
            p = env.predict(st, env.treatment(a))
            st = st.clone()
            st[:, 0] = p
        print(f"  demos {begin + sel.numel()}/{idx.numel()}", flush=True)
    return torch.cat(states), torch.cat(actions)


def allocate(policy: GaussianFieldPolicy, state: torch.Tensor, budget: float,
             env: FireEnv) -> torch.Tensor:
    """Policy output -> budget-normalised intensities.

    The planner's solution is an ALLOCATION of a fixed budget across patches: it
    spends its whole 3% (measured: treated 3.000%/day) and the decision is
    *where*. Cloning raw actions with an MSE does not learn that. With a 3%
    budget over 64 patches the planner leaves most patches near zero, so most
    target actions are `action_for(0) = -1`, and the MSE-optimal constant is
    therefore -1 everywhere -- which maps to zero intensity. The first run of
    this experiment did exactly that: BC loss fell 0.358 -> 0.064, and the
    resulting policy treated 6e-05% per day for a 0.0% reduction. It learned the
    right constant for the wrong objective.

    Normalising the intensities to the budget removes the degenerate solution:
    a constant output becomes a UNIFORM allocation (the planner's own
    initialisation), not an empty one, and what the network can still express is
    where to concentrate. Same network, same tanh -> intensity map as the
    amortised arm, so the capacity argument is untouched.
    """
    a, _ = policy.sample(state, deterministic=True)
    u = env.intensity(a)
    scale = u.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    u = (u / scale) * budget
    return project_budget(u.clamp(0, 1), budget)


def behaviour_clone(states: torch.Tensor, actions: torch.Tensor, in_ch: int,
                    action_dim: int, device, epochs: int, lr: float, budget: float,
                    env: FireEnv, batch: int = 64, seed: int = 0
                    ) -> GaussianFieldPolicy:
    """Clone the planner's ALLOCATION, in intensity space, after normalisation.

    The loss is MSE between budget-normalised intensities -- the same transform
    the policy is evaluated through -- so training and evaluation optimise the
    same quantity. Regressing raw actions instead collapses to no-treatment; see
    `allocate`.
    """
    seed_everything(seed)
    policy = GaussianFieldPolicy(in_ch, action_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = states.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for begin in range(0, n, batch):
            sel = perm[begin:begin + batch]
            st = states[sel].to(device)
            tgt_a = actions[sel].to(device)
            # Target allocation, normalised the same way the policy's is.
            tu = env.intensity(tgt_a)
            tu = (tu / tu.mean(dim=-1, keepdim=True).clamp(min=1e-8)) * budget
            pred, _ = policy.sample(st, deterministic=True)
            pu = env.intensity(pred)
            pu = (pu / pu.mean(dim=-1, keepdim=True).clamp(min=1e-8)) * budget
            loss = torch.nn.functional.mse_loss(pu, tu)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * sel.numel()
        print(f"  bc epoch {ep}: allocation mse {total / n:.8f}", flush=True)
    policy.eval()
    return policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--budget", type=float, default=0.03)
    ap.add_argument("--block", type=float, default=0.9)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    # train_unet (run_ndws_planning.py) reads args.lr and args.batch, so they
    # have to exist here with the same defaults the planning run uses -- the
    # surrogate must be the same model, or the arms are not comparable.
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--plan-steps", type=int, default=50)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--demo-fires", type=int, default=256,
                    help="training fires the planner solves to make demonstrations")
    ap.add_argument("--unet-ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/ndws_distill")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train = load_ndws(NDWSConfig(split="train", n_samples=args.n_train, grid=args.grid))
    val = load_ndws(NDWSConfig(split="eval", n_samples=args.n_eval, grid=args.grid))
    dstats = driver_stats(train)
    train["drivers"] = normalize_drivers(train["drivers"], dstats)
    val["drivers"] = normalize_drivers(val["drivers"], dstats)

    in_ch = 1 + train["drivers"].shape[1]
    if args.unet_ckpt:
        unet = make_fire_model("unet", in_ch, grid=args.grid, width=args.width).to(device)
        unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
        unet.eval()
        val_aucpr = float("nan")
    else:
        unet, val_aucpr = train_unet(train, val, args, device)
    print(f"surrogate val AUC-PR {val_aucpr:.4f}", flush=True)

    task = FireTask(cost_limit=args.budget, block=args.block)
    plan_env = FireEnv(unet, train, task, grid=args.grid, patches=args.patches,
                       horizon=args.horizon, device=device)
    test_env = FireEnv(unet, val, task, grid=args.grid, patches=args.patches,
                       horizon=args.horizon, device=device)
    held = torch.arange(test_env.n)

    # Demonstrations come from TRAINING patches. Cloning on the patches we then
    # score would measure memorisation, which is the failure mode §5.1 already
    # attributes to the amortised arm.
    n_demo = min(args.demo_fires, plan_env.n)
    demo_idx = torch.randperm(plan_env.n)[:n_demo]
    print(f"collecting demonstrations from {n_demo} training fires", flush=True)
    states, actions = collect_demonstrations(plan_env, demo_idx, args.horizon,
                                            args.plan_steps)
    print(f"  {states.shape[0]} (state, action) pairs", flush=True)
    torch.save({"states": states, "actions": actions}, root / "demos.pt")

    policy = behaviour_clone(states, actions, in_ch, plan_env.action_dim,
                             device, args.bc_epochs, args.bc_lr, args.budget,
                             plan_env, seed=args.seed)
    torch.save(policy.state_dict(), root / "distilled.pt")

    @torch.no_grad()
    def distilled(state, t=0):
        # Budget-normalised allocation, projected onto the same budget set the
        # planner's actions are projected onto. Two failures this guards, both
        # observed: raw cloned actions overspent 7.6x the budget on the first
        # smoke run (20.5% "reduction" that is not a result), and action-space
        # MSE then collapsed to no-treatment on the full run. See `allocate`.
        return FireEnv.action_for(allocate(policy, state, args.budget, test_env))

    plan_cache: dict = {}

    def per_instance(state, t=0):
        if t == 0:
            plan_cache["plan"] = per_instance_plan(test_env, state, args.horizon,
                                                   steps=args.plan_steps)
        return plan_cache["plan"][:, t]

    rows = []
    for name, act in (("pspe per-instance (planner)", per_instance),
                      ("distilled from planner", distilled)):
        rows.append({"policy": name,
                     **score_policy(test_env, act, held, args.horizon),
                     "demo fires": n_demo, "seed": args.seed})
        print(f"[{name}] {rows[-1]}", flush=True)

    planner = rows[0]["reduction vs none %"]
    dist = rows[1]["reduction vs none %"]
    # The number §9.7 turns on: how much of the planner's advantage a single
    # forward pass recovers. The from-scratch amortised arm sits at 10.4%
    # (§5.1), so that is the bar distillation has to clear to matter.
    summary = {
        "planner %": round(planner, 3),
        "distilled %": round(dist, 3),
        "scratch amortised % (from 5.1)": 10.4,
        "fraction of planner recovered": round(dist / planner, 3) if planner else None,
        "beats scratch amortised": bool(dist > 10.4),
        # Both arms must be budget-feasible for the comparison to mean
        # anything. If either is non-zero the row is not a result.
        "planner over budget": rows[0]["over budget"],
        "distilled over budget": rows[1]["over budget"],
        "planner treated %/day": round(rows[0]["treated % per day"], 3),
        "distilled treated %/day": round(rows[1]["treated % per day"], 3),
    }
    (root / "results.json").write_text(json.dumps(
        {"rows": rows, "summary": summary}, indent=2, default=float))
    (root / "results.md").write_text(markdown_table(rows) + "\n\n"
                                     + markdown_table([summary]) + "\n")
    print("\n" + markdown_table(rows))
    print("\n" + markdown_table([summary]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
