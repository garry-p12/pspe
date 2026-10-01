#!/usr/bin/env python
"""SafeDreamer, the third model-based safe-RL baseline, as a pre-registered test.

    python eval/run_safedreamer_baseline.py --testbed rdf --seed 0 --out runs/safedreamer/rdf_s0

§9.2 listed SafeDreamer [Huang et al., ICLR 2024] as unrun and the external ICML
read named it specifically. It is transplanted here the way CAP and SMBPO were:
same planner, dual, testbed, limit, evaluation protocol and real-sample budget,
with only the cost correction differing. Not a reimplementation.

**What distinguishes it from the other two.** CAP inflates by ensemble spread
(mean + k*sigma, k adapted); SMBPO by ensemble pessimism (max, truncated
horizon). Both are statistics OF the ensemble. SafeDreamer instead learns a
**cost critic on imagined rollouts** and feeds the Lagrangian that critic's
estimate -- a learned value, not a spread. That is a genuinely different
mechanism, which is what makes it worth running.

**Pre-registered prediction, recorded before the run.** Proposition 3 (§3.6)
says any margin measurable from the model-error distribution alone cannot hold
delta once rho > 1, and Corollary 3.3 says the theorem does not RANK such
methods. A cost critic fitted to surrogate rollouts is a functional of the
model, so SafeDreamer falls under it. Therefore:

    SafeDreamer should land near 12% violating -- indistinguishable from
    CAP (11.99%) and SMBPO (12.20%), and about 7 points worse than ours (5.00%).

If instead it lands near 5%, Proposition 3 is wrong about learned critics, and
that is the outcome worth knowing. §9.2a's earlier prediction failed because it
was an intuition written after the mechanism; this one is derived from the
theorem, which is the difference the SMBPO episode taught.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig, SimulateTrainer, ensure_dataset, make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


class CostCritic(nn.Module):
    """Predicts episode cost from an initial field. SafeDreamer's safety critic.

    Deliberately small: the point of the transplant is the *mechanism* -- a
    learned value standing in for a rollout -- not the capacity of the network.
    """

    def __init__(self, in_channels: int, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SafeDreamerTrainer(HybridPlannerTrainer):
    """The planner, with a learned cost critic in place of a conformal margin.

    The dual is fed the critic's estimate of episode cost. The critic is fitted
    online to costs the surrogate produces -- imagined rollouts, in SafeDreamer's
    terms -- so it never sees the real system except through the shared probe
    budget, exactly as CAP and SMBPO never do.
    """

    def __init__(self, *args, critic_lr=1e-3, critic_steps=20, **kw):
        super().__init__(*args, **kw)
        obs_c = self.env.obs_shape[0]
        # NOT self.critic: HybridPlannerTrainer already owns that name and its
        # critic returns (value, cost_value). Overwriting it fails at
        # trainer.py:296 with 'too many values to unpack'.
        self.sd_critic = CostCritic(obs_c).to(self.device)
        self.critic_opt = torch.optim.Adam(self.sd_critic.parameters(), lr=critic_lr)
        self.critic_steps = critic_steps
        self.critic_err: list[float] = []

    def _fit_critic(self, episodes: int = 8) -> float:
        """One round of imagined rollouts, fitted."""
        seed = self._probe_seed + 991
        state0 = self._probe_initial(self.env, episodes, seed)
        with torch.no_grad():
            target = self._rollout_costs(self.env, state0, seed)
        last = 0.0
        for _ in range(self.critic_steps):
            pred = self.sd_critic(state0)
            loss = nn.functional.mse_loss(pred, target)
            self.critic_opt.zero_grad(); loss.backward(); self.critic_opt.step()
            last = float(loss.detach())
        return last

    def _dual_input(self, surrogate_cost: float, it: int):
        """The critic's estimate, not the rollout's."""
        fit_loss = self._fit_critic()
        seed = self._probe_seed + 991
        state0 = self._probe_initial(self.env, 8, seed)
        with torch.no_grad():
            est = float(self.sd_critic(state0).mean())
        self.critic_err.append(abs(est - surrogate_cost))
        rec = {"sd/critic_estimate": est, "sd/surrogate_cost": surrogate_cost,
               "sd/critic_fit_loss": fit_loss,
               "sd/critic_minus_rollout": est - surrogate_cost}
        if self.cfg.real_cost_every and self.eval_env is not None \
                and it % self.cfg.real_cost_every == 0:
            # Same probe budget as every other arm; spent for reporting only,
            # since the critic is fitted on imagined rollouts by construction.
            real = self.probe_real_cost(self.cfg.real_cost_episodes)
            rec["dual/real_cost"] = real
            rec["sd/critic_minus_real"] = est - real
        return est, rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="rdf", choices=["dar", "swe", "rdf"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--surrogate", default="fno", choices=["fno", "deeponet", "gnot"])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--probe-episodes", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-steps", type=int, default=20)
    ap.add_argument("--sat-coef", type=float, default=1.0)
    ap.add_argument("--adv-floor", type=float, default=1e-2)
    ap.add_argument("--dual-ema", type=float, default=0.7)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/safedreamer")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    # One surrogate, not an ensemble: SafeDreamer's correction is a learned
    # critic, so an ensemble would be spending real samples the mechanism does
    # not use. The probe budget is matched to CAP and SMBPO regardless, which is
    # the quantity the comparison controls.
    seed_everything(args.seed)
    model = make_surrogate(args.surrogate, spec.n_channels, grid=args.grid)
    sim = SimulateTrainer(
        model, SimulateTrainConfig(testbed=args.testbed, surrogate=args.surrogate,
                                   epochs=args.epochs, grid=args.grid),
        data, RunLogger(root / "surrogate", use_tensorboard=False), device)
    st = sim.train()
    for q in sim.model.parameters():
        q.requires_grad_(False)
    print(f"  surrogate rel L2 {st['rel_l2_final']:.4f}", flush=True)

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    seed_everything(args.seed)
    train_env = make_env(dynamics="surrogate", surrogate=sim.model, **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    trainer = SafeDreamerTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                          seed=args.seed, saturation_coef=args.sat_coef,
                          advantage_std_floor=args.adv_floor, dual_ema=args.dual_ema,
                          ki=args.ki, eval_every=args.eval_every,
                          real_cost_every=args.probe_every,
                          real_cost_episodes=args.probe_episodes,
                          cost_margin_k=0.0, margin_conformal=False),
        eval_env=eval_env, logger=RunLogger(root / "sd", use_tensorboard=False),
        device=device, surrogate_train_transitions=fit_transitions,
        critic_lr=args.critic_lr, critic_steps=args.critic_steps)
    s = trainer.train()

    errs = trainer.critic_err
    row = {
        "method": "SafeDreamer (learned cost critic)", "testbed": args.testbed,
        "violating": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
        "worst": round(s.get("eval/cost_max_over_run", float("nan")), 4),
        "return": round(s["return"], 4), "limit": s["cost_limit"],
        "mean |critic - rollout|": round(sum(errs) / max(len(errs), 1), 4),
        "real samples": s["samples_real_env"],
        "probe samples": s.get("samples_real_probe", 0),
        "surrogate": args.surrogate, "seed": args.seed,
    }
    (root / "results.json").write_text(json.dumps([row], indent=2, default=float))
    (root / "results.md").write_text(markdown_table([row]) + "\n")
    print("\n" + markdown_table([row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
