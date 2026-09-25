#!/usr/bin/env python
"""CAP, the model-based safe-RL baseline our margin should be compared against.

    python eval/run_cap_baseline.py --testbed rdf --seed 0 --out runs/cap/rdf_s0

Every safe-RL baseline in this repository is model-free (CPO, PPO-Lagrangian,
Saute, primal-dual NPG), which makes "fewer real transitions than model-free
methods" close to tautological for a model-based planner. The comparison that
matters is against a method that also plans in a learned model and also
corrects for the model being wrong.

Conservative and Adaptive Penalty (Ma et al., AAAI 2022) is the closest such
method to ours: it inflates the cost estimate by an uncertainty term and adapts
the inflation from real data. Where we conformalise the realised cost against
the quantity the dual controls, CAP penalises by ensemble disagreement:

    c_CAP  =  c_mean  +  k * c_std        over an ensemble of surrogates
    k      adapted upward when real violations exceed the target, down otherwise

What is implemented here, stated so the comparison can be judged:

  * an ensemble of N independently seeded surrogates, cost taken as the mean
    plus k standard deviations across members;
  * k adapted by the same PID-style rule the paper describes, driven by the
    same periodic real-environment probe our method uses, at the SAME probe
    budget, so neither method is advantaged by seeing more of the true system;
  * the same planner, dual, testbed, limit and evaluation protocol as our arms.

What is not: CAP's original work uses its own model-based policy optimiser and
a different exploration scheme. This transplants its cost-penalty mechanism
onto our planner, which isolates the thing being compared (how the model's
error is corrected for) and removes everything else. That is the fair
comparison for this paper's claim and it is not a reimplementation of the
paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import markdown_table  # noqa: E402
from pspe.envs import make_env  # noqa: E402
from pspe.plan import GaussianFieldPolicy, HybridPlannerTrainer, PlannerConfig  # noqa: E402
from pspe.simulate import (  # noqa: E402
    SimulateTrainConfig, SimulateTrainer, ensure_dataset, make_surrogate,
)
from pspe.simulate.solvers import make_testbed  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


class CAPTrainer(HybridPlannerTrainer):
    """The planner, with CAP's cost penalty in place of a conformal margin.

    The dual is fed `mean + k*std` across the ensemble rather than a single
    model's cost, and `k` rises when the probe finds the real violation rate
    above target. The margin machinery is switched off so the two mechanisms
    are not stacked.
    """

    def __init__(self, *args, ensemble=(), k_init=1.0, k_lr=0.5, k_max=20.0,
                 target=0.1, **kw):
        super().__init__(*args, **kw)
        self.ensemble = list(ensemble)
        self.k, self.k_lr, self.k_max, self.target = k_init, k_lr, k_max, target
        self.probe_violations: list[float] = []

    @torch.no_grad()
    def _ensemble_cost(self, episodes: int = 8) -> tuple[float, float]:
        """Mean and spread of episode cost across ensemble members."""
        per_member = []
        keep_state, keep_t = self.env.state, getattr(self.env, "t", 0)
        base = self.env.surrogate
        try:
            for m in self.ensemble:
                self.env.surrogate = m
                seed = self._probe_seed + 77
                state0 = self._probe_initial(self.env, episodes, seed)
                per_member.append(float(self._rollout_costs(self.env, state0, seed).mean()))
        finally:
            self.env.surrogate = base
            self.env.state, self.env.t = keep_state, keep_t
        t = torch.tensor(per_member)
        return float(t.mean()), float(t.std(unbiased=False)) if len(t) > 1 else 0.0

    def _dual_input(self, surrogate_cost: float, it: int):
        """Penalised cost, and k adapted from the probe's real violation rate."""
        if not self.ensemble:
            return super()._dual_input(surrogate_cost, it)
        mean, std = self._ensemble_cost()
        penalised = mean + self.k * std
        rec = {"cap/ensemble_mean": mean, "cap/ensemble_std": std, "cap/k": self.k,
               "cap/penalised_cost": penalised}
        if self.cfg.real_cost_every and self.eval_env is not None \
                and it % self.cfg.real_cost_every == 0:
            # Same probe budget as our method: the comparison is about how the
            # error is corrected, not about who gets more of the true system.
            real = self.probe_real_cost(self.cfg.real_cost_episodes)
            self.probe_violations.append(float(real > self.env.task.cost_limit))
            recent = self.probe_violations[-10:]
            rate = sum(recent) / max(len(recent), 1)
            self.k = float(min(self.k_max, max(0.0,
                          self.k + self.k_lr * (rate - self.target))))
            rec.update({"dual/real_cost": real, "cap/violation_rate": rate})
            return penalised, rec
        return penalised, rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="rdf", choices=["dar", "swe", "rdf"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--probe-episodes", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--k-init", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=0.1)
    ap.add_argument("--sat-coef", type=float, default=1.0)
    ap.add_argument("--adv-floor", type=float, default=1e-2)
    ap.add_argument("--dual-ema", type=float, default=0.7)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/cap")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    ensemble = []
    for m in range(args.members):
        seed_everything(args.seed * 100 + m)
        model = make_surrogate("fno", spec.n_channels, grid=args.grid)
        sim = SimulateTrainer(
            model, SimulateTrainConfig(testbed=args.testbed, surrogate="fno",
                                       epochs=args.epochs, grid=args.grid),
            data, RunLogger(root / f"member{m}", use_tensorboard=False), device)
        s = sim.train()
        for q in sim.model.parameters():
            q.requires_grad_(False)
        ensemble.append(sim.model)
        print(f"  member {m}: rel L2 {s['rel_l2_final']:.4f}", flush=True)

    env_kwargs = dict(testbed=args.testbed, grid=args.grid, horizon=args.horizon,
                      n_actuators=9, device=device, batched=True)
    seed_everything(args.seed)
    train_env = make_env(dynamics="surrogate", surrogate=ensemble[0], **env_kwargs)
    eval_env = make_env(dynamics="truth", **env_kwargs)
    policy = GaussianFieldPolicy(train_env.obs_shape[0], train_env.action_dim)
    trainer = CAPTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                          seed=args.seed, saturation_coef=args.sat_coef,
                          advantage_std_floor=args.adv_floor, dual_ema=args.dual_ema,
                          ki=args.ki, eval_every=args.eval_every,
                          real_cost_every=args.probe_every,
                          real_cost_episodes=args.probe_episodes,
                          cost_margin_k=0.0, margin_conformal=False),
        eval_env=eval_env, logger=RunLogger(root / "cap", use_tensorboard=False),
        device=device, surrogate_train_transitions=fit_transitions,
        ensemble=ensemble, k_init=args.k_init, target=args.target)
    s = trainer.train()

    row = {
        "method": "CAP (ensemble penalty)", "testbed": args.testbed,
        "violating": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
        "worst": round(s.get("eval/cost_max_over_run", float("nan")), 4),
        "return": round(s["return"], 4), "limit": s["cost_limit"],
        "final k": round(trainer.k, 3),
        "real samples": s["samples_real_env"],
        "probe samples": s.get("samples_real_probe", 0),
        "members": args.members, "seed": args.seed,
    }
    (root / "results.json").write_text(json.dumps([row], indent=2, default=float))
    (root / "results.md").write_text(markdown_table([row]) + "\n")
    print("\n" + markdown_table([row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
