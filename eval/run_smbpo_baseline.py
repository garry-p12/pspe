#!/usr/bin/env python
"""SMBPO, the second model-based safe-RL baseline §9.2 says is still unrun.

    python eval/run_smbpo_baseline.py --testbed rdf --seed 0 --out runs/smbpo/rdf_s0

§9.2 reports CAP transplanted onto our planner (11.99% violating against our
7.80%, z = +2.08) and then concedes: "SMBPO and SafeDreamer unrun". §3.2 makes a
structural claim about *why* CAP loses -- ensemble disagreement is not the
quantity that breaches the limit -- and a structural claim earns its keep only
if it predicts the next method's result too. This runs that test.

SMBPO [Thomas et al., NeurIPS 2021, "Safe Reinforcement Learning by Imagining
the Near Future"] corrects for model error by *pessimism* rather than by an
adaptive inflation:

    c_SMBPO  =  max over ensemble members      (worst case, not mean + k*std)

over a truncated imagination horizon, with the penalty applied to states from
which a violation is reachable within that horizon. Their guarantee comes from
the model being accurate over a SHORT horizon and the penalty being large
enough; there is no adapted coefficient.

That makes it a clean contrast with CAP on the one axis this paper is about:

    ours   -- quantile of the REALISED cost distribution (what actually breaches)
    CAP    -- mean + k*std of model disagreement, k adapted from real violations
    SMBPO  -- max over model disagreement, truncated horizon, no adaptation

CAP and SMBPO both derive their correction from disagreement among models.
§3.2's argument says that is the wrong distribution to bound, so **SMBPO should
also fail to beat the conformal margin, and by a larger margin than CAP** --
`max` is a cruder statistic than `mean + k*std` and, having no adaptation,
cannot walk itself back when disagreement is uninformative. A pre-registered
prediction, recorded here before the run.

What is implemented, stated so the comparison can be judged:

  * the same independently seeded ensemble CAP gets, same member count, cost
    taken as the MAX across members rather than mean + k*std;
  * the truncated imagination horizon SMBPO relies on, as a fraction of the
    planning horizon (`--imagine-frac`), with the pessimistic cost accumulated
    over that prefix and extrapolated to the full episode, which is the
    "imagining the near future" mechanism;
  * the same planner, dual, testbed, limit, probe budget and evaluation
    protocol as every other arm, so nothing but the correction differs.

What is not: SMBPO's original work uses its own SAC-based optimiser, its own
replay scheme and a terminal-value penalty tied to a discount factor. This
transplants its pessimism mechanism onto our planner, exactly as
`run_cap_baseline.py` transplants CAP's. It is not a reimplementation.
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


class SMBPOTrainer(HybridPlannerTrainer):
    """The planner, with SMBPO's pessimistic cost in place of a conformal margin.

    The dual is fed the worst ensemble member's cost over a truncated horizon,
    scaled back up to a full episode. The margin machinery is off so the two
    mechanisms are not stacked, matching how CAP is run.
    """

    def __init__(self, *args, ensemble=(), imagine_frac=0.5, **kw):
        super().__init__(*args, **kw)
        self.ensemble = list(ensemble)
        self.imagine_frac = float(imagine_frac)
        self.pessimism_gap: list[float] = []

    @torch.no_grad()
    def _pessimism(self, episodes: int = 8) -> tuple[float, float]:
        """(max - mean) ensemble cost over the truncated horizon, and the mean.

        This returns a PENALTY, not a replacement cost estimate. The first
        version of this method returned `max(prefix) / imagine_frac` as the cost
        itself, and that was wrong in a way that inverted the experiment: rdf's
        cost accrues faster late in an episode (0.193 -> 0.229 per step), so
        linear extrapolation from a half-horizon prefix understates the episode
        cost by 4.7%, or -0.118 against a 2.53 episode cost. The pessimism term
        is only +0.03. Net, the dual was handed a cost BELOW the baseline's own
        estimate, under-constrained, and violated more (17.9% against a 14.1%
        no-margin baseline) -- a result about my arithmetic, not about SMBPO.

        SMBPO penalises states from which a violation is reachable inside a short
        horizon; it does not rescale the episode cost. So the faithful transplant
        keeps the full-horizon estimate the baseline uses and ADDS short-horizon
        disagreement to it. The penalty is >= 0 by construction, so this arm can
        only ever be more conservative than the baseline, never less.
        """
        full = self.cfg.horizon
        short = max(1, int(round(full * self.imagine_frac)))
        per_member = []
        keep_state, keep_t = self.env.state, getattr(self.env, "t", 0)
        base, base_h = self.env.surrogate, self.cfg.horizon
        try:
            self.cfg.horizon = short
            for m in self.ensemble:
                self.env.surrogate = m
                seed = self._probe_seed + 77
                state0 = self._probe_initial(self.env, episodes, seed)
                per_member.append(float(self._rollout_costs(self.env, state0, seed).mean()))
        finally:
            self.cfg.horizon = base_h
            self.env.surrogate = base
            self.env.state, self.env.t = keep_state, keep_t
        t = torch.tensor(per_member)
        mean = float(t.mean())
        # Scaled to the full horizon so the penalty is commensurate with a limit
        # defined on whole episodes. Scaling a DIFFERENCE is safe where scaling
        # the level was not: both members share the same accrual profile, so the
        # profile cancels.
        gap = (float(t.max()) - mean) / self.imagine_frac
        return gap, mean

    def _dual_input(self, surrogate_cost: float, it: int):
        """The baseline's own cost estimate, plus short-horizon pessimism."""
        if not self.ensemble:
            return super()._dual_input(surrogate_cost, it)
        gap, mean = self._pessimism()
        penalised = surrogate_cost + gap
        self.pessimism_gap.append(gap)
        rec = {"smbpo/pessimism": gap, "smbpo/ensemble_mean_prefix": mean,
               "smbpo/surrogate_cost": surrogate_cost,
               "smbpo/penalised_cost": penalised,
               "smbpo/imagine_frac": self.imagine_frac}
        if self.cfg.real_cost_every and self.eval_env is not None \
                and it % self.cfg.real_cost_every == 0:
            # Spent for reporting only: SMBPO has no adaptation to feed it into.
            # Matching CAP's probe budget keeps the sample accounting identical.
            real = self.probe_real_cost(self.cfg.real_cost_episodes)
            rec["dual/real_cost"] = real
            rec["smbpo/pessimism_covers_real"] = float(penalised >= real)
            return penalised, rec
        return penalised, rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="rdf", choices=["dar", "swe", "rdf"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--members", type=int, default=5)
    # fno matches run_cap_baseline.py, so the two share an ensemble
    # construction. deeponet is real-valued and exists as an escape hatch for
    # torch builds whose clip_grad_norm_ cannot handle FNO's complex weights
    # (pspe/simulate/trainer.py:109) -- a local-environment limit, not a
    # property of either method.
    ap.add_argument("--surrogate", default="fno",
                    choices=["fno", "deeponet", "gnot"])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--imagine-frac", type=float, default=0.5,
                    help="fraction of the horizon SMBPO trusts the model over")
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--probe-episodes", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--sat-coef", type=float, default=1.0)
    ap.add_argument("--adv-floor", type=float, default=1e-2)
    ap.add_argument("--dual-ema", type=float, default=0.7)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/smbpo")
    args = ap.parse_args()

    device = get_device(args.device)
    root = project_path(args.out); root.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(args.testbed, grid=args.grid)
    spec = make_testbed(args.testbed, grid=args.grid)
    fit_transitions = int(data["states"].shape[0] * data["states"].shape[1])

    # Same seeding as run_cap_baseline.py, so the two baselines share an
    # ensemble construction and differ only in how its spread is used.
    ensemble = []
    for m in range(args.members):
        seed_everything(args.seed * 100 + m)
        model = make_surrogate(args.surrogate, spec.n_channels, grid=args.grid)
        sim = SimulateTrainer(
            model, SimulateTrainConfig(testbed=args.testbed, surrogate=args.surrogate,
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
    trainer = SMBPOTrainer(
        train_env, policy,
        cfg=PlannerConfig(iterations=args.iterations, horizon=args.horizon,
                          seed=args.seed, saturation_coef=args.sat_coef,
                          advantage_std_floor=args.adv_floor, dual_ema=args.dual_ema,
                          ki=args.ki, eval_every=args.eval_every,
                          real_cost_every=args.probe_every,
                          real_cost_episodes=args.probe_episodes,
                          cost_margin_k=0.0, margin_conformal=False),
        eval_env=eval_env, logger=RunLogger(root / "smbpo", use_tensorboard=False),
        device=device, surrogate_train_transitions=fit_transitions,
        ensemble=ensemble, imagine_frac=args.imagine_frac)
    s = trainer.train()

    gaps = trainer.pessimism_gap
    row = {
        "method": "SMBPO (full-horizon cost + truncated pessimism)",
        "testbed": args.testbed,
        "violating": round(s.get("eval/violating_eval_fraction", float("nan")), 4),
        "worst": round(s.get("eval/cost_max_over_run", float("nan")), 4),
        "return": round(s["return"], 4), "limit": s["cost_limit"],
        "imagine frac": args.imagine_frac,
        "mean pessimism": round(sum(gaps) / max(len(gaps), 1), 4),
        "real samples": s["samples_real_env"],
        "probe samples": s.get("samples_real_probe", 0),
        "members": args.members, "surrogate": args.surrogate, "seed": args.seed,
    }
    (root / "results.json").write_text(json.dumps([row], indent=2, default=float))
    (root / "results.md").write_text(markdown_table([row]) + "\n")
    print("\n" + markdown_table([row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
