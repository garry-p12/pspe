#!/usr/bin/env python
"""Prop. 3 made concrete: a certified faithfulness floor with its coverage check.

    python eval/run_conformal.py --policy runs/plan/dar/policy.pt
    python eval/run_conformal.py --backbone Qwen/Qwen2.5-0.5B-Instruct --device cuda

Trains the explainer against a policy, then draws a calibration set and a
DISJOINT test set of fresh briefs, and reports, for each delta:

    f_min      the certified floor: P[F(b) >= f_min] >= 1 - delta
    coverage   the fraction of test briefs that actually met it

The pair is the result. A floor without its coverage is a claim; a floor whose
coverage matches the nominal level is a certificate that survived contact with
held-out data. A floor whose coverage falls short says the exchangeability
assumption failed — which, for a deployment guarantee, is the thing you most
need to know.
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
from pspe.explain import ExplainConfig, ExplainTrainConfig, ExplainTrainer  # noqa: E402
from pspe.explain.conformal import certify, minimum_calibration_size  # noqa: E402
from pspe.plan import GaussianFieldPolicy  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--backbone", default="tiny")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/conformal")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    for delta in args.deltas:
        need = minimum_calibration_size(delta)
        if args.n_cal < need:
            raise SystemExit(f"delta={delta} needs n_cal >= {need}; got {args.n_cal}")

    env = make_env(args.testbed, dynamics="truth", grid=args.grid,
                   horizon=args.horizon, device=device, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim).to(device)
    if args.policy:
        state = torch.load(args.policy, map_location=device)
        policy.load_state_dict(state["policy"] if "policy" in state else state)

    trainer = ExplainTrainer(
        env, policy, ExplainConfig(backbone=args.backbone),
        ExplainTrainConfig(iterations=args.iterations, seed=args.seed,
                           log_dir=str(root / "train")),
        RunLogger(root / "train", use_tensorboard=False), device,
    )
    summary = trainer.train()

    # Disjoint draws. The policy is fixed from here on, which is what makes the
    # calibration and test briefs exchangeable.
    cal = trainer.faithfulness_samples(args.n_cal)
    test = trainer.faithfulness_samples(args.n_test)

    rows = []
    for delta in args.deltas:
        cert = certify(cal, test, delta)
        rows.append({
            "delta": delta,
            "certified F floor": round(cert.f_min, 4),
            "empirical coverage": round(cert.empirical_coverage, 4),
            "nominal": round(1 - delta, 2),
            "p (one-sided)": round(cert.p_value, 3),
            "holds": cert.holds,
            "n_cal": cert.n_cal,
            "n_test": cert.n_test,
        })

    context = {
        "mean F (cal)": round(float(cal.mean()), 4),
        "mean F (test)": round(float(test.mean()), 4),
        "train eval F": round(summary["eval/faithfulness"], 4),
        "backbone": summary["backbone"],
        "stub": summary["backbone_is_stub"],
    }
    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n\n" + json.dumps(context) + "\n")
    (root / "results.json").write_text(json.dumps({"rows": rows, "context": context}, indent=2))
    print("\n" + table)
    print(json.dumps(context))
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
