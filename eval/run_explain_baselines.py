#!/usr/bin/env python
"""Section 7.2 explanation control: trained-in faithfulness vs post-hoc briefs.

    python eval/run_explain_baselines.py --policy runs/plan/dar/policy.pt
    python eval/run_explain_baselines.py --backbone Qwen/Qwen2.5-0.5B-Instruct --device cuda

Three arms against one fixed policy:

* **trained-in**  - Eq. 11: supervised brief loss + the faithfulness reward.
* **no-faithful** - the same training with the faithfulness term switched off.
  This is the *ablation*, not the baseline: it has still been trained on the
  policy's briefs.
* **post-hoc**    - the generator is never trained against this policy at all;
  briefs are produced after the fact. This is the actual control the paper's
  trained-in claim needs, and it was missing.

F(b) = exp(-KL(pi || pi-hat-from-brief)), so higher is better and 1.0 means the
brief implies exactly the policy that produced it.
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
from pspe.plan import GaussianFieldPolicy  # noqa: E402
from pspe.utils import RunLogger, get_device, project_path, seed_everything  # noqa: E402

ARMS = {
    "trained-in":  dict(use_faithfulness=True, posthoc=False),
    "no-faithful": dict(use_faithfulness=False, posthoc=False),
    # Kept, but it is a trained-vs-UNTRAINED comparison and the name hid that:
    # `posthoc=True` sets iterations = 0, so this arm scores a randomly
    # initialised prefix, embedding and head. Its briefs do not parse and its
    # score is the parser's fallback. The two arms below are the controls that
    # make "trained-in wins" mean something (audit item 12.1).
    "post-hoc-untrained": dict(use_faithfulness=False, posthoc=True),
    # Trained on a DIFFERENT policy's briefs, then asked to explain this one.
    # Isolates "training on the policy being explained is what matters".
    "post-hoc-transfer":  dict(use_faithfulness=True, posthoc_transfer=True),
    # The instruction-tuned backbone prompted in its own tokens, no training.
    # The only arm that uses the LM as a language model; needs a real backbone.
    "post-hoc-prompted":  dict(posthoc_prompted=True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", default="dar", choices=["dar", "swe", "rdf"])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--backbone", default="tiny")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--policy", default=None,
                        help="checkpoint of a *trained* policy; briefs about an "
                             "untrained policy are degenerate and not worth rating")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faith-weights", type=float, nargs="*", default=None,
                        help="extra trained-in arms at these faithfulness weights, e.g. "
                             "10 100 1000. At the default weight (1.0) the term's gradient is "
                             "real but two orders below the supervised term; this is the "
                             "sweep that says whether the objective can matter at all")
    parser.add_argument("--contrastive", type=float, default=0.0,
                        help="weight of the contrastive faithfulness term; adds a "
                             "'contrastive@w' arm on top of the selected arms")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="subset of arms, so each job fits a short backfill slot")
    parser.add_argument("--max-len", type=int, default=160,
                        help="brief length cap; the reference averages 136 tokens, "
                             "so the old default of 96 truncated every brief")
    parser.add_argument("--n-permutation", type=int, default=128,
                        help="briefs scored aligned and shuffled, for the state-specificity gap")
    parser.add_argument("--out", default="runs/explain_baselines")
    args = parser.parse_args()

    device = get_device(args.device)
    root = project_path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    env = make_env(args.testbed, dynamics="truth", grid=args.grid,
                   horizon=args.horizon, device=device, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim).to(device)
    if args.policy:
        state = torch.load(args.policy, map_location=device)
        policy.load_state_dict(state["policy"] if "policy" in state else state)
        print(f"loaded policy from {args.policy}")
    else:
        print("WARNING: untrained policy — F(b) is measured against a policy that "
              "does nothing interesting. Pass --policy for a reportable number.")

    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}
    if args.contrastive > 0:
        arms[f"contrastive@w{args.contrastive:g}"] = dict(
            use_faithfulness=True, posthoc=False, weight_contrastive=args.contrastive)
    for w in (args.faith_weights or []):
        arms[f"trained-in@w{w:g}"] = dict(use_faithfulness=True, posthoc=False, weight_faithful=w)

    rows = []
    for name, overrides in arms.items():
        seed_everything(args.seed)
        # The transfer arm needs a second, different policy to learn from. A
        # freshly initialised one is a legitimate source -- the control asks
        # whether training on THIS policy matters, not whether the source is
        # any good -- and it is reproducible from the arm's own seed.
        src = None
        if overrides.get("posthoc_transfer"):
            torch.manual_seed(args.seed + 7_000)
            src = GaussianFieldPolicy(env.obs_shape[0], env.action_dim).to(device)
            seed_everything(args.seed)
        trainer = ExplainTrainer(
            env, policy,
            # max_len 160: the reference briefs run past the default 96 and were
            # being cut before the cost/reward/confidence footer (defect 47).
            ExplainConfig(backbone=args.backbone, max_len=args.max_len),
            ExplainTrainConfig(iterations=args.iterations, batch=args.batch,
                               seed=args.seed, **overrides),
            RunLogger(root / name, use_tensorboard=False),
            device,
            source_policy=src,
        )
        summary = trainer.train()
        # Permutation control: the same briefs scored against another state's
        # action. A generator that learned the format and a generic action
        # scores the same shuffled as aligned; see docs/results/NDWS_EXPLAIN.md,
        # where that is exactly what happened on 64-dimensional fire plans.
        aligned = trainer.faithfulness_samples(args.n_permutation)
        shuffled = trainer.faithfulness_samples_permuted(args.n_permutation)
        rows.append({
            "arm": name,
            "F aligned": round(float(aligned.mean()), 4),
            "F shuffled": round(float(shuffled.mean()), 4),
            "state-specific gap": round(float(aligned.mean() - shuffled.mean()), 4),
            "F(b)": round(summary["eval/faithfulness"], 4),
            "F(b) reference": round(summary["eval/faithfulness_reference"], 4),
            "KL": round(summary["eval/kl"], 4),
            "train iters": summary["train_iterations"],
            "stub backbone": summary["backbone_is_stub"],
            "wall (s)": round(summary["wall_clock_s"], 1),
        })
        print(f"[{name}] F(b) = {rows[-1]['F(b)']}")

    table = markdown_table(rows)
    (root / "results.md").write_text(table + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + table)
    print(f"\nwritten to {root / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
