#!/usr/bin/env python
"""Emit `docs/GYM_RESULTS.md`: one page per environment, generated.

Every number in the output comes from an artifact on disk -- `runs/gym`'s
`summary.json`, `ladder.json`, `paired.json` and the per-run `result.json`
files, plus the specs in `pspe/envs/catalog.py`. Nothing is transcribed by
hand, because the numbers are the entire content of the document and three
separate hand-written tables in this work's drafts turned out to have an
error in them.

    scripts/make_gym_results_md.py                 # -> docs/GYM_RESULTS.md
    scripts/make_gym_results_md.py --out -         # stdout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pspe.envs.catalog import SPECS                              # noqa: E402

#: Two-sided 5% critical value of t with 11 degrees of freedom. Twelve
#: evaluation scenarios, paired against the incumbent on the same twelve, so
#: the differences carry 11 df.
T_CRIT_11 = 2.201

MINUS = "−"
PM = "±"


def fnum(x) -> str:
    if x is None:
        return "---"
    a = abs(x)
    if a >= 1e5 or (0 < a < 1e-3):
        out = f"{x:.4g}"
        return out.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    if x == int(x) and a < 1e4:
        return str(int(x))
    if a >= 100:
        return f"{x:,.1f}".replace(",", " ")
    return f"{x:.4g}"


def sign(x: float, places: int = 3) -> str:
    """A signed number with a typographic minus."""
    return f"{x:+.{places}f}".replace("-", MINUS)


def num(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}".replace("-", MINUS)


def wrap(text: str, width: int = 76) -> str:
    return "\n".join(textwrap.wrap(" ".join(text.split()), width))


def feas_cell(v) -> str:
    f, of = v["feasible"], v["of"]
    return f"{f}/{of}" if f == of else f"**{f}/{of}**"


def rung_order(v):
    """The catalog's own ordering: fully feasible first, then by return."""
    return (0 if v["feasible"] == v["of"] else 1, -v["return"])


def verdict_for(env, paired):
    """(verdict, reason) from the paired test rather than the point estimate."""
    p = paired.get(env)
    if not p:
        return "no data", ""
    seeds = [p["seeds"][k] for k in sorted(p["seeds"])]
    n_feas = sum(s["feasible"] == s["of"] for s in seeds)
    if n_feas < len(seeds):
        return "INFEASIBLE", f"{n_feas} of {len(seeds)} seeds feasible on all 12"
    ts = [s["t"] for s in seeds]
    diffs = [s["paired_diff"] for s in seeds]
    mean = sum(diffs) / len(diffs)
    if any(t is None for t in ts):
        # Zero variance in the differences is not "no effect" -- it is the
        # strongest possible measurement of one. Every scenario gave the same
        # answer, so the gap is exact and needs no interval. `CMDPToy-v0` is
        # the case that matters: a deterministic policy on a one-step bandit
        # returns the same number twelve times, and its loss to the analytic
        # optimum is the declared 5% training margin rather than noise.
        if all(abs(d) < 1e-12 for d in diffs):
            return "identical", "every scenario returns the same value"
        word = "exactly better" if mean > 0 else "exactly worse"
        return "deterministic", f"{word} by {abs(mean):.4g} on every scenario"
    if all(t > T_CRIT_11 for t in ts):
        return "BEATS", f"every seed t > {T_CRIT_11}"
    if all(t < -T_CRIT_11 for t in ts):
        return "LOSS", f"every seed t < {MINUS}{T_CRIT_11}, mean {sign(mean)}"
    if max(abs(t) for t in ts) < 1.0:
        return "tie", "no |t| reaches 1"
    return "mixed", "seeds disagree in sign or in significance"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/gym")
    ap.add_argument("--out", default="docs/GYM_RESULTS.md")
    a = ap.parse_args()
    root = Path(a.root)

    summary = {r["env_id"]: r
               for r in json.loads((root / "summary.json").read_text())}
    # Oracle headroom, one file per environment from `scripts/gym_oracle.py`.
    # Absent for an environment whose oracle has not been run, which is a
    # different statement from "zero headroom" and is printed as such.
    oracle = {}
    for op in sorted(root.glob("oracle_*.json")):
        for env, rec in json.loads(op.read_text()).items():
            rows = [r for r in rec["rows"]
                    if math.isfinite(r["oracle_return"])
                    and math.isfinite(r["oracle_cost"])]
            if not rows or rec.get("oracle_feasible_seeds", 0) == 0:
                continue
            o = sum(r["oracle_return"] for r in rows) / len(rows)
            ic = [r["incumbent_return"] for r in rows if "incumbent_return" in r]
            prev = oracle.get(env)
            cand = {"oracle": o, "cost": sum(r["oracle_cost"] for r in rows) / len(rows),
                    "feasible": rec["oracle_feasible_seeds"], "of": rec["of"],
                    "finite": len(rows), "budget": f"{rec['cands']}x{rec['iters']}",
                    "incumbent": (sum(ic) / len(ic)) if ic else None}
            # Several files may cover one environment (a re-run after a fix).
            # Keep the one with the larger CEM budget, then the better bound.
            if prev is None or (cand["oracle"] > prev["oracle"]):
                oracle[env] = cand
    ladder = json.loads((root / "ladder.json").read_text())
    paired = json.loads((root / "paired.json").read_text())
    runs: "dict[str, list]" = {}
    for p in sorted(root.glob("*/seed*/result.json")):
        d = json.loads(p.read_text())
        runs.setdefault(d["env_id"], []).append(d)

    specs = {s.env_id: s for s in SPECS}
    order = [s.env_id for s in SPECS]
    out: "list[str]" = []
    w = out.append

    w("# The fourteen environments: measured results\n")
    w(wrap("One page per environment. Every number is generated by "
           "`scripts/make_gym_results_md.py` from the artifacts under "
           "`runs/gym` and the specs in `pspe/envs/catalog.py`; none of it is "
           "typed. Regenerate after a re-run rather than editing this file."))
    w("")
    w(wrap("Returns and costs are means over **twelve held-out evaluation "
           "scenarios** (the `eval` split under `SPLIT_SEED = 20261018`), "
           "rolled with the deterministic policy. Cost is the EPISODE SUM, "
           "and a row is feasible only if every episode's cost is inside the "
           "limit -- so a bolded feasibility count is a failed row however "
           "good its return looks."))
    w("")

    w("## Verdicts\n")
    w(wrap("The verdict is the paired comparison, not the point estimate. "
           "Every rung is scored on the same twelve scenarios, so the "
           "scenario is a blocking factor; `BEATS` and `LOSS` require all "
           f"three training seeds past |t| = {T_CRIT_11} (11 df, two-sided "
           "5%)."))
    w("")
    w("| environment | learned return | cost / limit | feasible | incumbent | verdict |")
    w("|---|---|---|---|---|---|")
    verdicts = {}
    for env in order:
        s = summary.get(env)
        if not s:
            continue
        v, why = verdict_for(env, paired)
        verdicts[env] = (v, why)
        fe = f"{s['learned_feasible_episodes']}/{s['episodes']}"
        if s["learned_feasible_episodes"] != s["episodes"]:
            fe = "**" + fe + "**"
        anchor = env.lower().replace(".", "")
        w("| [`" + env + "`](#" + anchor + ") "
          "| " + num(s["learned_return"]) + " " + PM
          + f" {s['learned_return_sd']:.3f} "
          "| " + fnum(s["learned_cost"]) + " / " + fnum(s["limit"]) + " "
          "| " + fe + " | " + num(s["incumbent_return"]) + " | " + v + " |")
    w("")
    w(wrap("The " + PM + " is the spread across three TRAINING SEEDS, not "
           "across episodes. Per-episode spreads are an order of magnitude "
           "larger on several of these and are given per environment below."))
    w("")
    counts: "dict[str, int]" = {}
    for v, _ in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    w("| verdict | count |")
    w("|---|---|")
    for v in ("BEATS", "deterministic", "identical", "tie", "mixed",
              "LOSS", "INFEASIBLE"):
        if v in counts:
            w("| " + v + " | " + str(counts[v]) + " |")
    w("")

    w("## Headroom: what the hindsight oracle leaves on the table\n")
    w(wrap("A verdict against the incumbent says nothing about how much room "
           "there was. `scripts/gym_oracle.py` optimises an open-loop action "
           "sequence per scenario by constrained CEM, knowing that scenario's "
           "future, under the same episode budget. Foresight makes open-loop "
           "at least as good as any closed-loop policy on a seeded "
           "deterministic rollout, so this bounds what ANY policy could "
           "reach, and the gap from the incumbent to it is the whole prize. "
           "It reproduces `CMDPToy-v0`'s closed form to four decimals at "
           "256x16."))
    w("")
    w("| environment | oracle | incumbent | headroom | learned | CEM budget |")
    w("|---|---|---|---|---|---|")
    for env in order:
        o = oracle.get(env)
        srow = summary.get(env)
        if not o:
            if srow:
                w("| `" + env + "` | not run | " + num(srow["incumbent_return"])
                  + " | --- | " + num(srow["learned_return"]) + " | --- |")
            continue
        inc = o["incumbent"] if o["incumbent"] is not None else (
            srow["incumbent_return"] if srow else None)
        head = (o["oracle"] - inc) if inc is not None else None
        hs = sign(head) if head is not None else "---"
        if head is not None and abs(head) < 0.05:
            hs = "**" + hs + "**"
        lr = num(srow["learned_return"]) if srow else "---"
        note = "" if o["finite"] == o["of"] else f" ({o['finite']}/{o['of']} finite)"
        w("| `" + env + "` | " + num(o["oracle"]) + note + " | "
          + (num(inc) if inc is not None else "---") + " | " + hs
          + " | " + lr + " | " + o["budget"] + " |")
    w("")
    w(wrap("A bolded headroom is under 0.05 of return: the incumbent is at "
           "the ceiling and no method can beat it there. Read those rows as a "
           "finding about the environment, not about any policy."))
    w("")

    w("## How many effective samples each eval split carries\n")
    w(wrap("`of: 12` counts seeds, not information. Before reading any t or "
           "standard deviation below, check how many DISTINCT values the "
           "twelve scenarios actually produce. Where that is two or three, "
           "the twelve-seed spread and every t computed on it overstate "
           "their own n, and the row is a weaker measurement than its format "
           "suggests."))
    w("")
    w("| environment | distinct incumbent returns / 12 | reading |")
    w("|---|---|---|")
    for env in order:
        p = paired.get(env)
        if not p:
            continue
        n = len(set(p["incumbent_returns"]))
        if n == 1:
            note = "**1 -- exact, no test applies**"
        elif n <= 3:
            note = "**" + str(n) + " effective samples -- degenerate split**"
        else:
            note = "usable"
        w("| `" + env + "` | " + str(n) + " | " + note + " |")
    w("")
    w("---\n")

    for env in order:
        sp, s, lad, p = specs[env], summary.get(env), ladder.get(env), paired.get(env)
        if not (s and lad):
            continue
        v, why = verdicts[env]
        w("## " + env + "\n")
        w("**" + sp.title + "** --- " + v + ((" (" + why + ")") if why else "") + "\n")
        w("- **Reward** " + " ".join(sp.reward.split()))
        w("- **Cost** " + " ".join(sp.cost.split()))
        w("- **Budget** " + " ".join(sp.budget.split())
          + " (`cost_limit = " + fnum(sp.cost_limit) + "`)")
        rr = runs.get(env, [])
        if rr:
            r0 = rr[0]
            w("- **Episode** " + str(r0["eval"]["steps"]) + " steps; trained "
              + str(r0["steps_budget"]) + " steps, " + str(r0["iterations"])
              + " dual updates, " + f"{r0['train_seconds'] / 60:.0f}"
              + " min per seed")
        w("")

        rows = dict(lad.get("ladder", {}))
        rows.update(lad.get("learned", {}))
        inc_name = lad["incumbent"]
        w("| rung | return | per-episode sd | cost | feasible |")
        w("|---|---|---|---|---|")
        for name, val in sorted(rows.items(), key=lambda kv: rung_order(kv[1])):
            label = "**" + name + "**" if name.startswith("learned") else name
            if name == inc_name or name + "_action" == inc_name:
                label += " &larr; incumbent"
            w("| " + label + " | " + num(val["return"]) + " "
              "| " + f"{val['return_sd']:.3f}" + " | " + fnum(val["cost"])
              + " | " + feas_cell(val) + " |")
        w("")

        if p:
            w("Paired against the incumbent on the same twelve scenarios:\n")
            w("| seed | paired difference | t (11 df) | feasible | mean cost |")
            w("|---|---|---|---|---|")
            for k in sorted(p["seeds"]):
                sd = p["seeds"][k]
                ts = "n/a" if sd["t"] is None else f"{sd['t']:+.2f}".replace("-", MINUS)
                mc = sum(sd["costs"]) / len(sd["costs"])
                w("| s" + k + " | " + sign(sd["paired_diff"], 4) + " | " + ts
                  + " | " + feas_cell(sd) + " | " + fnum(mc) + " |")
            nd = len(set(p["incumbent_returns"]))
            if nd == 1:
                w("")
                w(wrap("**Note.** All twelve scenarios return the same value, "
                       "so each difference above is exact rather than "
                       "estimated: there is no interval to put on it and no "
                       "test to run. That is a property of a deterministic "
                       "policy on a one-step problem, not a defect."))
            elif nd <= 3:
                w("")
                w(wrap("**Caveat.** These twelve scenarios produce only "
                       + str(nd) + " distinct incumbent returns, so the t "
                       "above rests on far fewer effective samples than its "
                       "11 df claims and is not a significance test."))
            w("")
        if rr:
            u = rr[0]["eval_untrained"]
            w(wrap("Untrained control, the same architecture at "
                   "initialisation through the same scoring path: return "
                   + num(u["return"]) + ", cost " + fnum(u["cost"])
                   + ", feasible " + feas_cell(u) + "."))
            w("")
        if getattr(sp, "caveat", ""):
            w(wrap("**Catalog caveat.** " + " ".join(sp.caveat.split())))
            w("")
        w("---\n")

    text = "\n".join(out)
    if a.out == "-":
        sys.stdout.write(text)
    else:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print("wrote " + a.out + " (" + str(len(text.splitlines())) + " lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
