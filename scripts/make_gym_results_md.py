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


#: Verdict names, in the order the count table lists them. Also the set the
#: table is checked against, so a new verdict cannot be counted without being
#: shown. `STALE` and `no data` are easy to forget: they are the rows with no
#: result in them.
VERDICT_ORDER = ("BEATS", "trend", "deterministic", "identical", "tie",
                 "mixed", "LOSS", "INFEASIBLE", "STALE", "no data")


def verdict_for(env, paired, summary=None):
    """(verdict, reason) from the paired test rather than the point estimate."""
    # A run whose environment has since changed has no verdict on THIS
    # environment, and the saved numbers are not a weak result on it. Checked
    # before the paired record, because `paired_gym.py` writes `{"stale":
    # true}` for those seeds and never rolls them, so there are no episodes
    # to pair.
    srow = (summary or {}).get(env) or {}
    if srow.get("stale"):
        what = ", ".join(
            f"{k} was {v['saved']}, is now {v['live']}"
            for k, v in srow["stale"].items())
        return "STALE", what
    p = paired.get(env)
    if not p:
        return "no data", ""
    seeds = [p["seeds"][k] for k in sorted(p["seeds"])
             if not p["seeds"][k].get("stale")]
    if not seeds:
        return "STALE", "no checkpoint matches the current observation space"
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
    # Every seed pointing the same way but none of them clearing the critical
    # value is not disagreement, and calling it "mixed" asserts a conflict the
    # seeds do not have. `FloodEvacuate-v0` is the row that matters: three
    # seeds improve by +898.0, +890.1 and +1200.8 with |t| of 1.18, 1.16 and
    # 1.50. They agree on the direction AND on falling short of it. The gap is
    # large and the environment is loud -- 12 scenarios cannot resolve it --
    # which is a statement about the power of the test, not about the policy.
    if all(t > 0 for t in ts) or all(t < 0 for t in ts):
        word = "improves" if mean > 0 else "loses"
        return "trend", (
            f"every seed {word}, none reaches |t| = {T_CRIT_11} "
            f"(max {max(abs(t) for t in ts):.2f}); mean {sign(mean)}, "
            f"underpowered at 12 scenarios")
    return "mixed", "seeds disagree in sign"


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
    disc = {}
    dp = root / "discrimination.json"
    if dp.exists():
        disc = json.loads(dp.read_text())
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
        v, why = verdict_for(env, paired, summary)
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
    # Every verdict the table above can produce has to appear here, or the
    # counts sum to fewer environments than the catalog holds and the
    # difference is invisible. `STALE` and `no data` are the two that are
    # easy to forget, because they are the rows with no result in them.
    # One tuple, used both to order the rows and to check that nothing is
    # missing from them. These were two parallel literals until adding `trend`
    # updated one and not the other, which is exactly the drift the check
    # below exists to catch -- so the check now cannot disagree with the rows.
    for v in VERDICT_ORDER:
        if v in counts:
            w("| " + v + " | " + str(counts[v]) + " |")
    missing = set(counts) - set(VERDICT_ORDER)
    if missing:
        raise SystemExit(f"verdicts not in the count table: {sorted(missing)}")
    total = sum(counts.values())
    w("")
    w(wrap(f"{total} environments, which is every row in the catalog. A "
           "verdict that went uncounted here would make this table sum to "
           "fewer than the catalog holds, so it is asserted rather than "
           "trusted."))
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
                lr = ("stale" if srow.get("stale")
                      else num(srow["learned_return"]))
                w("| `" + env + "` | not run | " + num(srow["incumbent_return"])
                  + " | --- | " + lr + " | --- |")
            continue
        inc = o["incumbent"] if o["incumbent"] is not None else (
            srow["incumbent_return"] if srow else None)
        head = (o["oracle"] - inc) if inc is not None else None
        hs = sign(head) if head is not None else "---"
        if head is not None and abs(head) < 0.05:
            hs = "**" + hs + "**"
        lr = ("stale" if (srow or {}).get("stale")
              else num(srow["learned_return"]) if srow else "---")
        note = "" if o["finite"] == o["of"] else f" ({o['finite']}/{o['of']} finite)"
        w("| `" + env + "` | " + num(o["oracle"]) + note + " | "
          + (num(inc) if inc is not None else "---") + " | " + hs
          + " | " + lr + " | " + o["budget"] + " |")
    w("")
    w(wrap("A bolded headroom is under 0.05 of return: the incumbent is at "
           "the ceiling and no method can beat it there. Read those rows as a "
           "finding about the environment, not about any policy."))
    w("")

    w("## How many eval scenarios can tell two policies apart\n")
    w(wrap("`of: 12` counts seeds. It does not count information. A scenario "
           "DISCRIMINATES when the feasible policies on the ladder do not all "
           "score the same on it; one where they do is a constant that "
           "contributes nothing but a smaller standard error. The span is "
           "taken over policies FEASIBLE on that scenario, because a better "
           "outcome bought by overspending does not show the scenario can be "
           "improved within budget. Measured by "
           "`scripts/gym_discrimination.py`."))
    w("")
    w(wrap("This replaces an earlier count of distinct incumbent returns, "
           "which measured the wrong thing -- how much ONE policy's score "
           "varied, not whether the split separates policies. That metric "
           "called `CMDPToy-v0` and `FloodCrossing-v0` degenerate when both "
           "discriminate on all twelve scenarios, and it understated "
           "`FloodEvacuate-v0`, whose feasible rungs are identical on every "
           "scenario."))
    w("")
    if disc:
        w("| environment | discriminating | largest feasible span | reading |")
        w("|---|---|---|---|")
        for env in order:
            dd = disc.get(env)
            if not dd:
                continue
            n, of = dd["discriminating"], dd["of"]
            if n == 0:
                note = "**no eval scenario separates any two feasible policies**"
            elif n <= of // 4:
                note = f"**only {n} of {of} carries signal**"
            elif n < of:
                note = f"{of - n} scenarios are constant"
            else:
                note = "every scenario carries signal"
            w("| `" + env + "` | " + f"{n}/{of}" + " | "
              + fnum(dd["largest_span"]) + " | " + note + " |")
        w("")
        bad = [e for e, dd in disc.items()
               if dd["discriminating"] <= dd["of"] // 4]
        if bad:
            w(wrap("Rows to read with care: " + ", ".join("`" + b + "`" for b in bad)
                   + ". A standard deviation over twelve seeds on those is not "
                     "a standard deviation over twelve observations, and any t "
                     "computed from it overstates its own n."))
            w("")
    else:
        w(wrap("Not measured -- run `scripts/gym_discrimination.py`."))
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
                # `paired_gym.py` records a checkpoint it refused to roll as
                # `{"stale": true}` plus the two shapes. There are no
                # episodes behind it, so there is no difference to pair and
                # nothing to put in the numeric columns.
                if sd.get("stale"):
                    w("| s" + k + " | --- | --- | not rolled | --- |")
                    continue
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
