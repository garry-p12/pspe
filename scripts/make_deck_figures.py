#!/usr/bin/env python
"""Figures for the presentation deck (see docs/DECK_POINTERS.md).

    python scripts/make_deck_figures.py

Writes docs/figures/deck/*.{svg,pdf,png} in the palette the thesis figures
already use, so a slide can sit beside a paper figure without reformatting.

Two rules this file follows, both learned the hard way in this project:

* **Every number is measured and traceable.** Results computed on Vista are
  hard-coded with the job number and section in a comment beside them; results
  computed locally are READ FROM the JSON the run wrote, in `data/deck/`, so a
  re-run of the experiment changes the figure rather than leaving it stale.
* **Confidence intervals are drawn wherever they exist.** Four of the claims
  this project published were later cut or reversed because a five-seed mean
  was plotted without one. A bar with no interval is a bar that cannot be
  wrong, which is the problem.

Figures:

    deck_framework        the method: two gradients, the mixing rule, the dual
    deck_measurement      what closes the loop on reality, and what does not
    deck_matched_budget   §3.4n the headline: at equal data we beat all four
    deck_null_floor       §3.4l at full budget three of four do not beat zero
    deck_mixing_rule      §4.2a adaptive alpha is a variance effect
    deck_margin           §3.2b conformal margin, 14.1% to 5.0% violations
    deck_flood_frontier   the trigger family has three behaviours
    deck_collapse         the saturation collapse, and the guard that stops it
    deck_polder           controllers on the polder, feasibility marked
    deck_warning          measured warning time, and the floods with none
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "deck"
DATA = ROOT / "data" / "deck"

# The thesis figure palette, unchanged, so the two sets compose.
C = {
    "perceive": ("#dbe9f6", "#2b5d8c"),
    "simulate": ("#d7efe6", "#1f6b4f"),
    "plan":     ("#fbe6d0", "#b0561a"),
    "explain":  ("#e8dff3", "#5b3f85"),
    "world":    ("#eef1f4", "#37474f"),
    "good":     "#1f6b4f",
    "bad":      "#a12a2a",
    "warn":     "#b0561a",
    "ink":      "#222222",
    "muted":    "#666666",
    "rule":     "#c8ccd0",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",
    "axes.edgecolor": C["ink"],
    "axes.labelcolor": C["ink"],
    "text.color": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=220 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/figures/deck/{name}.{{svg,pdf,png}}")


def box(ax, x, y, w, h, title, lines, key, *, fs=8.5):
    fill, edge = C[key] if isinstance(C[key], tuple) else ("#ffffff", C[key])
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", zorder=3,
            fontsize=fs + 1.3, fontweight="bold", color=edge)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.52 - i * 0.26, ln, ha="center", va="top",
                zorder=3, fontsize=fs, color=C["ink"])


def arrow(ax, a, b, *, color=None, lw=1.4, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color or C["ink"], zorder=1,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------------------- #
# 1. the method
# --------------------------------------------------------------------------- #
def fig_framework():
    """What the method is, in one picture: two gradients and a rule to mix them."""
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.6); ax.axis("off")

    box(ax, 0.2, 3.5, 2.6, 1.6, "Differentiable model", [
        "roll the surrogate forward,", "differentiate through it"], "simulate")
    box(ax, 0.2, 1.2, 2.6, 1.6, "Real environment", [
        "sample, score, no gradient", "through the dynamics"], "world")

    box(ax, 3.3, 3.5, 2.5, 1.6, "Pathwise", [
        r"$g_{\mathrm{pw}}$", "low variance,", "biased by model error"], "simulate")
    box(ax, 3.3, 1.2, 2.5, 1.6, "Likelihood ratio", [
        r"$g_{\mathrm{lr}}$", "unbiased,", "high variance"], "world")

    box(ax, 6.3, 1.55, 2.6, 2.75, "Mixing rule", [
        r"$g = \alpha\, g_{\mathrm{pw}} + (1-\alpha)\, g_{\mathrm{lr}}$", "",
        "choose α to minimise", "the variance of g"], "plan")

    box(ax, 9.2, 3.5, 1.85, 1.6, "Policy", ["update"], "plan")
    box(ax, 9.2, 1.2, 1.85, 1.6, "PID dual", [r"$\lambda$ on the", "budget"], "explain")

    for y in (4.3, 2.0):
        arrow(ax, (2.8, y), (3.3, y))
    arrow(ax, (5.8, 4.3), (6.3, 3.7))
    arrow(ax, (5.8, 2.0), (6.3, 2.6))
    arrow(ax, (8.9, 3.7), (9.2, 4.3))
    arrow(ax, (8.9, 2.6), (9.2, 2.0))

    # Inside the box, where it belongs: the rule IS this expression, and set
    # beside the box it read as an unrelated aside.
    ax.text(7.6, 2.35, r"$\alpha^\ast=\dfrac{V_{\mathrm{lr}}-\mathrm{Cov}}"
                       r"{B^2+V_{\mathrm{pw}}+V_{\mathrm{lr}}-2\,\mathrm{Cov}}$",
            ha="center", va="center", fontsize=10.5, zorder=4)

    # The feedback path runs along the floor, clear of every box, and the
    # caption sits BELOW it rather than on top of it.
    ax.plot([10.1, 10.1, 1.5, 1.5], [1.2, 0.72, 0.72, 1.2],
            color=C["muted"], lw=1.2, ls="--", zorder=1)
    arrow(ax, (1.5, 0.95), (1.5, 1.2), color=C["muted"], lw=1.2, dashed=True)
    ax.text(5.8, 0.42, "the policy is re-measured on the REAL environment, "
                       "not on the model that trained it",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")
    ax.text(5.6, 5.42, "The estimator: a biased cheap gradient and an unbiased "
                       "expensive one, combined at the variance-minimising weight",
            ha="center", fontsize=10, fontweight="bold")
    save(fig, "deck_framework")


def fig_measurement():
    """The thesis in one picture: component metrics do not imply decision quality."""
    rows = [
        ("surrogate cost satisfied the dual", "7.3% of real evaluations violated",
         "probe the true environment"),
        ("one-step forecast skill high", "open-loop rollout invents a fire",
         "sync to observation"),
        ("faithfulness 0.47 vs 0.18, t = +21", "briefs carry zero state information",
         "permutation control"),
        ("reconstruction AP 0.594 vs 0.006", "decision worse in 0 of 9 configs",
         "score the belief on the decision"),
        ("solver matches reference, CSI 0.979", "reference matches satellite at 0.535",
         "score the reference too"),
        ("twin loop improved 8 of 8, t = 3.0", "at 607 fires, a third the size",
         "test at a size that can be wrong"),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 0.62 * len(rows) + 1.25))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 0.62 * len(rows) + 1.25); ax.axis("off")
    y0 = 0.62 * len(rows) + 0.3
    ax.text(1.75, y0 + 0.30, "the component metric said", ha="center",
            fontsize=9.5, fontweight="bold", color=C["good"])
    ax.text(5.55, y0 + 0.30, "the decision in fact", ha="center",
            fontsize=9.5, fontweight="bold", color=C["bad"])
    ax.text(9.55, y0 + 0.30, "what exposed it", ha="center",
            fontsize=9.5, fontweight="bold", color=C["plan"][1])
    for i, (ok, bad, fix) in enumerate(rows):
        y = y0 - 0.62 * i - 0.30
        ax.add_patch(FancyBboxPatch((0.1, y - 0.21), 3.3, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=1.0, edgecolor=C["good"],
                                    facecolor="#e8f3ee", zorder=2))
        ax.add_patch(FancyBboxPatch((3.9, y - 0.21), 3.3, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=1.0, edgecolor=C["bad"],
                                    facecolor="#f7e7e7", zorder=2))
        ax.add_patch(FancyBboxPatch((7.7, y - 0.21), 3.8, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=1.0, edgecolor=C["plan"][1],
                                    facecolor=C["plan"][0], zorder=2))
        ax.text(1.75, y, ok, ha="center", va="center", fontsize=8.1, zorder=3)
        ax.text(5.55, y, bad, ha="center", va="center", fontsize=8.1, zorder=3)
        ax.text(9.6, y, fix, ha="center", va="center", fontsize=8.1, zorder=3)
        arrow(ax, (3.42, y), (3.88, y), color=C["muted"], lw=1.0)
        arrow(ax, (7.22, y), (7.68, y), color=C["muted"], lw=1.0)
    ax.text(5.8, 0.12, "Each row: a component whose own metric certified it. "
                       "The map from component metric to decision quality is not "
                       "monotone, and in three cases it is inverted.",
            ha="center", fontsize=8.6, color=C["muted"], style="italic")
    save(fig, "deck_measurement")


# --------------------------------------------------------------------------- #
# 2. the headline results
# --------------------------------------------------------------------------- #
#: §3.4n, Vista job 1049073, 20 seeds, baselines capped at OUR real-sample count.
#: (label, effect vs do-nothing, ci_lo, ci_hi, violating %)
MATCHED_RDF = [
    ("probe + conformal (ours)", 0.5008, 0.451, 0.556, 5.0),
    ("Sauté RL",                 0.1073, 0.062, 0.156, 0.0),
    ("PPO-Lagrangian",           0.0843, 0.046, 0.126, 0.0),
    ("CPO",                      0.0405, 0.019, 0.064, 0.0),
    ("primal-dual NPG",          0.0405, 0.019, 0.065, 0.0),
]
MATCHED_DAR = [
    ("probe + margin (ours)", 0.0866, 0.077, 0.096, 0.5),
    ("Sauté RL",              0.0066, -0.002, 0.016, 0.0),
    ("CPO",                   0.0052, -0.001, 0.011, 0.0),
    ("primal-dual NPG",       0.0052, -0.001, 0.011, 0.0),
    ("PPO-Lagrangian",        0.0001, -0.009, 0.009, 0.0),
]


def _forest(ax, rows, title, xlabel):
    ys = np.arange(len(rows))[::-1]
    for y, (name, mu, lo, hi, viol) in zip(ys, rows):
        ours = "ours" in name
        col = C["plan"][1] if ours else C["muted"]
        ax.plot([lo, hi], [y, y], color=col, lw=2.6 if ours else 1.8,
                solid_capstyle="round", zorder=2)
        ax.plot([mu], [y], "o", color=col, ms=7 if ours else 5, zorder=3)
        ax.text(hi + 0.012 * (ax.get_xlim()[1] or 1), y, f"{viol:.1f}%",
                va="center", fontsize=7.6, color=C["muted"])
    ax.axvline(0, color=C["bad"], lw=1.1, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    for lbl in ax.get_yticklabels():
        if "ours" in lbl.get_text():
            lbl.set_fontweight("bold")
            lbl.set_color(C["plan"][1])
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_title(title, fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_matched_budget():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.1))
    _forest(axes[0], MATCHED_RDF,
            "rdf — matched at ≈8,300 real transitions each",
            "return gained over a do-nothing policy  (95% bootstrap CI)")
    _forest(axes[1], MATCHED_DAR,
            "dar — matched at ≈4,000 real transitions each",
            "return gained over a do-nothing policy  (95% bootstrap CI)")
    axes[0].set_xlim(-0.03, 0.72)
    axes[1].set_xlim(-0.02, 0.122)
    fig.suptitle("At an equal data budget, the method beats all four published "
                 "baselines; the dashed line is doing nothing",
                 fontsize=10.4, fontweight="bold", y=1.04)
    fig.text(0.5, -0.085, "20 seeds, seed-paired, 20,000-resample bootstrap. "
             "The percentage beside each interval is how often that arm violated "
             "the constraint — the trade we make, stated rather than hidden.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_matched_budget")


#: §3.4l, dar at FULL budget (38,400 transitions for the baselines).
FULL_DAR = [("Sauté RL", 0.0280, 0.0224, 0.0334, 20),
            ("PPO-Lagrangian", 0.0141, -0.0050, 0.0322, 13),
            ("CPO", 0.0116, -0.0012, 0.0238, 14),
            ("primal-dual NPG", 0.0096, -0.0039, 0.0226, 13)]


def fig_null_floor():
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ys = np.arange(len(FULL_DAR))[::-1]
    for y, (name, mu, lo, hi, nseed) in zip(ys, FULL_DAR):
        clears = lo > 0
        col = C["good"] if clears else C["bad"]
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot([mu], [y], "o", color=col, ms=6, zorder=3)
        ax.text(hi + 0.0015, y, f"{nseed}/20 seeds", va="center",
                fontsize=7.8, color=C["muted"])
    ax.axvline(0, color=C["bad"], lw=1.2, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in FULL_DAR], fontsize=8.8)
    ax.set_xlabel("return gained over doing nothing  (95% bootstrap CI)", fontsize=8.8)
    ax.set_title("At full budget on dar, three of four published baselines cannot "
                 "be shown to beat inaction", fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.set_xlim(-0.012, 0.047)
    fig.text(0.5, -0.1, "Three intervals straddle zero. A benchmark on which the "
             "baselines do not beat a null policy cannot rank methods — which is "
             "why the comparison above is run at a matched budget instead.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_null_floor")


#: §4.2a, 20 seeds, Vista job 1033239. Matched-budget arms only.
def fig_mixing_rule():
    arms = ["fixed α", "variance rule", "Eq. 8 (measured bias)"]
    viol_pct = [3.18, 0.45, 0.00]
    seeds = [5, 1, 0]
    samples = [5120, 5120, 8960]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.9))

    cols = [C["bad"], C["warn"], C["good"]]
    axes[0].bar(arms, viol_pct, color=cols, width=0.56, zorder=2)
    for i, (v, s) in enumerate(zip(viol_pct, seeds)):
        axes[0].text(i, v + 0.09, f"{v:.2f}%\n{s}/20 seeds", ha="center",
                     fontsize=8.2, color=C["ink"])
    axes[0].set_ylabel("evaluations violating the limit (%)", fontsize=8.6)
    axes[0].set_ylim(0, 4.0)
    axes[0].set_title("Adaptive α removes the tail", fontsize=9.6,
                      fontweight="bold", loc="left")
    axes[0].grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="x", labelsize=8.2)

    axes[1].bar(arms, samples, color=[C["muted"], C["muted"], C["bad"]],
                width=0.56, zorder=2)
    for i, s in enumerate(samples):
        axes[1].text(i, s + 140, f"{s:,}", ha="center", fontsize=8.2)
    axes[1].set_ylabel("real transitions used", fontsize=8.6)
    axes[1].set_ylim(0, 10500)
    axes[1].set_title("…but read the budget before the result",
                      fontsize=9.6, fontweight="bold", loc="left")
    axes[1].grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(axis="x", labelsize=8.2)

    fig.text(0.5, -0.12,
             "The only matched comparison is variance rule against fixed α: "
             "p = 0.083 at 20 seeds, which does not separate on the mean. "
             "Eq. 8 spends 75% more real data, so its zero is not attributable "
             "to the rule. What holds: α is a VARIANCE effect, not a mean one.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_mixing_rule")


#: §3.2b, 20 seeds, Vista job 1032033.
def fig_margin():
    arms = ["baseline", "margin only", "probe only", "probe + conformal"]
    mean = [14.1, 14.1, 10.9, 5.0]
    lo = [9.1, 9.1, 6.4, 0.9]
    hi = [17.3, 17.3, 15.5, 8.2]
    cols = [C["muted"], C["muted"], C["warn"], C["good"]]
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    x = np.arange(len(arms))
    ax.bar(x, mean, color=cols, width=0.56, zorder=2)
    ax.errorbar(x, mean, yerr=[np.array(mean) - np.array(lo),
                               np.array(hi) - np.array(mean)],
                fmt="none", ecolor=C["ink"], capsize=4, lw=1.1, zorder=3)
    for i, m in enumerate(mean):
        ax.text(i, hi[i] + 0.55, f"{m:.1f}%", ha="center", fontsize=8.6,
                fontweight="bold" if i == 3 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.4)
    ax.set_ylabel("evaluations violating the limit (%)", fontsize=8.6)
    ax.set_ylim(0, 20)
    ax.set_title("Conformal margin on the quantity that actually violates: "
                 "14.1% → 5.0%", fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.text(0.5, -0.08, "20 seeds. Seed-paired effect −9.1 points "
             "[−11.8, −5.5], t = −4.59, p = 0.0002, better on 15 of 20 seeds. "
             "A margin alone does nothing (0/20); the probe is what makes it bite.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_margin")


# --------------------------------------------------------------------------- #
# 3. the flood twin
# --------------------------------------------------------------------------- #
def fig_flood_frontier():
    d = json.loads((DATA / "evac_frontier.json").read_text())
    curve = sorted(d["curve"], key=lambda r: r[2])
    cost = [r[2] for r in curve]
    ret = [r[1] for r in curve]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(cost, ret, "o", color=C["muted"], ms=4, zorder=2,
            label="one threshold each (70 swept, 0.30 to 0.99)")
    for th, r, c in d["pareto"]:
        ax.plot([c], [r], "o", color=C["plan"][1], ms=9, zorder=4)
        ax.annotate(f"threshold {th:.2f}", (c, r), textcoords="offset points",
                    xytext=(10, -4), fontsize=8.2, color=C["plan"][1])
    lo = min(p[2] for p in d["pareto"])
    ax.axvspan(6.51, 15.19, color=C["bad"], alpha=0.07, zorder=1)
    ax.text(10.8, -3200, "between these costs the family\noffers exactly ONE return",
            ha="center", fontsize=8.2, color=C["bad"], style="italic")
    ax.set_xlabel("disruption spent  (people-intervals under evacuation order)",
                  fontsize=8.8)
    ax.set_ylabel("return  (higher is better)", fontsize=8.8)
    ax.set_title("A one-number trigger has three behaviours, not a frontier",
                 fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.legend(fontsize=8.0, frameon=False, loc="lower right")
    fig.text(0.5, -0.06, "Thirty consecutive thresholds return the same number "
             "to one decimal while their cost varies 2.3×. One degree of freedom "
             "on an eight-zone problem.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_flood_frontier")


def fig_collapse():
    v1 = json.loads((DATA / "collapse_v1.json").read_text())
    v2 = json.loads((DATA / "collapse_v2.json").read_text())

    def series(rows, key):
        return ([r["iteration"] for r in rows if r.get(key) is not None],
                [r[key] for r in rows if r.get(key) is not None])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    for ax, key, lab, logy in (
            (axes[0], "train/pre_tanh_mean_abs", "policy pre-tanh magnitude", True),
            (axes[1], "train/episode_cost", "cost spent (pump-hours)", False),
            (axes[2], "lambda", "dual multiplier λ", False)):
        x1, y1 = series(v1, key)
        x2, y2 = series(v2, key)
        if y1:
            ax.plot(x1, y1, color=C["bad"], lw=1.7, label="defaults")
        if y2:
            ax.plot(x2, y2, color=C["good"], lw=1.7, label="guard on")
        if logy:
            ax.set_yscale("log")
            ax.axhline(3.0, color=C["muted"], lw=1.0, ls=":")
            ax.text(76, 3.6, "beyond 3 the squash is flat:\nevery gradient vanishes",
                    fontsize=7.4, color=C["muted"])
        if key == "train/episode_cost":
            ax.axhline(2.0, color=C["ink"], lw=1.0, ls="--")
            ax.text(78, 2.18, "budget", fontsize=7.6, color=C["ink"])
        ax.set_xlabel("training iteration", fontsize=8.5)
        ax.set_ylabel(lab, fontsize=8.5)
        ax.grid(color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8.0, frameon=False, loc="lower right")
    fig.suptitle("The same planner, with one guard off and on: a policy that runs "
                 "to the boundary, against one that stays where gradients exist",
                 fontsize=10.2, fontweight="bold", y=1.06)
    fig.text(0.5, -0.145,
             "Defaults (red): pre-tanh reaches 1,436, cost pins at the ceiling "
             "from iteration 50, and λ winds up linearly\nagainst a quantity that "
             "can no longer respond. Guard on (green): pre-tanh settles near 1.9, "
             "cost comes inside\nthe budget, and λ rises then RELEASES. "
             "One seed each — the eight-seed version is still running.",
             ha="center", fontsize=8.4, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_collapse")


def fig_polder():
    rows = json.loads((DATA / "polder_controllers.json").read_text())
    want = [("do nothing", "fixed"), ("gate only", "fixed"),
            ("level deadband (incumbent)", "fixed"),
            ("pump while sealed", "fixed"), ("pump flat out", "fixed"),
            ("ppo_lagrangian", "rl"), ("cpo", "rl"), ("saute", "rl"),
            ("primal_dual_npg", "rl"), ("hybrid planner", "ours")]
    look = {r["name"]: r for r in rows if "return" in r}
    names, rets, costs, kinds = [], [], [], []
    for n, kind in want:
        if n in look:
            names.append(n.replace("_", "-"))
            rets.append(look[n]["return"])
            costs.append(look[n]["cost"])
            kinds.append(kind)
    budget = 2.0
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    # Hand-placed offsets. "gate only" and the deadband sit 0.44 pump-hours and
    # 1.8 return apart, so the default offset drew one label straight through
    # the other's marker; the two right-hand points ran off the axis.
    off = {"gate only": (8, -12), "level deadband (incumbent)": (8, 6),
           "do nothing": (8, -4), "pump while sealed": (-10, 9),
           "pump flat out": (-16, -14), "hybrid planner": (-14, 9),
           "ppo-lagrangian": (8, -12), "cpo": (8, 5),
           "primal-dual-npg": (8, 5), "saute": (8, -4)}
    ha = {"pump flat out": "right", "hybrid planner": "right",
          "pump while sealed": "right"}
    for r, c, k, n in zip(rets, costs, kinds, names):
        ok = c <= budget + 1e-6
        col = {"fixed": C["muted"], "rl": C["perceive"][1], "ours": C["plan"][1]}[k]
        ax.plot([c], [r], "o" if ok else "X", ms=9 if k == "ours" else 7,
                color=col, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(n, (c, r), textcoords="offset points",
                    xytext=off.get(n, (8, 5)), ha=ha.get(n, "left"),
                    fontsize=7.8, color=col,
                    fontweight="bold" if k == "ours" else "normal")
    ax.set_xlim(-0.45, max(costs) + 0.7)
    ax.axvline(budget, color=C["bad"], lw=1.3, ls="--", zorder=1)
    ax.axvspan(budget, max(costs) + 0.7, color=C["bad"], alpha=0.06, zorder=0)
    # Short, and kept in the lower third, where no controller sits.
    ax.text(budget + 0.11, min(rets) + 1.5, "over budget",
            fontsize=8.0, color=C["bad"], rotation=90, va="bottom")
    ax.set_xlabel("fuel spent (pump-hours)", fontsize=8.8)
    ax.set_ylabel("return (higher is better)", fontsize=8.8)
    ax.set_title("On the flood polder, the best FEASIBLE controller is the "
                 "incumbent deadband", fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.text(0.5, -0.07, "Circles are feasible, crosses are not. Three of four "
             "published safe-RL methods finish outside the budget — two of them "
             "on every evaluation episode — and still score below the deadband.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_polder")


#: Measured from the USGS record by service/app.py::warning, 2026-10-11.
WARNING = [("2013-10-31", 3822.8, None), ("2015-10-30", 3454.7, None),
           ("2019-05-04", 764.6, 285), ("2004-11-17", 535.2, 275),
           ("2015-05-24", 427.6, 335), ("2007-01-13", 404.9, 20)]


def fig_warning():
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ys = np.arange(len(WARNING))[::-1]
    for y, (date, q, lag) in zip(ys, WARNING):
        if lag is None:
            ax.barh(y, 480, color=C["bad"], alpha=0.16, height=0.6, zorder=2)
            ax.text(240, y, "no usable trace — the gauge was silent",
                    ha="center", va="center", fontsize=8.0,
                    color=C["bad"], fontweight="bold", zorder=4)
        else:
            col = C["warn"] if lag < 120 else C["good"]
            ax.barh(y, lag, color=col, height=0.6, zorder=2)
            ax.text(lag + 9, y, f"{lag} min", va="center", fontsize=8.2, color=col)
    ax.axvline(120, color=C["ink"], lw=1.2, ls="--", zorder=3)
    ax.text(126, len(WARNING) - 0.45,
            "120 min: below this the evacuation\nstudy finds no route out for everyone",
            fontsize=7.8, color=C["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d}   {q:,.0f} m³/s" for d, q, _ in WARNING], fontsize=8.3)
    ax.set_xlabel("warning time — lag between the upstream gauge's peak and the "
                  "outlet's (minutes)", fontsize=8.6)
    ax.set_xlim(0, 520)
    ax.set_title("Measured, not modelled: how much notice the gauge actually gave",
                 fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.text(0.5, -0.08, "Median 4 h 45 min — on the four it recorded. The two "
             "largest floods since the gauge was built left no trace at all, and "
             "one of the four it did see gave 20 minutes. A median over the days "
             "the instrument worked describes the warning available during the "
             "floods nobody needed warning for.",
             ha="center", fontsize=8.2, color=C["muted"], style="italic")
    fig.tight_layout()
    save(fig, "deck_warning")


def main() -> int:
    fig_framework()
    fig_measurement()
    fig_matched_budget()
    fig_null_floor()
    fig_mixing_rule()
    fig_margin()
    fig_flood_frontier()
    fig_collapse()
    fig_polder()
    fig_warning()
    print(f"\nall figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
