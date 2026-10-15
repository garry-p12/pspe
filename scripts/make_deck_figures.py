#!/usr/bin/env python
"""Figures for the presentation deck (see docs/DECK_POINTERS.md).

    python scripts/make_deck_figures.py

Writes docs/figures/deck/*.{svg,pdf,png} in the palette the thesis figures
already use, so a slide can sit beside a paper figure without reformatting.

Drawn for a reader who has never seen this repository: testbeds are named by
their physics rather than by their directory, quantities are named in words
rather than in code, and NO figure carries its own caption. A manuscript
figure's caption belongs in the text beside it, so the captions live in
`docs/DECK_POINTERS.md` and the images stay clean enough to drop into a paper
or a slide without cropping anything off.

Two further rules, both learned the hard way in this project:

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
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xlim(0, 11.2); ax.set_ylim(0.2, 5.3); ax.axis("off")

    box(ax, 0.2, 3.5, 2.6, 1.6, "Learned model", [
        "fast, differentiable", "approximation"], "simulate")
    box(ax, 0.2, 1.2, 2.6, 1.6, "Real system", [
        "slow, exact,", "no gradient"], "world")

    box(ax, 3.3, 3.5, 2.5, 1.6, "Gradient A", [
        r"$g_{\mathrm{pw}}$", "low noise", "biased"], "simulate")
    box(ax, 3.3, 1.2, 2.5, 1.6, "Gradient B", [
        r"$g_{\mathrm{lr}}$", "unbiased", "high noise"], "world")

    box(ax, 6.3, 1.55, 2.6, 2.75, "Combine", [
        r"$g = \alpha\, g_{\mathrm{pw}} + (1-\alpha)\, g_{\mathrm{lr}}$", "",
        "α minimises the", "noise in g"], "plan")

    box(ax, 9.2, 3.5, 1.85, 1.6, "Policy", ["update"], "plan")
    box(ax, 9.2, 1.2, 1.85, 1.6, "Budget", ["controller", r"$\lambda$"], "explain")

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
    ax.text(5.8, 0.42, "re-measured on the real system", ha="center",
            fontsize=8.8, color=C["muted"])

    save(fig, "deck_framework")


def fig_measurement():
    """The thesis in one picture: component metrics do not imply decision quality."""
    rows = [
        ("budget satisfied", "7.3% of real runs broke it", "test on the real system"),
        ("forecast accurate", "rollout invents an event", "re-anchor to observation"),
        ("explanations faithful", "carry no real information", "permutation control"),
        ("state estimate better", "decisions no better", "score on the decision"),
        ("matches the reference", "reference misses reality", "score the reference too"),
        ("improved 8 of 8 cases", "a third the size at 607", "test at a larger n"),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 0.62 * len(rows) + 1.25))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 0.62 * len(rows) + 1.25); ax.axis("off")
    y0 = 0.62 * len(rows) + 0.3
    ax.text(1.75, y0 + 0.30, "the metric said", ha="center",
            fontsize=10.5, fontweight="bold", color=C["good"])
    ax.text(5.55, y0 + 0.30, "reality was", ha="center",
            fontsize=10.5, fontweight="bold", color=C["bad"])
    ax.text(9.6, y0 + 0.30, "the test that found it", ha="center",
            fontsize=10.5, fontweight="bold", color=C["plan"][1])
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
        ax.text(1.75, y, ok, ha="center", va="center", fontsize=9.3, zorder=3)
        ax.text(5.55, y, bad, ha="center", va="center", fontsize=9.3, zorder=3)
        ax.text(9.6, y, fix, ha="center", va="center", fontsize=9.3, zorder=3)
        arrow(ax, (3.42, y), (3.88, y), color=C["muted"], lw=1.0)
        arrow(ax, (7.22, y), (7.68, y), color=C["muted"], lw=1.0)

    save(fig, "deck_measurement")


# --------------------------------------------------------------------------- #
# 2. the headline results
# --------------------------------------------------------------------------- #
#: §3.4n, Vista job 1049073, 20 seeds, baselines capped at OUR real-sample count.
#: (label, effect vs do-nothing, ci_lo, ci_hi, violating %)
MATCHED_RDF = [
    ("This work", 0.5008, 0.451, 0.556, 5.0),
    ("Sauté RL",                 0.1073, 0.062, 0.156, 0.0),
    ("PPO-Lagrangian",           0.0843, 0.046, 0.126, 0.0),
    ("CPO",                      0.0405, 0.019, 0.064, 0.0),
    ("primal-dual NPG",          0.0405, 0.019, 0.065, 0.0),
]
MATCHED_DAR = [
    ("This work", 0.0866, 0.077, 0.096, 0.5),
    ("Sauté RL",              0.0066, -0.002, 0.016, 0.0),
    ("CPO",                   0.0052, -0.001, 0.011, 0.0),
    ("primal-dual NPG",       0.0052, -0.001, 0.011, 0.0),
    ("PPO-Lagrangian",        0.0001, -0.009, 0.009, 0.0),
]


def _forest(ax, rows, title, xlabel):
    ys = np.arange(len(rows))[::-1]
    for y, (name, mu, lo, hi, viol) in zip(ys, rows):
        ours = name == "This work"
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
        if lbl.get_text() == "This work":
            lbl.set_fontweight("bold")
            lbl.set_color(C["plan"][1])
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_title(title, fontsize=9.6, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_matched_budget():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.1))
    _forest(axes[0], MATCHED_RDF,
            "(a)  Reaction–diffusion front",
            "improvement over doing nothing")
    _forest(axes[1], MATCHED_DAR,
            "(b)  Diffusion–advection–reaction",
            "improvement over doing nothing")
    axes[0].set_xlim(-0.03, 0.72)
    axes[1].set_xlim(-0.02, 0.122)
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
    ax.set_xlabel("improvement over doing nothing", fontsize=9.2)
    ax.set_title("Diffusion–advection–reaction, full data budget",
                 fontsize=10.0, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.set_xlim(-0.012, 0.047)
    fig.tight_layout()
    save(fig, "deck_null_floor")


#: §4.2a, 20 seeds, Vista job 1033239. Matched-budget arms only.
def fig_mixing_rule():
    arms = ["fixed blend", "variance rule", "measured-bias rule"]
    viol_pct = [3.18, 0.45, 0.00]
    seeds = [5, 1, 0]
    samples = [5120, 5120, 8960]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.9))

    cols = [C["bad"], C["warn"], C["good"]]
    axes[0].bar(arms, viol_pct, color=cols, width=0.56, zorder=2)
    for i, (v, s) in enumerate(zip(viol_pct, seeds)):
        axes[0].text(i, v + 0.09, f"{v:.2f}%\n{s}/20 seeds", ha="center",
                     fontsize=8.2, color=C["ink"])
    axes[0].set_ylabel("runs that broke the safety limit (%)", fontsize=9.0)
    axes[0].set_ylim(0, 4.0)
    axes[0].set_title("(a)  Adapting the blend removes the failures",
                      fontsize=10.0, fontweight="bold", loc="left")
    axes[0].grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="x", labelsize=8.2)

    axes[1].bar(arms, samples, color=[C["muted"], C["muted"], C["bad"]],
                width=0.56, zorder=2)
    for i, s in enumerate(samples):
        axes[1].text(i, s + 140, f"{s:,}", ha="center", fontsize=8.2)
    axes[1].set_ylabel("real-world samples used", fontsize=9.0)
    axes[1].set_ylim(0, 10500)
    axes[1].set_title("(b)  …but one arm paid more for them",
                      fontsize=10.0, fontweight="bold", loc="left")
    axes[1].grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7)
    axes[1].set_axisbelow(True)
    axes[1].tick_params(axis="x", labelsize=8.2)

    fig.tight_layout()
    save(fig, "deck_mixing_rule")


#: §3.2b, 20 seeds, Vista job 1032033.
def fig_margin():
    arms = ["no correction", "margin\nalone", "real-world\ntesting alone", "both"]
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
    ax.set_ylabel("runs that broke the safety limit (%)", fontsize=9.0)
    ax.set_ylim(0, 20)
    ax.set_title("Safety violations, 20 repeats", fontsize=10.0,
                 fontweight="bold", loc="left")
    ax.grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "deck_margin")


# --------------------------------------------------------------------------- #
# 3. the flood twin
# --------------------------------------------------------------------------- #
def fig_flood_frontier():
    """Three flat rows of dots. The picture IS the result, so keep it clear."""
    d = json.loads((DATA / "evac_frontier.json").read_text())
    curve = sorted(d["curve"], key=lambda r: r[2])
    cost = [r[2] for r in curve]
    ret = [r[1] for r in curve]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(cost, ret, "o", color=C["muted"], ms=4.5, zorder=2)

    # Label each row at ITS OWN right-hand end, where nothing is plotted.
    # Centring the label on the Pareto point drew it straight through the row.
    levels = sorted({round(r, 1) for r in ret}, reverse=True)
    for i, lv in enumerate(levels):
        xs = [c for c, r in zip(cost, ret) if round(r, 1) == lv]
        ax.plot([min(xs)], [lv], "o", color=C["plan"][1], ms=10, zorder=4)
        ax.annotate(f"behaviour {len(levels) - i}", (max(xs), lv),
                    textcoords="offset points", xytext=(12, -3),
                    fontsize=9.6, color=C["plan"][1], fontweight="bold")

    lo, hi = 6.51, 15.19
    span = max(ret) - min(ret)
    floor = min(ret) - 0.20 * span
    ax.set_ylim(floor - 0.06 * span, max(ret) + 0.12 * span)
    ax.set_xlim(min(cost) - 2.5, max(cost) + 9.5)
    ax.axvspan(lo, hi, color=C["bad"], alpha=0.07, zorder=1)
    ax.annotate("", xy=(lo, floor), xytext=(hi, floor),
                arrowprops=dict(arrowstyle="<->", color=C["bad"], lw=1.4))
    ax.text((lo + hi) / 2, floor + 0.035 * span,
            "cost varies 2.3× across this band,\noutcome does not change at all",
            ha="center", fontsize=9.0, color=C["bad"], fontweight="bold")

    ax.text(0.985, 0.045, "each dot is one setting of the rule",
            transform=ax.transAxes, ha="right", fontsize=8.8, color=C["muted"])
    ax.set_xlabel("disruption caused by evacuating early", fontsize=9.2)
    ax.set_ylabel("outcome  (higher is better)", fontsize=9.2)
    ax.set_title("A single-threshold rule has only three behaviours",
                 fontsize=10.0, fontweight="bold", loc="left")
    ax.grid(color=C["rule"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
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
            (axes[0], "train/pre_tanh_mean_abs", "how hard the policy pushes", True),
            (axes[1], "train/episode_cost", "energy used", False),
            (axes[2], "lambda", "safety penalty applied", False)):
        x1, y1 = series(v1, key)
        x2, y2 = series(v2, key)
        if y1:
            ax.plot(x1, y1, color=C["bad"], lw=1.9, label="guard off")
        if y2:
            ax.plot(x2, y2, color=C["good"], lw=1.9, label="guard on")
        if logy:
            ax.set_yscale("log")
            ax.axhline(3.0, color=C["muted"], lw=1.0, ls=":")
            ax.text(60, 4.4, "above this line the policy\nstops responding at all",
                    fontsize=8.4, color=C["muted"])
        if key == "train/episode_cost":
            ax.axhline(2.0, color=C["ink"], lw=1.0, ls="--")
            ax.text(78, 2.2, "budget", fontsize=8.6, color=C["ink"])
        ax.set_xlabel("training progress", fontsize=9.0)
        ax.set_ylabel(lab, fontsize=9.0)
        ax.grid(color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    axes[0].legend(fontsize=9.0, frameon=False, loc="lower right")
    for ax, t in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_title(t, fontsize=10.0, fontweight="bold", loc="left")
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
    ax.set_xlabel("energy used", fontsize=9.2)
    ax.set_ylabel("outcome  (higher is better)", fontsize=9.2)
    ax.set_title("Flood gate and pump control, against an energy budget",
                 fontsize=10.0, fontweight="bold", loc="left")
    ax.grid(color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
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
            ax.text(240, y, "gauge recorded nothing", ha="center", va="center",
                    fontsize=9.2, color=C["bad"], fontweight="bold", zorder=4)
        else:
            col = C["warn"] if lag < 120 else C["good"]
            ax.barh(y, lag, color=col, height=0.6, zorder=2)
            ax.text(lag + 9, y, f"{lag} min", va="center", fontsize=8.2, color=col)
    ax.axvline(120, color=C["ink"], lw=1.2, ls="--", zorder=3)
    # Below the shortest bar, where nothing is drawn. At the top of the axes
    # this label printed straight through the chart title.
    ax.text(128, -0.78, "minimum needed to evacuate", fontsize=9.0,
            color=C["ink"], fontweight="bold", va="center")
    ax.set_ylim(-1.3, len(WARNING) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d}   {q:,.0f} m³/s" for d, q, _ in WARNING], fontsize=8.3)
    ax.set_xlabel("warning time available  (minutes)", fontsize=9.2)
    ax.set_xlim(0, 520)
    ax.set_title("How much notice the river gauge actually gave, by flood",
                 fontsize=10.0, fontweight="bold", loc="left")
    ax.grid(axis="x", color=C["rule"], lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "deck_warning")



# --------------------------------------------------------------------------- #
# 4. what the framework offers, and how the twin and the gym work
# --------------------------------------------------------------------------- #
def fig_offers():
    """What the method buys, in three panels, before any numbers."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))
    for ax in axes:
        ax.axis("off")

    # (a) learns from less real data
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((0.6, 6.2), 8.8, 1.5,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=C["world"][0], edgecolor=C["muted"], lw=1.2))
    ax.text(5.0, 6.95, "standard methods", ha="center", va="center", fontsize=9.6)
    ax.add_patch(FancyBboxPatch((0.6, 3.9), 1.9, 1.5,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=C["plan"][0], edgecolor=C["plan"][1], lw=1.6))
    ax.text(1.55, 4.65, "ours", ha="center", va="center", fontsize=9.6,
            fontweight="bold", color=C["plan"][1])
    ax.annotate("", xy=(2.5, 3.15), xytext=(9.4, 3.15),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.3))
    ax.text(5.9, 2.65, "4.6× less real-world data\nfor the same result",
            ha="center", va="top", fontsize=9.6, fontweight="bold")
    ax.set_title("(a)  Learns from less", fontsize=10.4, fontweight="bold", loc="left")

    # (b) a stated failure rate
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    for i in range(20):
        x, y = 1.0 + (i % 5) * 1.75, 7.3 - (i // 5) * 1.45
        bad = i == 0
        ax.add_patch(FancyBboxPatch((x - 0.52, y - 0.45), 1.05, 0.9,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=C["bad"] if bad else "#e8f3ee",
                                    edgecolor=C["bad"] if bad else C["good"], lw=1.2))
    ax.text(5.0, 2.05, "we say in advance how often\nthe limit will be crossed",
            ha="center", va="top", fontsize=9.6, fontweight="bold")
    ax.set_title("(b)  States its own failure rate", fontsize=10.4,
                 fontweight="bold", loc="left")

    # (c) refuses
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((0.7, 4.3), 8.6, 3.4,
                                boxstyle="round,pad=0.06,rounding_size=0.25",
                                facecolor=C["explain"][0], edgecolor=C["explain"][1],
                                lw=1.6))
    ax.text(5.0, 6.6, "“No usable answer here.”", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=C["explain"][1])
    ax.text(5.0, 5.2, "the gauge was silent;\nthe plan moves risk rather than removing it",
            ha="center", va="center", fontsize=8.8, color=C["ink"])
    ax.text(5.0, 3.3, "a refusal is a result,\nnot a missing feature",
            ha="center", va="top", fontsize=9.6, fontweight="bold")
    ax.set_title("(c)  Refuses when it cannot answer", fontsize=10.4,
                 fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, "deck_offers")


def fig_twin_pipeline():
    """How the flood twin actually works, left to right, no jargon."""
    fig, ax = plt.subplots(figsize=(11.4, 3.5))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.5); ax.axis("off")

    ins = [("Ground elevation", "laser survey,\nbare earth"),
           ("Rainfall", "storm, or\nlive radar"),
           ("River level", "live gauges")]
    for i, (t, sub) in enumerate(ins):
        y = 2.45 - i * 0.92
        ax.add_patch(FancyBboxPatch((0.15, y - 0.33), 2.0, 0.76,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=C["perceive"][0],
                                    edgecolor=C["perceive"][1], lw=1.3))
        ax.text(1.15, y + 0.12, t, ha="center", fontsize=9.2, fontweight="bold",
                color=C["perceive"][1])
        ax.text(1.15, y - 0.19, sub, ha="center", fontsize=7.8, color=C["ink"])
        arrow(ax, (2.2, y), (3.0, 1.75), color=C["muted"], lw=1.1)

    ax.add_patch(FancyBboxPatch((3.05, 1.05), 2.3, 1.45,
                                boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=C["simulate"][0],
                                edgecolor=C["simulate"][1], lw=1.7))
    ax.text(4.2, 2.18, "Water model", ha="center", fontsize=10.2,
            fontweight="bold", color=C["simulate"][1])
    ax.text(4.2, 1.72, "shallow-water physics\non a 40 m grid", ha="center",
            fontsize=8.4)
    ax.text(4.2, 1.28, "18 hours in ~2 min", ha="center", fontsize=8.4,
            color=C["muted"])
    arrow(ax, (5.4, 1.75), (6.1, 1.75), lw=1.6)

    ax.add_patch(FancyBboxPatch((6.15, 1.2), 1.6, 1.1,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=C["world"][0], edgecolor=C["world"][1], lw=1.3))
    ax.text(6.95, 1.75, "Depth of water\neverywhere, hour\nby hour", ha="center",
            va="center", fontsize=8.4)

    outs = [("Which roads close", C["bad"]), ("Whose floor floods", C["bad"]),
            ("Where to build", C["plan"][1]), ("When to evacuate", C["plan"][1])]
    for i, (t, col) in enumerate(outs):
        y = 2.72 - i * 0.66
        ax.add_patch(FancyBboxPatch((8.3, y - 0.26), 2.95, 0.56,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor="white", edgecolor=col, lw=1.4))
        ax.text(9.78, y + 0.01, t, ha="center", va="center", fontsize=9.2,
                color=col, fontweight="bold")
        arrow(ax, (7.8, 1.75), (8.25, y), color=C["muted"], lw=1.0)

    # Under everything, not through it. Routed at box height this line ran
    # straight across the "River level" label.
    ax.plot([6.95, 6.95, 1.15, 1.15], [1.15, 0.42, 0.42, 1.34],
            color=C["muted"], lw=1.2, ls="--", zorder=0)
    arrow(ax, (1.15, 0.95), (1.15, 1.34), color=C["muted"], lw=1.2, dashed=True)
    ax.text(4.05, 0.52, "the live gauge corrects the model as the storm runs",
            ha="center", fontsize=8.8, color=C["muted"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    fig.tight_layout()
    save(fig, "deck_twin_pipeline")


def fig_gym_loop():
    """What turns a simulator into a decision problem."""
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.9); ax.axis("off")

    ax.add_patch(FancyBboxPatch((3.5, 1.85), 3.6, 1.5,
                                boxstyle="round,pad=0.05,rounding_size=0.16",
                                facecolor=C["simulate"][0],
                                edgecolor=C["simulate"][1], lw=1.8))
    ax.text(5.3, 2.95, "The flood twin", ha="center", fontsize=11.0,
            fontweight="bold", color=C["simulate"][1])
    ax.text(5.3, 2.45, "real terrain, real roads,\nreal river", ha="center",
            fontsize=8.8)

    ax.add_patch(FancyBboxPatch((0.25, 1.95), 2.55, 1.3,
                                boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=C["plan"][0], edgecolor=C["plan"][1], lw=1.6))
    ax.text(1.52, 2.88, "Controller", ha="center", fontsize=10.4,
            fontweight="bold", color=C["plan"][1])
    ax.text(1.52, 2.38, "learned, or a\nsimple rule", ha="center", fontsize=8.8)

    # Labels sit ABOVE their own arrow in the gap, not anchored to a box edge,
    # where they were drawn over the frames on both sides.
    arrow(ax, (3.45, 3.02), (2.80, 3.02), color=C["muted"], lw=1.4)
    ax.text(3.12, 3.14, "what it sees", ha="center", fontsize=8.8,
            color=C["muted"])
    arrow(ax, (2.80, 2.18), (3.45, 2.18), color=C["muted"], lw=1.4)
    ax.text(3.12, 2.30, "what it does", ha="center", fontsize=8.8,
            color=C["muted"])

    ax.add_patch(FancyBboxPatch((7.75, 2.5), 2.6, 0.75,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="#e8f3ee", edgecolor=C["good"], lw=1.4))
    ax.text(9.05, 2.87, "Outcome: less flooding", ha="center", va="center",
            fontsize=9.0, color=C["good"], fontweight="bold")
    ax.add_patch(FancyBboxPatch((7.75, 1.6), 2.6, 0.75,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="#f7e7e7", edgecolor=C["bad"], lw=1.4))
    ax.text(9.05, 1.97, "Cost: fuel, or disruption", ha="center", va="center",
            fontsize=9.0, color=C["bad"], fontweight="bold")
    arrow(ax, (7.1, 2.75), (7.7, 2.87), color=C["good"], lw=1.3)
    arrow(ax, (7.1, 2.3), (7.7, 1.97), color=C["bad"], lw=1.3)

    for x, title, sub in ((1.55, "Task 1 — Operate",
                           "when to open the gate,\nwhen to run the pump"),
                          (6.4, "Task 2 — Evacuate",
                           "which neighbourhoods to warn,\nand when")):
        ax.add_patch(FancyBboxPatch((x - 1.3, 0.12), 3.3, 1.0,
                                    boxstyle="round,pad=0.04,rounding_size=0.14",
                                    facecolor=C["explain"][0],
                                    edgecolor=C["explain"][1], lw=1.4))
        ax.text(x + 0.35, 0.86, title, ha="center", fontsize=9.6,
                fontweight="bold", color=C["explain"][1])
        ax.text(x + 0.35, 0.45, sub, ha="center", fontsize=8.4)
    ax.text(5.3, 1.42, "two decisions the twin is asked to make",
            ha="center", fontsize=8.8, color=C["muted"])
    fig.tight_layout()
    save(fig, "deck_gym_loop")


def fig_dem():
    """One number, two elevation datasets: the same storm, 54% more flood."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    labels = ["Surface model\n(default)", "Bare-earth model\n(correct)"]
    vals = [28.84, 18.71]
    bars = ax.bar(labels, vals, color=[C["bad"], C["good"]], width=0.5, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f} km²",
                ha="center", fontsize=11.0, fontweight="bold")
    ax.annotate("", xy=(0, 26.5), xytext=(1, 26.5),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.3))
    ax.text(0.5, 27.3, "54% more flood,\nfrom the elevation data alone",
            ha="center", fontsize=9.4, fontweight="bold")
    ax.set_ylabel("area flooded by the same storm", fontsize=9.2)
    ax.set_ylim(0, 33)
    ax.set_title("Trees and rooftops are not ground", fontsize=10.4,
                 fontweight="bold", loc="left")
    ax.grid(axis="y", color=C["rule"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.2)
    fig.tight_layout()
    save(fig, "deck_dem")



def fig_system():
    """The architecture, for a slide rather than for a document.

    `pspe_system` in make_thesis_figures.py places 56 pieces of text: an
    equation, the implementation, and the measured score inside every stage
    box. That is right for a figure a reader can stop at, and wrong for one
    that is on screen for ninety seconds -- a viewer reads the biggest thing,
    then one more, then looks back at the speaker.

    So this version keeps the SHAPE and drops the content: four stages, three
    measurement paths that return, one that does not. Two numbers in the whole
    figure, because the figure's own claim is that the loop closes, and every
    other number has a slide of its own.
    """
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.set_xlim(-0.05, 11.6); ax.set_ylim(0.02, 5.35); ax.axis("off")

    # -- the world, along the top ------------------------------------------ #
    ax.add_patch(FancyBboxPatch((0.25, 4.55), 11.1, 0.62,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C["world"][0], ec=C["world"][1], lw=1.4))
    ax.text(0.55, 4.86, "THE REAL WORLD", fontsize=11, fontweight="bold",
            va="center", color=C["world"][1])
    ax.text(3.15, 4.86, "satellites, river gauges, road networks",
            fontsize=9.6, va="center", color=C["ink"])

    # -- the four stages --------------------------------------------------- #
    stages = [("PERCEIVE", "what was seen,\nand what was not", "perceive"),
              ("SIMULATE", "push it\nforward", "simulate"),
              ("PLAN", "choose, under\na budget", "plan"),
              ("EXPLAIN", "say why, with\na guarantee", "explain")]
    x = 0.45
    for i, (title, body, key) in enumerate(stages):
        fill, edge = C[key]
        ax.add_patch(FancyBboxPatch((x, 2.75), 2.35, 1.35,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=1.7))
        ax.text(x + 1.175, 3.86, title, ha="center", fontsize=11.5,
                fontweight="bold", color=edge)
        ax.text(x + 1.175, 3.25, body, ha="center", va="center",
                fontsize=9.6, color=C["ink"])
        if i < 3:
            arrow(ax, (x + 2.35, 3.42), (x + 2.78, 3.42), lw=1.6)
        x += 2.78

    arrow(ax, (1.0, 4.55), (1.0, 4.12), color=C["world"][1], lw=1.5)
    ax.text(1.15, 4.33, "observe", fontsize=9.0, color=C["world"][1])
    arrow(ax, (9.3, 4.12), (9.3, 4.55), color=C["bad"], lw=1.5)
    ax.text(9.45, 4.33, "act", fontsize=9.0, color=C["bad"])

    # -- the paths that carry measurement BACK ----------------------------- #
    backs = [(0.45, "Re-anchor", "replace the guess with\nwhat was actually seen",
              "8× better forecast"),
             (3.23, "Re-learn", "one step on the model\nfrom each observed day", ""),
             (6.01, "Re-test", "run the policy on the\nreal system, not the model",
              "carries the guarantee")]
    for bx, title, body, gain in backs:
        ax.add_patch(FancyBboxPatch((bx, 0.75), 2.35, 1.35,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc="#e8f3ee", ec=C["good"], lw=1.6))
        ax.text(bx + 1.175, 1.86, title, ha="center", fontsize=10.8,
                fontweight="bold", color=C["good"])
        ax.text(bx + 1.175, 1.33, body, ha="center", va="center",
                fontsize=9.0, color=C["ink"])
        if gain:
            ax.text(bx + 1.175, 0.93, gain, ha="center", fontsize=9.2,
                    fontweight="bold", color=C["good"])
        ax.annotate("", xy=(bx + 1.6, 2.72), xytext=(bx + 1.175, 2.12),
                    arrowprops=dict(arrowstyle="-|>", color=C["good"], lw=1.6,
                                    connectionstyle="arc3,rad=0.22"))
    # One spine from the world into all three paths, with a head on it. Drawn
    # as a bare line ending at a box edge it read as a stray rule rather than
    # as "this is where the measurement comes from".
    ax.plot([0.12, 0.12], [4.55, 0.50], color=C["good"], lw=1.5, zorder=0)
    ax.plot([0.12, 8.36], [0.50, 0.50], color=C["good"], lw=1.5, zorder=0)
    for bx in (1.625, 4.405, 7.185):
        arrow(ax, (bx, 0.50), (bx, 0.72), color=C["good"], lw=1.5)


    # -- and the one that does not ----------------------------------------- #
    ax.add_patch(FancyBboxPatch((8.95, 0.75), 2.4, 1.35,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#fbeeee", ec=C["bad"], lw=1.6, ls="--"))
    ax.text(10.15, 1.86, "What does not close", ha="center", fontsize=10.8,
            fontweight="bold", color=C["bad"])
    ax.text(10.15, 1.25, "nobody recorded a fire\nwhere a break was cut\n"
                         "because we said so",
            ha="center", va="center", fontsize=9.0, color=C["ink"])

    ax.text(0.12, 0.14, "Green: the measurement comes back, and we check against it.    "
                        "Red dashes: it cannot, and we say so.",
            fontsize=9.2, color=C["muted"])
    fig.tight_layout(pad=0.2)
    save(fig, "deck_system")


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
    fig_offers()
    fig_twin_pipeline()
    fig_gym_loop()
    fig_dem()
    fig_system()
    print(f"\nall figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
