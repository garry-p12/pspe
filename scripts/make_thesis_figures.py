#!/usr/bin/env python
"""Figures for the revised thesis (see docs/THESIS_AND_RESULTS.md).

    python scripts/make_thesis_figures.py

The original Fig. 1 drew joint training as one of the two distinguishing
feedback paths. That claim did not survive (t = +0.66, +1.00, identical on
three PDE families), so this file replaces the whole set rather than patching
it. What distinguishes the system now is measurement: three paths that carry
real observations back into the loop, and one path that does not and cannot.

Writes docs/figures/*.{svg,pdf,png}:

    pspe_system            combined architecture, validated paths marked
    pspe_twin_loop         the three ways to carry state, with measured skill
    pspe_safety            reality probe and conformal margin, measured rates
    pspe_silent_failure    the thesis: component metric against decision
    pspe_mod_perceive      per module panels, each with what failed on it
    pspe_mod_simulate
    pspe_mod_plan
    pspe_mod_explain
    pspe_wildfire          budget Pareto and horizon ablation, measured

Every number annotated here is measured and traceable to a file named in the
appendix of the thesis document. Diagram prose uses words rather than dashes,
arrows or bullet dots; mathematics is set as equations.

Boxes size themselves from their contents (see `block_height`), so editing the
text cannot silently push a line outside its frame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

C = {
    "perceive": ("#dbe9f6", "#2b5d8c"),
    "simulate": ("#d7efe6", "#1f6b4f"),
    "plan":     ("#fbe6d0", "#b0561a"),
    "explain":  ("#e8dff3", "#5b3f85"),
    "world":    ("#eef1f4", "#37474f"),
    "good":     "#1f6b4f",
    "bad":      "#a12a2a",
    "ink":      "#222222",
    "muted":    "#666666",
    "rule":     "#c8ccd0",
}

FONT = dict(family="DejaVu Sans")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["svg.fonttype"] = "none"

BAR = 0.40          # coloured title bar
TOP = 0.30          # gap under the bar
BOT = 0.26          # gap above the lower edge
GAP = 0.20          # a blank entry


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def block_height(lines, lead, *, bar=True):
    """Exact height a titled box needs for `lines`, so nothing can overflow."""
    body = sum(GAP if not ln.strip("~+!") else lead for ln in lines)
    return (BAR if bar else 0) + TOP + body + BOT - (lead - 0.16)


def styled(line, body):
    """Prefix characters select a voice: ~ aside, + held up, ! did not."""
    if line[:1] == "~":
        return line[1:], dict(color=C["muted"], fontsize=body - 0.7, style="italic")
    if line[:1] == "+":
        return line[1:], dict(color=C["good"], fontsize=body - 0.2)
    if line[:1] == "!":
        return line[1:], dict(color=C["bad"], fontsize=body - 0.2)
    return line, dict(color=C["ink"], fontsize=body)


def box(ax, x, y, w, lines, fill, edge, title, *, body=9.3, lead=0.42, dashed=False,
        title_size=11.5, h=None):
    """Rounded box with a title bar, sized to its contents. Returns its height."""
    h = h if h is not None else block_height(lines, lead)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
                                fc=fill, ec=edge, lw=1.4, ls="--" if dashed else "-"))
    ax.add_patch(FancyBboxPatch((x, y + h - BAR), w, BAR,
                                boxstyle="round,pad=0,rounding_size=0.10", fc=edge, ec=edge, lw=0))
    ax.text(x + w / 2, y + h - BAR / 2, title, ha="center", va="center", color="white",
            fontsize=title_size, weight="bold", **FONT)
    ty = y + h - BAR - TOP
    for line in lines:
        if not line.strip("~+!"):
            ty -= GAP
            continue
        txt, st = styled(line, body)
        ax.text(x + 0.18, ty, txt, ha="left", va="center", **st, **FONT)
        ty -= lead
    return h


def chip(ax, x, y, text, edge, *, size=8.0, fill="white", weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=edge, weight=weight,
            **FONT, bbox=dict(boxstyle="round,pad=0.30", fc=fill, ec=edge, lw=1.0))


def arrow(ax, p, q, *, color=None, dashed=False, rad=0.0, lw=1.6, scale=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale, lw=lw,
                                 color=color or C["ink"], ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))


def elbow(ax, pts, *, color, lw=1.8, dashed=False, head=True):
    """Polyline with an optional head on the final segment, for routed paths."""
    for a, b in zip(pts[:-1], pts[1:-1] if head else pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                ls="--" if dashed else "-", solid_capstyle="round")
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=dashed)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=220 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/figures/{name}.{{svg,pdf,png}}")


# --------------------------------------------------------------------------- #
# 1. combined architecture
# --------------------------------------------------------------------------- #
def fig_system():
    W, Hc = 17.8, 10.6
    fig, ax = canvas(W, Hc)

    # -- the real system, along the top ------------------------------------ #
    ax.add_patch(FancyBboxPatch((0.45, 8.85), 16.9, 1.35,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc=C["world"][0], ec=C["world"][1], lw=1.5))
    ax.text(0.85, 9.82, "THE REAL SYSTEM", fontsize=12, weight="bold",
            color=C["world"][1], va="center", **FONT)
    ax.text(0.85, 9.30, "VIIRS active fire detections over 8 wildfires and 14 to 20 consecutive "
                        "days.  18k observed fire days with 11 remote sensing drivers.  "
                        "Gridded meteorology.",
            fontsize=9.4, color=C["ink"], va="center", **FONT)

    # -- the four modules -------------------------------------------------- #
    Y = 5.15
    xs = [1.45, 5.25, 9.15, 13.25]
    ws = [3.35, 3.45, 3.65, 3.55]
    mods = [
        ("perceive", "PERCEIVE", [
            r"$\hat u_t = m_t\,y_t + (1-m_t)\,\Phi$",
            "frozen backbone, LoRA, decoder",
            "fills what was not observed", "",
            "+60 percent hidden costs only",
            "+6.6 of 41 decision points",
            "!imputation lost in 0 of 9 runs"]),
        ("simulate", "SIMULATE", [
            r"$u_{t+1} = G_\theta(u_t, a_t)$",
            "U-Net fronts, FNO smooth fields",
            "learned gate, rollout and physics", "",
            "+0.3162 on the NDWS test split,",
            "+111 percent of the published",
            "~gate declines the spectral path"]),
        ("plan", "PLAN", [
            r"$\min_b \mathbb{E}\,[\,\mathrm{loss}\,]$  s.t.  $\bar b_t \leq B$",
            "projected gradient, decision time",
            "PID dual, probe, conformal margin", "",
            "+12.0 points over the operational",
            "+heuristic, paired t of 11.15",
            "+violations 7.3 percent down to 0"]),
        ("explain", "EXPLAIN", [
            r"$F(b) = \exp(-\mathrm{KL}/K)$",
            "brief parsed to a distribution",
            "split conformal certificate", "",
            "+coverage holds at 90 and 95",
            "+percent on 3 of 3 seeds",
            "!faithfulness claim withdrawn"]),
    ]
    H = max(block_height(m[2], 0.40) for m in mods)
    for (key, title, lines), x, w in zip(mods, xs, ws):
        box(ax, x, Y, w, lines, *C[key], title, body=9.0, lead=0.40, h=H)
    for i in range(3):
        arrow(ax, (xs[i] + ws[i], Y + H / 2), (xs[i + 1], Y + H / 2), lw=1.9)

    ax.text(2.55, 8.55, "observe", fontsize=9.6, color=C["world"][1], ha="left",
            va="center", **FONT)
    arrow(ax, (2.35, 8.85), (2.35, Y + H), color=C["world"][1], lw=1.9)

    # -- the three paths that carry real measurement back ------------------ #
    # The cards sit on a trunk drawn from the real system, so their provenance
    # reads without a legend: everything on this bus is a real measurement.
    cards = [
        ("state sync", "replace the belief with what\nthe satellite actually saw",
         "8.4 times better at one day\n8 of 8 fires, t of 15.9", (6.05, Y)),
        ("model adaptation", "one gradient step on the\nmodel from each observed day",
         "raises day 2 by 21 percent\nand day 3 by 30 percent", (7.75, Y)),
        ("reality probe", "roll the policy in the true\nsystem, feed that cost to the dual",
         "conformal margin holds the\nstated rate at delta of 0.1", (10.60, Y)),
    ]
    cw, cy, ch = 3.30, 1.55, 2.05
    for i, (title, mech, result, target) in enumerate(cards):
        cx = 0.95 + i * 3.60
        ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    fc="#eef7f2", ec=C["good"], lw=1.4))
        ax.text(cx + 0.26, cy + 1.72, title.upper(), fontsize=10.0, weight="bold",
                color=C["good"], va="center", **FONT)
        ax.text(cx + 0.26, cy + 1.10, mech, fontsize=8.7, color=C["ink"], va="center",
                linespacing=1.55, **FONT)
        ax.text(cx + 0.26, cy + 0.40, result, fontsize=8.7, color=C["good"], weight="bold",
                va="center", linespacing=1.55, **FONT)
        arrow(ax, (cx + cw / 2, cy + ch), target, color=C["good"], lw=1.9, rad=0.12)

    elbow(ax, [(0.45, 8.85), (0.45, cy + 1.00), (0.95, cy + 1.00)],
          color=C["good"], lw=1.6, head=False)
    for i in range(2):
        x = 0.95 + i * 3.60 + cw
        ax.plot([x, x + 0.30], [cy + 1.00, cy + 1.00], color=C["good"], lw=1.6)

    ax.text(0.95, 1.12, "PATHS THAT CARRY REAL MEASUREMENT BACK INTO THE LOOP",
            fontsize=10.2, weight="bold", color=C["good"], va="center", **FONT)

    # -- the path that does not close -------------------------------------- #
    arrow(ax, (12.20, Y + H), (12.20, 8.85), color=C["bad"], lw=2.0, dashed=True)
    ax.text(12.38, 8.58, "intervene", fontsize=9.6, color=C["bad"], ha="left",
            va="center", **FONT)
    ax.add_patch(FancyBboxPatch((12.00, cy), 5.35, ch,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc="#fdf0f0", ec=C["bad"], lw=1.4, ls="--"))
    ax.text(12.26, cy + 1.72, "THE PATH THAT DOES NOT CLOSE", fontsize=10.0, weight="bold",
            color=C["bad"], va="center", **FONT)
    ax.text(12.26, cy + 0.78, "No observational record contains a fire where a firebreak\n"
                              "was cut on our instruction. Intervention effects rest on a\n"
                              "stated action model, and no change of hazard or dataset\n"
                              "repairs that.",
            fontsize=8.7, color=C["ink"], va="center", linespacing=1.6, **FONT)

    ax.text(0.45, 0.45, "Green paths are validated against real observations. The dashed red path "
                        "is model based. That boundary is the honest scope of the twin claim.",
            fontsize=9.0, color=C["muted"], ha="left", va="center", style="italic", **FONT)
    save(fig, "pspe_system")


# --------------------------------------------------------------------------- #
# 2. the twin loop
# --------------------------------------------------------------------------- #
def fig_twin_loop():
    fig, ax = canvas(15.8, 9.2)

    ax.text(0.45, 8.85, "Three ways to carry a state across real satellite days",
            fontsize=13.5, weight="bold", color=C["ink"], va="center", **FONT)
    ax.text(0.45, 8.42, "Eight large wildfires, VIIRS detections, leave one fire out, 3 seeds. "
                        "Scored as average precision against the detections actually recorded.",
            fontsize=9.4, color=C["muted"], va="center", **FONT)

    x0, dx, n = 4.75, 1.78, 6
    cols = [x0 + dx * i for i in range(n)]
    observed = [True, True, False, True, True, True]          # day 2 has no overpass

    # -- what the satellite saw -------------------------------------------- #
    ax.add_patch(FancyBboxPatch((0.45, 6.85), 14.9, 1.16,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                fc=C["world"][0], ec=C["world"][1], lw=1.3))
    ax.text(0.75, 7.43, "what the\nsatellite saw", fontsize=9.6, weight="bold",
            color=C["world"][1], va="center", linespacing=1.5, **FONT)
    truth = [0.085, 0.115, 0.145, 0.175, 0.205, 0.235]
    for cx, r, obs in zip(cols, truth, observed):
        if obs:
            ax.add_patch(Rectangle((cx - 0.30, 7.13), 0.60, 0.60, fc="white",
                                   ec=C["world"][1], lw=1.1))
            ax.add_patch(Circle((cx, 7.43), r, fc=C["world"][1], ec="none", alpha=0.75))
        else:
            ax.add_patch(Rectangle((cx - 0.30, 7.13), 0.60, 0.60, fc="#f2f2f2",
                                   ec=C["muted"], lw=1.0, ls=":"))
            ax.text(cx, 7.43, "no\noverpass", fontsize=6.6, color=C["muted"], ha="center",
                    va="center", style="italic", linespacing=1.3, **FONT)
    for i, cx in enumerate(cols):
        ax.text(cx, 6.60, f"day {i}", fontsize=8.6, color=C["muted"], ha="center",
                va="center", **FONT)

    # -- the three tracks --------------------------------------------------- #
    tracks = [
        ("open loop", "#9aa0a6",
         "forecast from day 0 and\nnever look again",
         r"$\tilde u_{t+1} = G_\theta(\hat u_t, a_t)$",
         [0.085, 0.140, 0.190, 0.225, 0.250, 0.265], [False] * 6),
        ("state sync", "#2b5d8c",
         "replace the belief with\nwhat was observed",
         r"$\hat u_{t+1} = m\,y_{t+1} + (1-m)\,\Phi$",
         [0.085, 0.115, 0.185, 0.175, 0.205, 0.235], observed),
        ("state and model", C["good"],
         "also take one gradient\nstep on the model itself",
         r"$\theta \leftarrow \theta - \eta \nabla_\theta\, \ell(G_\theta,\, y_{t+1})$",
         [0.085, 0.115, 0.168, 0.175, 0.205, 0.235], observed),
    ]
    ys = [5.55, 3.95, 2.35]
    for (name, col, sub, eq, radii, obs), y in zip(tracks, ys):
        ax.add_patch(FancyBboxPatch((0.45, y - 0.62), 3.85, 1.30,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    fc="white", ec=col, lw=1.4))
        ax.text(0.72, y + 0.44, name, fontsize=10.6, weight="bold", color=col, va="center", **FONT)
        ax.text(0.72, y - 0.02, sub, fontsize=8.5, color=C["ink"], va="center",
                linespacing=1.5, **FONT)
        ax.text(0.72, y - 0.48, eq, fontsize=8.6, color=col, va="center", **FONT)

        for i, (cx, r) in enumerate(zip(cols, radii)):
            ax.add_patch(Rectangle((cx - 0.30, y - 0.30), 0.60, 0.60, fc="#fafbfc",
                                   ec=col, lw=1.1))
            ax.add_patch(Circle((cx, y), r, fc=col, ec="none", alpha=0.6))
            if i < n - 1:
                arrow(ax, (cx + 0.30, y), (cx + dx - 0.30, y), color=col, lw=1.3, scale=11)
            if i > 0 and obs[i]:
                # Every correction comes from the observation strip itself. The two
                # corrected tracks ride opposite sides of the column so neither
                # riser crosses a cell belonging to the track above it.
                rx = cx + (0.52 if col == "#2b5d8c" else -0.52)
                arrow(ax, (rx, 7.13), (rx, y + 0.18), color=col, lw=1.1, scale=9)
                ax.plot([rx, cx + (0.30 if rx > cx else -0.30)], [y + 0.18, y + 0.18],
                        color=col, lw=1.1)

    ax.text(4.45, 1.72, "Schematic: the open loop front drifts because nothing corrects it, and "
                        "on the unobserved day every mode must trust the model. The measured "
                        "skill is tabulated below.",
            fontsize=8.5, color=C["muted"], va="center", style="italic", **FONT)

    # -- measured skill ----------------------------------------------------- #
    ax.plot([0.45, 15.35], [1.40, 1.40], color=C["rule"], lw=1.0)
    ax.text(0.45, 1.10, "Average precision by lead time, and the paired test across the eight fires",
            fontsize=10.4, weight="bold", color=C["ink"], va="center", **FONT)
    vx = (4.30, 5.35, 6.40)
    for x, lab in zip(vx, ("day +1", "day +2", "day +3")):
        ax.text(x, 0.74, lab, fontsize=8.8, weight="bold", color=C["muted"], ha="center", **FONT)
    rows = [
        ("open loop", ("0.073", "0.047", "0.038"), "", "#9aa0a6"),
        ("state sync", ("0.608", "0.237", "0.085"),
         "gain of 0.535 at day 1, better on 8 of 8 fires, t of 15.9", "#2b5d8c"),
        ("state and model", ("0.613", "0.288", "0.110"),
         "adds 0.051 at day 2 and 0.025 at day 3, better on 8 of 8 fires", C["good"]),
    ]
    for k, (nm, vals, note, col) in enumerate(rows):
        yy = 0.44 - 0.28 * k
        ax.text(0.45, yy, nm, fontsize=8.9, color=col, weight="bold", va="center", **FONT)
        for x, v in zip(vx, vals):
            ax.text(x, yy, v, fontsize=8.9, color=C["ink"], ha="center", va="center", **FONT)
        if note:
            ax.text(7.15, yy, note, fontsize=8.6, color=col, va="center", **FONT)
    save(fig, "pspe_twin_loop")


# --------------------------------------------------------------------------- #
# 3. safety: the probe and the conformal margin
# --------------------------------------------------------------------------- #
def fig_safety():
    fig = plt.figure(figsize=(15.8, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=0.17)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(0.1, 6.72, "Why a satisfied dual still violates the limit",
            fontsize=12.8, weight="bold", color=C["ink"], va="center", **FONT)

    left = [r"the dual is fed $\hat J_C$ measured in $G_\theta$",
            r"violation is realised under the true $F$",
            "!nothing in the loop ever compares them",
            "!7.3 percent of evaluations violate"]
    right = [r"probe rolls $\pi_\phi$ in $F$ every $N$ iterations",
             r"plan against $d_{\mathrm{eff}}$, not against $d$",
             "+13 of 15 seed runs entirely clean",
             "+about 1 percent, and a stated rate"]
    h = max(block_height(left, 0.40), block_height(right, 0.40))
    box(ax, 0.1, 6.28 - h, 4.35, left, "#fdf0f0", C["bad"], "WITHOUT A PROBE",
        body=8.9, lead=0.40, h=h, title_size=10.6)
    box(ax, 4.85, 6.28 - h, 4.45, right, "#eef7f2", C["good"], "WITH PROBE AND MARGIN",
        body=8.9, lead=0.40, h=h, title_size=10.6)

    yt = 6.28 - h - 0.42
    ax.text(0.1, yt, "The margin, and the only property it needs",
            fontsize=10.8, weight="bold", color=C["ink"], va="center", **FONT)
    ax.add_patch(FancyBboxPatch((0.1, yt - 2.10), 9.2, 1.86,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#fafbfc", ec=C["rule"], lw=1.2))
    ax.text(4.7, yt - 0.55, r"$e_i \,=\, c_i - \bar c$   on the $n$ real episode costs "
                            r"the probe returned",
            fontsize=11.0, color=C["ink"], ha="center", va="center", **FONT)
    ax.text(4.7, yt - 1.12, r"$q \,=\, e_{(\lceil (n+1)(1-\delta) \rceil)}$"
                            "          "
                            r"$d_{\mathrm{eff}} \,=\, d - (\,q + \hat b\,)$",
            fontsize=13.0, color=C["good"], ha="center", va="center", **FONT)
    ax.text(4.7, yt - 1.70, r"exchangeability of the deviations gives  "
                            r"$\mathbb{P}\,[\,c > d\,] \,\leq\, \delta$",
            fontsize=10.6, color=C["ink"], ha="center", va="center", **FONT)

    yb = yt - 2.52
    ax.text(0.1, yb, "A margin is only as good as the distribution it bounds",
            fontsize=10.4, weight="bold", color=C["bad"], va="center", **FONT)
    ax.text(0.1, yb - 0.78, "The first attempt conformalised the surrogate's cost error. On the "
                            "reaction diffusion family the surrogate\nis accurate and the policy "
                            "is variable, so that quantile never saw what violates the limit: "
                            "violations\nrose to 18.2 percent against a 14.5 percent baseline. "
                            "Conformalising the episode costs holds the bound.",
            fontsize=8.8, color=C["ink"], va="center", linespacing=1.7, **FONT)

    # -- measured violation rates ------------------------------------------ #
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["as shipped", "saturation\nwall", "dual gain\nraised",
              "episode\nspread", "conformal\nmargin"]
    rates = [85.0, 44.0, 14.5, 5.5, 7.3]
    cols = [C["bad"], "#c96a2a", "#c9a02a", C["good"], C["good"]]
    bars = ax2.bar(range(5), rates, color=cols, width=0.66, zorder=3)
    for b, r in zip(bars, rates):
        ax2.text(b.get_x() + b.get_width() / 2, r + 2.0, f"{r}%", ha="center", va="bottom",
                 fontsize=10.4, weight="bold", color=b.get_facecolor(), **FONT)
    ax2.axhline(10.0, color=C["good"], ls="--", lw=1.4, zorder=2)
    ax2.annotate("the rate the bound\nstates, delta of 0.1", xy=(3.45, 10.0), xytext=(3.05, 30),
                 fontsize=8.6, color=C["good"], ha="center", va="bottom", linespacing=1.5,
                 arrowprops=dict(arrowstyle="-", color=C["good"], lw=1.0))
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(labels, fontsize=8.8)
    ax2.set_ylabel("evaluations violating the true limit (percent)", fontsize=9.6)
    ax2.set_ylim(0, 100)
    ax2.set_title("Reaction diffusion front, 5 seeds\n"
                  "three planner defects removed, then the bound",
                  fontsize=10.8, weight="bold", pad=12)
    ax2.grid(axis="y", color=C["rule"], lw=0.8, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(labelsize=8.8)
    save(fig, "pspe_safety")


# --------------------------------------------------------------------------- #
# 4. the thesis: silent failure
# --------------------------------------------------------------------------- #
def fig_silent_failure():
    fig, ax = canvas(15.8, 9.4)

    ax.text(0.4, 9.05, "Model based decision systems fail silently",
            fontsize=14.5, weight="bold", color=C["ink"], va="center", **FONT)
    ax.text(0.4, 8.50, "Every component metric below improved, or looked correct, while the "
                       "decision degraded or the constraint was violated. Each metric is "
                       "computed\ninside the model's own account of the world. The right hand "
                       "column is the measurement that exposed the gap.",
            fontsize=9.6, color=C["muted"], va="center", linespacing=1.7, **FONT)

    x1, x2, x3 = 0.55, 5.70, 10.30
    ax.text(x1, 7.72, "THE COMPONENT LOOKED CORRECT BY", fontsize=9.4, weight="bold",
            color=C["good"], va="center", **FONT)
    ax.text(x2, 7.72, "BUT IN FACT", fontsize=9.4, weight="bold", color=C["bad"], va="center",
            **FONT)
    ax.text(x3, 7.72, "THE MEASUREMENT THAT EXPOSED IT", fontsize=9.4, weight="bold",
            color=C["ink"], va="center", **FONT)
    ax.plot([x1 - 0.14, 15.4], [7.46, 7.46], color=C["ink"], lw=1.2)

    rows = [
        ("surrogate episode cost, dual satisfied",
         "7.3 percent of evaluations violated\nthe true limit",
         "probe the true environment periodically"),
        ("one step forecast skill",
         "open loop rollout invents a fire\nthat does not exist",
         "synchronise to observation, compare to open loop"),
        ("faithfulness 0.47 against 0.18, t of 21",
         "briefs carry zero state specific\ninformation",
         "permutation control, score against another state"),
        ("cost separates between doing nothing\nand acting greedily",
         "methods were ranked inside a\n7 percent achievable band",
         "return separation, not only cost separation"),
        ("four constrained RL baselines appeared\nto fail structurally",
         "they had barely moved off their\ninitialisation",
         "start every baseline where the method starts"),
        ("reconstruction precision 0.594 against\n0.006, about 100 times better",
         "the decision was worse in\n0 of 9 configurations",
         "score the belief on the decision it produces"),
        ("a margin cut violations about 7 times",
         "no stated failure rate, and the first\nconformal attempt made things worse",
         "conformalise the distribution that actually violates"),
    ]

    y, rh = 7.30, 0.80
    for i, (a, b, c) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((x1 - 0.14, y - rh), 15.54 - x1 + 0.14 - 0.05, rh,
                                   fc="#f7f8f9", ec="none", zorder=0))
        ax.add_patch(Rectangle((x1 - 0.14, y - rh), 0.055, rh, fc=C["good"], ec="none", zorder=1))
        ax.add_patch(Rectangle((x2 - 0.28, y - rh), 0.055, rh, fc=C["bad"], ec="none", zorder=1))
        for x, txt, col in ((x1 + 0.06, a, C["ink"]), (x2, b, C["bad"]), (x3, c, C["ink"])):
            ax.text(x, y - rh / 2, txt, fontsize=8.9, color=col, va="center",
                    linespacing=1.55, **FONT)
        y -= rh

    ax.add_patch(FancyBboxPatch((0.41, 0.35), 15.0, 1.02,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#eef7f2", ec=C["good"], lw=1.5))
    ax.text(7.91, 0.86, "The positive results share one shape: measure the quantity that "
                        "actually matters, which is real cost, observed\nstate, or decision "
                        "outcome, and the system works.",
            fontsize=10.4, color=C["good"], weight="bold", ha="center", va="center",
            linespacing=1.6, **FONT)
    save(fig, "pspe_silent_failure")


# --------------------------------------------------------------------------- #
# 5 to 8. per module panels
# --------------------------------------------------------------------------- #
def module_panel(name, key, equation, mech, held, failed, caption, fname):
    W = 11.0
    fig, ax = canvas(W, 8.0)
    fill, edge = C[key]

    ax.add_patch(FancyBboxPatch((0.25, 7.02), W - 0.5, 0.80,
                                boxstyle="round,pad=0,rounding_size=0.10", fc=edge, ec=edge, lw=0))
    ax.text(0.58, 7.42, name, fontsize=15.5, weight="bold", color="white", va="center", **FONT)
    ax.text(W - 0.58, 7.42, equation, fontsize=12.5, color="white", va="center",
            ha="right", **FONT)

    hm = 0.34 + 0.30 + len(mech) * 0.44
    ym = 6.80 - hm
    ax.add_patch(FancyBboxPatch((0.25, ym), W - 0.5, hm,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.2))
    ax.text(0.52, ym + hm - 0.30, "WHAT IT DOES", fontsize=9.2, weight="bold", color=edge,
            va="center", **FONT)
    for i, line in enumerate(mech):
        ax.text(0.52, ym + hm - 0.74 - i * 0.44, line, fontsize=9.0, color=C["ink"],
                va="center", **FONT)

    hl = 0.34 + 0.30 + max(len(held), len(failed)) * 0.38
    yl = ym - 0.42 - hl
    cw = (W - 0.5 - 0.30) / 2
    for x0, title, lines, col, bg in ((0.25, "WHAT HELD UP", held, C["good"], "#eef7f2"),
                                      (0.55 + cw, "WHAT DID NOT", failed, C["bad"], "#fdf0f0")):
        ax.add_patch(FancyBboxPatch((x0, yl), cw, hl, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=bg, ec=col, lw=1.3))
        ax.text(x0 + 0.27, yl + hl - 0.30, title, fontsize=9.2, weight="bold", color=col,
                va="center", **FONT)
        for i, line in enumerate(lines):
            ax.text(x0 + 0.27, yl + hl - 0.72 - i * 0.38, line, fontsize=8.8, color=C["ink"],
                    va="center", **FONT)

    cy = yl - 0.62
    ax.text(0.27, cy, caption, fontsize=8.9, color=C["muted"], va="center",
            linespacing=1.7, style="italic", **FONT)
    ax.set_ylim(cy - 0.52 - 0.13 * caption.count("\n"), 8.0)     # trim the dead margin
    save(fig, fname)


def fig_modules():
    module_panel(
        "PERCEIVE", "perceive",
        r"$\hat u_t = m_t\,y_t + (1-m_t)\,\Phi(\tilde u_t,\, y_t)$",
        ["A frozen vision backbone with LoRA adapters, a field decoder and a contrastive text "
         "head.",
         "Under occlusion a self supervised estimator fills the cells the satellite did not see.",
         "Occlusion is generated spatially correlated, so a blob covers the front, not stray "
         "pixels."],
        ["Losing 60 percent of the observation",
         "costs 6.6 of 41 points, and still beats",
         "the full state heuristic by a wide margin.",
         "The frozen backbone beats a trained one,",
         "t of 3.97, and buys alignment of 0.754."],
        ["Reconstruction is 100 times better on",
         "hidden cells, yet the decision was worse",
         "in 0 of 9 configurations. Of 4.3 points",
         "lost, 3.4 are a representation artefact",
         "and 0.86 a real cost under a hard budget."],
        "Bounded by one property of this field: fire covers 1.3 percent of cells, so assuming "
        "nothing burns where you cannot\nsee is close to the base rate and costs almost nothing. "
        "On a dense field such as flood depth this could invert,\nwhich is the next experiment "
        "worth running.",
        "pspe_mod_perceive")

    module_panel(
        "SIMULATE", "simulate",
        r"$u_{t+1} = G_\theta(u_t,\, a_t)$",
        [r"Fourier layers $v_{\ell+1} = \sigma(W v_\ell + \mathcal{F}^{-1}[R_\ell \cdot "
         r"\mathcal{F}[v_\ell]])$ with modes truncated at $k_{\max}$ of 12.",
         r"Trained on $\|G_\theta - u'\|^2$ with rollout and physics residual terms; spectral "
         r"normalisation bounds $L_G$.",
         "A U-Net path carries sharp fronts, combined through a gate that is free to take any "
         "value."],
        ["0.3162 on the NDWS test split, which is",
         "111 percent of the published benchmark.",
         "Resolution invariant to three decimals",
         "from 64 to 128 squared. On PDEBench",
         "shallow water, 0.0018 at one step."],
        ["Proposition 1 is vacuous in its infinite",
         "horizon form: 422 against a measured",
         "0.10. Report the finite horizon form and",
         "say that it is still 100 times loose.",
         "The physics loss buys speed, not accuracy."],
        "The learned gate over the spectral path, initialised at zero and free to grow, settles "
        "at 0.12 on every seed and split.\nOffered the Fourier operator on sharp fire fronts the "
        "model declines it, because a truncated spectrum cannot\nrepresent a front.",
        "pspe_mod_simulate")

    module_panel(
        "PLAN", "plan",
        r"$\min_b\ \mathbb{E}\,[\,\mathrm{loss}\,]\ \ \mathrm{s.t.}\ \ \bar b_t \leq B$",
        [r"Hybrid gradient $g(\alpha) = \alpha g_{\mathrm{pw}} + (1-\alpha) g_{\mathrm{LR}}$, "
         r"with $\alpha^* = (V_L - \mathrm{Cov}) / (B^2 + V_p + V_L - 2\,\mathrm{Cov})$.",
         r"A PID dual $\lambda_k = [\,K_p e_k + K_i I_k + K_d \max(0,\, e_k - e_{k-1})\,]_+$ "
         r"driven by measured constraint error.",
         "At decision time, projected gradient through the rollout with exact projection onto "
         "the budget."],
        ["12.0 points over the operational",
         "heuristic on 1500 held out fires,",
         "paired t of 11.15. The gap widens to",
         "13.6 with the hand written physics",
         "removed. Violations 7.3 percent to 0."],
        ["Equation 8 is degenerate: the measured",
         "bias squared is 0.0013, so it and the",
         "variance only rule pick the same alpha.",
         "The amortised policy reaches only 10.4",
         "percent, the same failure as safe RL."],
        "The mechanism is measured, not asserted. At a one day horizon the planner and the "
        "heuristic are indistinguishable,\nt of 1.40, exactly as theory requires since that "
        "objective is separable over cells. The gap reaches 14.3 points at\nfive days, and the "
        "heuristic degrades because it keeps optimising for today.",
        "pspe_mod_plan")

    module_panel(
        "EXPLAIN", "explain",
        r"$F(b) = \exp(-\mathrm{KL}(\pi_\phi \,\|\, \widehat\pi_b) / K)$",
        ["A brief is parsed by a frozen deterministic parser back into an action distribution, "
         "then scored against the policy.",
         r"Split conformal: with calibration scores $s_i = 1 - F(b_i)$, the "
         r"$\lceil (n+1)(1-\delta) \rceil$ order statistic bounds the floor.",
         "Per dimension KL is the only comparable form; summed KL sends every score to zero "
         "above about 16 actions."],
        ["Certificate coverage holds at 90 and 95",
         "percent on 3 of 3 seeds on real fire",
         "plans, with a floor of 0.706 that is not",
         "vacuous. It reports honestly on a weak",
         "generator rather than overstating it."],
        ["The faithfulness claim is withdrawn. A",
         "permutation control scores each brief",
         "against a different state's policy and",
         "finds a gap of zero, the signature of a",
         "generator that ignores its condition."],
        "Three fixes were tried and all were negative: wider conditioning, a differentiable "
        "contrastive objective, and 2.5 times\nthe training budget. The testbed also could not "
        "have answered the question, since its trained policy varies 1.2\npercent and yields 8 "
        "distinct actions across 64 states.",
        "pspe_mod_explain")


# --------------------------------------------------------------------------- #
# 9. wildfire results
# --------------------------------------------------------------------------- #
def fig_wildfire():
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8))

    ax = axes[0]
    budget = np.array([1, 2, 3, 5, 8])
    greedy, gerr = np.array([7.5, 12.8, 24.1, 36.1, 55.0]), np.array([1.4, 2.0, 3.8, 5.0, 5.5])
    pspe, perr = np.array([14.3, 26.2, 36.5, 53.0, 69.4]), np.array([0.5, 0.7, 1.0, 1.1, 0.6])
    ax.errorbar(budget, greedy, yerr=gerr, marker="s", ms=7, lw=2.0, capsize=4,
                color="#9aa0a6", label="forecast, then treat the riskiest cells")
    ax.errorbar(budget, pspe, yerr=perr, marker="o", ms=7, lw=2.4, capsize=4,
                color=C["plan"][1], label="PSPE, plan through the learned model")
    for b, p, r in zip(budget, pspe, pspe / greedy):
        ax.annotate(f"{r:.2f} times", (b, p), textcoords="offset points", xytext=(6, 12),
                    ha="left", fontsize=8.8, color=C["plan"][1], weight="bold")
    ax.set_xlabel("crew budget, percent of the sector treated per day", fontsize=9.8)
    ax.set_ylabel("population weighted burn reduction (percent)", fontsize=9.8)
    ax.set_title("The advantage is largest where crews are scarcest\n"
                 "1500 held out fires, 3 seeds", fontsize=11, weight="bold", pad=10)
    ax.legend(fontsize=9.0, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.set_xlim(0.45, 9.1); ax.set_ylim(0, 86)

    ax = axes[1]
    hz = np.array([1, 2, 3, 5])
    g, ge = np.array([24.8, 24.2, 22.0, 17.7]), np.array([1.1, 3.6, 4.4, 4.3])
    p, pe = np.array([25.4, 32.6, 34.1, 32.0]), np.array([1.7, 3.4, 4.4, 5.8])
    ax.errorbar(hz, g, yerr=ge, marker="s", ms=7, lw=2.0, capsize=4, color="#9aa0a6",
                label="forecast, then treat the riskiest cells")
    ax.errorbar(hz, p, yerr=pe, marker="o", ms=7, lw=2.4, capsize=4, color=C["plan"][1],
                label="PSPE, plan through the learned model")
    ax.fill_between(hz, g, p, color=C["plan"][1], alpha=0.11)
    # the edge labels are anchored inward so neither is clipped by the axes
    notes = [("t of 1.40, not significant", C["muted"], "normal", "left", (-10, -9)),
             ("t of 10.74", C["plan"][1], "bold", "center", (0, -9)),
             ("t of 11.08", C["plan"][1], "bold", "center", (0, -9)),
             ("t of 8.99", C["plan"][1], "bold", "right", (10, -9))]
    for h, gg, ee, (txt, col, wt, ha, off) in zip(hz, g, ge, notes):
        ax.annotate(txt, (h, gg - ee), textcoords="offset points", xytext=off, ha=ha,
                    va="top", fontsize=8.3, color=col, weight=wt, linespacing=1.4)
    ax.set_xlabel("planning horizon (days)", fontsize=9.8)
    ax.set_ylabel("population weighted burn reduction (percent)", fontsize=9.8)
    ax.set_title("The gap appears only once the decision is sequential\n"
                 "insignificant at one day, exactly where theory requires it to be",
                 fontsize=11, weight="bold", pad=10)
    ax.legend(fontsize=9.0, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    ax.set_xticks(hz); ax.set_xlim(0.55, 5.45); ax.set_ylim(6, 45)

    for ax in axes:
        ax.grid(color=C["rule"], lw=0.8, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9.0)
    fig.tight_layout()
    save(fig, "pspe_wildfire")


def main() -> None:
    fig_system()
    fig_twin_loop()
    fig_safety()
    fig_silent_failure()
    fig_modules()
    fig_wildfire()


if __name__ == "__main__":
    main()
