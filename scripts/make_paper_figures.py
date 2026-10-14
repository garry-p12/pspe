#!/usr/bin/env python
"""Publication figures for the conformal-margin paper.

    python scripts/make_paper_figures.py

Sized and styled for an ICML two-column layout, which is a different job from
the slide figures in `make_thesis_figures.py`: 3.25 in for a single column and
6.75 in across both, serif type to match the body text, 7 to 8 pt labels, and
no element that survives only as decoration.

Every number is measured. Provenance is in the docstring of each figure, and
the measurement it came from is named in `docs/THESIS_AND_RESULTS.md`.

Output is PDF (vector, for LaTeX) plus PNG at 400 dpi for quick viewing.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "paper"

COL, FULL = 3.25, 6.75          # ICML single and double column, inches

# Colourblind-safe (Okabe-Ito), and distinguishable in greyscale by marker.
C = {
    "none":        "#999999",
    "model_error": "#D55E00",   # vermilion
    "episode":     "#E69F00",   # orange
    "residual":    "#009E73",   # bluish green
    "rule":        "#CCCCCC",
    "ink":         "#000000",
}
MK = {"none": "s", "model_error": "^", "episode": "D", "residual": "o"}
LBL = {
    "none": "no margin",
    "model_error": "model error (default)",
    "episode": "deviations $+$ bias",
    "residual": "controlled quantity (ours)",
}


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "grid.linewidth": 0.4,
        "grid.color": C["rule"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,          # editable text, not outlines
        "ps.fonttype": 42,
    })


def finish(ax, *, grid=True, spines=("top", "right")) -> None:
    if grid:
        ax.grid(True, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
    for s in spines:
        ax.spines[s].set_visible(False)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"docs/figures/paper/{name}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
def fig_mechanism() -> None:
    """The claim, in two panels on one axis.

    (a) `runs/margin_synthetic/`: bias and spread dialled independently, 600
        calibration draws per point, nothing learned. Coverage of each recipe
        against rho = sigma/eps.
    (b) `runs/margin_choice/`: 5 surrogate qualities x 3 seeds on the
        reaction-diffusion front. The margin each recipe produces, against the
        rho measured in that run.

    (a) is the consequence, (b) the cause, on the same horizontal axis so the
    reader can carry the crossing from one to the other.
    """
    syn = json.loads((ROOT / "runs/margin_synthetic/results.json").read_text())
    pde = [r for f in glob.glob(str(ROOT / "runs/margin_choice/*/results.json"))
           for r in json.loads(Path(f).read_text())]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.05))

    ax = axes[0]
    rho = [r["rho = spread/model err"] for r in syn]
    for arm in ("none", "model_error", "episode", "residual"):
        ax.plot(rho, [100 * r[arm] for r in syn], marker=MK[arm], color=C[arm],
                label=LBL[arm], zorder=3)
    ax.axhline(10, color=C["residual"], ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.axvline(1, color=C["ink"], ls=":", lw=0.8, zorder=2)
    ax.annotate(r"$\delta=0.1$", xy=(0.12, 10), xytext=(0.12, 13.5),
                fontsize=6.5, color=C["residual"])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho=\sigma/\varepsilon$")
    ax.set_ylabel("violation rate (%)")
    ax.set_title(r"(a) coverage: the default fails once $\rho>1$")
    ax.set_ylim(-4, 96)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.86), handlelength=1.4,
              labelspacing=0.25, borderpad=0.2)
    finish(ax)

    ax = axes[1]
    for arm in ("model_error", "episode", "residual"):
        pts = sorted((r["rho = sigma/eps"], r["limit"] - r["d_eff"])
                     for r in pde if r["arm"] == arm and r.get("rho = sigma/eps") ==
                     r.get("rho = sigma/eps"))
        if not pts:
            continue
        x = np.array([q[0] for q in pts]); y = np.array([q[1] for q in pts])
        ax.scatter(x, y, marker=MK[arm], s=11, color=C[arm], label=LBL[arm],
                   zorder=3, linewidths=0)
        k = max(2, len(x) // 4)
        ax.plot(x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid"),
                color=C[arm], alpha=0.55, lw=1.0, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\rho=\sigma/\varepsilon$ (measured per run)")
    ax.set_ylabel("margin produced")
    ax.set_title("(b) why: the default's margin vanishes")
    ax.legend(loc="lower left", handlelength=1.4)
    finish(ax)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "fig1_mechanism")


# --------------------------------------------------------------------------- #
def fig_rdf() -> None:
    """Violation rate on the reaction-diffusion front (`docs/results/RDF_PLANNER_FIX.md`,
    `runs/constraint_fix_conf*/`, 5 seeds). Two planner defects are removed
    first, then three margins are compared in the regime that remains: an
    accurate surrogate and a dual that holds the mean.

    Horizontal bars because six labelled conditions do not fit across a single
    column, and rotating the labels costs more legibility than it saves.
    """
    rows = [
        ("as shipped",                      85.0, C["none"]),
        ("+ saturation wall",               44.0, C["none"]),
        ("+ integral gain",                 14.5, C["none"]),
        ("model-error conformal",           18.2, C["model_error"]),
        (r"hand-tuned $k\sigma$",           5.5,  C["episode"]),
        ("episode-cost conformal (ours)",   7.3,  C["residual"]),
    ]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows],
            height=0.66, zorder=3, linewidth=0)
    for yy, (_, v, c) in zip(y, rows):
        ax.text(v + 2.0, yy, f"{v}", va="center", fontsize=6.6, color=c)
    ax.axvline(10, color=C["residual"], ls=(0, (4, 2)), lw=0.9, zorder=4)
    ax.text(10.8, len(rows) - 0.35, r"$\delta=0.1$", fontsize=6.3,
            color=C["residual"], va="top")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax.set_xlabel("violating evaluations (%)")
    ax.set_xlim(0, 100)
    ax.set_title("the default is worse than no margin at all")
    ax.grid(True, axis="x", alpha=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "fig2_rdf_violations")


# --------------------------------------------------------------------------- #
def fig_precondition() -> None:
    """When a margin is usable at all.

    A margin must cover the model-reality bias `b` and fit the headroom `f*s`.
    Points below the diagonal are usable. Left: four learned-vs-learned
    configurations on the wildfire task (`runs/ndws_calib/`), none of which
    reach it. Right: the same task with the discrepancy dialled
    (`runs/ndws_perturb/`), which brackets the boundary.
    """
    learned = [(0.009468 * .5, 0.030637, "3%, 4ep"), (0.007328 * .5, 0.033054, "8%, 4ep"),
               (0.005069 * .5, 0.010344, "8%, 10ep"), (0.005492 * .5, 0.026921, "15%, 10ep")]
    dialled = [(0.003443, 0.001509, "0.01"), (0.003470, 0.001988, "0.02"),
               (0.003543, 0.003029, "0.04"), (0.003694, 0.005134, "0.08"),
               (0.003943, 0.009718, "0.16")]

    fig, ax = plt.subplots(figsize=(COL, 2.1))
    lim = [8e-4, 6e-2]
    ax.fill_between(lim, lim, [lim[0], lim[0]], color=C["residual"], alpha=0.10, zorder=1)
    ax.plot(lim, lim, color=C["ink"], lw=0.8, ls="--", zorder=2)
    ax.text(1.05e-2, 1.15e-3, "usable", fontsize=6.4, color=C["residual"], style="italic")
    ax.text(1.0e-3, 2.6e-2, "infeasible", fontsize=6.4, color=C["model_error"], style="italic")
    hx, by_, _ = zip(*learned)
    ax.scatter(hx, by_, marker="X", s=26, color=C["model_error"], zorder=4,
               label="second trained model", linewidths=0)
    hx, by_, _ = zip(*dialled)
    ax.scatter(hx, by_, marker="o", s=18, color=C["residual"], zorder=4,
               label="discrepancy dialled", linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(r"headroom $f\,s$")
    ax.set_ylabel(r"model-reality bias $b$")
    ax.set_title(r"a margin is usable only when $b<f\,s$")
    ax.legend(loc="lower right", handlelength=1.2, labelspacing=0.25,
              borderpad=0.25, bbox_to_anchor=(1.0, 0.10))
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "fig3_precondition")


# --------------------------------------------------------------------------- #
def fig_wildfire() -> None:
    """Firebreak planning on observed fire records (`docs/results/NDWS_PLANNING.md`).

    Left: budget Pareto, 3 seeds. Right: horizon ablation, 5 seeds, where the
    gap is absent at the horizon on which the heuristic is provably optimal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95))

    ax = axes[0]
    b = np.array([1, 2, 3, 5, 8])
    g, ge = np.array([7.5, 12.8, 24.1, 36.1, 55.0]), np.array([1.4, 2.0, 3.8, 5.0, 5.5])
    p, pe = np.array([14.3, 26.2, 36.5, 53.0, 69.4]), np.array([0.5, 0.7, 1.0, 1.1, 0.6])
    ax.errorbar(b, g, yerr=ge, marker="s", color=C["none"], capsize=1.6,
                elinewidth=0.7, label="forecast, treat riskiest")
    ax.errorbar(b, p, yerr=pe, marker="o", color=C["residual"], capsize=1.6,
                elinewidth=0.7, label="plan through the model")
    for bb, pp, rr in zip(b, p, p / g):
        ax.annotate(f"{rr:.2f}$\\times$", (bb, pp), textcoords="offset points",
                    xytext=(3.5, 5.5), fontsize=5.8, color=C["residual"])
    ax.set_xlabel("crew budget (% of sector/day)")
    ax.set_ylabel("burn reduction (%)")
    ax.set_title("(a) largest gain where crews are scarce")
    ax.set_xlim(0.4, 9.2); ax.set_ylim(0, 84)
    ax.legend(loc="upper left", handlelength=1.4)
    finish(ax)

    ax = axes[1]
    h = np.array([1, 2, 3, 5])
    g, ge = np.array([24.8, 24.2, 22.0, 17.7]), np.array([1.1, 3.6, 4.4, 4.3])
    p, pe = np.array([25.4, 32.6, 34.1, 32.0]), np.array([1.7, 3.4, 4.4, 5.8])
    ax.errorbar(h, g, yerr=ge, marker="s", color=C["none"], capsize=1.6, elinewidth=0.7)
    ax.errorbar(h, p, yerr=pe, marker="o", color=C["residual"], capsize=1.6, elinewidth=0.7)
    ax.fill_between(h, g, p, color=C["residual"], alpha=0.10)
    for hh, gg, ee, t, bold in zip(h, g, ge, ["n.s.", "10.7", "11.1", "9.0"],
                                   [False, True, True, True]):
        ax.annotate(("$t{=}$" + t) if bold else t, (hh, gg - ee),
                    textcoords="offset points", xytext=(0, -5.5), ha="center",
                    va="top", fontsize=5.8,
                    color=C["residual"] if bold else C["none"])
    ax.set_xlabel("planning horizon (days)")
    ax.set_ylabel("burn reduction (%)")
    ax.set_title("(b) the gap needs a sequential decision")
    ax.set_xticks(h); ax.set_xlim(0.6, 5.4); ax.set_ylim(6, 44)
    finish(ax)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "fig4_wildfire")


def fig_calibration() -> None:
    """Realised rate against the stated level (`runs/margin_coverage/`).

    The question a conformal method is finally judged on, and the one the other
    experiments do not ask: they fix delta = 0.1. A calibrated recipe tracks the
    diagonal at every level.

    Two regimes, because the claim is conditional. At rho = 0.25 the model's
    per-instance error dominates and the matched-pair recipe is merely
    over-conservative. At rho = 4 the policy's spread dominates, and its
    realised rate is nearly independent of the level requested: 0.30 when asked
    for 0.02, 0.47 when asked for 0.40. The number it states carries almost no
    information about what it does.
    """
    rows = json.loads((ROOT / "runs/margin_coverage/results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.05), sharey=True)
    for ax, rho, sub in zip(axes, (0.25, 4.0),
                            ("model error dominates", "policy spread dominates")):
        rs = [r for r in rows if r["rho"] == rho]
        d = [r["delta"] for r in rs]
        ax.plot([0, 0.45], [0, 0.45], color=C["ink"], ls="--", lw=0.8,
                zorder=2, label="calibrated")
        for arm in ("model_error", "episode", "residual"):
            ax.errorbar(d, [r[arm] for r in rs], yerr=[r[f"{arm} sd"] for r in rs],
                        marker=MK[arm], color=C[arm], label=LBL[arm],
                        capsize=1.5, elinewidth=0.7, zorder=3)
        ax.set_xlabel(r"stated level $\delta$")
        ax.set_title(rf"$\rho={rho}$: {sub}")
        ax.set_xlim(0, 0.43); ax.set_ylim(-0.02, 0.52)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        finish(ax)
    axes[0].set_ylabel("realised violation rate")
    axes[0].legend(loc="upper left", handlelength=1.4, labelspacing=0.25,
                   borderpad=0.2)
    axes[1].annotate("undercovers:\nrate barely responds\nto the level asked for",
                     xy=(0.30, 0.447), xytext=(0.16, 0.30), fontsize=6.2,
                     color=C["model_error"], linespacing=1.3,
                     arrowprops=dict(arrowstyle="->", color=C["model_error"], lw=0.6))
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig0_calibration")


def fig_tradeoff() -> None:
    """What the margin costs, paired per run (`runs/margin_choice/`, 15 runs).

    The first objection to any conservative method is that it buys safety by
    refusing to act. Differences are against the same run's `none` arm, so
    surrogate quality and seed cancel.

    Means with bootstrap intervals, not the individual runs. At 11 evaluations
    per run the per-run violation change is quantised to multiples of 1/11, so
    a scatter of the raw pairs shows binning artefacts rather than a
    relationship; the means remain estimable and are what the claim rests on.
    """
    rows = [r for f in glob.glob(str(ROOT / "runs/margin_choice/*/results.json"))
            for r in json.loads(Path(f).read_text())]
    base = {(r["fidelity"], r["seed"]): r for r in rows if r["arm"] == "none"}
    rng = np.random.default_rng(0)

    def boot(v, n=10000):
        d = rng.choice(np.asarray(v), size=(n, len(v)), replace=True).mean(1)
        return np.quantile(d, 0.025), np.quantile(d, 0.975)

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.axhline(0, color=C["rule"], lw=0.7, zorder=1)
    ax.axvline(0, color=C["rule"], lw=0.7, zorder=1)
    for arm in ("model_error", "episode", "residual"):
        dv, dr = [], []
        for r in rows:
            if r["arm"] != arm:
                continue
            b = base.get((r["fidelity"], r["seed"]))
            if b is None:
                continue
            dv.append(100 * (r["violating"] - b["violating"]))
            dr.append(r["return"] - b["return"])
        vlo, vhi = boot(dv); rlo, rhi = boot(dr)
        mv, mr = float(np.mean(dv)), float(np.mean(dr))
        ax.errorbar([mv], [mr], xerr=[[mv - vlo], [vhi - mv]],
                    yerr=[[mr - rlo], [rhi - mr]], marker=MK[arm], color=C[arm],
                    capsize=2, elinewidth=0.8, markersize=5.5, zorder=3,
                    label=LBL[arm])
    ax.set_xlabel("change in violation rate (points)")
    ax.set_ylabel("change in return")
    ax.set_title("safety costs 1.8% of return, not the policy")
    ax.legend(loc="lower left", handlelength=1.2, labelspacing=0.25, borderpad=0.25)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "fig5_tradeoff")


def fig_ablation() -> None:
    """Probe and margin, separately and together (`runs/constraint_fix/`, 5 seeds,
    dar, limit 0.936).

    The margin alone is bit-identical to the baseline: with no probe there is no
    measurement from which to size it. The probe alone lowers the worst case but
    not the rate. Only the pair reaches zero. Conservatism has to be derived from
    a measurement, not asserted.

    One scale, not two. A dual axis makes bars of different units look
    comparable, and the point here is a comparison within each metric.
    """
    arms = ["baseline", "probe only", "margin only", "probe $+$ margin"]
    viol = [1.8, 1.8, 1.8, 0.0]
    worst = [0.75, 0.56, 0.75, 0.29]
    cols = [C["none"], C["none"], C["none"], C["residual"]]

    fig, ax = plt.subplots(figsize=(COL, 1.8))
    y = np.arange(4)[::-1]
    ax.barh(y, viol, 0.6, color=cols, zorder=3, linewidth=0)
    for yy, v, w in zip(y, viol, worst):
        ax.text(v + 0.06, yy, f"{v}%", va="center", fontsize=6.5, color=C["ink"])
        ax.text(2.42, yy, f"{w:.2f}", va="center", ha="right", fontsize=6.5,
                color=C["ink"])
    ax.text(2.42, y[0] + 0.62, "worst cost\n(limit 0.94)", ha="right", va="bottom",
            fontsize=6.0, linespacing=1.2)
    ax.set_yticks(y); ax.set_yticklabels(arms, fontsize=6.6)
    ax.set_xlabel("violating evaluations (%)")
    ax.set_xlim(0, 2.6)
    ax.set_title("neither half works alone")
    ax.grid(True, axis="x", alpha=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "fig6_ablation")


# --------------------------------------------------------------------------- #
# Figures for the PSPE / digital-twin paper rather than the margin paper. Same
# style so they are interchangeable between the two.
# --------------------------------------------------------------------------- #
def fig_partial() -> None:
    """Planning under partial observation (`docs/results/NDWS_PARTIAL.md`,
    3 seeds, 800 held-out fires).

    The point is the divergence between the two panels. An estimator recovers
    hidden cells about a hundred times better by average precision (b) and
    produces a worse decision in all nine (rate, seed) configurations (a). The
    belief that assumes no fire where it cannot see does best, because fire
    covers 1.3% of cells and that assumption is close to the base rate.
    """
    occ = np.array([0, 15, 35, 60])
    dec = {
        "blind":         ([41.11, 39.72, 37.44, 34.52], [0.75, 0.77, 0.19, 0.56]),
        "persist":       ([41.11, 39.72, 37.41, 34.44], [0.75, 0.66, 0.29, 0.56]),
        "perceive":      ([41.11, 38.20, 33.30, 27.34], [0.75, 1.15, 1.98, 2.44]),
        "perceive-hard": ([41.11, 39.44, 36.63, 33.04], [0.75, 0.66, 0.93, 1.10]),
    }
    ap = {"blind": [0.006, 0.006, 0.005], "persist": [0.041, 0.037, 0.032],
          "perceive": [0.594, 0.489, 0.346], "perceive-hard": [0.284, 0.224, 0.147]}
    style = {"blind": (C["residual"], "o"), "persist": (C["none"], "s"),
             "perceive": (C["model_error"], "^"), "perceive-hard": (C["episode"], "D")}

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0))
    ax = axes[0]
    for k, (v, e) in dec.items():
        c, m = style[k]
        # `persist` lands within 0.1 points of `blind` at every occlusion rate,
        # which is the finding that carrying yesterday's mask forward buys
        # nothing. Dashed so the coincidence reads as a coincidence rather than
        # one line hiding the other.
        ax.errorbar(occ, v, yerr=e, marker=m, color=c, label=k, capsize=1.5,
                    elinewidth=0.7, ls="--" if k == "persist" else "-",
                    markerfacecolor="none" if k == "persist" else c)
    ax.set_xlabel("observation occluded (%)")
    ax.set_ylabel("burn reduction (%)")
    ax.set_title("(a) the decision")
    ax.set_xticks(occ); ax.set_ylim(24, 44)
    ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.25, borderpad=0.2)
    finish(ax)

    ax = axes[1]
    for k, v in ap.items():
        c, m = style[k]
        ax.plot(occ[1:], v, marker=m, color=c, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("observation occluded (%)")
    ax.set_ylabel("AP on hidden cells")
    ax.set_title("(b) the reconstruction, 100$\\times$ better")
    ax.set_xticks(occ[1:])
    finish(ax)
    fig.tight_layout(pad=0.3, w_pad=1.3)
    save(fig, "fig7_partial_observation")


def fig_twin() -> None:
    """Carrying a state across real satellite days (`docs/results/FIRMS_TWIN.md`,
    8 wildfires, VIIRS detections, leave one fire out, 3 seeds).

    Scored as average precision of the predicted fire against the detections
    actually recorded, on observed days only. Paired across fires, because the
    fires differ enormously in size and behaviour.
    """
    lead = np.array([1, 2, 3])
    modes = {
        "open loop":       ([0.073, 0.047, 0.038], [0.029, 0.026, 0.021], C["none"], "s"),
        "state sync":      ([0.608, 0.237, 0.085], [0.110, 0.093, 0.047], C["episode"], "D"),
        "state $+$ model": ([0.613, 0.288, 0.110], [0.112, 0.111, 0.054], C["residual"], "o"),
    }
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for k, (v, e, c, m) in modes.items():
        ax.errorbar(lead, v, yerr=e, marker=m, color=c, label=k, capsize=1.8,
                    elinewidth=0.7)
    ax.annotate(r"$8.4\times$, 8/8 fires" "\n" r"$t=15.9$", xy=(1, 0.608),
                xytext=(1.25, 0.60), fontsize=6.2, color=C["residual"],
                linespacing=1.3)
    ax.annotate("model adaptation\nadds 21% here", xy=(2, 0.288), xytext=(2.15, 0.40),
                fontsize=6.2, color=C["residual"], linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=C["residual"], lw=0.6))
    ax.set_xlabel("forecast lead time (days)")
    ax.set_ylabel("average precision")
    ax.set_title("syncing to observation, on real fire sequences")
    ax.set_xticks(lead); ax.set_ylim(0, 0.78)
    ax.legend(loc="upper right", handlelength=1.4, labelspacing=0.25, borderpad=0.2)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "fig8_twin_loop")


def fig_flood_elasticity() -> None:
    """Why a margin is hardest to calibrate exactly where the levee matters.

    Left: the amplification E = dlnD/dlnQ against how much of the berm the flood
    tops. E reaches 40x at the onset of over-topping and decays to 2.4x once the
    berm is broadly submerged.

    Right: sigma/s, the dimensionless form of the precondition
    b + z_delta*sigma < f*s (with b ~ 0), against flood magnitude. It has an
    interior minimum: threshold amplification bounds it from below in Q,
    protection saturation from above. The dashed rule is the feasibility
    threshold f/z_delta at f = 0.3.

    Provenance: runs/flood_elasticity/elasticity.json (§5.6). Cross-validated
    against a direct Monte Carlo at Q = 1.4e-2: 0.803 implied vs 0.834 measured.
    """
    src = ROOT / "runs" / "flood_elasticity" / "elasticity.json"
    if not src.exists():
        print("skip fig_flood_elasticity: no runs/flood_elasticity/elasticity.json")
        return
    d = json.loads(src.read_text())
    rows = [r for r in d["rows"] if r["elasticity"] == r["elasticity"]]
    q = np.array([r["inflow"] for r in rows]) * 1e3      # 1e-3 m/s units
    e = np.array([r["elasticity"] for r in rows])
    wet = np.array([r["berm_rows_wet"] for r in rows])
    sos = np.array([r["sigma_over_s_at_0.25"] for r in rows])
    need = d["f_fraction"] / d["z_delta"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.05))

    ax = axes[0]
    ax.plot(wet, e, marker="o", color=C["model_error"], zorder=3)
    ax.set_yscale("log")
    # A log axis defaults to decade ticks here, which leaves a single labelled
    # value and makes the panel unreadable. Label the values the reader needs.
    ax.set_yticks([2, 3, 5, 10, 20, 40])
    ax.set_yticklabels(["2", "3", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel("berm rows over-topped (of 96)")
    ax.set_ylabel("amplification $E=\\mathrm{d}\\ln D/\\mathrm{d}\\ln Q$")
    ax.set_title("forecast error is amplified at the crest", fontsize=7.5)
    finish(ax)

    ax = axes[1]
    ax.plot(q, sos, marker="o", color=C["residual"], zorder=3, label="$\\sigma/s$")
    ax.axhline(need, ls="--", lw=0.9, color=C["ink"], zorder=2,
               label=f"feasible below $f/z_\\delta={need:.2f}$")
    imin = int(np.argmin(sos))
    ax.annotate("minimum", (q[imin], sos[imin]), textcoords="offset points",
                xytext=(4, 9), fontsize=6.5, color=C["ink"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["ink"]))
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("channel inflow ($10^{-3}$ m s$^{-1}$)")
    ax.set_ylabel("$\\sigma/s$")
    ax.set_title("feasibility squeezed from both ends", fontsize=7.5)
    ax.legend(loc="upper center")
    finish(ax)

    save(fig, "fig9_flood_elasticity")


def fig_flood_precondition() -> None:
    """Proposition 2 as an acceptance test, across hydrograph spread.

    The required margin b + z_delta*sigma against the headroom f*s the actuator
    actually buys. The crossing gives the forecast accuracy a calibrated margin
    demands. Two curves, not one, because the span SHRINKS as spread widens --
    averaging over wider hydrographs pulls in floods the levee cannot stop -- so
    feasibility is squeezed from both sides at once.

    Provenance: runs/flood_precond/sigma_*/precondition.json (§5.6, job 1025566).
    """
    srcs = sorted(glob.glob(str(ROOT / "runs" / "flood_precond" / "*" / "precondition.json")))
    if not srcs:
        print("skip fig_flood_precondition: no runs/flood_precond/*/precondition.json")
        return
    rows = sorted((json.loads(Path(s).read_text()) for s in srcs),
                  key=lambda d: d["config"]["q_log_sigma"])
    x = np.array([d["config"]["q_log_sigma"] for d in rows]) * 100
    need = np.array([d["required_margin"] for d in rows])
    head = np.array([d["headroom"] for d in rows])
    ok = np.array([d["precondition"] == "PASS" for d in rows])

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ax.plot(x, need, marker="o", color=C["model_error"],
            label=r"required $b+z_\delta\sigma$", zorder=3)
    ax.plot(x, head, marker="s", color=C["residual"],
            label=r"headroom $f\!\cdot\!s$", zorder=3)
    ax.fill_between(x, head, need, where=need > head, color=C["model_error"],
                    alpha=0.13, lw=0, zorder=1)
    # Crossing, by linear interpolation between the last PASS and first FAIL.
    i = int(np.argmax(~ok)) if (~ok).any() else None
    if i:
        d0, d1 = (need - head)[i - 1], (need - head)[i]
        xc = x[i - 1] + (x[i] - x[i - 1]) * (-d0) / (d1 - d0)
        ax.axvline(xc, ls=":", lw=0.9, color=C["ink"], zorder=2)
        # Anchor the callout in AXES-fraction y, not data y. Two bugs here
        # already: at 62% of the height it printed through "headroom f.s", and
        # a data-coordinate placement read get_ylim() before set_yscale("log"),
        # resolving against linear limits to y=0 -- off-scale, so it vanished.
        ax.annotate(f"crossing {xc:.1f}%", xy=(xc, 0.0),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(4, 7), fontsize=6.8,
                    color=C["ink"])
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.4"])
    ax.minorticks_off()
    ax.set_xlabel("discharge forecast spread (%, log-normal)")
    ax.set_ylabel("depth (m)")
    ax.set_title("a margin fits only below ${\\sim}8\\%$ forecast spread", fontsize=7.5)
    ax.legend(loc="upper left")
    finish(ax)
    save(fig, "fig10_flood_precondition")


# --------------------------------------------------------------------------- #
def fig_matched_budget() -> None:
    """The head-to-head the sample-efficiency claim has to be made as.

    §3.4n, Vista job 1049073. Model-free baselines capped at the planner's OWN
    real-sample count rather than the 38,400 they are customarily given; 20
    seeds; every arm paired against the zero policy on its own initial
    conditions. 20,000-resample bootstrap.

    A ratio ("12.5x fewer transitions") is an input, not a result. This is the
    result: what each method achieves at the same budget.
    """
    rdf = [("ours (probe + conformal)", 0.5008, 0.451, 0.556, 5.0),
           ("Sauté RL",                 0.1073, 0.062, 0.156, 0.0),
           ("PPO-Lagrangian",           0.0843, 0.046, 0.126, 0.0),
           ("CPO",                      0.0405, 0.019, 0.064, 0.0),
           ("primal-dual NPG",          0.0405, 0.019, 0.065, 0.0)]
    dar = [("ours (probe + margin)", 0.0866, 0.077, 0.096, 0.5),
           ("Sauté RL",              0.0066, -0.002, 0.016, 0.0),
           ("CPO",                   0.0052, -0.001, 0.011, 0.0),
           ("primal-dual NPG",       0.0052, -0.001, 0.011, 0.0),
           ("PPO-Lagrangian",        0.0001, -0.009, 0.009, 0.0)]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95))
    for ax, rows, title, xlim in (
            (axes[0], rdf, r"(a) rdf, $\approx$8,300 real transitions each", (-0.03, 0.70)),
            (axes[1], dar, r"(b) dar, $\approx$4,000 real transitions each", (-0.02, 0.118))):
        ys = np.arange(len(rows))[::-1]
        for y, (name, mu, lo, hi, viol) in zip(ys, rows):
            ours = name.startswith("ours")
            col = C["residual"] if ours else C["none"]
            ax.plot([lo, hi], [y, y], color=col, lw=1.9 if ours else 1.2,
                    solid_capstyle="round", zorder=2)
            ax.plot([mu], [y], "o" if ours else "s", color=col,
                    ms=4.0 if ours else 2.8, zorder=3)
            ax.text(hi + 0.012 * (xlim[1] - xlim[0]), y, f"{viol:.1f}%",
                    va="center", fontsize=6.0, color=C["ink"])
        ax.axvline(0, color=C["model_error"], lw=0.8, ls="--", zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
        ax.set_xlim(*xlim)
        ax.set_xlabel("return over the zero policy (95% CI)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=C["rule"], lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig11_matched_budget")


def fig_trigger_frontier() -> None:
    """A one-degree-of-freedom controller has three reachable behaviours.

    `baselines/run_evac_frontier.py`, 70 thresholds over [0.30, 0.99], 4 seeds
    x 64 episodes each, on the Onion Creek egress table (§5.14x). Read from
    data/deck/evac_frontier.json so re-running the sweep updates the figure.
    """
    src = ROOT / "data" / "deck" / "evac_frontier.json"
    if not src.exists():
        print("skip fig12: no data/deck/evac_frontier.json")
        return
    d = json.loads(src.read_text())
    curve = sorted(d["curve"], key=lambda r: r[2])
    cost = [r[2] for r in curve]
    ret = [r[1] for r in curve]

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.plot(cost, ret, "o", color=C["none"], ms=2.4, zorder=2)
    levels = sorted({round(r, 1) for r in ret}, reverse=True)
    for lv in levels:
        xs = [c for c, r in zip(cost, ret) if round(r, 1) == lv]
        ax.plot([min(xs)], [lv], "o", color=C["residual"], ms=4.5, zorder=4)
    span = max(ret) - min(ret)
    ax.axvspan(6.51, 15.19, color=C["model_error"], alpha=0.10, zorder=1)
    ax.annotate("", xy=(6.51, min(ret) - 0.17 * span),
                xytext=(15.19, min(ret) - 0.17 * span),
                arrowprops=dict(arrowstyle="<->", color=C["model_error"], lw=0.8))
    ax.text(10.85, min(ret) - 0.13 * span, r"cost $\times 2.3$, outcome fixed",
            ha="center", fontsize=6.2, color=C["model_error"])
    ax.set_ylim(min(ret) - 0.24 * span, max(ret) + 0.10 * span)
    ax.set_xlabel("disruption (people-intervals under order)")
    ax.set_ylabel("episode return")
    ax.grid(color=C["rule"], lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig12_trigger_frontier")


def fig_saturation() -> None:
    """Why a constrained planner can diverge while its dual is working.

    One run of `HybridPlannerTrainer` on the polder environment (§5.14y), with
    the saturation guard off (the package default) and on. Traces read from
    data/deck/collapse_v{1,2}.json.
    """
    a = ROOT / "data" / "deck" / "collapse_v1.json"
    b = ROOT / "data" / "deck" / "collapse_v2.json"
    if not (a.exists() and b.exists()):
        print("skip fig13: no collapse traces")
        return
    v1, v2 = json.loads(a.read_text()), json.loads(b.read_text())

    def series(rows, key):
        xs = [r["iteration"] for r in rows if r.get(key) is not None]
        ys = [r[key] for r in rows if r.get(key) is not None]
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.75))
    panels = (("train/pre_tanh_mean_abs", r"$|\mu|$ before squashing", True,
               "(a) the policy leaves the responsive region"),
              ("train/episode_cost", "episode cost", False,
               "(b) and pins at the action ceiling"),
              ("lambda", r"dual multiplier $\lambda$", False,
               "(c) while the dual winds up against it"))
    for ax, (key, lab, logy, title) in zip(axes, panels):
        for rows, col, name in ((v1, C["model_error"], "guard off"),
                                (v2, C["residual"], "guard on")):
            xs, ys = series(rows, key)
            if ys:
                ax.plot(xs, ys, color=col, lw=1.1, label=name)
        if logy:
            ax.set_yscale("log")
            ax.axhline(3.0, color=C["ink"], lw=0.6, ls=":")
        if key == "train/episode_cost":
            ax.axhline(2.0, color=C["ink"], lw=0.6, ls="--")
        ax.set_xlabel("iteration")
        ax.set_ylabel(lab)
        ax.set_title(title, loc="left", fontsize=7.2)
        ax.grid(color=C["rule"], lw=0.4)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, "fig13_saturation")


def fig_warning_lead() -> None:
    """Instrument availability is part of the hazard model, not a detail.

    Lag between the upstream gauge's peak and the outlet's, on the six largest
    floods in the outlet's record since the upstream gauge was built (§5.14z).
    Measured from USGS instantaneous values by `service/app.py::warning`.
    """
    rows = [("2013-10-31", 3822.8, None), ("2015-10-30", 3454.7, None),
            ("2019-05-04", 764.6, 285), ("2004-11-17", 535.2, 275),
            ("2015-05-24", 427.6, 335), ("2007-01-13", 404.9, 20)]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ys = np.arange(len(rows))[::-1]
    for y, (date, q, lag) in zip(ys, rows):
        if lag is None:
            ax.barh(y, 470, color=C["model_error"], alpha=0.18, height=0.62, zorder=2)
            ax.text(235, y, "no usable record", ha="center", va="center",
                    fontsize=6.2, color=C["model_error"])
        else:
            col = C["episode"] if lag < 120 else C["residual"]
            ax.barh(y, lag, color=col, height=0.62, zorder=2)
            ax.text(lag + 10, y, f"{lag}", va="center", fontsize=6.2)
    ax.axvline(120, color=C["ink"], lw=0.8, ls="--", zorder=3)
    ax.set_ylim(-1.15, len(rows) - 0.4)
    ax.text(126, -0.8, "lead required", fontsize=6.2, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d[:4]}  {q:,.0f}" for d, q, _ in rows], fontsize=6.4)
    ax.set_xlabel("gauge lead time (minutes)")
    ax.set_xlim(0, 500)
    ax.grid(axis="x", color=C["rule"], lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig14_warning_lead")


def fig_polder_controllers() -> None:
    """Feasibility is a precondition for ranking, not a column in the table.

    Polder gate-and-pump control at a 2.0 pump-hour budget (§5.14y). Costs are
    `episode_cost` of the final policy -- NOT the mean over training, which is
    a different policy and was the statistic an earlier draft reported.
    """
    src = ROOT / "data" / "deck" / "polder_controllers.json"
    if not src.exists():
        print("skip fig15: no polder comparison")
        return
    look = {r["name"]: r for r in json.loads(src.read_text()) if "return" in r}
    want = [("do nothing", "fixed"), ("gate only", "fixed"),
            ("level deadband (incumbent)", "fixed"),
            ("pump while sealed", "fixed"), ("pump flat out", "fixed"),
            ("ppo_lagrangian", "rl"), ("cpo", "rl"), ("saute", "rl"),
            ("primal_dual_npg", "rl"), ("hybrid planner", "ours")]
    budget = 2.0
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    off = {"gate only": (4, -8), "level deadband (incumbent)": (-4, 5),
           "do nothing": (5, -3), "pump while sealed": (-4, 5),
           "pump flat out": (-4, -9), "hybrid planner": (-4, 5),
           "ppo_lagrangian": (4, -8), "cpo": (4, 3),
           "primal_dual_npg": (4, 3), "saute": (4, -3)}
    for n, kind in want:
        if n not in look:
            continue
        r, c = look[n]["return"], look[n]["cost"]
        ok = c <= budget + 1e-6
        col = {"fixed": C["none"], "rl": C["episode"], "ours": C["residual"]}[kind]
        ax.plot([c], [r], "o" if ok else "x", ms=4.2 if ok else 5.0,
                color=col, mew=1.2, zorder=3)
        ax.annotate(n.replace("_", "-"), (c, r), textcoords="offset points",
                    xytext=off.get(n, (4, 4)), fontsize=5.8, color=col)
    ax.axvline(budget, color=C["model_error"], lw=0.9, ls="--")
    ax.axvspan(budget, 6.8, color=C["model_error"], alpha=0.06)
    ax.set_xlim(-0.5, 6.8)
    ax.set_xlabel("pump-hours spent")
    ax.set_ylabel("episode return")
    ax.grid(color=C["rule"], lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save(fig, "fig15_polder_controllers")


def main() -> None:
    paper_style()
    fig_calibration()
    fig_mechanism()
    fig_rdf()
    fig_precondition()
    fig_wildfire()
    fig_tradeoff()
    fig_ablation()
    fig_partial()
    fig_twin()
    fig_flood_elasticity()
    fig_flood_precondition()
    fig_matched_budget()
    fig_trigger_frontier()
    fig_saturation()
    fig_warning_lead()
    fig_polder_controllers()


if __name__ == "__main__":
    main()
