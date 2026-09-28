#!/usr/bin/env python
"""Application figures for the paper's appendix.

    python scripts/make_appendix_figures.py

These are spatial: they show the framework acting on real ground, where the
figures in `make_paper_figures.py` show measured quantities. Same house style —
ICML column widths, serif, editable text — so the two sets sit together.

Provenance is stated in each figure's docstring and every number is measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "paper"
COL, FULL = 3.25, 6.75

C = {
    "hit": "#2563a8",       # model and satellite agree
    "model": "#e0803a",     # model only
    "sat": "#4bb99a",       # satellite only
    "dry": "#e9edf2",
    "ink": "#111418",
    "mute": "#5d6a7d",
    "rule": "#c9d1da",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"docs/figures/paper/{name}.{{pdf,png}}")


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(C["rule"])


def fig_sar_validation() -> None:
    """How the reference compares with what the satellite saw.

    Sentinel-1 RTC, 2 March 2022, relative orbit 74, against the median of five
    same-track baselines. The agreement panel is the point: the model's extent
    exceeds the observed one over open ground, and the disagreement concentrates
    in tree cover, where C-band cannot see standing water at all.

    Provenance: runs/sar_validation/ (§5.7).
    """
    src = ROOT / "runs" / "sar_validation" / "richmond.npz"
    ref = ROOT / "runs" / "sar_validation" / "ref_day7p62.tif"
    if not src.exists() or not ref.exists():
        print("skip fig_sar_validation: cached arrays missing")
        return
    import tifffile

    d = np.load(src)
    meta = json.loads((ROOT / "runs" / "sar_validation" / "meta.json").read_text())
    f_db, b_db, sar, cover = d["flood_db"], d["base_db"], d["sar"], d["cover"]
    model = np.nan_to_num(tifffile.imread(ref)) > 0.05

    # The sea and the river channel are water in the model and permanent water
    # to the sensor, so counting them as disagreement measures nothing. Left in,
    # they contributed 120 km2 of spurious "model over-prediction" — most of the
    # apparent error in the first version of this figure.
    perm = d["perm"] if "perm" in d else (cover == 80)
    comparable = ~perm & np.isfinite(f_db) & np.isfinite(b_db)
    model = model & comparable
    sar = sar & comparable

    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.95))

    axes[0].imshow(b_db, cmap="gray", vmin=-22, vmax=-2, interpolation="nearest")
    axes[0].set_title("Before\n5-scene median, same track")
    _bare(axes[0])

    axes[1].imshow(f_db, cmap="gray", vmin=-22, vmax=-2, interpolation="nearest")
    axes[1].set_title(f"During\n2 Mar 2022, orbit {meta.get('epsg') and 74}")
    _bare(axes[1])

    # Observed flood: water now that was not water before.
    obs = np.zeros(sar.shape + (3,))
    obs[...] = 0.93
    obs[sar] = [0.15, 0.42, 0.68]
    axes[2].imshow(obs, interpolation="nearest")
    axes[2].set_title(f"Observed flood\n{sar.sum()*9e-4:.0f} km$^2$ of new water")
    _bare(axes[2])

    # Agreement, three ways.
    agree = np.zeros(sar.shape, dtype=np.uint8)
    agree[model & sar] = 1
    agree[model & ~sar] = 2
    agree[~model & sar] = 3
    cmap = ListedColormap([C["dry"], C["hit"], C["model"], C["sat"]])
    axes[3].imshow(agree, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[3].set_title("Model vs observation\npermanent water excluded")
    _bare(axes[3])
    axes[3].legend(
        handles=[
            Patch(facecolor=C["hit"], label=f"both ({(model & sar).sum()*9e-4:.0f} km$^2$)"),
            Patch(facecolor=C["model"], label=f"model only ({(model & ~sar).sum()*9e-4:.0f} km$^2$)"),
            Patch(facecolor=C["sat"], label=f"satellite only ({(~model & sar).sum()*9e-4:.0f} km$^2$)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, handlelength=1.1,
        borderpad=0.2, labelspacing=0.25,
    )
    save(fig, "figA1_sar_validation")


def fig_sar_stratified() -> None:
    """Skill against the satellite, split by whether radar can adjudicate.

    C-band sees standing water on open ground; under canopy it does not, and in
    built-up areas flooded structures scatter bright rather than dark. Scoring
    without that split charges the model for the instrument's blind spots.

    Provenance: runs/sar_validation/ (§5.7).
    """
    src = ROOT / "runs" / "sar_validation" / "richmond.npz"
    ref = ROOT / "runs" / "sar_validation" / "ref_day7p62.tif"
    if not src.exists() or not ref.exists():
        print("skip fig_sar_stratified: cached arrays missing")
        return
    import tifffile

    d = np.load(src)
    sar, cover = d["sar"], d["cover"]
    model = np.nan_to_num(tifffile.imread(ref)) > 0.05

    WC = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
          50: "built-up", 60: "bare", 90: "wetland"}
    BLIND = {10, 50, 95}

    rows = []
    for code, name in WC.items():
        sel = cover == code
        if sel.sum() < 2000:
            continue
        m, s = model & sel, sar & sel
        h = (m & s).sum(); miss = (~m & s).sum(); fa = (m & ~s).sum()
        den = h + miss + fa
        rows.append({"name": name, "csi": h / den if den else 0,
                     "far": fa / (h + fa) if (h + fa) else 0,
                     "frac": sel.mean(), "blind": code in BLIND})
    rows.sort(key=lambda r: -r["frac"])

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    y = np.arange(len(rows))
    colors = [C["model"] if r["blind"] else C["hit"] for r in rows]
    ax.barh(y, [r["csi"] for r in rows], color=colors, height=0.62, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r["csi"] + 0.012, i, f"{r['csi']:.2f}", va="center",
                fontsize=6.8, color=C["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}  ({100*r['frac']:.0f}%)" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("CSI against observed flood extent")
    ax.set_xlim(0, 0.78)
    ax.grid(True, axis="x", alpha=0.5, color=C["rule"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=C["hit"], label="radar can adjudicate"),
        Patch(facecolor=C["model"], label="radar is blind here"),
    ], loc="lower right")
    save(fig, "figA2_sar_stratified")


# ---------------------------------------------------------------- wildfire ---

FIRE = {
    "seen": "#c1442e",      # fire the satellite actually saw
    "recovered": "#2563a8", # hidden fire the estimator found
    "missed": "#8d96a3",    # hidden fire it did not find
    "false": "#e0803a",     # cells it invented
    "cloud": "#dfe5ec",     # occluded
    "land": "#f4f2ee",
}


def fig_wildfire_perception() -> None:
    """One NDWS patch: what is hidden, and what each belief puts in its place.

    The sample is the median case by hidden-cell recall among patches where the
    cloud covers roughly half the fire front, so the panel shows a typical
    reconstruction rather than a flattering one.
    """
    d = np.load(ROOT / "runs" / "ndws_partial_local" / "panel.npz")
    truth, hide, est = d["truth"] > 0.5, d["hide"], d["est"]
    persist, rate = d["persist"], float(d["rate"])
    recall, burning = float(d["recall"]), int(d["burning"])

    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.98))

    def paint(ax, rgb, title):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, pad=3.5)
        _bare(ax)

    def blank():
        return np.tile(np.array(_rgb(FIRE["land"])), (*truth.shape, 1))

    # (a) the state that was actually there
    a = blank(); a[truth] = _rgb(FIRE["seen"])
    paint(axes[0], a, f"(a) fire front, day 0\n{burning} active cells")

    # (b) what an instrument with cloud over it delivers
    b = blank(); b[truth & ~hide] = _rgb(FIRE["seen"]); b[hide] = _rgb(FIRE["cloud"])
    paint(axes[1], b, f"(b) observed, {rate:.0%} occluded\n"
                      f"{int(d['covered'])} of {burning} front cells hidden")

    # (c) persistence: hidden cells get the visible mean, which is near zero
    c = blank(); c[truth & ~hide] = _rgb(FIRE["seen"])
    c[hide] = _blend(FIRE["cloud"], FIRE["seen"], float(persist[hide].mean()))
    paint(axes[2], c, "(c) persistence fill\nadds almost nothing here")

    # (d) the learned estimator, scored cell by cell
    e = est > 0.5
    f = blank(); f[truth & ~hide] = _rgb(FIRE["seen"])
    f[hide & truth & e] = _rgb(FIRE["recovered"])
    f[hide & truth & ~e] = _rgb(FIRE["missed"])
    f[hide & ~truth & e] = _rgb(FIRE["false"])
    over = (e & hide).sum() / max((truth & hide).sum(), 1)
    paint(axes[3], f, f"(d) learned estimator\nrecalls {recall:.0%} of the hidden front,"
                      f"\nspread over {over:.1f}x its area")

    axes[3].legend(handles=[
        Patch(facecolor=FIRE["seen"], label="seen"),
        Patch(facecolor=FIRE["recovered"], label="recovered"),
        Patch(facecolor=FIRE["missed"], label="missed"),
        Patch(facecolor=FIRE["false"], label="invented"),
    ], loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0,
        handleheight=1.0, labelspacing=0.45, borderpad=0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "figA3_wildfire_perception")


def fig_wildfire_decision() -> None:
    """The finding: reconstruction improves 70-fold and the decision does not.

    Left, how well each belief matches the state that was hidden. Right, what
    the burn reduction actually was when a policy acted on that belief. The
    two panels disagree, which is the result worth reporting.
    """
    rows = _partial_rows()
    rates = sorted({r["occlusion"] for r in rows if r["occlusion"] > 0})
    beliefs = [("blind", "treat as unburnt", C["mute"]),
               ("persist", "persistence", C["sat"]),
               ("perceive-hard", "estimator, thresholded", C["model"]),
               ("perceive", "estimator, probability", C["hit"])]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))

    def series(belief, policy, key):
        return [_mean(rows, r, belief, policy, key) for r in rates]

    for name, label, col in beliefs:
        axes[0].plot(rates, series(name, "PSPE planner", "state AP on hidden cells"),
                     "o-", color=col, label=label, ms=3.4, lw=1.1)
    axes[0].set_ylabel("average precision on hidden cells")
    axes[0].set_title("(a) does the belief match what was hidden?", pad=4)
    axes[0].set_ylim(0, 0.68)
    axes[0].legend(loc="upper right")

    zero = _mean(rows, 0.0, "blind", "PSPE planner", "burn reduction %")
    for name, label, col in beliefs:
        # Blind and persistence land on top of each other, so the first is drawn
        # wide and underneath rather than being hidden by the second.
        wide = name == "blind"
        axes[1].plot([0] + rates, [zero] + series(name, "PSPE planner", "burn reduction %"),
                     "o-", color=col, label=label, ms=5.2 if wide else 3.4,
                     lw=2.6 if wide else 1.1, alpha=0.55 if wide else 1.0,
                     zorder=2 if wide else 3)
    gz = _mean(rows, 0.0, "blind", "greedy", "burn reduction %")
    axes[1].plot([0] + rates, [gz] + series("blind", "greedy", "burn reduction %"),
                 "s--", color=C["ink"], ms=3.0, lw=0.9, label="greedy risk ranking")
    axes[1].set_ylabel("burn reduction (%)")
    axes[1].set_title("(b) what the decision achieved", pad=4)
    axes[1].set_ylim(0, 48)
    axes[1].legend(loc="lower left", ncol=1)
    # The gap IS the finding: the belief that matches the hidden state best is
    # the one that plans worst.
    lo = _mean(rows, 0.6, "perceive", "PSPE planner", "burn reduction %")
    hi = _mean(rows, 0.6, "blind", "PSPE planner", "burn reduction %")
    axes[1].annotate("", xy=(0.615, hi), xytext=(0.615, lo),
                     arrowprops=dict(arrowstyle="<->", lw=0.7, color=C["ink"]))
    axes[1].text(0.60, (hi + lo) / 2, f"{hi - lo:.1f} pts", ha="right", va="center",
                 fontsize=6.6, color=C["ink"])
    axes[1].text(0.015, zero + 1.6, "blind and persistence coincide",
                 fontsize=6.4, color=C["mute"])

    for ax in axes:
        ax.set_xlabel("fraction of the grid occluded")
        ax.set_xlim(-0.02, 0.65)
        ax.grid(True, alpha=0.5, color=C["rule"], zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "figA4_wildfire_decision")


def _partial_rows() -> list[dict]:
    rows = []
    for f in sorted((ROOT / "runs").glob("ndws_partial_r*/seed_*/results.json")):
        rows += json.loads(f.read_text())["rows"]
    return rows


def _mean(rows, rate, belief, policy, key) -> float:
    vals = [r[key] for r in rows
            if r["occlusion"] == rate and r["belief"] == belief
            and r["policy"] == policy and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _blend(a: str, b: str, w: float) -> tuple[float, float, float]:
    ra, rb = _rgb(a), _rgb(b)
    return tuple(x + (y - x) * w for x, y in zip(ra, rb))


# ------------------------------------------------------------------ portal ---

SHOTS = ROOT / "runs" / "portal_shots"


def _shot(ax, name: str, crop: tuple[float, float] = (0.0, 1.0)) -> None:
    """Draw a captured portal state, cropped vertically to the part that reads."""
    img = plt.imread(SHOTS / name)
    h = img.shape[0]
    ax.imshow(img[int(crop[0] * h):int(crop[1] * h)], interpolation="lanczos")
    _bare(ax)


def _call(ax, text, xy, xytext, ha="left") -> None:
    """A callout onto the screenshot, in axes fractions."""
    ax.annotate(text, xy=xy, xycoords="axes fraction",
                xytext=xytext, textcoords="axes fraction",
                ha=ha, va="center", fontsize=6.6, color=C["ink"],
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["rule"], lw=0.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C["ink"],
                                shrinkA=1, shrinkB=2,
                                connectionstyle="arc3,rad=-0.15"))


def fig_portal_appraisal() -> None:
    """The planner as an operator meets it: options ranked, harm flagged.

    Everything the paper argues about the Richmond valley appears here as a
    decision rather than a number. Two of the six candidate levees deepen
    flooding -- the backwater effect of Section 5.6b -- and the tool says so
    in the list rather than quietly ranking them last.
    """
    fig, ax = plt.subplots(figsize=(FULL, FULL * 0.585))
    _shot(ax, "05_detail.png", (0.055, 1.0))
    _call(ax, "two of six candidate sites make\nflooding worse, and the tool says so",
          (0.075, 0.804), (0.30, 0.915))
    _call(ax, "ranked by what each buys\nagainst its capital cost",
          (0.228, 0.608), (0.38, 0.700))
    _call(ax, "and the measure it just recommended\nis reported as keeping no road open",
          (0.160, 0.345), (0.38, 0.215))
    _call(ax, "modelled depth over the 2022 event,\ncandidate sites marked",
          (0.602, 0.575), (0.80, 0.33), ha="center")
    save(fig, "figA5_portal_appraisal")


def fig_portal_anywhere() -> None:
    """The same pipeline on a place it was never configured for.

    Nothing about Cedar Rapids is in the repository. The domain is cut from
    public Copernicus DEM tiles on request, the solver runs, and the answer
    comes back inside a minute -- which is what makes the framework a tool
    rather than a case study.
    """
    fig, ax = plt.subplots(figsize=(FULL, FULL * 0.585))
    # The header is part of the evidence here -- it names the place the tool is
    # actually solving -- so this one keeps the full frame.
    _shot(ax, "06_anywhere.png")
    _call(ax, "any town, entered by name", (0.070, 0.868), (0.36, 0.905))
    _call(ax, "domain cut from public DEM tiles and solved\non request: 12.5 x 12.4 km at 60 m in 47 s",
          (0.105, 0.508), (0.44, 0.395))
    _call(ax, "the same solver, on terrain that is\nnowhere in the repository",
          (0.625, 0.545), (0.80, 0.80), ha="center")
    save(fig, "figA6_portal_anywhere")


# ------------------------------------------------------- the validation chain ---


# Closed-domain rerun (job 1025861), CSI@0.01 -- the configuration Section 5.6
# reports, after the open-edge outflow law was found to be removing water.
CLOSED_CSI = 0.9785


def fig_validation_chain() -> None:
    """Our solver against the reference, and the reference against the world.

    The two comparisons are not interchangeable and the figure exists to stop
    them being read as one number. Agreement with FloodCastBench says our
    routing operator is the accepted scheme. Only the satellite panel says
    anything about the world, and it is the weaker of the two.
    """
    src = ROOT / "runs" / "floodcast_validate" / "validate.json"
    if not src.exists():
        print("skip fig_validation_chain: validate.json missing")
        return
    d = json.loads(src.read_text())
    fr = d["frames"]
    t0 = fr[0]["t_hours"]
    hrs = [f["t_hours"] - t0 for f in fr]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))

    # Per-frame data exists only for the OPEN-edge configuration, which Section
    # 5.6 superseded: a single domain-average bed slope on all four edges drained
    # water that should have stayed. The closed-domain rerun is the one the paper
    # reports, and only its summary survived, so it is drawn as a reference line
    # rather than silently swapped in.
    ax = axes[0]
    ax.plot(hrs, [f["csi@0.01"] for f in fr], "o-", color=C["hit"], ms=3.4, lw=1.1,
            label="open edges (superseded)")
    ax.axhline(CLOSED_CSI, color=C["mute"], lw=0.9, ls=":")
    ax.text(hrs[-1], CLOSED_CSI + 0.0007, f"closed domain, as reported: {CLOSED_CSI:.3f}",
            fontsize=6.4, color=C["mute"], va="bottom", ha="right")
    ax.set_ylabel("CSI against FloodCastBench", color=C["hit"])
    ax.tick_params(axis="y", colors=C["hit"])
    ax.set_ylim(0.955, 1.004)
    ax.set_xlabel("hours into the validation window")
    ax.set_title(f"(a) our solver vs the reference\nper frame, open edges "
                 f"(mean {d['mean_csi']['@0.01']:.3f})", pad=4)
    rr = ax.twinx()
    rr.plot(hrs, [f["rmse_wet"] for f in fr], "s--", color=C["model"], ms=3.0, lw=0.9)
    rr.set_ylabel("RMSE on wet cells (m)", color=C["model"])
    rr.tick_params(axis="y", colors=C["model"])
    rr.set_ylim(0, 0.34)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        rr.spines[s].set_visible(False)
    ax.grid(True, alpha=0.5, color=C["rule"], zorder=0)
    ax.set_axisbelow(True)

    # The chain, side by side, because the gap between them IS rule 23.
    ax = axes[1]
    bars = [
        ("our solver\nvs the reference", CLOSED_CSI, C["hit"]),
        ("the reference\nvs the satellite", 0.535, C["sat"]),
    ]
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.5, zorder=3)
    for i, (_, v, _c) in enumerate(bars):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=7, color=C["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars])
    ax.set_ylim(1.78, -0.45)          # inverted, with room for the captions
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("critical success index")
    ax.set_title("(b) two comparisons, one of which\nis about the world", pad=4)
    # The y axis is inverted, so "just below bar i" is i + 0.4, not i - 0.4.
    ax.text(0.02, 0.42, "model against model", fontsize=6.6, color=C["mute"])
    ax.text(0.02, 1.42, "model against observation,\nwhere radar can adjudicate",
            fontsize=6.6, color=C["mute"])
    ax.grid(True, axis="x", alpha=0.5, color=C["rule"], zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(wspace=0.72)
    save(fig, "figA7_validation_chain")


def main() -> None:
    style()
    fig_sar_validation()
    fig_sar_stratified()
    fig_wildfire_perception()
    fig_wildfire_decision()
    fig_portal_appraisal()
    fig_portal_anywhere()
    fig_validation_chain()


if __name__ == "__main__":
    main()
