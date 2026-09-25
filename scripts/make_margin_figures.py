#!/usr/bin/env python
"""Figures for the conformal-margin head-to-head.

    python scripts/make_margin_figures.py

Panel (a) is the controlled study (`eval/run_margin_synthetic.py`): bias and
spread dialled independently, nothing learned, so the crossover can be seen
against the algebra. Panel (b) is the PDE fidelity sweep
(`eval/run_margin_choice.py`), drawn only once its results exist.

The claim the figure has to make legible: matched-pair conformalisation covers
when the model's per-instance error is at least the policy's episode spread,
and undercovers once the policy is the more variable of the two. Coverage at
level delta needs a margin of bias + z_delta * sigma; the matched-pair quantile
supplies about bias + z_delta * eps; the bias cancels, so the crossover sits at
rho = sigma / eps = 1.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
STYLE = {
    "none":        ("#9aa0a6", "s", "no margin"),
    "model_error": ("#a12a2a", "^", "conformalise model error (the field's default)"),
    "episode":     ("#c9a02a", "D", "conformalise deviations, plus bias"),
    "residual":    ("#1f6b4f", "o", "conformalise against the controlled quantity"),
}
FONT = dict(family="DejaVu Sans")
DELTA = 0.1


def panel_synthetic(ax, rows):
    rho = [r["rho = spread/model err"] for r in rows]
    for arm, (c, m, label) in STYLE.items():
        ax.plot(rho, [100 * r[arm] for r in rows], marker=m, ms=6, lw=2.0,
                color=c, label=label)
    ax.axhline(100 * DELTA, color="#1f6b4f", ls="--", lw=1.4)
    ax.axvline(1.0, color="#444444", ls=":", lw=1.4)
    ax.text(1.12, 62, "crossover predicted\nat rho = 1", fontsize=8.4,
            color="#444444", va="center", linespacing=1.4, **FONT)
    ax.text(0.02, 100 * DELTA + 1.5, f" stated rate, delta = {DELTA}", fontsize=8.6,
            color="#1f6b4f", va="bottom", transform=ax.get_yaxis_transform(), **FONT)
    ax.set_xscale("log")
    ax.set_xlabel("rho = policy episode spread / model per-instance error", fontsize=9.6)
    ax.set_ylabel("realised violation rate (%)", fontsize=9.6)
    ax.set_title("(a) Controlled study: the crossover is where the algebra puts it\n"
                 "600 calibration draws per point, nothing learned",
                 fontsize=10.6, weight="bold", pad=10)
    ax.set_ylim(-3, 95)
    ax.legend(fontsize=8.2, frameon=False, loc="center left")


def panel_pde(ax, rows):
    """The margin each recipe produces, against rho, on a real planner.

    Not the violation rate. With eval_every 20 over 200 iterations there are 11
    evaluations per run, so every rate is a multiple of 1/11 and delta sits
    between 1/11 and 2/11; pooled over 15 runs the ordering comes out right
    (none 0.133, model_error 0.115, episode 0.091, residual 0.085) but at
    z = 1.41 it is not separable from noise. Plotting it produced a scatter
    with no structure, which is an honest picture of an underpowered metric and
    a misleading picture of the claim.

    The margin is continuous, immune to that resolution limit, and measures the
    mechanism rather than its consequence: as the model improves relative to the
    policy's spread, a quantile over model error shrinks toward nothing while a
    quantile against the controlled quantity does not.
    """
    by = defaultdict(list)
    for r in rows:
        rho = r.get("rho = sigma/eps")
        if rho is None or rho != rho or r["arm"] == "none":
            continue
        by[r["arm"]].append((rho, r["limit"] - r["d_eff"]))
    for arm, (c, m, label) in STYLE.items():
        pts = sorted(by.get(arm, []))
        if not pts:
            continue
        xs = np.array([q[0] for q in pts]); ys = np.array([q[1] for q in pts])
        ax.scatter(xs, ys, marker=m, s=46, color=c, label=label, zorder=3, alpha=0.85)
        if len(xs) >= 4:
            k = max(2, len(xs) // 4)
            ax.plot(xs[k - 1:], np.convolve(ys, np.ones(k) / k, mode="valid"),
                    lw=1.8, color=c, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rho = policy episode spread / model per-instance error, measured per run",
                  fontsize=9.6)
    ax.set_ylabel("margin the recipe produces (limit is 3.26)", fontsize=9.6)
    ax.set_title("(b) Reaction-diffusion front: the default's margin collapses\n"
                 "15 runs; ours is flat because it tracks the policy, not the model",
                 fontsize=10.6, weight="bold", pad=10)
    ax.legend(fontsize=8.2, frameon=False, loc="lower left")


def panel_perturb(ax, rows):
    """Violation rate against b/s, the feasibility ratio.

    A margin has to cover the systematic bias b and still leave the tightened
    limit reachable, which needs b < s. Left of the dashed line the recipes
    that bound the right distribution should hold delta; right of it every
    recipe must fail, because the constraint is unachievable rather than
    miscalibrated. That distinction is invisible in a violation rate alone,
    which is why the ratio is on the axis.
    """
    by = defaultdict(list)
    for r in rows:
        ratio = r.get("bias", float("nan")) / max(r.get("span", float("nan")), 1e-12)
        by[(round(ratio, 3), r["arm"])].append(r)
    ratios = sorted({k[0] for k in by})
    for arm, (c, m, label) in STYLE.items():
        xs = [t for t in ratios if (t, arm) in by]
        if not xs:
            continue
        ys = [100 * np.mean([r["violating batches"] for r in by[(t, arm)]]) for t in xs]
        es = [100 * np.std([r["violating batches"] for r in by[(t, arm)]]) for t in xs]
        ax.errorbar(xs, ys, yerr=es, marker=m, ms=6, lw=2.0, capsize=3, color=c, label=label)
    ax.axhline(100 * DELTA, color="#1f6b4f", ls="--", lw=1.4)
    ax.axvline(1.0, color="#a12a2a", ls="--", lw=1.6)
    ax.text(1.06, 55, "b = s\nbeyond this the limit\nis unreachable at all",
            fontsize=8.2, color="#a12a2a", va="center", linespacing=1.4, **FONT)
    ax.set_xscale("log")
    ax.set_xlabel("b / s : model-reality bias over the span the planner controls",
                  fontsize=9.6)
    ax.set_ylabel("evaluation batches violating the limit (%)", fontsize=9.6)
    ax.set_title("(c) Real fire dynamics: a margin needs room to act\n"
                 "discrepancy dialled, not hoped for", fontsize=10.6, weight="bold", pad=10)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")


def main() -> int:
    syn = ROOT / "runs" / "margin_synthetic" / "results.json"
    # Prefer the eval_every 5 rerun: at 11 evaluations per run the violation
    # rate is quantised to multiples of 1/11 and delta is not attainable.
    hires = sorted((ROOT / "runs" / "margin_choice_hires").glob("*/results.json"))
    pde = hires or sorted((ROOT / "runs" / "margin_choice").glob("*/results.json"))
    if hires:
        print(f"  (using {len(hires)} high-resolution runs)")
    if not syn.exists():
        print("run eval/run_margin_synthetic.py first"); return 1
    srows = json.loads(syn.read_text())
    prows = [r for f in pde for r in json.loads(f.read_text())]
    prt = sorted((ROOT / "runs" / "margin_perturb").glob("*/results.json")) + \
          sorted((ROOT / "runs" / "ndws_perturb").glob("*/results.json"))
    qrows = [r for f in prt for r in json.loads(f.read_text()).get("rows", [])]

    panels = [("a", panel_synthetic, srows)]
    if prows:
        panels.append(("b", panel_pde, prows))
    if qrows:
        panels.append(("c", panel_perturb, qrows))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0 * len(panels), 5.0), squeeze=False)
    for ax, (_, fn, data) in zip(axes[0], panels):
        fn(ax, data)
    if not prows:
        print("  (no PDE fidelity results yet)")
    if not qrows:
        print("  (no perturbation results yet)")
    for ax in axes[0]:
        ax.grid(color="#c8ccd0", lw=0.8, alpha=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=9.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"pspe_margin_choice.{ext}", bbox_inches="tight",
                    dpi=220 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote docs/figures/pspe_margin_choice.* "
          f"({len(srows)} synthetic, {len(prows)} PDE, {len(qrows)} perturbation rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
