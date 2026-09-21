#!/usr/bin/env python
"""Render the PSPE architecture figure (paper Fig. 1 replacement).

    python scripts/make_architecture_figure.py

Writes docs/figures/pspe_architecture.{svg,pdf,png}. Pure matplotlib, so the
figure regenerates from this file rather than living as an opaque asset; edit
the layout constants below and rerun.

Layout: the four modules left to right along the top, the true environment
along the bottom, and the two feedback paths that distinguish the current
system from the paper's original diagram drawn explicitly:

  * the reality probe - real cost and measured pathwise bias fed back into the
    dual and the mixing coefficient, which is what closed the 7.3% violation;
  * the joint-training gradient - the planning loss flowing back into the
    surrogate, data-anchored, which is contribution #1 made concrete.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

# Palette — one hue per module, muted enough to print.
C = {
    "perceive": ("#dbe9f6", "#2b5d8c"),
    "simulate": ("#d7efe6", "#1f6b4f"),
    "plan":     ("#fbe6d0", "#b0561a"),
    "explain":  ("#e8dff3", "#5b3f85"),
    "env":      ("#f0f0f0", "#444444"),
    "probe":    ("#fde2e2", "#a12a2a"),
    "ink":      "#222222",
    "muted":    "#666666",
}

FONT = dict(family="DejaVu Sans")
plt.rcParams["mathtext.fontset"] = "cm"


def box(ax, x, y, w, h, fill, edge, title, lines, *, dashed=False, title_size=11, body_size=10.4):
    """Rounded module box with a coloured title bar and body text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.3, ls="--" if dashed else "-"))
    bar_h = 0.42
    ax.add_patch(FancyBboxPatch((x, y + h - bar_h), w, bar_h,
                                boxstyle="round,pad=0,rounding_size=0.12", fc=edge, ec=edge, lw=0))
    ax.text(x + w / 2, y + h - bar_h / 2, title, ha="center", va="center", color="white",
            fontsize=title_size, weight="bold", **FONT)
    ty = y + h - bar_h - 0.22
    for line in lines:
        style = dict(color=C["ink"], fontsize=body_size)
        if line.startswith("~"):          # muted annotation
            line, style = line[1:], dict(color=C["muted"], fontsize=body_size - 0.6, style="italic")
        ax.text(x + 0.14, ty, line, ha="left", va="top", **style, **FONT)
        ty -= 0.44


def chip(ax, x, y, w, text, edge, size=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.3, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="white", ec=edge, lw=1.0))
    ax.text(x + w / 2, y + 0.15, text, ha="center", va="center", fontsize=size, color=edge, **FONT)


def arrow(ax, p, q, label=None, *, color=None, dashed=False, rad=0.0, lw=1.4, label_dy=0.13,
          label_size=8):
    color = color or C["ink"]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom", fontsize=label_size,
                color=color, **FONT,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))


def stacked_frames(ax, x, y, w, h, n=3):
    for i in range(n):
        off = 0.09 * (n - 1 - i)
        ax.add_patch(Rectangle((x + off, y - off), w, h, fc="#f7f7f7", ec=C["muted"], lw=0.9))
    # a crude "field": diagonal gradient bands
    for k in range(4):
        ax.add_patch(Rectangle((x + 0.08 + 0.14 * k, y + 0.12), 0.1, h - 0.24,
                               fc=["#c9dff2", "#9cc2e5", "#5f97c9", "#2b5d8c"][k], ec="none"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # -- observations ------------------------------------------------------- #
    stacked_frames(ax, 0.45, 4.35, 1.1, 1.25)
    ax.text(1.0, 4.05, r"$O_t$ : imagery, sensors", ha="center", va="top", fontsize=8.2,
            color=C["ink"], **FONT)

    # -- the four modules --------------------------------------------------- #
    Y, H = 3.55, 2.7
    box(ax, 2.2, Y, 2.7, H, *C["perceive"], "PERCEIVE",
        ["frozen vision backbone", "+ LoRA adapters", r"+ field decoder $\to \hat z_t$",
         "+ contrastive text head"])
    box(ax, 5.55, Y, 2.95, H, *C["simulate"], "SIMULATE",
        [r"$\hat u_{t+1} = G_\theta(\hat u_t, a_t)$",
         "FNO  —  smooth fields", "U-Net  —  sharp fronts",
         "loss: data · physics · rollout"])
    box(ax, 9.15, Y, 2.95, H, *C["plan"], "PLAN",
        [r"$\pi_\phi(a_t \mid \hat z_t)$ over actuators",
         r"$\nabla = \alpha\,\nabla_{\rm path} + (1-\alpha)\,\nabla_{\rm LR}$",
         r"$\alpha^* = V_L\,/\,(B^2 + V_p + V_L)$",
         r"PID dual $\lambda \leftarrow$ real cost"])
    box(ax, 12.75, Y, 2.85, H, *C["explain"], "EXPLAIN",
        [r"LM + LoRA $\to$ brief $b_t$", r"frozen parser $\to \hat\pi_b$",
         r"$F(b) = \exp[-\mathrm{KL}(\pi_\phi \,\|\, \hat\pi_b)]$",
         r"$\Pr[\,F \geq 1-\hat s\,] \geq 1-\delta$"])

    # constraint chips under PLAN
    chip(ax, 9.25, Y - 0.42, 0.9, "budget", C["plan"][1])
    chip(ax, 10.2, Y - 0.42, 0.9, "safety", C["plan"][1])
    chip(ax, 11.15, Y - 0.42, 0.95, "equity", C["plan"][1])

    # -- forward arrows along the top ---------------------------------------- #
    mid = Y + H / 2
    arrow(ax, (1.65, mid + 0.2), (2.2, mid + 0.2), r"$O_t$", label_size=9)
    arrow(ax, (4.9, mid + 0.25), (5.55, mid + 0.25), r"$\hat z_t$", label_size=9)
    arrow(ax, (8.5, mid + 0.25), (9.15, mid + 0.25), r"$\hat u_{t+1:H}$", label_size=9)
    arrow(ax, (12.1, mid + 0.25), (12.75, mid + 0.25), r"$a_t,\ \hat z_t$", label_size=8.5)
    arrow(ax, (15.6, mid + 0.25), (16.3, mid + 0.25))
    ax.text(15.95, mid + 0.5, r"$b_t$", ha="center", fontsize=9.5, **FONT)

    # joint-training gradient: PLAN -> SIMULATE (backward, dashed)
    arrow(ax, (9.15, Y + 0.95), (8.5, Y + 0.95), color=C["plan"][1], dashed=True, rad=0.0)
    ax.text(8.82, Y + 0.82, r"$\beta\,\nabla_{\!\theta}\mathcal{L}_{\rm plan}$",
            ha="center", va="top", fontsize=8.5, color=C["plan"][1], **FONT)

    # -- the true environment ----------------------------------------------- #
    ex, ey, ew, eh = 4.6, 0.45, 8.9, 1.75
    box(ax, ex, ey, ew, eh, *C["env"], "TRUE ENVIRONMENT  F   (numerical solver  /  real system)",
        [r"$u_{t+1} = F(u_t, a_t) + \eta$        probed, never differentiated through"],
        title_size=9.5)

    # action applied to the environment
    arrow(ax, (10.7, Y - 0.75), (10.7, ey + eh), r"apply $a_t$", rad=0.0, label_dy=-0.02, label_size=8.5)
    # observations come back to the front
    ax.plot([ex + 0.6, ex + 0.6, 1.0], [ey, 0.18, 0.18], color=C["perceive"][1], lw=1.4)
    arrow(ax, (1.0, 0.18), (1.0, 3.7), color=C["perceive"][1], rad=0.0)
    ax.text(3.0, 0.22, r"new observation  $O_{t+1}$", ha="center", va="bottom", fontsize=7.6,
            color=C["perceive"][1], **FONT)

    # -- the reality probe --------------------------------------------------- #
    px, py, pw, ph = 13.9, 0.45, 2.3, 1.75
    box(ax, px, py, pw, ph, *C["probe"], "REALITY PROBE",
        [r"real cost $J_C \to \lambda$", r"margin $d - k\sigma$",
         r"bias $B^2 \to \alpha^*$"], title_size=9, body_size=9)
    arrow(ax, (ex + ew, ey + eh / 2), (px, py + ph / 2), color=C["probe"][1])
    arrow(ax, (px + pw / 2, py + ph), (11.9, Y - 0.6), color=C["probe"][1], rad=-0.25)
    ax.text(13.2, 2.75, r"$\lambda,\ B^2$", ha="center", fontsize=8, color=C["probe"][1], **FONT,
            bbox=dict(fc="white", ec="none", pad=1.2))

    # -- legend -------------------------------------------------------------- #
    lx, ly = 13.3, 6.32
    ax.add_patch(Rectangle((lx, ly + 0.9), 0.35, 0.18, fc="white", ec=C["ink"], lw=1.2))
    ax.text(lx + 0.45, ly + 0.99, "trainable", va="center", fontsize=7.6, **FONT)
    ax.add_patch(Rectangle((lx, ly + 0.55), 0.35, 0.18, fc="white", ec=C["ink"], lw=1.2, ls="--"))
    ax.text(lx + 0.45, ly + 0.64, "frozen (backbone, parser)", va="center", fontsize=7.6, **FONT)
    ax.plot([lx, lx + 0.35], [ly + 0.3, ly + 0.3], color=C["plan"][1], ls="--", lw=1.4)
    ax.text(lx + 0.45, ly + 0.3, "gradient of the planning loss", va="center", fontsize=7.6, **FONT)
    ax.plot([lx, lx + 0.35], [ly + 0.02, ly + 0.02], color=C["probe"][1], lw=1.4)
    ax.text(lx + 0.45, ly + 0.02, "measured from the true environment", va="center",
            fontsize=7.6, **FONT)

    # frozen sub-outlines inside Perceive and Explain (dashed)
    ax.add_patch(FancyBboxPatch((2.32, Y + 1.7), 2.45, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="none", ec=C["perceive"][1], lw=0.9, ls="--"))
    ax.add_patch(FancyBboxPatch((12.87, Y + 1.26), 2.6, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="none", ec=C["explain"][1], lw=0.9, ls="--"))

    ax.text(8.0, 7.2, "PSPE — Perceive · Simulate · Plan · Explain", ha="center", va="center",
            fontsize=13, weight="bold", color=C["ink"], **FONT)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"pspe_architecture.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print("wrote", ", ".join(str(OUT / f"pspe_architecture.{e}") for e in ("svg", "pdf", "png")))


if __name__ == "__main__":
    main()
