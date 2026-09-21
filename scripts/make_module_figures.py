#!/usr/bin/env python
"""Render the four per-module detail figures.

    python scripts/make_module_figures.py

Writes docs/figures/pspe_{perceive,simulate,plan,explain}.{svg,pdf,png}. Each
expands one box of the overview into its internal dataflow: what is frozen,
what trains, where each loss attaches, and what is measured from the true
environment. Deliberately sparse — structure and equations only; the numbers
live in the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_architecture_figure import C, FONT, arrow, chip  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
LOSS = ("#fff6d5", "#8a6d00")
MEAS = C["probe"]


def node(ax, x, y, w, h, label, edge, fill="white", *, dashed=False, size=10, sub=None, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fill, ec=edge, lw=lw, ls="--" if dashed else "-"))
    cy = y + h / 2 + (0.12 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=size, color=C["ink"], **FONT)
    if sub:
        ax.text(x + w / 2, cy - 0.3, sub, ha="center", va="center", fontsize=size - 2,
                color=C["muted"], style="italic", **FONT)


def loss(ax, x, y, w, label, sub=None, h=0.6):
    node(ax, x, y, w, h, label, LOSS[1], LOSS[0], size=9.5, sub=sub)


def measured(ax, x, y, w, label, sub=None, h=0.6):
    node(ax, x, y, w, h, label, MEAS[1], MEAS[0], size=9.5, sub=sub)


def header(ax, title, edge, xmax):
    ax.add_patch(Rectangle((0, 6.55), xmax, 0.55, fc=edge, ec="none"))
    ax.text(0.25, 6.82, title, ha="left", va="center", color="white", fontsize=13, weight="bold", **FONT)


def legend(ax, x, y, items):
    for i, (kind, text) in enumerate(items):
        yy = y - 0.32 * i
        style = {"solid": dict(fc="white", ec=C["ink"]), "dashed": dict(fc="white", ec=C["ink"], ls="--"),
                 "loss": dict(fc=LOSS[0], ec=LOSS[1]), "meas": dict(fc=MEAS[0], ec=MEAS[1]),
                 "joint": dict(fc="white", ec=C["plan"][1], ls="--")}[kind]
        ax.add_patch(Rectangle((x, yy - 0.08), 0.32, 0.17, lw=1.1, **style))
        ax.text(x + 0.42, yy, text, va="center", fontsize=8, **FONT)


def new_canvas(xmax=13.0, ymax=7.1):
    fig, ax = plt.subplots(figsize=(xmax, ymax))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================== #
def perceive():
    f, e = C["perceive"]
    fig, ax = new_canvas()
    header(ax, "PERCEIVE  —  imagery to physical field", e, 13)

    node(ax, 0.3, 4.1, 1.5, 1.2, r"$O_t$", e, f, sub="imagery", size=12)
    node(ax, 2.5, 3.0, 3.2, 3.2, "", e, "#f4f8fc", dashed=True)
    ax.text(4.1, 5.95, "frozen backbone (SigLIP)", ha="center", fontsize=10, weight="bold", color=e, **FONT)
    for i, yy in enumerate((5.0, 4.3, 3.6)):
        node(ax, 2.75, yy, 1.5, 0.5, f"attention {i + 1}", e, "white", dashed=True, size=9)
        node(ax, 4.45, yy, 1.05, 0.5, "LoRA", e, f, size=9)
        arrow(ax, (4.25, yy + 0.25), (4.45, yy + 0.25), lw=1.0)
    ax.text(4.1, 3.25, "⋮", ha="center", fontsize=12, **FONT)
    arrow(ax, (1.8, 4.7), (2.5, 4.7))

    node(ax, 6.3, 4.25, 1.3, 0.9, "tokens", e, "white", size=10)
    arrow(ax, (5.7, 4.7), (6.3, 4.7))

    node(ax, 8.3, 5.2, 2.0, 0.8, "field decoder", e, f, size=10)
    arrow(ax, (7.6, 4.9), (8.3, 5.6), rad=-0.15)
    node(ax, 10.9, 5.2, 1.7, 0.8, r"$\hat z_t$", e, "white", size=12)
    arrow(ax, (10.3, 5.6), (10.9, 5.6))
    loss(ax, 10.9, 4.2, 1.7, r"$\mathcal{L}_{\rm reg}$", sub=r"rel. $L_2$")
    arrow(ax, (11.75, 5.2), (11.75, 4.8), lw=1.0)

    node(ax, 8.3, 3.2, 2.0, 0.8, "projection head", e, f, size=10)
    arrow(ax, (7.6, 4.5), (8.3, 3.6), rad=0.15)
    node(ax, 10.9, 3.2, 1.7, 0.8, r"$e_{\rm img}$", e, "white", size=11)
    arrow(ax, (10.3, 3.6), (10.9, 3.6))
    node(ax, 6.3, 1.7, 1.3, 0.8, "captions", e, "white", size=10)
    node(ax, 8.3, 1.7, 2.0, 0.8, "text encoder", e, f, size=10)
    arrow(ax, (7.6, 2.1), (8.3, 2.1))
    node(ax, 10.9, 1.7, 1.7, 0.8, r"$e_{\rm txt}$", e, "white", size=11)
    arrow(ax, (10.3, 2.1), (10.9, 2.1))
    loss(ax, 10.9, 0.6, 1.7, r"$\mathcal{L}_{\rm con}$", sub="InfoNCE")
    arrow(ax, (11.75, 3.2), (11.75, 2.5), lw=1.0)
    arrow(ax, (11.75, 1.7), (11.75, 1.2), lw=1.0)

    ax.text(0.3, 1.2, r"$\mathcal{L} = w_{\rm reg}\,\mathcal{L}_{\rm reg} + w_{\rm con}\,\mathcal{L}_{\rm con}$",
            fontsize=11, va="center", **FONT)
    legend(ax, 0.35, 2.4, [("dashed", "frozen"), ("solid", "trainable"), ("loss", "loss")])
    save(fig, "pspe_perceive")


# =========================================================================== #
def simulate():
    f, e = C["simulate"]
    fig, ax = new_canvas()
    header(ax, r"SIMULATE  —  differentiable surrogate  $G_\theta$", e, 13)

    node(ax, 0.3, 4.6, 1.2, 0.7, r"$u_t$", e, "white", size=12)
    node(ax, 0.3, 3.7, 1.2, 0.7, r"$a_t$", e, "white", size=12)
    node(ax, 0.3, 2.8, 1.2, 0.7, r"$(x, y)$", e, "white", size=11)
    node(ax, 2.1, 3.7, 1.1, 0.7, "concat", e, "white", size=10)
    for yy in (4.95, 4.05, 3.15):
        arrow(ax, (1.5, yy), (2.1, 4.05), lw=1.0)
    node(ax, 3.7, 3.7, 1.1, 0.7, "lift", e, f, size=10)
    arrow(ax, (3.2, 4.05), (3.7, 4.05))

    node(ax, 5.3, 2.35, 4.4, 3.25, "", e, "#f3faf7")
    ax.text(7.5, 5.38, r"FNO block  ×4      $h \leftarrow h + \mathrm{GELU}(\mathcal{K}h + Wh)$",
            ha="center", fontsize=10, weight="bold", color=e, **FONT)
    node(ax, 5.55, 4.45, 0.9, 0.6, r"$\mathcal{F}$", e, "white", size=11)
    node(ax, 6.65, 4.45, 1.3, 0.6, r"keep $k \leq 12$", e, "white", size=9.5)
    node(ax, 8.15, 4.45, 1.3, 0.6, r"$W_k$", e, f, size=11)
    arrow(ax, (6.45, 4.75), (6.65, 4.75), lw=1.0)
    arrow(ax, (7.95, 4.75), (8.15, 4.75), lw=1.0)
    node(ax, 8.35, 3.5, 0.9, 0.6, r"$\mathcal{F}^{-1}$", e, "white", size=11)
    arrow(ax, (8.8, 4.45), (8.8, 4.1), lw=1.0)
    node(ax, 5.55, 2.6, 1.5, 0.6, r"$W$  1×1", e, f, size=10)
    node(ax, 8.15, 2.6, 1.3, 0.6, "+  GELU", e, "white", size=10)
    arrow(ax, (7.05, 2.9), (8.15, 2.9), lw=1.0)
    arrow(ax, (8.8, 3.5), (8.8, 3.2), lw=1.0)
    ax.plot([5.3, 5.3], [2.9, 4.75], color=C["ink"], lw=1.0)
    arrow(ax, (5.3, 4.75), (5.55, 4.75), lw=1.0)
    arrow(ax, (5.3, 2.9), (5.55, 2.9), lw=1.0)
    ax.plot([9.45, 9.7], [2.9, 2.9], color=C["ink"], lw=1.0)
    ax.plot([9.7, 9.7], [2.9, 4.05], color=C["ink"], lw=1.0)
    arrow(ax, (4.8, 4.05), (5.3, 4.05))

    node(ax, 10.3, 3.7, 1.1, 0.7, "project", e, f, size=10)
    arrow(ax, (9.7, 4.05), (10.3, 4.05))
    node(ax, 11.7, 3.7, 1.1, 0.7, r"$\hat u_{t+1}$", e, "white", size=12)
    arrow(ax, (11.4, 4.05), (11.7, 4.05))

    loss(ax, 5.3, 1.3, 1.3, r"$\mathcal{L}_{\rm data}$")
    loss(ax, 6.85, 1.3, 2.7, r"$\mathcal{L}_{\rm phys} = \|\dot{\hat u} - \mathrm{rhs}(\hat u, a)\|$")
    loss(ax, 9.8, 1.3, 1.6, r"$\mathcal{L}_{\rm roll}$", sub=r"$H$ steps")
    ax.text(0.3, 1.6, r"$\mathcal{L} = \mathcal{L}_{\rm data} + \lambda_p \mathcal{L}_{\rm phys} + \lambda_r \mathcal{L}_{\rm roll}$",
            fontsize=11, va="center", **FONT)
    measured(ax, 0.3, 5.75, 4.5, r"Lipschitz mode:  $\|W_k\|_2 \leq 1$,  $L_G$ measured", h=0.55)
    ax.text(0.3, 0.6, r"sharp fronts:  $\ell = \ell_{\rm UNet} + g\cdot\ell_{\rm FNO}$,  gate $g$ learned",
            fontsize=10, va="center", **FONT)
    legend(ax, 11.4, 6.0, [("solid", "trainable"), ("loss", "loss"), ("meas", "measured")])
    save(fig, "pspe_simulate")


# =========================================================================== #
def plan():
    f, e = C["plan"]
    fig, ax = new_canvas(14.0)
    header(ax, "PLAN  —  constrained policy, hybrid gradient", e, 14)

    node(ax, 0.3, 4.7, 1.3, 0.8, r"$\hat z_t$", e, "white", size=12)
    node(ax, 2.1, 4.5, 2.0, 1.2, r"policy $\pi_\phi$", e, f, sub=r"$\mu,\ \log\sigma$", size=11)
    arrow(ax, (1.6, 5.1), (2.1, 5.1))
    node(ax, 4.6, 4.7, 1.9, 0.8, r"$a_t = \mu + \sigma\,\xi$", e, "white", size=10.5)
    arrow(ax, (4.1, 5.1), (4.6, 5.1))

    node(ax, 7.0, 4.4, 2.6, 1.4, "", e, "#fdf5ec")
    ax.text(8.3, 5.5, r"rollout through $G_\theta$", ha="center", fontsize=10, weight="bold", color=e, **FONT)
    ax.text(8.3, 4.85, r"$\mathcal{L}_{\rm plan} = -\sum_t\,(r_t - \lambda\, c_t)$", ha="center", fontsize=11, **FONT)
    arrow(ax, (6.5, 5.1), (7.0, 5.1))

    node(ax, 10.1, 5.05, 1.7, 0.7, r"$\nabla_{\rm path}$", e, f, size=11)
    node(ax, 10.1, 4.2, 1.7, 0.7, r"$\nabla_{\rm LR}$", e, f, size=11)
    arrow(ax, (9.6, 5.3), (10.1, 5.4), rad=-0.1)
    arrow(ax, (9.6, 4.9), (10.1, 4.55), rad=0.1)
    node(ax, 12.2, 4.5, 1.5, 1.0, r"$\alpha\nabla_{\rm path}$" "\n" r"$+(1-\alpha)\nabla_{\rm LR}$", e, "white", size=10)
    arrow(ax, (11.8, 5.4), (12.2, 5.15), rad=-0.1)
    arrow(ax, (11.8, 4.55), (12.2, 4.85), rad=0.1)

    node(ax, 10.0, 2.85, 3.9, 0.95, "", e, "#fdf5ec")
    ax.text(11.95, 3.32, r"$\alpha^* = \dfrac{V_L - \mathrm{Cov}}{B^2 + V_p + V_L - 2\,\mathrm{Cov}}$",
            ha="center", va="center", fontsize=11, **FONT)
    arrow(ax, (11.95, 3.8), (12.6, 4.5), lw=1.0)

    node(ax, 0.3, 2.6, 3.3, 1.3, "", e, "#fdf5ec")
    ax.text(1.95, 3.62, "PID-Lagrangian dual", ha="center", fontsize=10, weight="bold", color=e, **FONT)
    ax.text(1.95, 3.22, r"$\lambda = [\,K_p e + K_i \Sigma e + K_d\,\Delta e^+]_+$", ha="center", fontsize=10, **FONT)
    ax.text(1.95, 2.82, r"$e = \bar J_C - (d - k\sigma)$", ha="center", fontsize=10, **FONT)
    arrow(ax, (3.6, 3.9), (7.0, 4.5), rad=0.0, color=e)
    ax.text(5.1, 4.38, r"$\lambda$", fontsize=10, color=e, **FONT)

    measured(ax, 4.2, 2.6, 5.2, "", h=1.3)
    ax.text(6.8, 3.62, "reality probe", ha="center", fontsize=10, weight="bold", color=MEAS[1], **FONT)
    ax.text(6.8, 3.2, r"roll $\pi_\phi$ in $F$:   real cost $J_C$", ha="center", fontsize=10, **FONT)
    ax.text(6.8, 2.82, r"$B^2 = \|\nabla_{\rm path}^{G_\theta} - \nabla_{\rm path}^{F}\|^2$", ha="center", fontsize=10, **FONT)
    arrow(ax, (4.2, 3.25), (3.6, 3.25), color=MEAS[1])
    ax.text(3.9, 3.45, r"$J_C,\ \sigma$", ha="center", fontsize=9, color=MEAS[1], **FONT)
    arrow(ax, (9.4, 3.25), (10.0, 3.3), color=MEAS[1])
    ax.text(9.85, 3.02, r"$B^2$", ha="center", fontsize=9, color=MEAS[1], **FONT)

    arrow(ax, (9.55, 4.4), (9.55, 1.85), color=e, dashed=True)
    node(ax, 8.1, 1.05, 3.4, 0.8, r"$\beta\,\nabla_\theta \mathcal{L}_{\rm plan} \to G_\theta$", e, "white",
         sub="joint training, data-anchored", size=10.5, dashed=True)

    chip(ax, 0.3, 1.6, 0.95, "budget", e); chip(ax, 1.35, 1.6, 0.95, "safety", e); chip(ax, 2.4, 1.6, 0.95, "equity", e)
    ax.text(0.3, 1.25, r"$g_j \leq d_j$", fontsize=10, **FONT)
    legend(ax, 12.1, 2.3, [("solid", "trainable"), ("meas", "measured from F"), ("joint", "joint gradient")])
    save(fig, "pspe_plan")


# =========================================================================== #
def explain():
    f, e = C["explain"]
    fig, ax = new_canvas()
    header(ax, "EXPLAIN  —  briefs trained against the policy", e, 13)

    node(ax, 0.3, 4.6, 1.7, 1.1, "condition", e, "white", sub=r"$\hat z_t,\ a_t,\ \lambda,\ c_t$", size=10.5)
    node(ax, 2.5, 4.75, 1.4, 0.8, "prefix", e, f, size=10.5)
    arrow(ax, (2.0, 5.15), (2.5, 5.15))
    node(ax, 4.4, 4.1, 2.6, 2.1, "", e, "#f6f2fb", dashed=True)
    ax.text(5.7, 5.92, "frozen LM · Qwen2.5-0.5B", ha="center", fontsize=9, weight="bold", color=e, **FONT)
    for i, yy in enumerate((5.05, 4.45)):
        node(ax, 4.6, yy, 1.2, 0.45, f"layers {'1–12' if i == 0 else '13–24'}", e, "white", dashed=True, size=8.5)
        node(ax, 5.95, yy, 0.9, 0.45, "LoRA", e, f, size=9)
        arrow(ax, (5.8, yy + 0.22), (5.95, yy + 0.22), lw=1.0)
    arrow(ax, (3.9, 5.15), (4.4, 5.15))
    node(ax, 7.5, 4.75, 1.7, 0.8, r"brief $b_t$", e, "white", size=11)
    arrow(ax, (7.0, 5.15), (7.5, 5.15))
    node(ax, 9.7, 4.75, 1.5, 0.8, "parser", e, "white", dashed=True, size=10.5)
    arrow(ax, (9.2, 5.15), (9.7, 5.15))
    node(ax, 11.7, 4.75, 1.1, 0.8, r"$\hat\pi_b$", e, "white", size=12)
    arrow(ax, (11.2, 5.15), (11.7, 5.15))

    node(ax, 9.7, 3.2, 3.1, 0.9, r"$F(b) = \exp[-\mathrm{KL}(\pi_\phi \,\|\, \hat\pi_b)]$", e, f, size=11)
    arrow(ax, (12.25, 4.75), (12.25, 4.1), lw=1.0)
    node(ax, 7.5, 3.3, 1.7, 0.7, r"$\pi_\phi(\cdot\mid \hat z_t)$", e, "white", size=10.5)
    arrow(ax, (9.2, 3.65), (9.7, 3.65), lw=1.0)

    loss(ax, 0.3, 2.2, 3.3, r"$\mathcal{L}_{\rm sup}$ = NLL(reference)", h=0.65)
    loss(ax, 3.9, 2.2, 4.6, r"$\mathcal{L}_{\rm faith} = -(F(b) - F(b_{\rm greedy}))\,\log p(b)$", h=0.65)
    arrow(ax, (9.7, 3.5), (8.5, 2.6), rad=-0.25, lw=1.0)
    ax.text(0.3, 1.5, r"$\mathcal{L} = \mathcal{L}_{\rm sup} + w_f\,\mathcal{L}_{\rm faith}$", fontsize=11, va="center", **FONT)

    measured(ax, 7.3, 0.6, 5.5, "", h=1.2)
    ax.text(10.05, 1.5, "conformal certificate", ha="center", fontsize=10, weight="bold", color=MEAS[1], **FONT)
    ax.text(10.05, 1.0, r"$s = 1 - F$;   $\Pr[\,F(b) \geq 1 - \hat s\,] \geq 1 - \delta$", ha="center", fontsize=10.5, **FONT)

    legend(ax, 11.3, 6.3, [("dashed", "frozen"), ("solid", "trainable"), ("meas", "deployment-time")])
    save(fig, "pspe_explain")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "cm"
    for fn in (perceive, simulate, plan, explain):
        fn()
    print("wrote", ", ".join(f"pspe_{n}.{{svg,pdf,png}}" for n in ("perceive", "simulate", "plan", "explain")))
