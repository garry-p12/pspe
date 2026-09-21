#!/usr/bin/env python
"""Render the application / digital-twin figures.

    python scripts/make_application_figures.py

Writes docs/figures/pspe_{digital_twin,twin_timeline,applications,
app_wildfire,app_flood,app_heat}.{svg,pdf,png}. These place the framework in a
climate digital-twin setting: how observations sync the twin, where what-if
rollouts and constrained planning sit, how the real system is probed, and what
each module means for wildfire, flood and urban-heat operations. Same helpers
and palette as the architecture figures.
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
from make_module_figures import OUT, header, legend, new_canvas, node, save  # noqa: E402

TWIN = "#2f3e46"
DOM = {
    "wildfire": ("#fde2e2", "#a12a2a"),
    "flood":    ("#dbe9f6", "#2b5d8c"),
    "heat":     ("#fbe6d0", "#b0561a"),
}
MODULES = ("perceive", "simulate", "plan", "explain")


def module(ax, x, y, w, h, name, sub, *, size=11):
    f, e = C[name]
    node(ax, x, y, w, h, name.capitalize(), e, f, size=size, sub=sub, lw=1.5)


def polyline(ax, pts, *, color, dashed=False, lw=1.4):
    """Orthogonal route with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls="--" if dashed else "-",
            solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, dashed=dashed, lw=lw)


# =========================================================================== #
def digital_twin():
    fig, ax = new_canvas()
    header(ax, "PSPE as a climate digital twin: observe, sync, rehearse, intervene, explain", TWIN, 13.0)
    fe, ee = C["env"]
    pe = C["probe"][1]

    # physical system
    node(ax, 0.3, 2.6, 2.0, 2.2, "", ee, fe, lw=1.5)
    ax.text(1.3, 4.5, "physical system", ha="center", fontsize=10.5, weight="bold", **FONT)
    for i, t in enumerate(["satellites and radar", "gauges and weather", "grid and sensor feeds", "actuators"]):
        ax.text(1.3, 4.05 - 0.4 * i, t, ha="center", fontsize=8.8, color=C["muted"], **FONT)

    # twin container
    ax.add_patch(FancyBboxPatch((2.9, 1.75), 7.5, 3.65, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#fafafa", ec=TWIN, lw=1.4, ls="--"))
    ax.text(3.1, 5.2, "PSPE digital twin", fontsize=10.5, weight="bold", color=TWIN, **FONT)

    xs = {"perceive": 3.15, "simulate": 5.0, "plan": 6.85, "explain": 8.7}
    subs = {"perceive": r"sync $O_t$ into $\hat z_t$", "simulate": "what if rollouts",
            "plan": r"constrained $a_t$", "explain": "certified brief"}
    for n in MODULES:
        module(ax, xs[n], 3.0, 1.5, 1.3, n, subs[n])

    arrow(ax, (2.3, 3.75), (3.15, 3.75), r"$O_t$")
    arrow(ax, (4.65, 3.75), (5.0, 3.75), r"$\hat z_t$", label_dy=0.1)
    arrow(ax, (6.5, 3.95), (6.85, 3.95), lw=1.2)
    arrow(ax, (6.85, 3.35), (6.5, 3.35), lw=1.2)
    ax.text(6.675, 4.42, "rollouts", ha="center", fontsize=7.6, color=C["muted"], **FONT)
    ax.text(6.675, 2.82, "candidate $a_t$", ha="center", fontsize=7.6, color=C["muted"], **FONT)
    arrow(ax, (8.35, 3.75), (8.7, 3.75), r"$a_t,\ \lambda$", label_dy=0.1)

    for i, (t, w) in enumerate([("budget", 0.7), ("safety", 0.7), ("equity", 0.7)]):
        chip(ax, 6.85 + 0.8 * i - 0.55, 2.35, w, t, C["plan"][1])
    ax.text(7.6, 2.05, r"$g_j \leq d_j$", ha="center", fontsize=8.5, **FONT)

    # operator
    node(ax, 10.9, 3.0, 1.8, 1.3, "operator", ee, fe, sub="approve or override", size=10.5, lw=1.5)
    arrow(ax, (10.2, 3.75), (10.9, 3.75), r"brief $b_t$", label_dy=0.1)

    # feedback: operator -> actuators
    polyline(ax, [(11.8, 3.0), (11.8, 1.2), (1.3, 1.2), (1.3, 2.6)], color=C["ink"])
    ax.text(6.5, 1.32, "approved interventions are sent to the actuators", ha="center", fontsize=8.5, **FONT,
            bbox=dict(fc="white", ec="none", pad=1.5))

    # reality probe: physical -> plan, bypassing the surrogate
    polyline(ax, [(1.3, 4.8), (1.3, 5.9), (7.6, 5.9), (7.6, 4.3)], color=pe, dashed=True)
    ax.text(4.4, 6.02, r"reality probe every $N$ steps: real cost $J_C$, bias $B^2$, margin $d - k\sigma$",
            ha="center", fontsize=8.5, color=pe, **FONT, bbox=dict(fc="white", ec="none", pad=1.5))

    legend(ax, 10.9, 5.75, [("dashed", "twin boundary"), ("meas", "measured in the world")])
    save(fig, "pspe_digital_twin")


# =========================================================================== #
def twin_timeline():
    fig, ax = new_canvas(13.5, 5.6)
    ax.add_patch(Rectangle((0, 5.05), 13.5, 0.55, fc=TWIN, ec="none"))
    ax.text(0.25, 5.32, "twin operating cycle: sync, rehearse, act, verify", ha="left", va="center",
            color="white", fontsize=13, weight="bold", **FONT)
    fe, ee = C["env"]
    pe = C["probe"][1]

    # lanes
    ax.add_patch(Rectangle((1.4, 2.75), 11.9, 2.0, fc="#fafafa", ec=TWIN, lw=1.0, ls="--"))
    ax.add_patch(Rectangle((1.4, 0.2), 11.9, 1.85, fc=fe, ec=ee, lw=1.0))
    ax.text(0.7, 3.75, "digital\ntwin", ha="center", va="center", fontsize=10.5, weight="bold", color=TWIN, **FONT)
    ax.text(0.7, 1.2, "physical\nworld", ha="center", va="center", fontsize=10.5, weight="bold", color=ee, **FONT)

    ticks = [1.9, 5.5, 9.1]
    labels = [r"$t$", r"$t+1$", r"$t+2$"]
    w = 3.2
    for k, (x0, lab) in enumerate(zip(ticks, labels)):
        # world: observe
        node(ax, x0, 1.15, 1.1, 0.6, "observe", ee, "white", size=9.5)
        ax.text(x0 + 0.55, 0.95, r"$O_{%s}$" % lab.strip("$"), ha="center", va="top", fontsize=9, **FONT)
        # twin: four-stage cycle as one strip
        cy = 3.35
        for i, n in enumerate(MODULES):
            f, e = C[n]
            xx = x0 + 0.05 + 0.82 * i
            node(ax, xx, cy, 0.76, 0.65, ["sync", "what if", "plan", "brief"][i], e, f, size=8.5, lw=1.2)
        # up: observation -> sync ; down: action -> world
        arrow(ax, (x0 + 0.55, 1.75), (x0 + 0.4, 3.35), lw=1.2)
        arrow(ax, (x0 + 2.3, 3.35), (x0 + 2.75, 1.75), lw=1.2)
        ax.text(x0 + 2.7, 2.55, r"$a_{%s}$" % lab.strip("$"), ha="left", va="center", fontsize=9, **FONT)
        node(ax, x0 + 2.35, 1.15, 0.85, 0.6, "act", ee, "white", size=9.5)
        # world advances
        if k < len(ticks) - 1:
            arrow(ax, (x0 + 3.2, 1.45), (ticks[k + 1], 1.45), lw=1.2)
        # tick label
        ax.text(x0 + 0.55, 4.6, lab, ha="center", va="center", fontsize=10, **FONT)

    ax.text(12.8, 1.45, "next\ncycle", ha="center", va="center", fontsize=8, color=C["muted"], **FONT)
    ax.text(12.8, 3.65, "next\ncycle", ha="center", va="center", fontsize=8, color=C["muted"], **FONT)

    # reality probe at cycle t+1: truth rollout of the current policy
    x0 = ticks[1]
    polyline(ax, [(x0 + 1.05, 1.15), (x0 + 1.05, 0.6), (x0 + 1.85, 0.6), (x0 + 1.85, 3.35)],
             color=pe, dashed=True, lw=1.3)
    ax.text(x0 + 1.45, 0.28, r"reality probe: roll $\pi_\phi$ in $F$ every $N$ steps", ha="center",
            va="bottom", fontsize=8, color=pe, **FONT, bbox=dict(fc="white", ec="none", pad=1.2))
    ax.text(x0 + 1.75, 2.3, "$J_C,\\ B^2$\nupdate $\\lambda,\\ \\alpha$", ha="right", va="center", fontsize=8.5,
            color=pe, **FONT, linespacing=1.4)

    save(fig, "pspe_twin_timeline")


# =========================================================================== #
def applications():
    fig, ax = new_canvas(13.0, 7.1)
    header(ax, "one framework, three climate twins", TWIN, 13.0)

    cols = ["wildfire", "flood", "heat"]
    titles = {"wildfire": "wildfire", "flood": "flood", "heat": "urban heat and infrastructure"}
    cx = {"wildfire": 2.65, "flood": 6.1, "heat": 9.55}
    cw = 3.3
    for c in cols:
        f, e = DOM[c]
        node(ax, cx[c], 5.75, cw, 0.55, titles[c], e, f, size=10.5, lw=1.5)

    rows = list(MODULES) + ["testbed"]
    ry = {"perceive": 4.6, "simulate": 3.5, "plan": 2.4, "explain": 1.3, "testbed": 0.45}
    rh = {"perceive": 1.0, "simulate": 1.0, "plan": 1.0, "explain": 1.0, "testbed": 0.7}
    for r in rows:
        if r == "testbed":
            node(ax, 0.2, ry[r], 2.3, rh[r], "PDE testbed", C["muted"], "white", size=10, lw=1.3)
        else:
            f, e = C[r]
            node(ax, 0.2, ry[r], 2.3, rh[r], r.capitalize(), e, f, size=11, lw=1.5)

    cell = {
        ("perceive", "wildfire"): "satellite fire mask, weather, fuel\nand terrain give the fire front field",
        ("perceive", "flood"):    "radar water extent, gauges and\nrainfall nowcast give the water depth field",
        ("perceive", "heat"):     "satellite surface temperature and\ngrid load give the temperature field",
        ("simulate", "wildfire"): "next day fire spread with\nthe hybrid FNO and U-Net model",
        ("simulate", "flood"):    "shallow water routing\nwith an FNO surrogate",
        ("simulate", "heat"):     "heat and pollutant transport\nwith an FNO surrogate",
        ("plan", "wildfire"):     "firebreaks, crews and evacuation\nwithin crew budget, crew safety\nand equity limits",
        ("plan", "flood"):        "gates, pumps and barriers\nwithin energy budget, hospital\ndepth cap and equity limits",
        ("plan", "heat"):         "cooling centres and load shedding\nwithin energy budget, exposure\ncap and equity limits",
        ("explain", "wildfire"):  "incident action brief with\na faithfulness certificate",
        ("explain", "flood"):     "operations brief for\nemergency managers",
        ("explain", "heat"):      "public advisory and\nutility dispatch note",
        ("testbed", "wildfire"):  "reaction diffusion (rdf), real NDWS data",
        ("testbed", "flood"):     "shallow water equations (swe, PDEBench)",
        ("testbed", "heat"):      "diffusion advection reaction (dar)",
    }
    for r in rows:
        for c in cols:
            node(ax, cx[c], ry[r], cw, rh[r], "", "#cccccc", "white", lw=0.9)
            ax.text(cx[c] + cw / 2, ry[r] + rh[r] / 2, cell[(r, c)], ha="center", va="center",
                    fontsize=8.2 if r != "testbed" else 8.0, color=C["ink"], **FONT, linespacing=1.35)

    save(fig, "pspe_applications")


# =========================================================================== #
def domain_card(name, title, data, texts, chips, outputs, testbed):
    fig, ax = new_canvas(13.0, 7.1)
    fd, ed = DOM[name]
    header(ax, title, ed, 13.0)
    fe, ee = C["env"]

    # data sources
    ax.text(0.2, 5.95, "observations", fontsize=10, weight="bold", color=ee, **FONT)
    for i, t in enumerate(data):
        node(ax, 0.2, 5.1 - 0.7 * i, 2.65, 0.55, t, ee, fe, size=8.3, lw=1.2)
        arrow(ax, (2.85, 5.375 - 0.7 * i), (3.1, 4.4), lw=1.0)

    # modules
    xs = [3.1, 5.55, 8.0, 10.45]
    for x, n in zip(xs, MODULES):
        f, e = C[n]
        node(ax, x, 3.6, 2.2, 1.6, "", e, f, lw=1.5)
        ax.text(x + 1.1, 4.9, n.capitalize(), ha="center", va="center", fontsize=11.5, weight="bold",
                color=e, **FONT)
        ax.text(x + 1.1, 4.2, texts[n], ha="center", va="center", fontsize=8.8, color=C["ink"],
                **FONT, linespacing=1.35)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 2.2, 4.4), (b, 4.4))
    arrow(ax, (xs[2] + 0.3, 3.6), (xs[1] + 1.9, 3.6), rad=-0.4, lw=1.1)
    ax.text(xs[1] + 2.2 + 0.125, 3.12, "candidate actions", ha="center", fontsize=7.6, color=C["muted"], **FONT)

    # under simulate: testbed; under plan: constraint chips; under explain: outputs
    ax.text(xs[1] + 1.1, 2.55, testbed, ha="center", fontsize=8.4, color=C["simulate"][1], style="italic", **FONT)
    ax.text(xs[2] + 1.1, 2.55, "constraints", ha="center", fontsize=8.6, weight="bold", color=C["plan"][1], **FONT)
    for i, t in enumerate(chips):
        chip(ax, xs[2] + 0.1, 2.05 - 0.4 * i, 2.0, t, C["plan"][1], size=8)
    ax.text(xs[3] + 1.1, 2.55, "decision outputs", ha="center", fontsize=8.6, weight="bold", color=C["explain"][1], **FONT)
    for i, t in enumerate(outputs):
        node(ax, xs[3], 2.0 - 0.45 * i, 2.2, 0.38, t, C["explain"][1], "white", size=8.4, lw=1.1)

    # loop back to the world
    polyline(ax, [(xs[3] + 1.1, 0.62), (xs[3] + 1.1, 0.5), (1.5, 0.5), (1.5, 2.95)], color=C["ink"], lw=1.2)
    ax.text(6.5, 0.62, "actuate, then observe, then re-sync the twin", ha="center", fontsize=8.5, **FONT,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(1.65, 1.9, r"probe: real cost $J_C$", ha="left", va="center", fontsize=8, color=C["probe"][1], **FONT)

    save(fig, f"pspe_app_{name}")


def wildfire():
    domain_card(
        "wildfire", "wildfire twin: next day spread, firebreaks, crews",
        ["active fire (VIIRS, MODIS)", "wind, temperature, humidity", "fuel (NDVI) and elevation", "previous day fire mask"],
        {"perceive": "fire front field $\\hat z_t$\nfrom 12 input drivers",
         "simulate": "next day fire spread\nhybrid FNO and U-Net",
         "plan": "firebreaks, crew dispatch\nand evacuation timing",
         "explain": "incident action brief\nwith a certificate"},
        ["crew hour budget", "crew safety (flame length)", "equity across communities"],
        ["firebreak placement map", "crew allocation", "evacuation order", "brief for incident command"],
        "real NDWS data, rdf testbed",
    )


def flood():
    domain_card(
        "flood", "flood twin: routing, gates, barriers",
        ["radar water extent (Sentinel-1)", "river gauges, rainfall nowcast", "elevation model and land cover", "current gate and pump states"],
        {"perceive": "water depth\nfield $\\hat z_t$",
         "simulate": "shallow water routing\nwith an FNO surrogate",
         "plan": "gate and pump schedule\nand barrier placement",
         "explain": "operations brief\nwith a certificate"},
        ["pump energy budget", "depth cap at hospitals", "neighbourhood equity"],
        ["gate and pump schedule", "barrier sites", "evacuation zones", "emergency manager brief"],
        "PDEBench shallow water, swe testbed",
    )


def heat():
    domain_card(
        "heat", "urban heat and infrastructure twin: exposure, cooling, grid load",
        ["surface temperature (Landsat)", "weather and air quality", "building and grid load", "population vulnerability"],
        {"perceive": "surface temperature\nfield $\\hat z_t$",
         "simulate": "heat and pollutant\ntransport with an FNO",
         "plan": "cooling centres, load\nshedding and greening",
         "explain": "public advisory and\nutility dispatch note"},
        ["energy budget", "heat exposure cap", "equity across districts"],
        ["cooling centre siting", "load shedding schedule", "public advisory", "utility dispatch note"],
        "dar testbed",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "cm"
    names = []
    for fn, n in ((digital_twin, "digital_twin"), (twin_timeline, "twin_timeline"),
                  (applications, "applications"), (wildfire, "app_wildfire"),
                  (flood, "app_flood"), (heat, "app_heat")):
        fn()
        names.append(n)
    print("wrote", ", ".join(f"pspe_{n}.{{svg,pdf,png}}" for n in names))
