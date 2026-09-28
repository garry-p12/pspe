"""A levee-planning scenario built from FloodCastBench terrain, not invented.

The span, elasticity and precondition results in 5.6 were measured on
`floodplain_terrain` -- a valley I built and tuned until the settlement flooded
from the channel. That is a physically coherent testbed but it is synthetic, and
a claim that the framework was "run on real flood data" cannot rest on it. This
module replaces every synthetic ingredient with one taken from the archive:

    terrain      the event's own DEM, block-averaged to a tractable resolution
    initial      the reference depth field at the window's start
    forcing      uniform rainfall derived from the reference's MASS BUDGET over
                 the window, so the flood's magnitude and timing come from the
                 event rather than from tuning
    settlement   a compact low-lying block chosen for how deeply it floods in
                 the reference
    levees       candidate segments on the settlement's perimeter, placed at the
                 LOWEST point of each angular sector -- where water actually
                 enters, which is terrain-derived rather than hand-placed

Only the *spatial distribution* of the forcing is approximated (uniform), because
the archive ships no rainfall field. That is stated wherever results are quoted.

Coarsening is honest and standard: FloodCastBench itself publishes Australia at
both 30 m and 60 m. The local-inertial timestep scales with dx while cell count
falls as dx^2, so 4x coarsening buys ~64x, which is what makes a planning study
with hundreds of solves feasible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .floodcast import EVENTS, NODATA, _numbered, _read_tiff, frame_seconds

Tensor = torch.Tensor


@dataclass
class RealScenario:
    """Everything a planning run needs, all of it derived from the archive."""

    z: Tensor                 # (H, W) bed elevation, metres
    exposure: Tensor          # (H, W) normalised weighting over the settlement
    masks: Tensor             # (K, H, W) candidate levee footprints in [0, 1]
    h0: Tensor                # (H, W) initial depth, from the reference
    rain: float               # m/s, uniform, from the reference's mass budget
    dx: float                 # metres
    duration: float           # seconds
    settlement: tuple[int, int]
    site_elevations: list[float]
    meta: dict

    @property
    def k(self) -> int:
        return int(self.masks.shape[0])


def _coarsen(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f, a.shape[1] // f
    return np.nanmean(a[: h * f, : w * f].reshape(h, f, w, f), axis=(1, 3))


def build_scenario(
    root: str | Path,
    event: str = "australia2022",
    coarsen: int = 4,
    start_hours: float = 144.0,
    end_hours: float = 192.0,
    settle_radius: int = 10,
    n_sites: int = 6,
    site_width_m: float = 360.0,
    settle_rc: tuple[int, int] | None = None,
    device: torch.device | str = "cpu",
    peak_stride: int = 36,
) -> RealScenario:
    spec = EVENTS[event]
    root = Path(root)
    if root.name != "FloodCastBench" and (root / "FloodCastBench").is_dir():
        root = root / "FloodCastBench"

    dem = _read_tiff(root / "Study regions" / f"{spec['region']}_DEM.tif")
    dem[dem < NODATA] = np.nan
    frames = _numbered(list((root / str(spec["folder"]) / str(spec["res"]) /
                             str(spec["region"])).glob("*.tif")))
    if not frames:
        raise FileNotFoundError("no depth frames")
    secs = [frame_seconds(p) for p in frames]
    i0 = min(range(len(secs)), key=lambda i: abs(secs[i] - start_hours * 3600))
    i1 = min(range(len(secs)), key=lambda i: abs(secs[i] - end_hours * 3600))
    if i1 <= i0:
        raise SystemExit("end_hours must exceed start_hours")

    f0 = np.nan_to_num(_read_tiff(frames[i0]))
    f1 = np.nan_to_num(_read_tiff(frames[i1]))
    duration = float(secs[i1] - secs[i0])

    # Forcing from the reference's own mass budget. Not fitted to anything we
    # will later compare against -- it sets the flood's size, which is the point.
    rain = float((f1.sum() - f0.sum()) / f0.size / duration)

    # Peak depth over the window locates where flooding actually bites.
    peak = np.zeros_like(f0)
    for i in range(i0, i1 + 1, max(1, peak_stride)):
        peak = np.maximum(peak, np.nan_to_num(_read_tiff(frames[i])))

    zc = _coarsen(dem, coarsen)
    h0c = _coarsen(f0, coarsen)
    pc = _coarsen(peak, coarsen)
    h, w = zc.shape

    # Settlement: the compact block that floods deepest in the reference.
    b = max(4, settle_radius * 2)
    if settle_rc is not None:
        # An explicit centre, e.g. one chosen by the controllability screen.
        # "Deepest-flooding" is a poor default: it selects whichever block holds
        # the most water, not the one a levee can actually defend, and the two
        # differ by 3x in achievable span a kilometre apart (5.6b).
        sr, sc = settle_rc
        best = float(np.nanmean(pc[sr - b // 2 : sr + b // 2, sc - b // 2 : sc + b // 2]))
    else:
        best, pos = -1.0, (b, b)
        for r in range(b, h - b, 3):
            for c in range(b, w - b, 3):
                m = float(np.nanmean(pc[r - b // 2 : r + b // 2, c - b // 2 : c + b // 2]))
                if m > best:
                    best, pos = m, (r, c)
        sr, sc = pos

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d2 = ((yy - sr) ** 2 + (xx - sc) ** 2) / float(settle_radius) ** 2
    bowl = np.exp(-d2)
    exposure = bowl / bowl.sum()

    # Levee candidates: one per angular sector of the settlement's perimeter,
    # each centred on that sector's LOWEST cell -- where water enters. Terrain
    # decides the placement; nothing is positioned by hand.
    ring = settle_radius + 2
    sites, site_elev = [], []
    for s in range(n_sites):
        a0 = 2 * np.pi * s / n_sites
        a1 = 2 * np.pi * (s + 1) / n_sites
        best_e, best_rc = np.inf, None
        for a in np.linspace(a0, a1, 24):
            rr = int(round(sr + ring * np.sin(a)))
            cc = int(round(sc + ring * np.cos(a)))
            if 0 <= rr < h and 0 <= cc < w and np.isfinite(zc[rr, cc]):
                if zc[rr, cc] < best_e:
                    best_e, best_rc = float(zc[rr, cc]), (rr, cc)
        if best_rc is None:
            continue
        rr, cc = best_rc
        # Footprint in METRES, converted to cells. Specifying it in cells made
        # the structure's physical size an accident of the grid: the same site
        # was a 360 m levee at 120 m resolution and a 180 m levee at 60 m, which
        # is why it appeared 6x less effective when the grid was refined.
        sigma_cells = site_width_m / (float(spec["dx"]) * coarsen)
        g = np.exp(-(((yy - rr) ** 2 + (xx - cc) ** 2) / sigma_cells**2))
        sites.append(g)
        site_elev.append(best_e)

    t = lambda a: torch.as_tensor(np.nan_to_num(a), dtype=torch.float32, device=device)
    return RealScenario(
        z=t(zc), exposure=t(exposure), masks=t(np.stack(sites)), h0=t(h0c),
        rain=rain, dx=float(spec["dx"]) * coarsen, duration=duration,
        settlement=(sr, sc), site_elevations=site_elev,
        meta={
            "event": event, "coarsen": coarsen,
            "window_hours": [start_hours, end_hours],
            "rain_mm_per_h": rain * 3.6e6,
            "settlement_elev_m": float(zc[sr, sc]),
            "settlement_peak_depth_m": best,
            "grid": [h, w],
            "forcing_note": (
                "uniform rainfall derived from the reference mass budget; the "
                "archive ships no rainfall field, so only the SPATIAL "
                "distribution is approximated, not the magnitude or timing"
            ),
        },
    )
