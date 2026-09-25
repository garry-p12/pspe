"""Section-7 metric collection.

One place that knows which metric belongs to which module, so the ablation
runner and the results table agree on names and directions.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from typing import Any

# metric key -> (display name, lower_is_better)
METRICS: dict[str, tuple[str, bool]] = {
    # Surrogate fidelity (Phase 1)
    "rel_l2_1step": ("rel L2 (1 step)", True),
    "rel_l2_final": ("rel L2 (rollout)", True),
    # Planning performance (Phase 2)
    "return": ("return", False),
    "episode_cost": ("episode cost", True),
    "violation_rate": ("violation rate", True),
    "samples": ("env samples", True),
    # Perception quality (Phase 3)
    "val/loss/regression": ("perception rel L2", True),
    "val/metric/retrieval_acc": ("retrieval acc", False),
    "params/trainable_fraction": ("trainable params", True),
    # Explanation faithfulness (Phase 4)
    "eval/faithfulness": ("faithfulness F(b)", False),
    "eval/kl": ("brief KL", True),
    # Cross-domain transfer (Phase 5)
    "transfer_gap": ("transfer gap", True),
    # Provenance: True means the run used an offline stand-in backbone, not an
    # open-weight model. Surfaced in the table so it cannot be missed.
    "backbone_is_stub": ("stub backbone", True),
    # Compute budget
    "wall_clock_s": ("wall clock (s)", True),
    "peak_memory_mb": ("peak memory (MB)", True),
}


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def collect(root: str | Path) -> dict[str, dict[str, Any]]:
    """Every `summary.json` under `root`, keyed by its directory name."""
    root = Path(root)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("summary.json")):
        out[str(path.parent.relative_to(root))] = json.loads(path.read_text())
    return out


def to_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, summary in summaries.items():
        row: dict[str, Any] = {"run": name}
        for key, (display, _) in METRICS.items():
            if key in summary:
                row[display] = summary[key]
        rows.append(row)
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_no runs found_"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def _is_number(value: Any) -> bool:
    """Bools count: `backbone_is_stub` averages to the fraction of stub runs."""
    return isinstance(value, (int, float)) and not isinstance(value, complex)


def aggregate_seeds(
    per_seed: dict[int, list[dict[str, Any]]], key: str = "run"
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Collapse per-seed result rows into one mean +/- std row per run.

    `per_seed` maps seed -> the rows that seed produced (the `results.json`
    written by `eval/run_ablations.py`). Rows are matched across seeds by `key`,
    so a run missing from one seed is averaged over the seeds that have it and
    its `seeds` count says so rather than silently borrowing another run's data.

    Returns `(display_rows, stats)`: the rows carry formatted `mean +/- std`
    cells for the markdown table, `stats` keeps mean, std, n and the raw
    per-seed values for every metric so nothing is lost to formatting.

    Std is the *sample* std (ddof=1) — these are seeds drawn from a population
    of runs, not the population itself. With one seed there is no spread to
    report, so the cell stays a bare number instead of claiming +/- 0.
    """
    order: list[str] = []
    values: dict[str, dict[str, list[Any]]] = {}
    for seed in sorted(per_seed):
        for row in per_seed[seed]:
            name = str(row.get(key, ""))
            if name not in values:
                values[name] = {}
                order.append(name)
            for metric, value in row.items():
                if metric == key:
                    continue
                values[name].setdefault(metric, []).append(value)

    display_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    for name in order:
        row: dict[str, Any] = {key: name}
        run_stats: dict[str, Any] = {}
        seen_seeds = max((len(v) for v in values[name].values()), default=0)
        row["seeds"] = seen_seeds
        for metric, raw in values[name].items():
            numbers = [float(v) for v in raw if _is_number(v)]
            if not numbers:
                unique = {str(v) for v in raw}
                row[metric] = unique.pop() if len(unique) == 1 else "mixed"
                run_stats[metric] = {"values": raw, "n": len(raw)}
                continue
            mean = statistics.fmean(numbers)
            std = statistics.stdev(numbers) if len(numbers) > 1 else None
            # A constant column (horizon, parameter count, an ablation flag) is
            # not a measurement with zero spread; printing "16 ± 0" invites the
            # reader to treat it as one.
            spread = std is not None and std > 0.0
            row[metric] = f"{mean:.4g} ± {std:.2g}" if spread else f"{mean:.4g}"
            run_stats[metric] = {
                "mean": mean, "std": std, "n": len(numbers), "values": numbers,
            }
        display_rows.append(row)
        stats[name] = run_stats
    return display_rows, stats


# --------------------------------------------------------------------------- #
# Reporting that matches what we cite
#
# Agarwal et al. (NeurIPS 2021) argue that point estimates over a handful of
# runs hide most of the uncertainty, and recommend the interquartile mean with
# stratified bootstrap intervals. This project cites that paper and then
# reports mean +/- std over 3 to 5 seeds, which is the practice it argues
# against. These helpers close that gap; `seed_report` is the one to call.
# --------------------------------------------------------------------------- #
def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% of runs.

    More robust than the mean to a single diverged seed and less wasteful than
    the median, which is why it is the recommended summary.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if v.size < 4:                       # too few to trim meaningfully
        return float(v.mean())
    lo, hi = int(np.floor(v.size * 0.25)), int(np.ceil(v.size * 0.75))
    return float(v[lo:hi].mean())


def bootstrap_ci(values: Sequence[float], *, alpha: float = 0.05,
                 n_boot: int = 10_000, statistic=iqm, seed: int = 0
                 ) -> tuple[float, float]:
    """Percentile bootstrap interval for `statistic` over runs.

    Resampling is over runs, which is the unit of independence here. With very
    few runs the interval is wide; that is the honest outcome rather than a
    defect of the method.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    stats = np.array([statistic(d) for d in draws])
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def seed_report(values: Sequence[float], *, alpha: float = 0.05,
                digits: int = 4, seed: int = 0) -> str:
    """`IQM [lo, hi]` over runs, for tables. Falls back to the raw value at n=1."""
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        return f"{v[0]:.{digits}f} (1 run)"
    lo, hi = bootstrap_ci(v, alpha=alpha, seed=seed)
    return f"{iqm(v):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
