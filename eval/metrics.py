"""Section-7 metric collection.

One place that knows which metric belongs to which module, so the ablation
runner and the results table agree on names and directions.
"""

from __future__ import annotations

import json
from pathlib import Path
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
