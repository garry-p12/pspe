"""Local-only run logging: TensorBoard scalars plus a JSONL mirror.

TensorBoard is the default tracker (fully local, free). The JSONL mirror makes
the ablation runner and results table independent of the event files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # tensorboard is in requirements, but keep the import soft for CPU smoke runs
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - exercised only when tensorboard is absent
    SummaryWriter = None  # type: ignore[assignment]


class RunLogger:
    """Writes scalars to TensorBoard and appends them to `metrics.jsonl`."""

    def __init__(self, run_dir: str | Path, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "metrics.jsonl").open("a")
        self._writer = (
            SummaryWriter(str(self.run_dir))
            if use_tensorboard and SummaryWriter is not None
            else None
        )

    def log(self, step: int, **scalars: float) -> None:
        record: dict[str, Any] = {"step": step}
        for key, value in scalars.items():
            value = float(value)
            record[key] = value
            if self._writer is not None:
                self._writer.add_scalar(key, value, step)
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

    def log_summary(self, **values: Any) -> None:
        """Write the run's final metrics to `summary.json`."""
        path = self.run_dir / "summary.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2, default=float))

    def close(self) -> None:
        self._jsonl.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
