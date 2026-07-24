#!/usr/bin/env python
"""Generate PDE testbed trajectories.

    python scripts/generate_data.py testbed=dar
    python scripts/generate_data.py testbed=swe data.n_trajectories=512
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pspe.simulate.dataset import DataConfig, dataset_path, generate_dataset
from pspe.utils import get_device, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    data_cfg = DataConfig(testbed=cfg.testbed, seed=cfg.seed, **cfg.data)
    out_path = dataset_path(cfg.testbed, data_cfg.grid)

    data = generate_dataset(data_cfg, device=device, out_path=out_path)
    print(
        f"[{cfg.testbed}] states {data['states'].shape} controls {data['controls'].shape} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
