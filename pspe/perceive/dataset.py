"""Proxy perception dataset: (image, field, weak caption) triples.

Two sources, one interface:

* `source="real"` - free satellite/aerial imagery the user downloads themselves
  (Sentinel-2 via Copernicus, Landsat via USGS EarthExplorer, or NAIP). Expected
  layout under `data/perception_proxy/`:

      images/<id>.png        RGB patch, any size (resized to image_size)
      fields/<id>.npy        (C, grid, grid) target field, e.g. an index band
      captions.json          optional {"<id>": "weak caption"}; synthesised
                             from the field when absent

  See `data/perception_proxy/README.md` for the download recipes.

* `source="synthetic"` - a renderer that turns PDE testbed fields into
  sensor-like RGB (colour map + point-spread blur + shot noise + vignetting).
  Kept deliberately separate from the PDE training data used by Simulate, and
  it is what makes Phase 3 runnable with no downloads at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..simulate.solvers import make_testbed
from .text import describe_field

Tensor = torch.Tensor

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "perception_proxy"


@dataclass
class PerceptionDataConfig:
    source: str = "synthetic"       # "synthetic" | "real"
    testbed: str = "dar"            # synthetic source only
    n_samples: int = 256
    image_size: int = 64
    grid: int = 64
    noise: float = 0.05
    blur_sigma: float = 1.0
    seed: int = 0
    root: str | None = None


def render_field(field: Tensor, noise: float = 0.05, blur_sigma: float = 1.0,
                 generator: torch.Generator | None = None) -> Tensor:
    """Field (C, H, W) -> sensor-like RGB image (3, H, W) in [0, 1].

    The forward sensor model is deliberately lossy - blur, noise, saturation and
    vignetting - so that recovering the field is a real inverse problem rather
    than a colour-map inversion.
    """
    u = field[0:1]
    u = (u - u.amin()) / (u.amax() - u.amin()).clamp_min(1e-6)

    # Three-band response with different sensitivities (a crude spectral index).
    bands = torch.cat([u**0.7, u, 1.0 - u**1.4], dim=0)

    if blur_sigma > 0:
        radius = max(1, int(3 * blur_sigma))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=u.device)
        kernel = torch.exp(-(coords**2) / (2 * blur_sigma**2))
        kernel = kernel / kernel.sum()
        bands = F.conv2d(
            F.pad(bands.unsqueeze(0), (radius, radius, 0, 0), mode="circular"),
            kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3,
        )
        bands = F.conv2d(
            F.pad(bands, (0, 0, radius, radius), mode="circular"),
            kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3,
        ).squeeze(0)

    grid = bands.shape[-1]
    lin = torch.linspace(-1, 1, grid, device=u.device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    vignette = 1.0 - 0.25 * (xx**2 + yy**2)
    bands = bands * vignette

    if noise > 0:
        shot = torch.randn(bands.shape, generator=generator) * noise
        bands = bands + shot.to(bands.device) * bands.clamp_min(0.05).sqrt()
    return bands.clamp(0.0, 1.0)


class PerceptionDataset(Dataset):
    """Returns (image, field, caption)."""

    def __init__(self, cfg: PerceptionDataConfig, split: str = "train",
                 train_frac: float = 0.8) -> None:
        self.cfg = cfg
        self.split = split
        if cfg.source == "real":
            self.items = self._load_real(train_frac)
        else:
            self.items = self._build_synthetic(train_frac)

    # -- synthetic ---------------------------------------------------------- #
    def _build_synthetic(self, train_frac: float) -> list[tuple[Tensor, Tensor, str]]:
        cfg = self.cfg
        generator = torch.Generator().manual_seed(cfg.seed + (0 if self.split == "train" else 1))
        testbed = make_testbed(cfg.testbed, grid=cfg.grid)
        n = cfg.n_samples if self.split == "train" else max(8, int(cfg.n_samples * (1 - train_frac)))

        state = testbed.initial_condition(n, generator)
        # A few solver steps so the fields carry real PDE structure, not just noise.
        for _ in range(4):
            state = testbed.step(state)

        items = []
        for i in range(n):
            field = state[i, :1]
            image = render_field(field, cfg.noise, cfg.blur_sigma, generator)
            if image.shape[-1] != cfg.image_size:
                image = F.interpolate(
                    image.unsqueeze(0), size=(cfg.image_size, cfg.image_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            items.append((image, field, describe_field(field)))
        return items

    # -- real imagery ------------------------------------------------------- #
    def _load_real(self, train_frac: float) -> list[tuple[Tensor, Tensor, str]]:
        root = Path(self.cfg.root or DATA_ROOT)
        image_dir, field_dir = root / "images", root / "fields"
        if not image_dir.exists():
            raise FileNotFoundError(
                f"no proxy imagery at {image_dir}. See data/perception_proxy/README.md "
                "for free Sentinel-2 / Landsat / NAIP download recipes, or use "
                "source=synthetic."
            )
        caption_path = root / "captions.json"
        captions = json.loads(caption_path.read_text()) if caption_path.exists() else {}

        ids = sorted(p.stem for p in image_dir.glob("*.png"))
        cut = max(1, int(train_frac * len(ids)))
        ids = ids[:cut] if self.split == "train" else ids[cut:]

        from PIL import Image  # imported lazily: only the real path needs pillow

        items = []
        for sample_id in ids:
            img = Image.open(image_dir / f"{sample_id}.png").convert("RGB")
            img = img.resize((self.cfg.image_size, self.cfg.image_size))
            image = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
            field = torch.from_numpy(np.load(field_dir / f"{sample_id}.npy")).float()
            if field.ndim == 2:
                field = field.unsqueeze(0)
            caption = captions.get(sample_id) or describe_field(field)
            items.append((image, field, caption))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        return self.items[idx]


def collate(batch: list[tuple[Tensor, Tensor, str]]) -> tuple[Tensor, Tensor, list[str]]:
    images = torch.stack([b[0] for b in batch])
    fields = torch.stack([b[1] for b in batch])
    return images, fields, [b[2] for b in batch]
