"""EuroSAT (Sentinel-2) as a real image -> physical-field perception task.

The synthetic perception path renders a PDE field to imagery and asks the
encoder to invert the renderer. This module is the real-data counterpart: it
asks the encoder to recover a **vegetation index field (NDVI)** from a
Sentinel-2 RGB patch.

Why this is a genuine inverse problem, not a lookup:

    NDVI = (NIR - Red) / (NIR + Red)

NDVI needs the near-infrared band (B08), which is *not present* in an RGB image
(B04/B03/B02). So mapping RGB -> NDVI is a real reconstruction of a physical
quantity the input does not directly contain — exactly the "recover a physical
field from imagery" task in Section 4.1 of the paper, on real satellite data
instead of a synthetic renderer.

EuroSAT is fully open, hosted on the HuggingFace Hub, and needs no Copernicus
account or API key — so it runs on a free Colab/Kaggle GPU with one
`datasets.load_dataset` call.

The heavy dependency (`datasets`) is imported lazily inside the loader; the pure
functions here (NDVI, band extraction, captioning) have no such dependency and
are unit-tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# EuroSAT MSI band order (13 bands), 0-indexed:
#   0:B01 1:B02(blue) 2:B03(green) 3:B04(red) 4:B05 5:B06 6:B07
#   7:B08(NIR) 8:B08A 9:B09 10:B10 11:B11 12:B12
BAND_BLUE, BAND_GREEN, BAND_RED, BAND_NIR = 1, 2, 3, 7

# EuroSAT land-cover classes, used only for weak text supervision.
EUROSAT_CLASSES = [
    "annual crop", "forest", "herbaceous vegetation", "highway", "industrial",
    "pasture", "permanent crop", "residential", "river", "sea or lake",
]


@dataclass
class EuroSATConfig:
    hf_id: str = "blanchon/EuroSAT_MSI"  # 13-band multispectral EuroSAT on the Hub
    split: str = "train"
    n_samples: int = 512
    image_size: int = 64
    train_frac: float = 0.8
    seed: int = 0
    cache_dir: str | None = None


# --------------------------------------------------------------------------- #
# Pure functions (no `datasets` dependency; unit-tested offline)
# --------------------------------------------------------------------------- #
def compute_ndvi(bands: Tensor, eps: float = 1e-6) -> Tensor:
    """(>=8, H, W) reflectance stack -> (1, H, W) NDVI field in [-1, 1].

    NDVI = (NIR - Red) / (NIR + Red). The target the encoder must reconstruct
    from RGB alone.
    """
    nir = bands[BAND_NIR].float()
    red = bands[BAND_RED].float()
    ndvi = (nir - red) / (nir + red + eps)
    return ndvi.clamp(-1.0, 1.0).unsqueeze(0)


def extract_rgb(bands: Tensor) -> Tensor:
    """(>=4, H, W) stack -> (3, H, W) RGB in [0, 1], per-image max-normalised.

    Sentinel-2 reflectance is unbounded-ish; a per-image max keeps the network
    input in a stable range without leaking the NDVI target (which uses NIR,
    dropped here).
    """
    rgb = torch.stack([bands[BAND_RED], bands[BAND_GREEN], bands[BAND_BLUE]], dim=0).float()
    return (rgb / rgb.amax().clamp_min(1e-6)).clamp(0.0, 1.0)


def ndvi_caption(ndvi: Tensor, label: int | None = None) -> str:
    """Weak natural-language supervision from NDVI statistics and land cover."""
    mean = float(ndvi.mean())
    high = float((ndvi > 0.3).float().mean())  # vegetated fraction
    if mean > 0.4:
        cover = "dense vegetation"
    elif mean > 0.15:
        cover = "moderate vegetation"
    elif mean > 0.0:
        cover = "sparse vegetation"
    else:
        cover = "bare or water surface"
    scene = f"{EUROSAT_CLASSES[label]}; " if label is not None and 0 <= label < len(EUROSAT_CLASSES) else ""
    return (
        f"{scene}{cover}, mean NDVI {mean:.2f}, "
        f"{high * 100:.0f}% of the scene vegetated"
    )


def _resize(x: Tensor, size: int) -> Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(0)


def bands_to_example(
    bands: Tensor, label: int | None, image_size: int
) -> tuple[Tensor, Tensor, str]:
    """One 13-band patch -> (rgb_image, ndvi_field, caption), all at image_size."""
    ndvi = _resize(compute_ndvi(bands), image_size)
    rgb = _resize(extract_rgb(bands), image_size)
    return rgb, ndvi, ndvi_caption(ndvi, label)


# --------------------------------------------------------------------------- #
# HuggingFace loader (lazy `datasets` import)
# --------------------------------------------------------------------------- #
def load_eurosat(cfg: EuroSATConfig, split: str) -> list[tuple[Tensor, Tensor, str]]:
    """Return a list of (rgb, ndvi, caption). `split` is "train" or "val".

    Downloads once from the Hub (no credentials). Geographic leakage between
    train and val is not controlled here — EuroSAT ships no geo split — so
    reported numbers should be read as in-distribution reconstruction, and a
    held-out region is the stronger protocol when the metadata allows it.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - Colab has it
        raise ImportError(
            "EuroSAT needs the `datasets` library: pip install datasets. "
            "It is included in requirements-notebook.txt for the Colab path."
        ) from exc

    raw = load_dataset(cfg.hf_id, split=cfg.split, cache_dir=cfg.cache_dir)
    n = min(cfg.n_samples, len(raw))
    cut = max(1, int(cfg.train_frac * n))
    indices = range(0, cut) if split == "train" else range(cut, n)

    items: list[tuple[Tensor, Tensor, str]] = []
    for i in indices:
        row = raw[i]
        bands = _row_to_bands(row)
        label = row.get("label") if isinstance(row, dict) else None
        items.append(bands_to_example(bands, label, cfg.image_size))
    return items


def _row_to_bands(row: dict) -> Tensor:
    """Normalise the several shapes EuroSAT_MSI rows appear in into (13, H, W).

    Depending on the Hub revision, the multispectral image arrives as a numpy
    array, a list of PIL bands, or a dict of named bands. Handle the common
    cases and fail loudly on the rest rather than silently mis-indexing.
    """
    img = row.get("image", row.get("img", row))
    if isinstance(img, np.ndarray):
        arr = img
    elif hasattr(img, "numpy"):
        arr = img.numpy()
    elif isinstance(img, (list, tuple)):
        arr = np.stack([np.asarray(b) for b in img], axis=0)
    else:
        arr = np.asarray(img)

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if tensor.ndim == 2:
        raise ValueError("EuroSAT row is single-band; the MSI (13-band) config is required")
    # Move the band axis to front if it is last (H, W, C) -> (C, H, W).
    if tensor.shape[0] > tensor.shape[-1] and tensor.shape[-1] <= 13:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] < 8:
        raise ValueError(
            f"EuroSAT patch has {tensor.shape[0]} bands; need >= 8 for the NIR band. "
            "Use the multispectral config (hf_id='blanchon/EuroSAT_MSI'), not RGB."
        )
    return tensor
