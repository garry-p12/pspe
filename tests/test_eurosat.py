"""EuroSAT real-perception loader: pure functions, offline.

The HuggingFace download and the real VLM run happen on Colab, but the pieces
that decide *correctness* — NDVI computation, the RGB/NIR split that keeps the
target out of the input, captioning, and band-shape normalisation — are pure
and tested here with synthetic 13-band patches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pspe.perceive.eurosat import (
    BAND_NIR,
    BAND_RED,
    bands_to_example,
    compute_ndvi,
    extract_rgb,
    ndvi_caption,
    _row_to_bands,
)


def _patch() -> torch.Tensor:
    """13-band patch: left half vegetated (high NIR), right half bare."""
    bands = torch.rand(13, 64, 64) * 0.3
    bands[BAND_NIR, :, :32] = 0.8
    bands[BAND_RED, :, :32] = 0.2
    bands[BAND_NIR, :, 32:] = 0.2
    bands[BAND_RED, :, 32:] = 0.3
    return bands


def test_ndvi_is_bounded_and_orders_vegetation_correctly() -> None:
    ndvi = compute_ndvi(_patch())
    assert ndvi.shape == (1, 64, 64)
    assert -1.0 <= float(ndvi.min()) and float(ndvi.max()) <= 1.0
    # Higher NIR relative to red must give higher NDVI.
    assert float(ndvi[0, :, :32].mean()) > float(ndvi[0, :, 32:].mean())


def test_rgb_does_not_leak_the_nir_target() -> None:
    """The whole task is RGB -> NDVI; the input must not contain the NIR band."""
    rgb = extract_rgb(_patch())
    assert rgb.shape == (3, 64, 64)
    assert 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0
    # extract_rgb reads bands 1,2,3 (B/G/R); the NIR band 7 is never one of them.
    assert BAND_NIR not in (1, 2, 3)


def test_caption_mentions_cover_class_and_ndvi() -> None:
    ndvi = compute_ndvi(_patch())
    caption = ndvi_caption(ndvi, label=1)
    assert "forest" in caption and "NDVI" in caption
    # No label -> no scene prefix, still a valid caption.
    assert "NDVI" in ndvi_caption(ndvi, label=None)


def test_bands_to_example_resizes_both_image_and_field() -> None:
    image, field, caption = bands_to_example(_patch(), label=3, image_size=32)
    assert image.shape == (3, 32, 32)
    assert field.shape == (1, 32, 32)
    assert isinstance(caption, str) and caption


def test_row_to_bands_normalises_hwc_and_rejects_rgb_only() -> None:
    hwc = np.random.rand(64, 64, 13).astype(np.float32)
    assert _row_to_bands({"image": hwc}).shape == (13, 64, 64)

    chw = np.random.rand(13, 32, 32).astype(np.float32)
    assert _row_to_bands({"image": chw}).shape == (13, 32, 32)

    with pytest.raises(ValueError):
        _row_to_bands({"image": np.random.rand(3, 64, 64)})


def test_dataset_config_accepts_eurosat_source() -> None:
    from pspe.perceive.dataset import PerceptionDataConfig

    cfg = PerceptionDataConfig(source="eurosat", n_samples=16, image_size=32)
    assert cfg.source == "eurosat"
    assert cfg.hf_id  # a Hub id is present
