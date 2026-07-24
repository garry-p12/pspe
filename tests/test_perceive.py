"""Phase 3: LoRA plumbing, rendering, losses, and the frozen-backbone check."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pspe.perceive import (
    LoRALinear,
    PerceiveConfig,
    PerceiveModule,
    PerceiveTrainConfig,
    PerceiveTrainer,
    PerceptionDataConfig,
    PerceptionDataset,
    assert_lora_only,
    collate,
    info_nce,
    inject_lora,
    mark_only_lora_trainable,
    render_field,
)
from pspe.perceive.text import WeakTextEncoder, describe_field
from pspe.simulate import make_testbed
from pspe.utils import RunLogger, count_parameters

GRID = 16


def test_lora_is_identity_at_initialisation() -> None:
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=4)
    x = torch.randn(3, 8)
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)


def test_lora_injection_freezes_the_base_weights() -> None:
    module = nn.Sequential(nn.Linear(8, 8), nn.GELU())
    module[0] = nn.Linear(8, 8)
    holder = nn.Module()
    holder.proj = nn.Linear(8, 8)
    replaced = inject_lora(holder, r=2)
    mark_only_lora_trainable(holder)

    assert replaced == 1
    trainable, total = count_parameters(holder)
    assert 0 < trainable < total
    assert not holder.proj.base.weight.requires_grad


def test_perceive_module_backbone_stays_frozen() -> None:
    model = PerceiveModule(PerceiveConfig(image_size=GRID, out_grid=GRID))
    stats = assert_lora_only(model)
    assert stats["params/backbone_trainable"] < stats["params/backbone_total"]
    assert stats["params/trainable_fraction"] < 1.0


def test_unfreezing_the_backbone_trips_the_check() -> None:
    model = PerceiveModule(PerceiveConfig(image_size=GRID, out_grid=GRID))
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    with pytest.raises(AssertionError):
        assert_lora_only(model)


def test_render_field_produces_a_lossy_rgb_image() -> None:
    testbed = make_testbed("dar", grid=GRID)
    field = testbed.initial_condition(1, torch.Generator().manual_seed(0))[0]
    image = render_field(field, noise=0.05, blur_sigma=1.0)
    assert image.shape == (3, GRID, GRID)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_describe_field_mentions_a_direction_and_a_level() -> None:
    testbed = make_testbed("dar", grid=GRID)
    field = testbed.initial_condition(1, torch.Generator().manual_seed(0))[0]
    caption = describe_field(field)
    assert "concentration" in caption and "mean level" in caption


def test_info_nce_is_lower_for_aligned_embeddings() -> None:
    torch.manual_seed(0)
    embeds = torch.randn(8, 16)
    scale = torch.tensor(2.6592)
    aligned, aligned_acc = info_nce(embeds, embeds.clone(), scale)
    shuffled, _ = info_nce(embeds, embeds[torch.randperm(8)], scale)
    assert float(aligned) < float(shuffled)
    assert aligned_acc == 1.0


def test_weak_text_encoder_is_deterministic() -> None:
    encoder = WeakTextEncoder(embed_dim=16).eval()
    texts = ["high concentration with peak in the north", "low concentration"]
    with torch.no_grad():
        first, second = encoder(texts), encoder(texts)
    assert torch.allclose(first, second)


def test_synthetic_dataset_shapes() -> None:
    cfg = PerceptionDataConfig(n_samples=8, image_size=GRID, grid=GRID)
    dataset = PerceptionDataset(cfg, "train")
    images, fields, captions = collate([dataset[i] for i in range(4)])
    assert images.shape == (4, 3, GRID, GRID)
    assert fields.shape == (4, 1, GRID, GRID)
    assert len(captions) == 4


def test_trainer_runs_and_reports_held_out_error(tmp_path) -> None:
    trainer = PerceiveTrainer(
        PerceiveConfig(image_size=GRID, out_grid=GRID),
        PerceptionDataConfig(n_samples=8, image_size=GRID, grid=GRID),
        PerceiveTrainConfig(epochs=1, batch=4),
        RunLogger(tmp_path / "perceive"),
    )
    summary = trainer.train()
    assert "val/loss/regression" in summary
    assert summary["params/trainable_fraction"] < 1.0
    assert summary["wall_clock_s"] > 0.0


def test_stub_backbone_is_watermarked_in_the_summary(tmp_path) -> None:
    """A stand-in-backbone run must be impossible to mistake for a VLM run."""
    import json

    trainer = PerceiveTrainer(
        PerceiveConfig(image_size=GRID, out_grid=GRID),
        PerceptionDataConfig(n_samples=8, image_size=GRID, grid=GRID),
        PerceiveTrainConfig(epochs=1, batch=4),
        RunLogger(tmp_path / "perceive"),
    )
    summary = trainer.train()
    assert summary["backbone_is_stub"] is True
    assert summary["backbone"] == "tiny"

    on_disk = json.loads((tmp_path / "perceive" / "summary.json").read_text())
    assert on_disk["backbone_is_stub"] is True
