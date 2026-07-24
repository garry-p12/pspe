"""Phase 3 trainer: LoRA-only adaptation of a frozen perception backbone.

The acceptance check is enforced here, not just reported: `assert_lora_only`
raises if any backbone parameter is trainable, and the trainable/total split is
written to the log on the first line of every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..utils.common import count_parameters, peak_memory_mb, timer
from ..utils.logging import RunLogger
from .dataset import PerceptionDataConfig, PerceptionDataset, collate
from .encoder import PerceiveConfig, PerceiveModule
from .losses import perceive_loss
from .text import WeakTextEncoder

Tensor = torch.Tensor


@dataclass
class PerceiveTrainConfig:
    epochs: int = 10
    batch: int = 16
    lr: float = 1e-3
    w_regression: float = 1.0
    w_contrastive: float = 0.2
    freeze_encoder: bool = True     # ablation switch: frozen vs fine-tuned encoder
    seed: int = 0
    log_dir: str = "runs/perceive"


def assert_lora_only(module: PerceiveModule) -> dict[str, float]:
    """Verify the backbone is frozen; return the parameter-count breakdown."""
    backbone_trainable = [
        n for n, p in module.backbone.named_parameters()
        if p.requires_grad and "lora" not in n
    ]
    if backbone_trainable:
        raise AssertionError(
            f"{len(backbone_trainable)} backbone parameters are trainable, "
            f"e.g. {backbone_trainable[:3]}; expected LoRA adapters only"
        )
    trainable, total = count_parameters(module)
    bb_trainable, bb_total = count_parameters(module.backbone)
    return {
        "params/trainable": float(trainable),
        "params/total": float(total),
        "params/trainable_fraction": trainable / max(total, 1),
        "params/backbone_trainable": float(bb_trainable),
        "params/backbone_total": float(bb_total),
    }


class PerceiveTrainer:
    def __init__(
        self,
        model_cfg: PerceiveConfig | None = None,
        data_cfg: PerceptionDataConfig | None = None,
        train_cfg: PerceiveTrainConfig | None = None,
        logger: RunLogger | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = train_cfg or PerceiveTrainConfig()
        self.device = torch.device(device)
        self.model = PerceiveModule(model_cfg).to(self.device)
        self.text_encoder = WeakTextEncoder(embed_dim=self.model.cfg.embed_dim).to(self.device)

        if not self.cfg.freeze_encoder:
            # Ablation: unfreeze the backbone. Reported, never silent.
            for p in self.model.backbone.parameters():
                p.requires_grad_(True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        params += list(self.text_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)

        data_cfg = data_cfg or PerceptionDataConfig()
        self.train_set = PerceptionDataset(data_cfg, "train")
        self.val_set = PerceptionDataset(data_cfg, "val")
        self.logger = logger or RunLogger(self.cfg.log_dir)

    def train(self) -> dict[str, float]:
        param_stats = (
            assert_lora_only(self.model)
            if self.cfg.freeze_encoder
            else {"params/trainable_fraction": 1.0}
        )
        self.logger.log(0, **param_stats)

        loader = DataLoader(
            self.train_set, batch_size=self.cfg.batch, shuffle=True, collate_fn=collate
        )
        step = 0
        with timer() as clock:
            for epoch in range(self.cfg.epochs):
                self.model.train()
                for images, fields, captions in loader:
                    images = images.to(self.device)
                    fields = fields.to(self.device)
                    field_pred, image_embed = self.model(images)
                    text_embed = self.text_encoder(captions)
                    loss, components = perceive_loss(
                        field_pred, fields, image_embed, text_embed,
                        self.model.logit_scale,
                        self.cfg.w_regression, self.cfg.w_contrastive,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    step += 1
                    self.logger.log(step, epoch=epoch, **components)

        metrics = self.evaluate()
        summary = {
            **metrics,
            **param_stats,
            # Watermark: a stand-in-backbone number must never be cited as a
            # Qwen2-VL/moondream2 result in a draft.
            "backbone": self.model.cfg.backbone,
            "backbone_is_stub": self.model.is_stub_backbone,
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
        }
        self.logger.log_summary(**summary)
        return summary

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Field-reconstruction error on the held-out split."""
        self.model.eval()
        loader = DataLoader(
            self.val_set, batch_size=self.cfg.batch, shuffle=False, collate_fn=collate
        )
        totals: dict[str, float] = {}
        batches = 0
        for images, fields, captions in loader:
            images, fields = images.to(self.device), fields.to(self.device)
            field_pred, image_embed = self.model(images)
            text_embed = self.text_encoder(captions)
            _, components = perceive_loss(
                field_pred, fields, image_embed, text_embed, self.model.logit_scale,
                self.cfg.w_regression, self.cfg.w_contrastive,
            )
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        return {f"val/{k}": v / max(batches, 1) for k, v in totals.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "text_encoder": self.text_encoder.state_dict(),
                "model_config": self.model.cfg.__dict__,
            },
            path,
        )
