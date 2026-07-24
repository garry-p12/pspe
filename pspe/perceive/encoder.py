"""Perception backbone: frozen open-weight VLM + LoRA adapters + field head.

Two backbone paths, one interface:

* `backbone="tiny"`   - an in-tree patch-transformer vision encoder. No
  download, runs on CPU, and still exercises the frozen-backbone/LoRA-only
  training path so the Phase 3 acceptance check is meaningful offline.

* `backbone="Qwen/Qwen2-VL-2B-Instruct"` (or `vikhyatk/moondream2`, or any
  HF vision model) - loaded through `transformers`, backbone frozen, LoRA
  attached with `peft`. 4-bit quantisation via `bitsandbytes` is requested only
  on CUDA; bitsandbytes has no macOS/arm64 wheels, so on Apple silicon the
  model loads in bf16 instead of failing.

Both produce a patch-token grid, which `FieldDecoder` turns back into the
physical field the Simulate module consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .lora import inject_lora, mark_only_lora_trainable

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Offline backbone
# --------------------------------------------------------------------------- #
class _TinyBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.heads = heads
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (b, n, self.heads, d // self.heads)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q.view(shape).transpose(1, 2),
            k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2),
        )
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class TinyVisionBackbone(nn.Module):
    """Patch-transformer stand-in for the VLM vision tower."""

    hidden_size: int

    def __init__(
        self, image_size: int = 64, patch: int = 8, width: int = 128,
        depth: int = 4, heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.grid_tokens = image_size // patch
        self.hidden_size = width
        self.embed = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid_tokens**2, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_TinyBlock(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)

    def forward(self, images: Tensor) -> Tensor:
        x = self.embed(images).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # (B, n_tokens, width)


# --------------------------------------------------------------------------- #
# Field decoder
# --------------------------------------------------------------------------- #
class FieldDecoder(nn.Module):
    """Patch tokens -> physical field (B, C, grid, grid)."""

    def __init__(
        self, hidden_size: int, out_channels: int, token_grid: int, out_grid: int,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.out_grid = out_grid
        self.project = nn.Linear(hidden_size, width)
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        b, n, _ = tokens.shape
        side = int(n**0.5)
        if side * side != n:  # VLM towers may prepend a CLS/register token
            tokens = tokens[:, n - (int(n**0.5) ** 2) :]
            side = int(tokens.shape[1] ** 0.5)
        x = self.project(tokens).transpose(1, 2).reshape(b, -1, side, side)
        x = torch.nn.functional.interpolate(
            x, size=(self.out_grid, self.out_grid), mode="bilinear", align_corners=False
        )
        return self.net(x)


# --------------------------------------------------------------------------- #
# Full perception module
# --------------------------------------------------------------------------- #
@dataclass
class PerceiveConfig:
    backbone: str = "tiny"
    image_size: int = 64
    out_grid: int = 64
    out_channels: int = 1
    lora_r: int = 8
    lora_alpha: int = 16
    embed_dim: int = 128        # shared space for the contrastive term
    quant: str = "none"         # "none" | "4bit" (4bit is CUDA-only)


class PerceiveModule(nn.Module):
    """Frozen backbone + LoRA + field head + projection head for contrastive loss."""

    def __init__(self, cfg: PerceiveConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PerceiveConfig()
        self._backbone_is_hf = self.cfg.backbone != "tiny"
        self.backbone, hidden, token_grid = self._build_backbone()
        self.decoder = FieldDecoder(
            hidden, self.cfg.out_channels, token_grid, self.cfg.out_grid
        )
        self.image_projection = nn.Sequential(
            nn.Linear(hidden, self.cfg.embed_dim), nn.GELU(),
            nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1 / 0.07)

    @property
    def is_stub_backbone(self) -> bool:
        """True when running the offline stand-in rather than an open-weight VLM.

        Every run's `summary.json` carries this, so a stub number can never be
        mistaken for a Qwen2-VL/moondream2 number in a later draft.
        """
        return self.cfg.backbone == "tiny"

    # -- construction ------------------------------------------------------- #
    def _build_backbone(self) -> tuple[nn.Module, int, int]:
        if self.cfg.backbone == "tiny":
            backbone = TinyVisionBackbone(image_size=self.cfg.image_size)
            inject_lora(backbone, self.cfg.lora_r, self.cfg.lora_alpha)
            mark_only_lora_trainable(backbone)
            return backbone, backbone.hidden_size, backbone.grid_tokens
        return self._build_hf_backbone()

    def _build_hf_backbone(self) -> tuple[nn.Module, int, int]:
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
            from transformers import AutoModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                'backbone "%s" needs the optional extra: pip install -e ".[llm]"'
                % self.cfg.backbone
            ) from exc

        kwargs: dict[str, object] = {"trust_remote_code": True}
        if self.cfg.quant == "4bit" and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16

        import inspect

        model = AutoModel.from_pretrained(self.cfg.backbone, **kwargs)
        vision = getattr(model, "vision_model", None) or getattr(model, "visual", None) or model

        # Reject packed-patch vision towers (Qwen2-VL, some moondream builds):
        # their forward takes `grid_thw` and expects processor-packed pixel
        # patches, not a plain (B, C, H, W) tensor. Driving them needs the
        # model's image processor, which this generic field-regression encoder
        # does not wire. Point the user at vision towers that DO take pixel_values.
        forward_params = inspect.signature(vision.forward).parameters
        if "grid_thw" in forward_params or "image_grid_thw" in forward_params:
            raise ValueError(
                f"backbone '{self.cfg.backbone}' has a packed-patch vision tower "
                "(needs grid_thw); it cannot be driven as a plain image encoder "
                "here. Use a standard-ViT open-weight vision backbone instead, e.g. "
                "google/siglip-base-patch16-224, openai/clip-vit-base-patch32, or "
                "facebook/dinov2-small — all take pixel_values and run with LoRA "
                "on a T4."
            )

        for p in vision.parameters():
            p.requires_grad_(False)

        peft_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                            "qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
        )
        vision = get_peft_model(vision, peft_cfg)

        cfg = getattr(getattr(vision, "config", None), "vision_config", None) or vision.config
        hidden = int(getattr(cfg, "hidden_size", 768))
        input_size = int(getattr(cfg, "image_size", 224))
        patch = int(getattr(cfg, "patch_size", 16))
        # The token grid is set by the backbone's own resolution, not ours: the
        # encoder resizes our imagery to `input_size` before the tower sees it.
        token_grid = input_size // patch
        self._hf_input_size = input_size
        return vision, hidden, token_grid

    # -- forward ------------------------------------------------------------ #
    def encode(self, images: Tensor) -> Tensor:
        if self._backbone_is_hf:
            # Standard ViT vision towers want their trained resolution and
            # roughly [-1, 1] normalised pixels; resize + normalise here so raw
            # [0, 1] imagery of any size works.
            size = getattr(self, "_hf_input_size", 224)
            pixel_values = torch.nn.functional.interpolate(
                images, size=(size, size), mode="bilinear", align_corners=False
            )
            pixel_values = (pixel_values - 0.5) / 0.5
            pixel_values = pixel_values.to(next(self.backbone.parameters()).dtype)
            out = self.backbone(pixel_values=pixel_values)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            return hidden.float()
        out = self.backbone(images)
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        return out

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return (field_estimate, image_embedding)."""
        tokens = self.encode(images)
        field = self.decoder(tokens)
        embedding = self.image_projection(tokens.mean(dim=1))
        return field, embedding
