"""Explanation model: frozen small LM + LoRA, conditioned on the planner state.

Conditioning is a learned prefix: the planner's state/action summary is
projected into `n_prefix` embedding vectors that are prepended to the token
embeddings. This works identically for the in-tree LM and for a HuggingFace
causal LM (which accepts `inputs_embeds`), so the training loop does not branch.

Backbone paths:

* `backbone="tiny"` - a small causal transformer, randomly initialised, frozen,
  with LoRA adapters plus a trainable embedding/output head. Offline, CPU-fast,
  and it exercises the whole faithfulness objective end to end. It is a
  stand-in for the language model, not a claim about one.

* `backbone="Qwen/Qwen2.5-1.5B-Instruct"` (or `microsoft/Phi-3.5-mini-instruct`)
  - loaded via `transformers`, frozen, LoRA attached via `peft`, 4-bit
  quantisation requested only when CUDA is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..perceive.lora import inject_lora
from .tokenizer import WordTokenizer

Tensor = torch.Tensor


@dataclass
class ExplainConfig:
    backbone: str = "tiny"
    width: int = 128
    depth: int = 4
    heads: int = 4
    max_len: int = 96
    n_prefix: int = 4
    cond_dim: int = 64
    lora_r: int = 8
    lora_alpha: int = 16
    quant: str = "none"
    temperature: float = 0.8


class _CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.heads = heads

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (b, n, self.heads, d // self.heads)
        attn = F.scaled_dot_product_attention(
            q.view(shape).transpose(1, 2),
            k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2),
            is_causal=True,
        )
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class TinyCausalLM(nn.Module):
    """Word-level causal transformer over the brief vocabulary."""

    def __init__(self, vocab_size: int, cfg: ExplainConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.width)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len + cfg.n_prefix, cfg.width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_CausalBlock(cfg.width, cfg.heads) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, vocab_size)
        self.width = cfg.width

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        x = inputs_embeds + self.pos[:, : inputs_embeds.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class ExplainModule(nn.Module):
    """Prefix-conditioned brief generator with a frozen backbone."""

    def __init__(
        self,
        tokenizer: WordTokenizer,
        cfg: ExplainConfig | None = None,
        cond_features: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ExplainConfig()
        self.tokenizer = tokenizer
        self.backbone, self.width, self._is_hf = self._build_backbone(len(tokenizer))
        self.prefix = nn.Sequential(
            nn.Linear(cond_features, self.cfg.cond_dim), nn.GELU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.n_prefix * self.width),
        )

    @property
    def is_stub_backbone(self) -> bool:
        """True when running the offline stand-in rather than an open-weight LM.

        Recorded in every run's `summary.json`: a randomly-initialised 4-layer
        transformer's faithfulness score is not a Qwen2.5/Phi-3.5 result, and
        the two must never be confused in a results table.
        """
        return self.cfg.backbone == "tiny"

    # -- construction ------------------------------------------------------- #
    def _build_backbone(self, vocab_size: int) -> tuple[nn.Module, int, bool]:
        if self.cfg.backbone == "tiny":
            model = TinyCausalLM(vocab_size, self.cfg)
            inject_lora(model, self.cfg.lora_r, self.cfg.lora_alpha)
            # Freeze the transformer body; train LoRA adapters plus the
            # embedding/output head (the "LoRA head" of the plan).
            for name, param in model.named_parameters():
                trainable = (
                    "lora_a" in name
                    or "lora_b" in name
                    or name.startswith("embedding")
                    or name.startswith("head")
                )
                param.requires_grad_(trainable)
            return model, model.width, False
        return self._build_hf_backbone()

    def _build_hf_backbone(self) -> tuple[nn.Module, int, bool]:
        try:
            from peft import (  # type: ignore[import-not-found]
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                'backbone "%s" needs the optional extra: pip install -e ".[llm]"'
                % self.cfg.backbone
            ) from exc

        kwargs: dict[str, object] = {"trust_remote_code": True}
        use_4bit = self.cfg.quant == "4bit" and torch.cuda.is_available()
        if use_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(self.cfg.backbone, **kwargs)

        if use_4bit:
            # THE canonical QLoRA setup. Doing this by hand (freeze + manual
            # gradient_checkpointing_enable) is what caused the 1.5B backbone to
            # train as if it were full-precision (~38 GB, 16 bytes/param with
            # Adam). prepare_model_for_kbit_training freezes the base correctly,
            # casts norms/head to fp32, enables input grads, and wires gradient
            # checkpointing so only LoRA + the prefix ever carry optimizer state.
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
            for p in model.parameters():
                p.requires_grad_(False)
            try:  # non-quantised path: still checkpoint to save activation memory
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except Exception:
                pass

        model = get_peft_model(
            model,
            LoraConfig(
                r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM",
            ),
        )
        width = int(model.config.hidden_size)
        return model, width, True

    # -- embedding plumbing -------------------------------------------------- #
    def _embed_tokens(self, ids: Tensor) -> Tensor:
        if self._is_hf:
            # The frozen HF backbone is loaded in bf16/4-bit, but the trainable
            # prefix, embedding-cat and heads all run in float32. Return token
            # embeddings in float32 so `cat([prefix, tokens])` has one dtype;
            # `_logits` casts to the backbone's dtype only at its boundary. This
            # is what avoids "mat1 and mat2 have the same dtype: float != BFloat16".
            return self.backbone.get_input_embeddings()(ids).float()
        return self.backbone.embedding(ids)

    def _logits(self, inputs_embeds: Tensor) -> Tensor:
        if self._is_hf:
            weight_dtype = self.backbone.get_input_embeddings().weight.dtype
            out = self.backbone(inputs_embeds=inputs_embeds.to(weight_dtype))
            return out.logits.float()
        return self.backbone(inputs_embeds)

    def _prefix_embeds(self, condition: Tensor) -> Tensor:
        return self.prefix(condition).view(condition.shape[0], self.cfg.n_prefix, self.width)

    # -- training forward ---------------------------------------------------- #
    def forward(self, condition: Tensor, token_ids: Tensor, mask: Tensor) -> Tensor:
        """Per-sample NLL of `token_ids` given `condition`. Shape (B,)."""
        prefix = self._prefix_embeds(condition)
        embeds = torch.cat([prefix, self._embed_tokens(token_ids)], dim=1)
        logits = self._logits(embeds)[:, self.cfg.n_prefix - 1 : -1]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        picked = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        picked = picked * mask
        return -picked.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)

    # -- sampling ------------------------------------------------------------ #
    def generate(
        self,
        condition: Tensor,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> tuple[list[str], Tensor]:
        """Sample one brief per condition row.

        Returns (texts, sequence_log_prob). The log-prob keeps its graph, which
        is what the REINFORCE term in the faithfulness objective differentiates.
        """
        max_new_tokens = max_new_tokens or self.cfg.max_len
        temperature = temperature or self.cfg.temperature
        batch = condition.shape[0]
        device = condition.device

        prefix = self._prefix_embeds(condition)
        ids = torch.full((batch, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        total_logprob = torch.zeros(batch, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            embeds = torch.cat([prefix, self._embed_tokens(ids)], dim=1)
            logits = self._logits(embeds)[:, -1].float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            if greedy:
                next_id = log_probs.argmax(dim=-1)
            else:
                next_id = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            step_logprob = log_probs.gather(-1, next_id.unsqueeze(-1)).squeeze(-1)
            total_logprob = total_logprob + step_logprob * (~finished).float()

            ids = torch.cat([ids, next_id.unsqueeze(-1)], dim=1)
            finished = finished | (next_id == self.tokenizer.eos_id)
            if bool(finished.all()):
                break

        texts = [self.tokenizer.decode(row[1:]) for row in ids]
        return texts, total_logprob

    def _score_sequence(self, condition: Tensor, seq_ids: Tensor, seq_mask: Tensor) -> Tensor:
        """Summed log-prob of a fixed token sequence under the policy. (B,).

        One teacher-forced forward, differentiable through the prefix and LoRA.
        This is what the REINFORCE term backprops through — NOT the autoregressive
        sampling loop.
        """
        prefix = self._prefix_embeds(condition)
        embeds = torch.cat([prefix, self._embed_tokens(seq_ids)], dim=1)
        logits = self._logits(embeds)[:, self.cfg.n_prefix - 1 : -1]
        logp = F.log_softmax(logits.float(), dim=-1)
        picked = logp.gather(-1, seq_ids.unsqueeze(-1)).squeeze(-1)
        return (picked * seq_mask).sum(dim=-1)

    def sample_and_score(
        self,
        condition: Tensor,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], Tensor]:
        """Memory-efficient REINFORCE rollout: sample without grad, score once.

        The autoregressive loop in `generate` retains the graph across every
        step — O(L) full-model forwards with all activations kept, which explodes
        memory for a billion-parameter backbone. Here the sampling runs under
        `no_grad` (cheap), then the sampled sequence is scored with a single
        teacher-forced forward that carries the gradient. Identical REINFORCE
        estimator, bounded memory.
        """
        with torch.no_grad():
            texts, _ = self.generate(condition, max_new_tokens, temperature, greedy=False)
        seq_ids, mask = self.tokenizer.batch_encode(texts, max_len=self.cfg.max_len)
        seq_ids = seq_ids.to(condition.device)
        mask = mask.to(condition.device).float()
        return texts, self._score_sequence(condition, seq_ids, mask)
