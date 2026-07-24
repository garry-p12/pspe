"""Closed-vocabulary word tokenizer for briefs.

Briefs live in a templated sub-language, so a word-level vocabulary built from
`build_vocabulary` covers them exactly. This is what lets Phase 4 run offline
with no tokenizer download; the HuggingFace path uses the model's own tokenizer.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


class WordTokenizer:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, vocabulary: list[str]) -> None:
        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        words = specials + [w for w in vocabulary if w not in specials]
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        unk = self.stoi[self.UNK]
        ids = [self.stoi.get(tok, unk) for tok in text.split()]
        return [self.bos_id] + ids + [self.eos_id] if add_special else ids

    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return " ".join(self.itos[i] for i in ids if i not in skip)

    def batch_encode(self, texts: list[str], max_len: int | None = None) -> tuple[Tensor, Tensor]:
        """Return (token_ids, attention_mask), right-padded."""
        encoded = [self.encode(t) for t in texts]
        length = max_len or max(len(e) for e in encoded)
        ids = torch.full((len(encoded), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(encoded), length, dtype=torch.bool)
        for i, seq in enumerate(encoded):
            seq = seq[:length]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = True
        return ids, mask

    def unknown_rate(self, text: str) -> float:
        """Fraction of tokens outside the vocabulary - a template-drift check."""
        tokens = text.split()
        if not tokens:
            return 0.0
        return sum(t not in self.stoi for t in tokens) / len(tokens)
