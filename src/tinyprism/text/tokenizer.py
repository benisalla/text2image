"""Word-level tokenizer over the closed vocabulary of the shapes captions.

Real systems use a frozen CLIP / T5 / LLM encoder; for a toy grammar a word
vocabulary is exact and needs no downloads. Index 0 is padding, and an
all-padding sequence doubles as the "null" prompt for classifier-free guidance.
"""

from __future__ import annotations

import torch

from tinyprism.data.shapes import VOCAB_WORDS

PAD, UNK = "<pad>", "<unk>"


class WordTokenizer:
    def __init__(self, words: tuple[str, ...] = VOCAB_WORDS, max_len: int = 10):
        self.itos = [PAD, UNK, *words]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, captions: list[str]) -> torch.Tensor:
        out = torch.zeros(len(captions), self.max_len, dtype=torch.long)
        for i, text in enumerate(captions):
            ids = [self.stoi.get(w, 1) for w in text.lower().replace(",", " ").split()][: self.max_len]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def decode(self, tokens: torch.Tensor) -> list[str]:
        return [" ".join(self.itos[t] for t in row.tolist() if t != 0) for row in tokens]
