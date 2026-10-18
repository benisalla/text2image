"""Small trainable text encoder (stand-in for CLIP / T5 / an LLM encoder)."""

from __future__ import annotations

import torch
from torch import nn

from text2image.nn.layers import TransformerBlock


class TextEncoder(nn.Module):
    """Token embedding + learned positions + a few bidirectional transformer blocks.

    Returns the per-token sequence (for cross-attention / joint attention) and a
    pooled vector (for adaLN / global conditioning), mirroring how SD3 and FLUX
    use both a T5 sequence and a CLIP pooled embedding.
    """

    def __init__(self, vocab_size: int, max_len: int, dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok(tokens) + self.pos[:, : tokens.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, x.mean(dim=1)
