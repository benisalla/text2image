"""Decoder-only autoregressive transformer over VQ tokens (DALL-E 1, Parti, LlamaGen).

The text embeddings form a prefix; image codes are then predicted one at a time
in raster order with a causal transformer, exactly like a language model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from text2image.models.base import VQModel
from text2image.nn.layers import TransformerBlock


class AutoregressiveTransformer(VQModel):
    family = "autoregressive (discrete, next-token)"
    paper = "DALL-E (Ramesh et al. 2021), Parti (Yu et al. 2022), LlamaGen (Sun et al. 2024)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        vq_config: dict,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        text_dim: int = 128,
        text_depth: int = 2,
    ):
        super().__init__(vocab_size, text_len, vq_config, text_dim, text_depth)
        self.num_codes = self.vq.codebook_size
        self.num_tokens = self.vq.latent_size**2
        self.text_len = text_len
        self.prefix_proj = nn.Linear(text_dim, dim)
        self.tok = nn.Embedding(self.num_codes, dim)
        self.pos = nn.Parameter(torch.zeros(1, text_len + self.num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.num_codes)

    def forward(self, idx, seq):
        """Logits for image positions 0..len(idx) given the prefix and codes idx[:, :len]."""
        h = torch.cat([self.prefix_proj(seq), self.tok(idx)], dim=1)
        h = h + self.pos[:, : h.shape[1]]
        for blk in self.blocks:
            h = blk(h, is_causal=True)
        return self.head(self.norm(h[:, self.text_len - 1 :]))

    def loss(self, images, tokens):
        idx = self.vq.encode(images)[-1]
        seq, _ = self.encode_text(self.drop_prompts(tokens))
        logits = self(idx[:, :-1], seq)
        return F.cross_entropy(logits.reshape(-1, self.num_codes), idx.reshape(-1))

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, temperature=1.0, top_k=0, generator=None):
        b = tokens.shape[0]
        seq, _ = self.encode_text(tokens, with_null=True)
        idx = torch.zeros(b, 0, dtype=torch.long, device=tokens.device)
        for _ in range(self.num_tokens):
            logits = self.guide(self(torch.cat([idx, idx]), seq)[:, -1], cfg_scale) / temperature
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return self.vq.decode([idx])
