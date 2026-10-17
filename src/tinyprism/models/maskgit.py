"""MaskGIT / Muse: masked generative transformer over VQ tokens.

Training: mask a random (cosine-scheduled) fraction of the code grid and
predict the masked codes with a *bidirectional* transformer that cross-attends
to the text. Sampling: start fully masked and, over a handful of steps, commit
the most confident predictions while re-masking the rest (parallel decoding).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.models.base import VQModel
from tinyprism.nn.layers import TransformerBlock


class MaskGIT(VQModel):
    family = "masked generative (discrete)"
    paper = "MaskGIT (Chang et al. 2022), Muse (Chang et al. 2023)"

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
        self.mask_id = self.num_codes
        self.tok = nn.Embedding(self.num_codes + 1, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, context_dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.num_codes)

    def forward(self, idx, seq):
        ctx = self.ctx_proj(seq)
        h = self.tok(idx) + self.pos
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.head(self.norm(h))

    def loss(self, images, tokens):
        idx = self.vq.encode(images)[-1]
        b, n = idx.shape
        seq, _ = self.encode_text(self.drop_prompts(tokens))
        ratio = torch.cos(torch.rand(b, 1, device=idx.device) * math.pi / 2)
        n_mask = (ratio * n).ceil().clamp(min=1)
        rank = torch.rand(b, n, device=idx.device).argsort(dim=1).argsort(dim=1)
        mask = rank < n_mask
        logits = self(torch.where(mask, self.mask_id, idx), seq)
        return F.cross_entropy(logits[mask], idx[mask])

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, steps=10, temperature=4.5, generator=None):
        b, n = tokens.shape[0], self.num_tokens
        seq, _ = self.encode_text(tokens, with_null=True)
        idx = torch.full((b, n), self.mask_id, dtype=torch.long, device=tokens.device)
        for s in range(steps):
            unknown = idx == self.mask_id
            logits = self.guide(self(torch.cat([idx, idx]), seq), cfg_scale)
            probs = logits.softmax(-1)
            sampled = torch.multinomial(probs.view(-1, self.num_codes), 1, generator=generator).view(b, n)
            sampled = torch.where(unknown, sampled, idx)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).log()
            # Gumbel noise on the confidences, annealed to zero (MaskGIT's "choice temperature").
            u = torch.rand(b, n, generator=generator, device=idx.device).clamp(1e-9, 1 - 1e-9)
            conf = conf + temperature * (1 - (s + 1) / steps) * -torch.log(-torch.log(u))
            conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
            n_mask = int(math.floor(n * math.cos(math.pi / 2 * (s + 1) / steps)))
            n_mask = max(0, min(n_mask, int(unknown.sum(1).min()) - 1))
            idx = sampled
            if n_mask > 0:
                remask = conf.argsort(dim=1)[:, :n_mask]
                idx = idx.scatter(1, remask, self.mask_id)
        return self.vq.decode([idx])
