"""VAR: visual autoregressive modelling by next-*scale* prediction.

Instead of predicting one token at a time in raster order, VAR predicts a whole
token map per step, from 1x1 up to the full latent resolution. All tokens of a
scale are produced in parallel and attend to every token of the same or
coarser scales (block-causal mask). Text enters through a start token built
from the pooled embedding and cross-attention to the text sequence, as in the
text-to-image follow-ups (Infinity, STAR, Switti).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.models.base import VQModel
from tinyprism.nn.layers import TransformerBlock


class VAR(VQModel):
    family = "autoregressive (discrete, next-scale)"
    paper = "VAR (Tian et al., NeurIPS 2024 best paper), Infinity (Han et al. 2025)"

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
        self.scales = self.vq.scales
        self.num_codes = self.vq.codebook_size
        lens = self.vq.tokens_per_scale
        self.total = sum(lens)
        self.sos = nn.Linear(text_dim, dim)
        self.word_embed = nn.Linear(self.vq.z_dim, dim)
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.lvl_embed = nn.Embedding(len(self.scales), dim)
        self.pos = nn.Parameter(torch.zeros(1, self.total, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, context_dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.num_codes)
        lvl = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
        self.register_buffer("lvl", lvl, persistent=False)
        self.register_buffer("attn_mask", lvl[:, None] >= lvl[None, :], persistent=False)

    def forward(self, x_in, seq, pooled):
        """x_in: (B, L-1, z_dim) teacher-forcing inputs for scales >= 1; returns logits (B, L, K)."""
        h = torch.cat([self.sos(pooled).unsqueeze(1), self.word_embed(x_in)], dim=1)
        n = h.shape[1]
        h = h + self.pos[:, :n] + self.lvl_embed(self.lvl[:n])
        ctx = self.ctx_proj(seq)
        mask = self.attn_mask[:n, :n]
        for blk in self.blocks:
            h = blk(h, ctx, attn_mask=mask)
        return self.head(self.norm(h))

    def loss(self, images, tokens):
        idx_list = self.vq.encode(images)
        seq, pooled = self.encode_text(self.drop_prompts(tokens))
        logits = self(self.vq.var_inputs(idx_list), seq, pooled)
        return F.cross_entropy(logits.reshape(-1, self.num_codes), torch.cat(idx_list, dim=1).reshape(-1))

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, temperature=1.0, top_k=0, generator=None):
        b = tokens.shape[0]
        seq, pooled = self.encode_text(tokens, with_null=True)
        latent = self.vq.latent_size
        f_hat = torch.zeros(b, self.vq.z_dim, latent, latent, device=tokens.device)
        x_in = torch.zeros(2 * b, 0, self.vq.z_dim, device=tokens.device)
        idx_list, start = [], 0
        for si, pn in enumerate(self.scales):
            logits = self(x_in, seq, pooled)[:, start : start + pn * pn]
            logits = self.guide(logits, cfg_scale) / temperature
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.multinomial(logits.softmax(-1).view(-1, self.num_codes), 1, generator=generator).view(b, -1)
            idx_list.append(idx)
            f_hat, nxt = self.vq.quantizer.next_scale(si, f_hat, idx)
            if nxt is not None:
                x_in = torch.cat([x_in, torch.cat([nxt, nxt])], dim=1)
            start += pn * pn
        return self.vq.decode(idx_list)
