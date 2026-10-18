"""Single-stream DiT: one shared transformer over [text tokens; image tokens].

This is the design of FLUX single-stream blocks and of Z-Image's S3-DiT: text
and image tokens are simply concatenated and processed with *shared* weights,
which is the most parameter-efficient option (cf. DiT-Air). Each block runs
attention and MLP in parallel from a single fused input projection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from text2image.models.base import FlowModel
from text2image.models.dit import FinalLayer, PatchEmbed
from text2image.nn.layers import TimestepEmbedder, merge_heads, modulate, split_heads, unpatchify


class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.lin1 = nn.Linear(dim, 3 * dim + self.hidden)
        self.q_norm = nn.RMSNorm(dim // heads)
        self.k_norm = nn.RMSNorm(dim // heads)
        self.lin2 = nn.Linear(dim + self.hidden, dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        qkv, m = self.lin1(modulate(self.norm(h), shift, scale)).split([3 * self.dim, self.hidden], dim=-1)
        q, k, v = (split_heads(z, self.heads) for z in qkv.chunk(3, dim=-1))
        attn = merge_heads(F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v))
        out = self.lin2(torch.cat([attn, F.gelu(m, approximate="tanh")], dim=-1))
        return h + gate.unsqueeze(1) * out


class SingleStreamDiT(FlowModel):
    paper = "FLUX.1 single-stream blocks, Z-Image S3-DiT (2025)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        text_dim: int = 128,
        text_depth: int = 2,
        prediction: str = "v",
    ):
        super().__init__(vocab_size, text_len, text_dim, text_depth, prediction=prediction)
        self.image_size, self.patch = image_size, patch
        self.embed = PatchEmbed(image_size, patch, dim)
        self.t_embed = TimestepEmbedder(dim)
        self.pooled_proj = nn.Linear(text_dim, dim)
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.type_emb = nn.Parameter(torch.zeros(2, dim))  # 0 = text, 1 = image
        self.blocks = nn.ModuleList([SingleStreamBlock(dim, heads) for _ in range(depth)])
        self.final = FinalLayer(dim, patch)

    def forward(self, x, t, text):
        seq, pooled = text
        c = self.t_embed(t) + self.pooled_proj(pooled)
        y = self.ctx_proj(seq) + self.type_emb[0]
        h = torch.cat([y, self.embed(x) + self.type_emb[1]], dim=1)
        for blk in self.blocks:
            h = blk(h, c)
        return unpatchify(self.final(h[:, y.shape[1] :], c), self.patch)
