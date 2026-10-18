"""SANA: an efficiency-first diffusion transformer.

Three changes relative to DiT / PixArt, all visible below:

* **Linear attention** for image self-attention: softmax(QK^T)V is replaced by
  ReLU(Q) (ReLU(K)^T V), normalised by ReLU(Q) sum(ReLU(K)). Cost grows linearly
  with the number of tokens instead of quadratically (the point of SANA at 4K).
* **Mix-FFN**: the MLP becomes 1x1 conv -> 3x3 depthwise conv -> GLU -> 1x1 conv,
  restoring the local detail that linear attention is weak at.
* **NoPE + adaLN-single**: no positional embedding at all (the depthwise conv, with
  zero padding, tells tokens where they are), and one shared adaLN MLP for all blocks
  plus a small learned per-block table, as in PixArt-alpha.

Text enters through ordinary (softmax) cross-attention. Trained with rectified flow.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.models.base import FlowModel
from tinyprism.nn.layers import Attention, TimestepEmbedder, merge_heads, modulate, split_heads, unpatchify


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float = 1e-6):
        super().__init__()
        self.heads, self.eps = heads, eps
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = (split_heads(z, self.heads) for z in self.qkv(x).chunk(3, dim=-1))
        q, k = F.relu(q), F.relu(k)
        kv = k.transpose(-2, -1) @ v  # (B, H, d, d): independent of sequence length
        num = q @ kv
        den = q @ k.sum(dim=2, keepdim=True).transpose(-2, -1)
        return self.proj(merge_heads(num / (den + self.eps)))


class MixFFN(nn.Module):
    def __init__(self, dim: int, grid: int, ratio: float = 2.5):
        super().__init__()
        hidden = int(dim * ratio)
        self.grid = grid
        self.inverted = nn.Conv2d(dim, 2 * hidden, 1)
        self.depthwise = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.point = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, n, d = x.shape
        h = x.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        h = self.depthwise(F.silu(self.inverted(h)))
        a, gate = h.chunk(2, dim=1)
        return self.point(a * F.silu(gate)).flatten(2).transpose(1, 2)


class SanaBlock(nn.Module):
    def __init__(self, dim: int, heads: int, grid: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = LinearAttention(dim, heads)
        self.cross = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = MixFFN(dim, grid)
        self.table = nn.Parameter(torch.randn(6, dim) / math.sqrt(dim))  # adaLN-single per-block offsets

    def forward(self, x, ctx, mod):
        sh1, sc1, g1, sh2, sc2, g2 = (self.table[None] + mod).unbind(dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1))
        x = x + self.cross(x, ctx)
        return x + g2.unsqueeze(1) * self.ffn(modulate(self.norm2(x), sh2, sc2))


class Sana(FlowModel):
    family = "flow matching (linear-attention DiT)"
    paper = "SANA (Xie et al., ICLR 2025), SANA 1.5 / Sprint (2025)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 96,
        depth: int = 4,
        heads: int = 4,
        text_dim: int = 64,
        text_depth: int = 1,
        prediction: str = "v",
    ):
        super().__init__(vocab_size, text_len, text_dim, text_depth, prediction=prediction)
        self.image_size, self.patch = image_size, patch
        grid = image_size // patch
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)  # NoPE: no positional embedding
        self.t_embed = TimestepEmbedder(dim)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))  # adaLN-single, shared
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([SanaBlock(dim, heads, grid) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_table = nn.Parameter(torch.randn(2, dim) / math.sqrt(dim))
        self.out = nn.Linear(dim, patch * patch * 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, text):
        seq, _ = text
        temb = self.t_embed(t)
        mod = self.t_block(temb).view(-1, 6, temb.shape[-1])
        ctx = self.ctx_proj(seq)
        h = self.embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h, ctx, mod)
        shift, scale = (self.final_table[None] + temb.unsqueeze(1)).unbind(dim=1)
        return unpatchify(self.out(modulate(self.norm(h), shift, scale)), self.patch)
