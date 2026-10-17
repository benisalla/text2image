"""MMDiT: the two-stream multimodal diffusion transformer of SD3 / FLUX double blocks.

Text and image tokens keep *separate* weights (norms, QKV, MLP, adaLN) but are
concatenated for a single *joint* attention, so information flows both ways:
the text representation is refined by the image as denoising progresses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.models.base import FlowModel
from tinyprism.models.dit import FinalLayer, PatchEmbed
from tinyprism.nn.layers import MLP, TimestepEmbedder, merge_heads, modulate, split_heads, unpatchify


class StreamParams(nn.Module):
    """Per-modality weights of one MMDiT block.

    ``pre_only`` streams only contribute keys/values to the joint attention and
    are not updated (SD3's ``context_pre_only`` for the text stream of the last block).
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, pre_only: bool = False):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(dim // heads)
        self.k_norm = nn.RMSNorm(dim // heads)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            self.mlp = MLP(dim, mlp_ratio)
        self.n_mod = 2 if pre_only else 6
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, self.n_mod * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def pre_attn(self, x, c):
        mod = self.ada(c).chunk(self.n_mod, dim=-1)
        q, k, v = self.qkv(modulate(self.norm1(x), mod[0], mod[1])).chunk(3, dim=-1)
        q = self.q_norm(split_heads(q, self.heads))
        k = self.k_norm(split_heads(k, self.heads))
        return (q, k, split_heads(v, self.heads)), mod

    def post_attn(self, x, attn_out, mod):
        _, _, g1, sh2, sc2, g2 = mod
        x = x + g1.unsqueeze(1) * self.proj(merge_heads(attn_out))
        return x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))


class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, last: bool = False):
        super().__init__()
        self.img = StreamParams(dim, heads)
        self.txt = StreamParams(dim, heads, pre_only=last)
        self.last = last

    def forward(self, x, y, c):
        (qx, kx, vx), mod_x = self.img.pre_attn(x, c)
        (qy, ky, vy), mod_y = self.txt.pre_attn(y, c)
        n_txt = y.shape[1]
        out = F.scaled_dot_product_attention(torch.cat([qy, qx], dim=2), torch.cat([ky, kx], dim=2), torch.cat([vy, vx], dim=2))
        if not self.last:
            y = self.txt.post_attn(y, out[:, :, :n_txt], mod_y)
        x = self.img.post_attn(x, out[:, :, n_txt:], mod_x)
        return x, y


class MMDiT(FlowModel):
    paper = "Stable Diffusion 3 / MMDiT (Esser et al. 2024), FLUX.1 double-stream blocks"

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
        self.blocks = nn.ModuleList([MMDiTBlock(dim, heads, last=i == depth - 1) for i in range(depth)])
        self.final = FinalLayer(dim, patch)

    def forward(self, x, t, text):
        seq, pooled = text
        c = self.t_embed(t) + self.pooled_proj(pooled)
        y = self.ctx_proj(seq)
        h = self.embed(x)
        for blk in self.blocks:
            h, y = blk(h, y, c)
        return unpatchify(self.final(h, c), self.patch)
