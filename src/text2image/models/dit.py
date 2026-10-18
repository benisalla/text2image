"""DiT with cross-attention text conditioning (PixArt-alpha style).

Image patches are tokens; each block applies adaLN-Zero modulated
self-attention and MLP (conditioned on timestep + pooled text) and a plain
cross-attention to the text sequence. Trained with rectified flow.

With ``prediction="x"`` and pixel patches, this is also the JiT recipe
(Li & He, 2025): a plain transformer on raw pixels that predicts the clean image.
"""

from __future__ import annotations

import torch
from torch import nn

from text2image.models.base import FlowModel
from text2image.nn.layers import MLP, Attention, TimestepEmbedder, modulate, sincos_2d, unpatchify


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch: int, dim: int, channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, patch, stride=patch)
        self.register_buffer("pos", sincos_2d(dim, image_size // patch)[None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos


class FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm + linear to patch pixels, zero-initialised."""

    def __init__(self, dim: int, patch: int, channels: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * channels)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, qk_norm=True)
        self.norm_c = nn.LayerNorm(dim)
        self.cross = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, mlp_ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, ctx, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1))
        x = x + self.cross(self.norm_c(x), ctx)
        return x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))


class DiT(FlowModel):
    paper = "DiT (Peebles & Xie 2023) + PixArt-alpha cross-attention (Chen et al. 2023)"

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
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final = FinalLayer(dim, patch)

    def condition(self, t, pooled, **kwargs):
        """Global conditioning vector for adaLN; subclasses add extra inputs (e.g. MeanFlow's r)."""
        return self.t_embed(t) + self.pooled_proj(pooled)

    def forward(self, x, t, text, **kwargs):
        seq, pooled = text
        c = self.condition(t, pooled, **kwargs)
        ctx = self.ctx_proj(seq)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, ctx, c)
        return unpatchify(self.final(h, c), self.patch)


class JiT(DiT):
    family = "flow matching (x-prediction)"
    paper = "JiT: Back to Basics, Let Denoising Generative Models Denoise (Li & He 2025)"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("prediction", "x")
        super().__init__(*args, **kwargs)
