"""Building blocks shared by every architecture in the zoo."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: per-sample affine transform of a (B, N, D) sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, n, d = x.shape
    return x.view(b, n, heads, d // heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


class Attention(nn.Module):
    """Multi-head self- or cross-attention with optional QK-norm."""

    def __init__(self, dim: int, heads: int, context_dim: int | None = None, qk_norm: bool = False):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim or dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        hd = dim // heads
        self.q_norm = nn.RMSNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(hd) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        ctx = x if context is None else context
        q = self.q_norm(split_heads(self.q(x), self.heads))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k, v = self.k_norm(split_heads(k, self.heads)), split_heads(v, self.heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.proj(merge_heads(out))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, out_dim: int | None = None):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, out_dim or dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN block: self-attention, optional cross-attention to text, MLP.

    Used by the token-based models (MaskGIT, AR, VAR, MAR) where conditioning
    enters through cross-attention or a text prefix rather than adaLN.
    """

    def __init__(self, dim: int, heads: int, context_dim: int | None = None, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.cross = None
        if context_dim is not None:
            self.norm_c = nn.LayerNorm(dim)
            self.cross = Attention(dim, heads, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x, context=None, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        if self.cross is not None:
            x = x + self.cross(self.norm_c(x), context)
        return x + self.mlp(self.norm2(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar t in [0, 1] (or integer step / T) followed by an MLP."""

    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine positional embedding, shape (grid * grid, dim)."""
    assert dim % 4 == 0
    coords = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))

    def emb(pos):
        out = pos.reshape(-1, 1) * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    return torch.cat([emb(yy), emb(xx)], dim=1)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/p * W/p, p*p*C)."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // p) * (w // p), p * p * c)


def unpatchify(x: torch.Tensor, p: int, channels: int = 3) -> torch.Tensor:
    """(B, N, p*p*C) -> (B, C, H, W) for a square grid."""
    b, n, _ = x.shape
    g = int(math.isqrt(n))
    x = x.view(b, g, g, p, p, channels).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, g * p, g * p)


class ResBlock(nn.Module):
    """Conv residual block (GroupNorm + SiLU) with an optional time-embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int | None = None, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch) if emb_dim else None
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.emb is not None:
            h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h
