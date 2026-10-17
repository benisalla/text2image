"""MAR: masked autoregressive generation of *continuous* tokens with a diffusion head.

There is no vector quantizer. Images are cut into continuous patch tokens; a
bidirectional transformer sees the tokens generated so far (others replaced by
a [MASK] embedding) and outputs a conditioning vector z per position. A small
MLP head then models p(token | z) with a denoising objective; here we use the
flow-matching variant of that head, as in NextStep-1 (ICLR 2026) and later MAR
follow-ups. Generation reveals tokens in a random order over a few iterations.

The original MAR models VAE latents; with 32x32 toy images raw pixel patches
work fine and keep the model self-contained.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.generation import flow
from tinyprism.models.base import T2IModel
from tinyprism.nn.layers import TimestepEmbedder, TransformerBlock, modulate, patchify, sincos_2d, unpatchify


class FlowHeadBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    """Per-token velocity predictor v(x_t, t | z): MAR's 'SimpleMLPAdaLN'."""

    def __init__(self, token_dim: int, cond_dim: int, width: int = 256, depth: int = 3):
        super().__init__()
        self.t_embed = TimestepEmbedder(width)
        self.cond = nn.Linear(cond_dim, width)
        self.inp = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList([FlowHeadBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, token_dim)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, z):
        c = self.t_embed(t) + self.cond(z)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(h).unsqueeze(1), shift, scale).squeeze(1))


class MAR(T2IModel):
    family = "masked autoregressive (continuous, diffusion/flow head)"
    paper = "MAR (Li et al., NeurIPS 2024), NextStep-1 flow head (2025)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        head_width: int = 256,
        head_depth: int = 3,
        text_dim: int = 128,
        text_depth: int = 2,
        diffusion_batch_mul: int = 4,
    ):
        super().__init__(vocab_size, text_len, text_dim, text_depth)
        self.image_size, self.patch = image_size, patch
        self.num_tokens = (image_size // patch) ** 2
        self.token_dim = 3 * patch * patch
        self.diffusion_batch_mul = diffusion_batch_mul
        self.inp = nn.Linear(self.token_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(dim, image_size // patch)[None], persistent=False)
        self.ctx_proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, context_dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = FlowHead(self.token_dim, dim, head_width, head_depth)

    def forward(self, x, known, seq):
        """x: (B, N, token_dim) tokens (values at unknown positions are ignored)."""
        h = torch.where(known.unsqueeze(-1), self.inp(x), self.mask_token) + self.pos
        ctx = self.ctx_proj(seq)
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.norm(h)

    def loss(self, images, tokens):
        x = patchify(images, self.patch)
        b, n, _ = x.shape
        seq, _ = self.encode_text(self.drop_prompts(tokens))
        # Mask ratio from a truncated normal centred at 1.0 (min 0.7), as in MAR.
        ratio = (1 - torch.randn(b, 1, device=x.device).abs() * 0.25).clamp(0.7, 1.0)
        rank = torch.rand(b, n, device=x.device).argsort(dim=1).argsort(dim=1)
        masked = rank < (ratio * n).ceil()
        z = self(x, ~masked, seq)[masked]
        target = x[masked]
        z, target = z.repeat(self.diffusion_batch_mul, 1), target.repeat(self.diffusion_batch_mul, 1)
        t = flow.sample_t(target.shape[0], x.device)
        noise = torch.randn_like(target)
        return F.mse_loss(self.head(flow.interpolate(target, noise, t), t, z), noise - target)

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, num_iter=16, head_steps=25, generator=None):
        b, n, dev = tokens.shape[0], self.num_tokens, tokens.device
        seq, _ = self.encode_text(tokens, with_null=True)
        x = torch.zeros(b, n, self.token_dim, device=dev)
        known = torch.zeros(b, n, dtype=torch.bool, device=dev)
        order = torch.rand(b, n, generator=generator, device=dev).argsort(dim=1).argsort(dim=1)
        done = 0
        for it in range(num_iter):
            target = n - int(math.floor(n * math.cos(math.pi / 2 * (it + 1) / num_iter)))
            target = max(target, done + 1)
            now = (order >= done) & (order < target)
            z = self(torch.cat([x, x]), torch.cat([known, known]), seq)
            z_now = torch.cat([z[:b][now], z[b:][now]])

            def v_fn(xt, t, z_now=z_now):
                v = self.head(torch.cat([xt, xt]), torch.cat([t, t]), z_now)
                return self.guide(v, cfg_scale)

            noise = torch.randn(z_now.shape[0] // 2, self.token_dim, generator=generator, device=dev)
            x[now] = flow.euler_sample(v_fn, noise, head_steps)
            known |= now
            done = target
            if done >= n:
                break
        return unpatchify(x, self.patch).clamp(-1, 1)
