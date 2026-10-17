"""Text-conditioned U-Net trained as a DDPM (the Stable Diffusion 1.x / 2.x / SDXL lineage).

Convolutional ResBlocks receive the timestep (+ pooled text, as in SDXL) and
spatial transformer blocks cross-attend to the text sequence at the lower
resolutions. Trained with epsilon-prediction, sampled with DDIM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.generation.ddpm import DDPMSchedule
from tinyprism.models.base import T2IModel
from tinyprism.nn.layers import ResBlock, TimestepEmbedder, TransformerBlock


class SpatialTransformer(nn.Module):
    """Flatten the feature map, run self+cross attention, reshape back (zero-init residual)."""

    def __init__(self, ch: int, heads: int, context_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.proj_in = nn.Linear(ch, ch)
        self.block = TransformerBlock(ch, heads, context_dim=context_dim)
        self.proj_out = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, ctx):
        b, c, h, w = x.shape
        z = self.proj_in(self.norm(x).flatten(2).transpose(1, 2))
        z = self.proj_out(self.block(z, ctx))
        return x + z.transpose(1, 2).view(b, c, h, w)


class Level(nn.Module):
    """``num_res`` ResBlocks, each optionally followed by a SpatialTransformer."""

    def __init__(self, in_ch, out_ch, emb_dim, ctx_dim, attn: bool, num_res: int, heads: int):
        super().__init__()
        self.res = nn.ModuleList([ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim) for i in range(num_res)])
        self.attn = nn.ModuleList([SpatialTransformer(out_ch, heads, ctx_dim) for _ in range(num_res)]) if attn else None

    def forward(self, x, emb, ctx):
        for i, res in enumerate(self.res):
            x = res(x, emb)
            if self.attn is not None:
                x = self.attn[i](x, ctx)
        return x


class UNetDDPM(T2IModel):
    family = "diffusion (DDPM, epsilon)"
    paper = "DDPM (Ho et al. 2020) + Latent Diffusion / Stable Diffusion U-Net (Rombach et al. 2022)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        image_size: int = 32,
        ch: int = 64,
        mults: tuple[int, ...] = (1, 2, 2),
        attn_levels: tuple[int, ...] = (1, 2),
        num_res: int = 2,
        heads: int = 4,
        text_dim: int = 128,
        text_depth: int = 2,
        timesteps: int = 1000,
    ):
        super().__init__(vocab_size, text_len, text_dim, text_depth)
        self.image_size = image_size
        self.schedule = DDPMSchedule(timesteps)
        emb = ch * 4
        self.t_embed = TimestepEmbedder(emb)
        self.pooled_proj = nn.Linear(text_dim, emb)
        chs = [ch * m for m in mults]

        self.conv_in = nn.Conv2d(3, chs[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        in_ch = chs[0]
        for i, c in enumerate(chs):
            self.down.append(Level(in_ch, c, emb, text_dim, i in attn_levels, num_res, heads))
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
            in_ch = c
        self.mid = Level(in_ch, in_ch, emb, text_dim, True, 2, heads)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chs))):
            self.up.append(Level(in_ch + chs[i], chs[i], emb, text_dim, i in attn_levels, num_res, heads))
            self.upsample.append(nn.Conv2d(chs[i], chs[i], 3, padding=1) if i > 0 else nn.Identity())
            in_ch = chs[i]
        self.norm_out = nn.GroupNorm(8, in_ch)
        self.conv_out = nn.Conv2d(in_ch, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t_int, text):
        seq, pooled = text
        emb = self.t_embed(t_int.float() / self.schedule.timesteps) + self.pooled_proj(pooled)
        h = self.conv_in(x)
        skips = []
        for level, down in zip(self.down, self.downsample):
            h = level(h, emb, seq)
            skips.append(h)
            h = down(h)
        h = self.mid(h, emb, seq)
        for i, (level, up) in enumerate(zip(self.up, self.upsample)):
            h = level(torch.cat([h, skips.pop()], dim=1), emb, seq)
            if i < len(self.up) - 1:
                h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))

    def loss(self, images, tokens):
        text = self.encode_text(self.drop_prompts(tokens))
        noise = torch.randn_like(images)
        t = self.schedule.sample_t(images.shape[0], images.device)
        x_t = self.schedule.q_sample(images, t, noise)
        return F.mse_loss(self(x_t, t, text), noise)

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, steps=50, generator=None):
        b = tokens.shape[0]
        seq, pooled = self.encode_text(tokens, with_null=True)
        noise = torch.randn(b, 3, self.image_size, self.image_size, generator=generator, device=tokens.device)

        def eps_fn(x, t):
            return self.guide(self(torch.cat([x, x]), torch.cat([t, t]), (seq, pooled)), cfg_scale)

        return self.schedule.ddim_sample(eps_fn, noise, steps).clamp(-1, 1)
