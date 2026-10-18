"""MeanFlow: one-step generation by learning the *average* velocity of the flow.

A flow-matching model learns the instantaneous velocity v(z_t, t), so sampling
must integrate an ODE in many small steps. MeanFlow instead learns the average
velocity over an interval,

    u(z_t, r, t) = 1 / (t - r) * integral_r^t v(z_s, s) ds,

so a single evaluation jumps from noise to data: x = z_1 - u(z_1, 0, 1).
Differentiating the definition gives the "MeanFlow identity"

    u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t),

whose right-hand side is the training target; the total derivative d/dt u is a
Jacobian-vector product with tangent (v, 0, 1) in (z, r, t), computed by
forward-mode autodiff. Classifier-free guidance is folded into the target at
training time, so sampling stays at one network evaluation.

Backbone: the DiT of ``models/dit.py`` with an extra embedding of the interval t - r.
"""

from __future__ import annotations

import torch
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from text2image.generation import flow
from text2image.models.dit import DiT
from text2image.nn.layers import TimestepEmbedder


class MeanFlow(DiT):
    family = "flow matching, one-step (average velocity)"
    paper = "MeanFlow (Geng et al., NeurIPS 2025), improved MeanFlow (CVPR 2026)"

    def __init__(
        self,
        *args,
        ratio_r_neq_t: float = 0.25,
        guidance: float = 2.0,
        adaptive_p: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.h_embed = TimestepEmbedder(self.t_embed.mlp[-1].out_features)
        self.ratio_r_neq_t = ratio_r_neq_t
        self.guidance = guidance
        self.adaptive_p = adaptive_p

    def condition(self, t, pooled, r=None):
        h = torch.zeros_like(t) if r is None else t - r
        return self.t_embed(t) + self.h_embed(h) + self.pooled_proj(pooled)

    def average_velocity(self, z, r, t, text):
        return self(z, t, text, r=r)

    def sample_times(self, n: int, device):
        """(r, t) with r <= t; a fraction ``1 - ratio_r_neq_t`` has r = t (plain flow matching)."""
        a = torch.sigmoid(torch.randn(n, device=device) - 0.4)
        b = torch.sigmoid(torch.randn(n, device=device) - 0.4)
        t, r = torch.maximum(a, b), torch.minimum(a, b)
        same = torch.rand(n, device=device) >= self.ratio_r_neq_t
        return torch.where(same, t, r), t

    def loss(self, images, tokens):
        b = images.shape[0]
        tokens = self.drop_prompts(tokens)
        text = self.encode_text(tokens)
        r, t = self.sample_times(b, images.device)
        noise = torch.randn_like(images)
        z = flow.interpolate(images, noise, t)
        v = noise - images
        if self.guidance != 1.0:
            # Training-time CFG: v~ = w * v + (1 - w) * u(z_t, t, t | null) for prompted samples.
            with torch.no_grad():
                u_null = self.average_velocity(z, t, t, self.encode_text(torch.zeros_like(tokens)))
            prompted = tokens.any(dim=1).view(-1, 1, 1, 1)
            v = torch.where(prompted, self.guidance * v + (1 - self.guidance) * u_null, v)

        def fn(z_, r_, t_):
            return self.average_velocity(z_, r_, t_, text)

        # Forward-mode AD is only implemented for the reference ("math") attention kernel.
        with sdpa_kernel(SDPBackend.MATH):
            u, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        target = (v - (t - r).view(-1, 1, 1, 1) * dudt).detach()
        err = (u - target).pow(2).mean(dim=(1, 2, 3))
        weight = 1.0 / (err.detach() + 1e-3) ** self.adaptive_p  # adaptive loss weighting
        return (weight * err).mean()

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=1.0, steps=1, generator=None):
        """``steps`` jumps of the learned average velocity (1 = one-step generation).

        Guidance is already learned during training, so ``cfg_scale=1`` uses a single
        network evaluation per step; values > 1 add standard CFG on top.
        """
        b = tokens.shape[0]
        seq, pooled = self.encode_text(tokens, with_null=cfg_scale != 1.0)
        z = torch.randn(b, 3, self.image_size, self.image_size, generator=generator, device=tokens.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
        for t, r in zip(ts[:-1], ts[1:]):
            tb, rb = t.expand(b), r.expand(b)
            if cfg_scale != 1.0:
                u = self.guide(
                    self.average_velocity(torch.cat([z, z]), torch.cat([rb, rb]), torch.cat([tb, tb]), (seq, pooled)), cfg_scale
                )
            else:
                u = self.average_velocity(z, rb, tb, (seq, pooled))
            z = z - (t - r) * u
        return z.clamp(-1, 1)
