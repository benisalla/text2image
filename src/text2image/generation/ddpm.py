"""Discrete-time Gaussian diffusion (DDPM training, DDIM sampling)."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


class DDPMSchedule(nn.Module):
    """Cosine noise schedule (Nichol & Dhariwal, 2021) with epsilon prediction."""

    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        super().__init__()
        self.timesteps = timesteps
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0).float(), persistent=False)

    def sample_t(self, n: int, device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (n,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        steps: int = 50,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta = 0). ``eps_fn(x, t_int)`` predicts the noise."""
        x = noise
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=x.device).round().long()
        for i, t in enumerate(ts):
            a = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            eps = eps_fn(x, t.expand(x.shape[0]))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
