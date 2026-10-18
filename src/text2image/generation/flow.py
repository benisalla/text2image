"""Rectified flow / flow matching (SD3, FLUX, Z-Image, NextStep-1 heads).

Convention: ``x_t = (1 - t) * x0 + t * noise`` so t = 0 is data and t = 1 is
noise, and the target velocity is ``v = noise - x0``. Sampling integrates the
ODE from t = 1 down to t = 0 with Euler steps.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def sample_t(n: int, device, logit_normal: bool = True) -> torch.Tensor:
    """Logit-normal timestep density from SD3: focuses training on mid-range noise."""
    if logit_normal:
        return torch.sigmoid(torch.randn(n, device=device))
    return torch.rand(n, device=device)


def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - t) * x0 + t * noise


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    steps: int = 30,
    shift: float = 1.0,
) -> torch.Tensor:
    """Integrate dx/dt = v from t = 1 to t = 0.

    ``shift > 1`` spends more steps at high noise (the SD3 / FLUX timestep shift).
    """
    ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    ts = shift * ts / (1 + (shift - 1) * ts)
    x = noise
    for t, t_next in zip(ts[:-1], ts[1:]):
        v = velocity_fn(x, t.expand(x.shape[0]))
        x = x + (t_next - t) * v
    return x
