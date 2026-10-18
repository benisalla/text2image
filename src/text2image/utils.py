"""Small helpers: EMA, LR schedule, image grids, fixed demo prompts."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

DEMO_PROMPTS = [
    "a large red circle at the center",
    "a small blue square at the top left",
    "a large yellow triangle at the bottom",
    "a small green cross at the right",
    "a large purple square at the bottom right",
    "a small cyan circle at the top",
    "a large green triangle at the left",
    "a small red cross at the bottom left",
]


class EMA:
    """Exponential moving average of a model's weights (with the usual warm-up)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))) * 0.9 + 0.1

    return f


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) in [-1, 1] -> (B, H, W, 3) uint8."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int,
    row_labels: list[str] | None = None,
    upscale: int = 3,
    pad: int = 2,
    label_width: int = 230,
) -> None:
    """Save a grid of images (rows of ``nrow``) with optional text labels on the left."""
    arr = to_uint8(images)
    n, h, w, _ = arr.shape
    rows = math.ceil(n / nrow)
    h, w = h * upscale, w * upscale
    lw = label_width if row_labels else 0
    canvas = Image.new("RGB", (lw + nrow * (w + pad) + pad, rows * (h + pad) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        r, c = divmod(i, nrow)
        tile = Image.fromarray(arr[i]).resize((w, h), Image.NEAREST)
        canvas.paste(tile, (lw + pad + c * (w + pad), pad + r * (h + pad)))
    if row_labels:
        for r, text in enumerate(row_labels[:rows]):
            draw.text((6, pad + r * (h + pad) + h // 2 - 6), text, fill=(20, 20, 20))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
