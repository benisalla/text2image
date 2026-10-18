"""Synthetic captioned-shapes dataset.

Every image is one coloured shape on a dark background, described by a caption
such as ``"a large red circle at the top left"``. Four independent attributes
(colour, shape, position, size) give 6 * 4 * 9 * 2 = 432 distinct prompts, which
is small enough to learn on a CPU in minutes but still forces a model to bind
several words to the right visual properties.

Because the ground truth is known exactly, generated images can be scored
automatically by an "oracle" classifier (see ``text2image.evaluate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (230, 45, 45),
    "green": (45, 200, 70),
    "blue": (50, 95, 240),
    "yellow": (240, 215, 40),
    "purple": (170, 65, 220),
    "cyan": (40, 215, 220),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "cross")
# (x, y) centre as a fraction of the image side.
POSITIONS: dict[str, tuple[float, float]] = {
    "top left": (0.28, 0.28),
    "top": (0.50, 0.28),
    "top right": (0.72, 0.28),
    "left": (0.28, 0.50),
    "center": (0.50, 0.50),
    "right": (0.72, 0.50),
    "bottom left": (0.28, 0.72),
    "bottom": (0.50, 0.72),
    "bottom right": (0.72, 0.72),
}
# Shape "radius" as a fraction of the image side.
SIZES: dict[str, float] = {"small": 0.13, "large": 0.23}
BACKGROUND = (18, 18, 26)

COLOR_NAMES = tuple(COLORS)
POSITION_NAMES = tuple(POSITIONS)
SIZE_NAMES = tuple(SIZES)
ATTRIBUTES = ("color", "shape", "position", "size")
NUM_CLASSES = (len(COLOR_NAMES), len(SHAPES), len(POSITION_NAMES), len(SIZE_NAMES))

VOCAB_WORDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        ["a", "at", "the"] + list(SIZE_NAMES) + list(COLOR_NAMES) + list(SHAPES) + [w for p in POSITION_NAMES for w in p.split()]
    )
)


def caption(color: int, shape: int, position: int, size: int) -> str:
    return f"a {SIZE_NAMES[size]} {COLOR_NAMES[color]} {SHAPES[shape]} at the {POSITION_NAMES[position]}"


def _shape_mask(shape: str, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Boolean mask in normalised coordinates (the shape spans roughly [-1, 1])."""
    if shape == "circle":
        return dx**2 + dy**2 <= 1.0
    if shape == "square":
        return np.maximum(np.abs(dx), np.abs(dy)) <= 0.82
    if shape == "triangle":  # apex up, flat base
        return (dy <= 0.8) & (np.abs(dx) <= (dy + 1.0) / 1.8 * 1.0)
    if shape == "cross":
        return ((np.abs(dx) <= 0.3) & (np.abs(dy) <= 1.0)) | ((np.abs(dy) <= 0.3) & (np.abs(dx) <= 1.0))
    raise ValueError(shape)


def render(
    color: int,
    shape: int,
    position: int,
    size: int,
    res: int = 32,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    supersample: int = 3,
) -> np.ndarray:
    """Render one image as uint8 ``(res, res, 3)`` with anti-aliasing."""
    hi = res * supersample
    yy, xx = np.mgrid[0:hi, 0:hi].astype(np.float32) + 0.5
    cx, cy = POSITIONS[POSITION_NAMES[position]]
    cx, cy = (cx + offset[0]) * hi, (cy + offset[1]) * hi
    r = SIZES[SIZE_NAMES[size]] * scale * hi
    mask = _shape_mask(SHAPES[shape], (xx - cx) / r, (yy - cy) / r).astype(np.float32)
    cover = mask.reshape(res, supersample, res, supersample).mean(axis=(1, 3))[..., None]
    fg = np.array(COLORS[COLOR_NAMES[color]], np.float32)
    bg = np.array(BACKGROUND, np.float32)
    return (cover * fg + (1.0 - cover) * bg).round().astype(np.uint8)


@dataclass
class Batch:
    images: torch.Tensor  # (B, 3, H, W) float in [-1, 1]
    captions: list[str]
    labels: torch.Tensor  # (B, 4) long: color, shape, position, size


class ShapesDataset:
    """Infinite stream of (image, caption) pairs.

    ``jitter`` adds small random offsets / scale changes so that the model has
    to learn a distribution rather than memorise 432 fixed images. To keep the
    data pipeline essentially free on a CPU, ``sample`` draws from a bank of
    ``variants`` pre-rendered jittered copies of every prompt (built lazily).
    """

    def __init__(self, res: int = 32, seed: int = 0, jitter: bool = True, variants: int = 16):
        self.res = res
        self.jitter = jitter
        self.variants = variants if jitter else 1
        self.rng = np.random.default_rng(seed)
        self._bank: torch.Tensor | None = None

    def random_labels(self, n: int) -> np.ndarray:
        return np.stack([self.rng.integers(0, k, size=n) for k in NUM_CLASSES], axis=1)

    def render_labels(self, labels: np.ndarray) -> Batch:
        """Render the given (N, 4) label rows from scratch."""
        imgs = []
        for c, s, p, z in labels:
            if self.jitter:
                offset = tuple(self.rng.uniform(-0.03, 0.03, size=2))
                scale = float(self.rng.uniform(0.9, 1.1))
            else:
                offset, scale = (0.0, 0.0), 1.0
            imgs.append(render(c, s, p, z, self.res, offset, scale))
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return Batch(x, [caption(*row) for row in labels], torch.from_numpy(labels).long())

    @staticmethod
    def flat_index(labels: np.ndarray) -> np.ndarray:
        return np.ravel_multi_index(labels.T, NUM_CLASSES)

    def bank(self) -> torch.Tensor:
        """uint8 tensor (num_prompts, variants, 3, H, W) of pre-rendered images."""
        if self._bank is None:
            all_labels = np.stack(np.unravel_index(np.arange(int(np.prod(NUM_CLASSES))), NUM_CLASSES), axis=1)
            imgs = [self.render_labels(all_labels).images for _ in range(self.variants)]
            self._bank = ((torch.stack(imgs, dim=1) + 1) * 127.5).round().to(torch.uint8)
        return self._bank

    def sample(self, batch_size: int) -> Batch:
        labels = self.random_labels(batch_size)
        variant = torch.from_numpy(self.rng.integers(0, self.variants, size=batch_size))
        images = self.bank()[torch.from_numpy(self.flat_index(labels)), variant].float() / 127.5 - 1.0
        return Batch(images, [caption(*row) for row in labels], torch.from_numpy(labels).long())
