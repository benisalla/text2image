"""Common interface for every text-to-image model in the zoo.

A model owns its text encoder and exposes exactly two methods:

* ``loss(images, tokens)``: scalar training loss on a batch of images in [-1, 1].
* ``generate(tokens, cfg_scale=..., **kw)``: images in [-1, 1] for the prompts.

Classifier-free guidance (CFG) is supported everywhere by randomly replacing
prompts with the all-padding "null" prompt during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.generation import flow
from tinyprism.text.encoder import TextEncoder


class T2IModel(nn.Module):
    family: str = ""
    paper: str = ""

    def __init__(self, vocab_size: int, text_len: int, text_dim: int = 128, text_depth: int = 2, cfg_dropout: float = 0.1):
        super().__init__()
        self.text = TextEncoder(vocab_size, text_len, dim=text_dim, depth=text_depth)
        self.text_dim = text_dim
        self.cfg_dropout = cfg_dropout

    def drop_prompts(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cfg_dropout <= 0:
            return tokens
        keep = torch.rand(tokens.shape[0], 1, device=tokens.device) >= self.cfg_dropout
        return tokens * keep

    def encode_text(self, tokens: torch.Tensor, with_null: bool = False):
        """Encode prompts; with ``with_null`` the null prompts are appended for CFG."""
        if with_null:
            tokens = torch.cat([tokens, torch.zeros_like(tokens)])
        return self.text(tokens)

    @staticmethod
    def guide(out: torch.Tensor, scale: float) -> torch.Tensor:
        """Split a [cond; uncond] batch and apply classifier-free guidance."""
        cond, uncond = out.chunk(2)
        return uncond + scale * (cond - uncond)

    def loss(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, cfg_scale: float = 3.0, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VQModel(T2IModel):
    """A model that generates the discrete codes of a frozen, pre-trained VQ tokenizer.

    The tokenizer lives inside the model (so a checkpoint is self-contained) but
    is never trained here; ``load_tokenizer`` copies in the weights before training.
    """

    def __init__(self, vocab_size: int, text_len: int, vq_config: dict, text_dim: int = 128, text_depth: int = 2):
        super().__init__(vocab_size, text_len, text_dim, text_depth)
        from tinyprism.tokenizers.vq import VQVAE

        self.vq_config = dict(vq_config)
        self.vq = VQVAE(**self.vq_config).requires_grad_(False).eval()

    def load_tokenizer(self, state_dict: dict) -> None:
        self.vq.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vq.eval()
        return self


class FlowModel(T2IModel):
    """Rectified-flow training / sampling around a network ``forward(x_t, t, text)``.

    ``prediction="v"`` regresses the velocity directly (SD3 / FLUX).
    ``prediction="x"`` predicts the clean image and converts it to a velocity,
    keeping the v-loss (JiT, "Back to Basics: Let Denoising Generative Models Denoise").
    """

    family = "flow matching"

    def __init__(self, *args, prediction: str = "v", **kwargs):
        super().__init__(*args, **kwargs)
        assert prediction in ("v", "x")
        self.prediction = prediction

    def velocity(self, x_t, t, text):
        out = self(x_t, t, text)
        if self.prediction == "v":
            return out
        return (x_t - out) / t.clamp(min=0.05).view(-1, 1, 1, 1)

    def loss(self, images, tokens):
        text = self.encode_text(self.drop_prompts(tokens))
        noise = torch.randn_like(images)
        t = flow.sample_t(images.shape[0], images.device)
        x_t = flow.interpolate(images, noise, t)
        return F.mse_loss(self.velocity(x_t, t, text), noise - images)

    @torch.no_grad()
    def generate(self, tokens, cfg_scale=3.0, steps=30, shift=1.0, generator=None):
        b = tokens.shape[0]
        seq, pooled = self.encode_text(tokens, with_null=True)
        noise = torch.randn(b, 3, self.image_size, self.image_size, generator=generator, device=tokens.device)

        def v_fn(x, t):
            v = self.velocity(torch.cat([x, x]), torch.cat([t, t]), (seq, pooled))
            return self.guide(v, cfg_scale)

        return flow.euler_sample(v_fn, noise, steps, shift).clamp(-1, 1)
