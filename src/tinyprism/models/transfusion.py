"""Transfusion: one transformer that is both a language model and an image denoiser.

There is no separate text encoder: caption tokens and noisy image patches form a
single sequence processed by the *same* transformer. Text positions use a causal
mask and are trained with next-token cross-entropy (so the model is a genuine LM
over captions); image positions attend to the whole caption and to each other
bidirectionally, and are trained with the flow-matching loss. Total loss:
``L = L_LM + lambda * L_flow`` (lambda = 5 in the paper).

Patches enter and leave the transformer through small U-Net down/up blocks with
timestep conditioning (``patchifier="unet"``, the variant the paper found clearly
better than a linear patchify, which is kept as ``patchifier="linear"``).

This is the recipe behind unified multimodal generators such as Transfusion,
Show-o-style hybrids and HunyuanImage 3.0's MoE+Transfusion design.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.models.base import FlowModel
from tinyprism.models.dit import FinalLayer, PatchEmbed
from tinyprism.nn.layers import ResBlock, TimestepEmbedder, TransformerBlock, sincos_2d, unpatchify


class UNetPatchifier(nn.Module):
    """Conv down block -> patch tokens, and patch tokens -> conv up block (with a skip), both t-conditioned."""

    def __init__(self, image_size: int, patch: int, dim: int, emb_dim: int, ch: int = 32):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.down = ResBlock(ch, ch, emb_dim)
        self.to_tokens = nn.Conv2d(ch, dim, patch, stride=patch)
        self.register_buffer("pos", sincos_2d(dim, image_size // patch)[None], persistent=False)
        self.from_tokens = nn.ConvTranspose2d(dim, ch, patch, stride=patch)
        self.up = ResBlock(2 * ch, ch, emb_dim)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def encode(self, x, temb):
        skip = self.down(self.conv_in(x), temb)
        return self.to_tokens(skip).flatten(2).transpose(1, 2) + self.pos, skip

    def decode(self, tokens, skip, temb):
        b, n, d = tokens.shape
        g = skip.shape[-1] // (self.from_tokens.stride[0])
        h = self.from_tokens(tokens.transpose(1, 2).reshape(b, d, g, g))
        h = self.up(torch.cat([h, skip], dim=1), temb)
        return self.conv_out(F.silu(self.norm_out(h)))


class Transfusion(FlowModel):
    family = "unified LM + flow matching (one transformer)"
    paper = "Transfusion (Zhou et al., ICLR 2025); HunyuanImage 3.0 (2025)"

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 96,
        depth: int = 4,
        heads: int = 4,
        flow_weight: float = 5.0,
        text_dim: int | None = None,
        text_depth: int = 0,
        prediction: str = "v",
        patchifier: str = "unet",
    ):
        # The "text encoder" is reduced to token + position embeddings (text_depth=0):
        # all contextual processing of the caption happens inside the shared transformer.
        super().__init__(vocab_size, text_len, text_dim or dim, text_depth, prediction=prediction)
        self.image_size, self.patch = image_size, patch
        self.text_len, self.flow_weight = text_len, flow_weight
        assert patchifier in ("unet", "linear")
        self.t_embed = TimestepEmbedder(dim)
        if patchifier == "unet":
            self.patchifier = UNetPatchifier(image_size, patch, dim, dim)
        else:
            self.embed = PatchEmbed(image_size, patch, dim)
            self.final = FinalLayer(dim, patch)
        self.patchifier_kind = patchifier
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        n_img = (image_size // patch) ** 2
        n = text_len + n_img
        mask = torch.zeros(n, n, dtype=torch.bool)
        mask[:text_len, :text_len] = torch.tril(torch.ones(text_len, text_len, dtype=torch.bool))  # causal text
        mask[text_len:, :] = True  # image sees the caption and the whole image
        self.register_buffer("attn_mask", mask, persistent=False)

    def backbone(self, x, t, text):
        """Returns (caption states, image output in pixel space, timestep embedding)."""
        seq, _ = text
        temb = self.t_embed(t)
        if self.patchifier_kind == "unet":
            tokens, skip = self.patchifier.encode(x, temb)
        else:
            tokens = self.embed(x)
        h = torch.cat([seq, tokens + temb.unsqueeze(1)], dim=1)
        for blk in self.blocks:
            h = blk(h, attn_mask=self.attn_mask)
        h = self.norm(h)
        h_txt, h_img = h[:, : self.text_len], h[:, self.text_len :]
        if self.patchifier_kind == "unet":
            out = self.patchifier.decode(h_img, skip, temb)
        else:
            out = unpatchify(self.final(h_img, temb), self.patch)
        return h_txt, out, temb

    def forward(self, x, t, text):
        return self.backbone(x, t, text)[1]

    @staticmethod
    def lm_targets(tokens: torch.Tensor) -> torch.Tensor:
        """Next-token targets: caption words, then one padding token as end-of-caption; the rest ignored."""
        targets = tokens[:, 1:].clone()
        length = (tokens != 0).sum(dim=1, keepdim=True)
        pos = torch.arange(1, tokens.shape[1], device=tokens.device)[None]
        targets[pos > length] = -100
        targets[length.expand_as(targets) == 0] = -100  # dropped (null) prompts: nothing to predict
        return targets

    def loss(self, images, tokens):
        tokens = self.drop_prompts(tokens)
        text = self.encode_text(tokens)
        noise = torch.randn_like(images)
        t = torch.sigmoid(torch.randn(images.shape[0], device=images.device))
        x_t = (1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * noise
        h_txt, out, _ = self.backbone(x_t, t, text)
        if self.prediction == "x":
            out = (x_t - out) / t.clamp(min=0.05).view(-1, 1, 1, 1)
        flow_loss = F.mse_loss(out, noise - images)
        targets = self.lm_targets(tokens)
        if not (targets != -100).any():
            return self.flow_weight * flow_loss
        logits = self.lm_head(h_txt[:, :-1])
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return lm_loss + self.flow_weight * flow_loss

    @torch.no_grad()
    def complete_caption(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """Greedy text generation: keep tokens[:, :start], let the model write the rest.

        Text never attends to the image, so a dummy image is enough.
        """
        tokens = tokens.clone()
        dummy = torch.zeros(tokens.shape[0], 3, self.image_size, self.image_size, device=tokens.device)
        ones = torch.ones(tokens.shape[0], device=tokens.device)
        done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        for i in range(start, self.text_len):
            tokens[:, i:] = 0
            h_txt, _, _ = self.backbone(dummy, ones, self.encode_text(tokens))
            nxt = self.lm_head(h_txt[:, i - 1]).argmax(-1)
            done |= nxt == 0  # end-of-caption
            tokens[:, i] = torch.where(done, 0, nxt)
        return tokens
