"""VQ-VAE image tokenizer with an optional multi-scale residual quantizer.

* ``scales=(8,)``: a standard VQ-VAE (VQGAN without the GAN), producing one
  8x8 grid of discrete codes. Used by the autoregressive and MaskGIT models.
* ``scales=(1, 2, 3, 4, 6, 8)``: the multi-scale residual quantizer of VAR.
  The latent is explained coarse-to-fine: each scale quantizes the downsampled
  *residual* left by the previous scales, and the decoder sees the sum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyprism.nn.layers import ResBlock


class MultiScaleQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, scales: tuple[int, ...], beta: float = 0.25, phi_ratio: float = 0.5):
        super().__init__()
        self.scales = tuple(scales)
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        # VAR's "phi": a light conv that refines each upsampled scale before accumulation.
        self.phi = nn.Conv2d(dim, dim, 3, padding=1) if len(scales) > 1 else None
        self.phi_ratio = phi_ratio
        self.register_buffer("usage", torch.zeros(codebook_size))

    def _phi(self, h):
        if self.phi is None:
            return h
        return (1 - self.phi_ratio) * h + self.phi_ratio * self.phi(h)

    def lookup(self, f: torch.Tensor) -> torch.Tensor:
        """Nearest code index for every position of a (B, C, h, w) map -> (B, h*w)."""
        z = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        d = z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.weight.T + self.codebook.weight.pow(2).sum(1)[None]
        return d.argmin(1).view(f.shape[0], -1)

    def embed(self, idx: torch.Tensor, pn: int) -> torch.Tensor:
        """(B, pn*pn) indices -> (B, C, pn, pn) code vectors."""
        return self.codebook(idx).transpose(1, 2).reshape(idx.shape[0], -1, pn, pn)

    def _up(self, h, size):
        return h if h.shape[-1] == size else F.interpolate(h, size=(size, size), mode="bicubic")

    def forward(self, f: torch.Tensor):
        """Returns (straight-through f_hat, vq loss, list of per-scale indices)."""
        size = f.shape[-1]
        f_ng = f.detach()
        rest, f_hat = f_ng.clone(), torch.zeros_like(f_ng)
        loss, idx_list = 0.0, []
        for pn in self.scales:
            r = rest if pn == size else F.interpolate(rest, size=(pn, pn), mode="area")
            idx = self.lookup(r)
            idx_list.append(idx)
            h = self._phi(self._up(self.embed(idx, pn), size))
            f_hat = f_hat + h
            rest = rest - h
            loss = loss + F.mse_loss(f_hat.detach(), f) * self.beta + F.mse_loss(f_hat, f_ng)
            if self.training:
                with torch.no_grad():
                    self.usage.mul_(0.99).add_(torch.bincount(idx.flatten(), minlength=self.usage.numel()).float(), alpha=0.01)
        f_hat = (f_hat.detach() - f_ng) + f  # straight-through estimator
        return f_hat, loss / len(self.scales), idx_list

    @torch.no_grad()
    def restart_dead_codes(self, f: torch.Tensor, threshold: float = 0.01) -> int:
        """Re-seed rarely used codes with random encoder outputs (keeps the codebook alive)."""
        dead = (self.usage < threshold).nonzero().flatten()
        if dead.numel() == 0:
            return 0
        z = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        pick = z[torch.randint(0, z.shape[0], (dead.numel(),), device=z.device)]
        self.codebook.weight.data[dead] = pick + 0.01 * torch.randn_like(pick)
        self.usage[dead] = 1.0
        return dead.numel()

    def next_scale(self, si: int, f_hat: torch.Tensor, idx: torch.Tensor):
        """Add scale ``si``'s codes to f_hat; return (f_hat, VAR input for scale si+1 or None)."""
        size = f_hat.shape[-1]
        f_hat = f_hat + self._phi(self._up(self.embed(idx, self.scales[si]), size))
        if si == len(self.scales) - 1:
            return f_hat, None
        pn = self.scales[si + 1]
        nxt = f_hat if pn == size else F.interpolate(f_hat, size=(pn, pn), mode="area")
        return f_hat, nxt.flatten(2).transpose(1, 2)


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        ch: int = 64,
        z_dim: int = 32,
        codebook_size: int = 256,
        scales: tuple[int, ...] = (8,),
        beta: float = 0.25,
    ):
        super().__init__()
        self.image_size, self.z_dim, self.codebook_size = image_size, z_dim, codebook_size
        self.latent_size = image_size // 4
        assert scales[-1] == self.latent_size, "last scale must equal the latent resolution"
        self.scales = tuple(scales)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            ResBlock(ch, ch),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1),
            ResBlock(ch, 2 * ch),
            nn.Conv2d(2 * ch, 2 * ch, 4, stride=2, padding=1),
            ResBlock(2 * ch, 2 * ch),
            nn.GroupNorm(8, 2 * ch),
            nn.SiLU(),
            nn.Conv2d(2 * ch, z_dim, 1),
        )
        self.quantizer = MultiScaleQuantizer(codebook_size, z_dim, scales, beta)
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, 2 * ch, 3, padding=1),
            ResBlock(2 * ch, 2 * ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1),
            ResBlock(2 * ch, ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, padding=1),
            ResBlock(ch, ch),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    @property
    def tokens_per_scale(self) -> list[int]:
        return [pn * pn for pn in self.scales]

    def forward(self, x):
        f = self.encoder(x)
        f_hat, vq_loss, idx = self.quantizer(f)
        return self.decoder(f_hat), vq_loss, idx, f

    def loss(self, x):
        recon, vq_loss, _, f = self(x)
        rec = F.mse_loss(recon, x) + F.l1_loss(recon, x)
        return rec + vq_loss, {"recon": rec.item(), "vq": float(vq_loss.detach())}, f

    @torch.no_grad()
    def encode(self, x) -> list[torch.Tensor]:
        """Images -> list of (B, pn*pn) code indices, one entry per scale."""
        return self.quantizer(self.encoder(x))[2]

    @torch.no_grad()
    def decode(self, idx_list: list[torch.Tensor]) -> torch.Tensor:
        b = idx_list[0].shape[0]
        f_hat = torch.zeros(b, self.z_dim, self.latent_size, self.latent_size, device=idx_list[0].device)
        for si, idx in enumerate(idx_list):
            f_hat, _ = self.quantizer.next_scale(si, f_hat, idx)
        return self.decoder(f_hat).clamp(-1, 1)

    @torch.no_grad()
    def var_inputs(self, idx_list: list[torch.Tensor]) -> torch.Tensor:
        """Teacher-forcing inputs for VAR: for every scale k >= 1, the accumulated
        f_hat of scales < k downsampled to scale k. Shape (B, sum_{k>=1} pn_k^2, z_dim)."""
        b = idx_list[0].shape[0]
        f_hat = torch.zeros(b, self.z_dim, self.latent_size, self.latent_size, device=idx_list[0].device)
        outs = []
        for si in range(len(self.scales) - 1):
            f_hat, nxt = self.quantizer.next_scale(si, f_hat, idx_list[si])
            outs.append(nxt)
        return torch.cat(outs, dim=1)
