"""Name -> (class, default kwargs, default training recipe) for every model in the zoo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from tinyprism.models.autoregressive import AutoregressiveTransformer
from tinyprism.models.base import T2IModel, VQModel
from tinyprism.models.dit import DiT, JiT
from tinyprism.models.mar import MAR
from tinyprism.models.maskgit import MaskGIT
from tinyprism.models.meanflow import MeanFlow
from tinyprism.models.mmdit import MMDiT
from tinyprism.models.sana import Sana
from tinyprism.models.single_stream import SingleStreamDiT
from tinyprism.models.transfusion import Transfusion
from tinyprism.models.unet import UNetDDPM
from tinyprism.models.var import VAR
from tinyprism.text.tokenizer import WordTokenizer
from tinyprism.tokenizers.vq import VQVAE


@dataclass
class Recipe:
    steps: int = 2000
    batch_size: int = 32
    lr: float = 5e-4
    warmup: int = 200
    ema: float = 0.999
    tokenizer: str | None = None  # name of the VQ tokenizer a model depends on
    sample_kwargs: dict = field(default_factory=dict)


@dataclass
class Spec:
    cls: type
    kwargs: dict
    recipe: Recipe


# Default sizes are a "CPU preset": every architectural ingredient is present but
# the models are ~1M parameters so they train in minutes without a GPU. Scale up
# with e.g. ``tinyprism-train dit --set dim=384 depth=12 heads=6``.
TRANSFORMER = dict(dim=96, depth=4, heads=4, text_dim=64, text_depth=1)
UNET = dict(ch=32, num_res=1, attn_levels=(2,), heads=4, text_dim=64, text_depth=1)
VQ = dict(ch=32, z_dim=16, codebook_size=128)

TOKENIZERS: dict[str, Spec] = {
    "vq": Spec(VQVAE, dict(VQ, scales=(8,)), Recipe(steps=1500, lr=1e-3, ema=0)),
    "msvq": Spec(VQVAE, dict(VQ, scales=(1, 2, 3, 4, 6, 8)), Recipe(steps=1500, lr=1e-3, ema=0)),
}

MODELS: dict[str, Spec] = {
    "unet": Spec(UNetDDPM, UNET, Recipe(sample_kwargs=dict(steps=50))),
    "unet-eps": Spec(UNetDDPM, dict(UNET, prediction="eps"), Recipe(sample_kwargs=dict(steps=50))),
    "dit": Spec(DiT, TRANSFORMER, Recipe(sample_kwargs=dict(steps=30))),
    "jit": Spec(JiT, TRANSFORMER, Recipe(sample_kwargs=dict(steps=30))),
    "mmdit": Spec(MMDiT, TRANSFORMER, Recipe(sample_kwargs=dict(steps=30))),
    "single-stream": Spec(SingleStreamDiT, TRANSFORMER, Recipe(sample_kwargs=dict(steps=30))),
    "sana": Spec(Sana, TRANSFORMER, Recipe(sample_kwargs=dict(steps=30))),
    "meanflow": Spec(MeanFlow, TRANSFORMER, Recipe(sample_kwargs=dict(steps=1, cfg_scale=1.0))),
    "transfusion": Spec(Transfusion, dict(dim=96, depth=4, heads=4), Recipe(sample_kwargs=dict(steps=30))),
    "ar": Spec(AutoregressiveTransformer, TRANSFORMER, Recipe(tokenizer="vq")),
    "maskgit": Spec(MaskGIT, TRANSFORMER, Recipe(tokenizer="vq", sample_kwargs=dict(steps=12))),
    "var": Spec(VAR, TRANSFORMER, Recipe(tokenizer="msvq")),
    "mar": Spec(MAR, dict(TRANSFORMER, head_width=96), Recipe(sample_kwargs=dict(num_iter=16, head_steps=25))),
}


def build_tokenizer(name: str, **overrides) -> VQVAE:
    spec = TOKENIZERS[name]
    return spec.cls(**{**spec.kwargs, **overrides})


def build_model(name: str, text_tokenizer: WordTokenizer | None = None, vq_config: dict | None = None, **overrides) -> T2IModel:
    spec = MODELS[name]
    tok = text_tokenizer or WordTokenizer()
    kwargs = {**spec.kwargs, **overrides}
    if issubclass(spec.cls, VQModel):
        kwargs["vq_config"] = vq_config if vq_config is not None else {**TOKENIZERS[spec.recipe.tokenizer].kwargs}
    return spec.cls(vocab_size=tok.vocab_size, text_len=tok.max_len, **kwargs)


def save_checkpoint(path: str | Path, kind: str, name: str, config: dict, model: torch.nn.Module, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "name": name, "config": config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, map_location="cpu"):
    """Rebuild a model / tokenizer / oracle from a checkpoint written by ``tinyprism.train``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    kind, name, cfg = ckpt["kind"], ckpt["name"], ckpt["config"]
    if kind == "model":
        model = build_model(name, **cfg)
    elif kind == "tokenizer":
        model = build_tokenizer(name, **cfg)
    elif kind == "oracle":
        from tinyprism.evaluate import Oracle

        model = Oracle()
    else:
        raise ValueError(kind)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
