"""Automatic evaluation with an "oracle" attribute classifier.

Because the dataset is synthetic, a small CNN can recognise colour, shape,
position and size of real images almost perfectly. Running it on *generated*
images measures how faithfully a model follows the prompt (a toy analogue of
GenEval / CLIP-score). We report per-attribute accuracy and the fraction of
images where all four attributes are right.

Usage::

    text2image-eval --ckpt runs/dit/model.pt --oracle runs/oracle/model.pt --n 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from text2image.data.shapes import ATTRIBUTES, NUM_CLASSES, ShapesDataset
from text2image.text.tokenizer import WordTokenizer


class Oracle(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(4 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(4 * w * 16, 256), nn.GELU(),
        )  # fmt: skip
        self.heads = nn.ModuleList([nn.Linear(256, k) for k in NUM_CLASSES])

    def forward(self, x):
        h = self.features(x)
        return [head(h) for head in self.heads]

    def loss(self, x, labels):
        return sum(F.cross_entropy(lg, labels[:, i]) for i, lg in enumerate(self(x)))

    @torch.no_grad()
    def predict(self, x) -> torch.Tensor:
        return torch.stack([lg.argmax(-1) for lg in self(x)], dim=1)


def score(pred: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    correct = pred == labels
    out = {a: correct[:, i].float().mean().item() for i, a in enumerate(ATTRIBUTES)}
    out["all"] = correct.all(1).float().mean().item()
    return out


@torch.no_grad()
def evaluate_model(model, oracle, n: int = 256, batch_size: int = 64, seed: int = 1234, **sample_kwargs) -> dict:
    data = ShapesDataset(seed=seed, jitter=False)
    tok = WordTokenizer()
    labels = torch.from_numpy(data.random_labels(n)).long()
    gen = torch.Generator().manual_seed(seed)
    preds = []
    t0 = time.time()
    for i in range(0, n, batch_size):
        lab = labels[i : i + batch_size]
        tokens = tok.encode(data.render_labels(lab.numpy()).captions)
        images = model.generate(tokens, generator=gen, **sample_kwargs)
        preds.append(oracle.predict(images))
    secs = (time.time() - t0) / n
    return {**score(torch.cat(preds), labels), "sec_per_image": secs, "n": n}


@torch.no_grad()
def evaluate_real(oracle, n: int = 1024, seed: int = 1234) -> dict:
    batch = ShapesDataset(seed=seed, jitter=True).sample(n)
    return score(oracle.predict(batch.images), batch.labels)


def main(argv=None):
    from text2image.registry import MODELS, load_checkpoint

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--cfg", type=float, default=None, help="guidance scale (default: the model's own)")
    p.add_argument("--out", default=None, help="write metrics JSON here (default: next to the checkpoint)")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    model, ckpt = load_checkpoint(args.ckpt)
    oracle, _ = load_checkpoint(args.oracle)
    kwargs = dict(MODELS[ckpt["name"]].recipe.sample_kwargs)
    if args.cfg is not None:
        kwargs["cfg_scale"] = args.cfg
    metrics = {"model": ckpt["name"], "params": model.num_params, **evaluate_model(model, oracle, args.n, **kwargs)}
    print(json.dumps(metrics, indent=2))
    out = Path(args.out) if args.out else Path(args.ckpt).with_name("metrics.json")
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
