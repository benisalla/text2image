"""Generate images from text prompts with a trained checkpoint.

Example::

    tinyprism-sample --ckpt runs/mmdit/model.pt --prompt "a large red circle at the top left" -n 8 --out red.png
"""

from __future__ import annotations

import argparse

import torch

from tinyprism.registry import MODELS, load_checkpoint
from tinyprism.text.tokenizer import WordTokenizer
from tinyprism.utils import save_grid


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", action="append", required=True, help="repeatable")
    p.add_argument("-n", type=int, default=4, help="samples per prompt")
    p.add_argument("--cfg", type=float, default=None, help="guidance scale (default: the model's own)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples.png")
    args = p.parse_args(argv)

    model, ckpt = load_checkpoint(args.ckpt)
    tok = WordTokenizer()
    tokens = tok.encode([pr for pr in args.prompt for _ in range(args.n)])
    kwargs = dict(MODELS[ckpt["name"]].recipe.sample_kwargs)
    if args.cfg is not None:
        kwargs["cfg_scale"] = args.cfg
    images = model.generate(tokens, generator=torch.Generator().manual_seed(args.seed), **kwargs)
    save_grid(images, args.out, nrow=args.n, row_labels=args.prompt)
    print(f"saved {len(images)} images to {args.out}")


if __name__ == "__main__":
    main()
