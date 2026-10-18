"""Collect metrics from a runs/ directory into results/: a markdown table and a comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tinyprism.registry import MODELS, load_checkpoint
from tinyprism.text.tokenizer import WordTokenizer
from tinyprism.utils import DEMO_PROMPTS, save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    rows, grid, labels = [], [], []
    tok = WordTokenizer()
    tokens = tok.encode(DEMO_PROMPTS)
    for name in MODELS:
        mfile, cfile = runs / name / "metrics.json", runs / name / "model.pt"
        if not (mfile.exists() and cfile.exists()):
            print(f"missing {name}, skipped")
            continue
        m = json.loads(mfile.read_text())
        hist = json.loads((runs / name / "history.json").read_text())
        model, ckpt = load_checkpoint(cfile)
        spec = MODELS[name]
        rows.append(
            f"| `{name}` | {spec.cls.family} | {m['params'] / 1e6:.2f}M | {m['color']:.0%} | {m['shape']:.0%} | "
            f"{m['position']:.0%} | {m['size']:.0%} | **{m['all']:.0%}** | {hist['train_seconds'] / 60:.0f} min | "
            f"{m['sec_per_image'] * 1000:.0f} ms |"
        )
        images = model.generate(tokens, generator=torch.Generator().manual_seed(0), **spec.recipe.sample_kwargs)
        grid.append(images)
        labels.append(name)

    oracle_metrics = torch.load(runs / "oracle" / "model.pt", weights_only=False)["metrics"]
    header = (
        "| model | family | trainable params | color | shape | position | size | all 4 correct "
        "| train time (CPU) | sampling / image |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    real = (
        f"| *real images (oracle ceiling)* | - | - | {oracle_metrics['color']:.0%} | {oracle_metrics['shape']:.0%} | "
        f"{oracle_metrics['position']:.0%} | {oracle_metrics['size']:.0%} | **{oracle_metrics['all']:.0%}** | - | - |"
    )
    table = header + "\n".join(rows) + "\n" + real + "\n"
    (out / "results.md").write_text(table)
    print(table)
    if grid:
        save_grid(torch.cat(grid), out / "comparison.png", nrow=len(DEMO_PROMPTS), row_labels=labels, upscale=3, label_width=110)
        for name in labels:
            src = runs / name / "samples.png"
            if src.exists():
                (out / f"samples_{name}.png").write_bytes(src.read_bytes())
    for name in ("vq", "msvq"):
        src = runs / name / "reconstructions.png"
        if src.exists():
            (out / f"reconstructions_{name}.png").write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
