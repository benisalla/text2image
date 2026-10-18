"""Train a text-to-image model, a VQ tokenizer, or the evaluation oracle.

Examples::

    tinyprism-train oracle --out runs/oracle
    tinyprism-train vq     --out runs/vq
    tinyprism-train dit    --out runs/dit
    tinyprism-train maskgit --tokenizer runs/vq/model.pt --out runs/maskgit
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tinyprism.data.shapes import ShapesDataset
from tinyprism.evaluate import Oracle, evaluate_real
from tinyprism.registry import MODELS, TOKENIZERS, Recipe, build_model, build_tokenizer, load_checkpoint, save_checkpoint
from tinyprism.text.tokenizer import WordTokenizer
from tinyprism.utils import DEMO_PROMPTS, EMA, lr_lambda, save_grid


def parse_overrides(pairs: list[str]) -> dict:
    """``--set dim=128 depth=4`` -> {"dim": 128, "depth": 4} (values parsed as JSON when possible)."""
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def train_loop(model, loss_fn, recipe: Recipe, args, log_every: int, on_step=None):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=recipe.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(recipe.warmup, recipe.steps))
    ema = EMA(model, recipe.ema) if recipe.ema else None
    model.train()
    t0, running, history = time.time(), 0.0, []
    for step in range(1, recipe.steps + 1):
        loss = loss_fn(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema:
            ema.update(model)
        if on_step:
            on_step(step)
        running += loss.item()
        if step % log_every == 0 or step == recipe.steps:
            avg = running / (log_every if step % log_every == 0 else step % log_every)
            history.append({"step": step, "loss": avg, "elapsed": time.time() - t0})
            lr = sched.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"[{args.name}] step {step:5d}/{recipe.steps}  loss {avg:.4f}  lr {lr:.2e}  {elapsed:6.0f}s", flush=True)
            running = 0.0
    final = ema.model if ema else model
    return final.eval(), history, time.time() - t0


def main(argv=None):
    names = ["oracle", *TOKENIZERS, *MODELS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", choices=names)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--tokenizer", help="path to a trained VQ tokenizer checkpoint (token-based models)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override model constructor kwargs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = ShapesDataset(seed=args.seed)
    tok = WordTokenizer()
    overrides = parse_overrides(args.set)

    if args.name == "oracle":
        spec_recipe = Recipe(steps=800, batch_size=64, lr=1e-3, ema=0)
    elif args.name in TOKENIZERS:
        spec_recipe = TOKENIZERS[args.name].recipe
    else:
        spec_recipe = MODELS[args.name].recipe
    recipe = Recipe(**{**asdict(spec_recipe)})
    recipe.steps = args.steps or recipe.steps
    recipe.batch_size = args.batch_size or recipe.batch_size
    recipe.lr = args.lr or recipe.lr

    def batch():
        return data.sample(recipe.batch_size)

    if args.name == "oracle":
        model = Oracle()

        def loss_fn(step):
            b = batch()
            x = b.images + 0.08 * torch.randn_like(b.images) * torch.rand(len(b.images), 1, 1, 1)
            return model.loss(x, b.labels)

        model, history, secs = train_loop(model, loss_fn, recipe, args, args.log_every)
        metrics = evaluate_real(model)
        print(f"[oracle] accuracy on held-out real images: {metrics}")
        save_checkpoint(out / "model.pt", "oracle", "oracle", {}, model, metrics=metrics)

    elif args.name in TOKENIZERS:
        model = build_tokenizer(args.name, **overrides)
        last_f = {}

        def loss_fn(step):
            loss, _, f = model.loss(batch().images)
            last_f["f"] = f.detach()
            return loss

        def on_step(step):
            if step % 100 == 0 and step <= recipe.steps * 0.8:
                n = model.quantizer.restart_dead_codes(last_f["f"])
                if n:
                    print(f"[{args.name}] restarted {n} dead codes", flush=True)

        model, history, secs = train_loop(model, loss_fn, recipe, args, args.log_every, on_step)
        with torch.no_grad():
            real = data.sample(16).images
            recon = model(real)[0]
            used = torch.cat([i.flatten() for i in model.encode(data.sample(512).images)]).unique().numel()
        mse = torch.mean((recon - real) ** 2).item()
        print(f"[{args.name}] recon mse {mse:.5f}, codes used {used}/{model.codebook_size}")
        save_grid(torch.cat([real[:8], recon[:8]]), out / "reconstructions.png", nrow=8, row_labels=["real", "reconstruction"])
        save_checkpoint(out / "model.pt", "tokenizer", args.name, overrides, model, recon_mse=mse, codes_used=used)

    else:
        vq_config = None
        if MODELS[args.name].recipe.tokenizer:
            if not args.tokenizer:
                p.error(f"{args.name} needs --tokenizer (a trained '{MODELS[args.name].recipe.tokenizer}' checkpoint)")
            vq, vq_ckpt = load_checkpoint(args.tokenizer)
            wanted = MODELS[args.name].recipe.tokenizer
            if vq_ckpt["kind"] != "tokenizer" or vq_ckpt["name"] != wanted:
                p.error(f"{args.name} needs a '{wanted}' tokenizer, got a '{vq_ckpt['name']}' checkpoint")
            vq_config = {**TOKENIZERS[vq_ckpt["name"]].kwargs, **vq_ckpt["config"]}
        model = build_model(args.name, tok, vq_config=vq_config, **overrides)
        if vq_config is not None:
            model.load_tokenizer(vq.state_dict())
        print(f"[{args.name}] {type(model).__name__}: {model.num_params / 1e6:.2f}M trainable params")

        def loss_fn(step):
            b = batch()
            return model.loss(b.images, tok.encode(b.captions))

        model, history, secs = train_loop(model, loss_fn, recipe, args, args.log_every)
        config = {**overrides, **({"vq_config": vq_config} if vq_config else {})}
        save_checkpoint(out / "model.pt", "model", args.name, config, model, train_seconds=secs)
        tokens = tok.encode([p_ for p_ in DEMO_PROMPTS for _ in range(4)])
        images = model.generate(tokens, generator=torch.Generator().manual_seed(0), **recipe.sample_kwargs)
        save_grid(images, out / "samples.png", nrow=4, row_labels=DEMO_PROMPTS)

    (out / "history.json").write_text(json.dumps({"history": history, "train_seconds": secs}, indent=2))
    print(f"[{args.name}] done in {secs:.0f}s -> {out}")


if __name__ == "__main__":
    main()
