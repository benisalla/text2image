# tiny-prism

**Nine modern text-to-image architectures (plus an ε-prediction ablation), written from scratch, small enough to train and compare on a laptop CPU.**

A prism turns one beam of light into a full spectrum. `tiny-prism` takes one prompt and
sends it through the full spectrum of today's generative architectures, from the
Stable Diffusion U-Net to FLUX-style MMDiT, VAR's next-scale prediction and MAR's
continuous-token autoregression. All nine share the same text encoder, data,
training loop and automatic evaluation, so the *only* thing that differs is the architecture.

| family | models |
|---|---|
| **Diffusion / flow matching** | `unet` (Stable Diffusion U-Net, v-prediction; `unet-eps` ablation) · `dit` (DiT / PixArt-α) · `jit` (JiT pixel x-prediction) · `mmdit` (SD3 / FLUX double-stream) · `single-stream` (FLUX single blocks / Z-Image S3-DiT) |
| **Autoregressive / masked tokens** | `ar` (DALL-E / Parti / LlamaGen) · `maskgit` (MaskGIT / Muse) · `var` (VAR next-scale) · `mar` (MAR + flow head, NextStep-1 style) |

See [docs/architectures.md](docs/architectures.md) for the survey of the field (2025-2026),
the reasoning behind this selection, and exactly where each idea lives in the code.

## Results

Every model was trained for **2,000 steps at batch size 32 on a CPU** (no GPU), then asked to
draw 128 random prompts. The oracle scores each generated image (see below):

| model | family | params | color | shape | position | size | all 4 correct | train time (CPU) | sampling / image |
|---|---|---|---|---|---|---|---|---|---|
| `unet-eps` | diffusion (DDPM) | 1.17M | 62% | 38% | 11% | 60% | **3%** | 72 min | 2659 ms |
| `dit` | flow matching | 0.95M | 100% | 32% | 100% | 90% | **25%** | 37 min | 113 ms |
| `jit` | flow matching (x-prediction) | 0.95M | 100% | 50% | 100% | 100% | **50%** | 23 min | 111 ms |
| `mmdit` | flow matching | 1.34M | 100% | 34% | 100% | 98% | **34%** | 37 min | 119 ms |
| `single-stream` | flow matching | 0.68M | 100% | 30% | 100% | 88% | **23%** | 20 min | 121 ms |
| `ar` | autoregressive (discrete, next-token) | 0.54M | 83% | 47% | 80% | 86% | **30%** | 14 min | 101 ms |
| `maskgit` | masked generative (discrete) | 0.69M | 91% | 84% | 100% | 100% | **77%** | 15 min | 40 ms |
| `var` | autoregressive (discrete, next-scale) | 0.69M | 92% | 72% | 97% | 97% | **65%** | 23 min | 15 ms |
| `mar` | masked autoregressive (continuous, diffusion/flow head) | 0.87M | 100% | 74% | 100% | 96% | **70%** | 34 min | 66 ms |
| *real images (oracle ceiling)* | - | - | 100% | 100% | 100% | 100% | **100%** | - | - |

<p align="center"><img src="results/comparison.png" width="760" alt="Each model's sample for 8 prompts"></p>

*Columns of the grid, left to right:* “a large red circle at the center” · “a small blue square at the top left” · “a large yellow triangle at the bottom” · “a small green cross at the right” · “a large purple square at the bottom right” · “a small cyan circle at the top” · “a large green triangle at the left” · “a small red cross at the bottom left”.
Per-model 4-sample grids are in [`results/`](results/); tokenizer reconstructions in
[`results/reconstructions_vq.png`](results/reconstructions_vq.png) and [`results/reconstructions_msvq.png`](results/reconstructions_msvq.png).

**What the numbers say** (at this tiny scale and budget):

* **Every architecture learns text conditioning.** All transformer models get colour and position near 100%;
  random guessing would give 17% and 11%.
* **Token models win on shape at equal steps.** The VQ decoder handles the fine pixel detail, so the transformer
  only has to pick 64 codes. MaskGIT (77%), MAR (70%) and VAR (65%) lead. VAR also samples fastest:
  six parallel steps, one per scale.
* **x-prediction doubles DiT's score.** JiT and DiT use the same network and differ only in what it predicts.
  Predicting the clean image instead of the velocity takes all-correct from 25% to 50%,
  which agrees with the claim of *Back to Basics* (Li & He, 2025) for pixel-space transformers.
* **Flow transformers get layout right but blur shapes.** At 2,000 steps the pixel-space DiT, MMDiT and single-stream
  models produce blobby shapes; shape is the attribute they have learned least.
* **ε-prediction fails at high noise.** `unet-eps` (the original DDPM / SD 1.x objective) denoises *better* than DiT
  at low and medium noise, but its x₀ estimate degrades sharply near pure noise, where ε-prediction must divide by
  √ᾱ ≈ 0. That is when the global layout is decided, and position accuracy drops to chance (11%). Lowering the
  guidance scale makes it worse (0-3%), which rules out over-guidance. The default `unet` therefore uses
  **v-prediction** (SD 2.x).

> Caveats: train and sampling times are wall-clock on a CPU shared with other heavy jobs, so compare
> them within the table only. Scores at n=128 carry about ±4% standard error, and none of these models
> has converged. Rankings at real scale (billions of parameters, VAE latents, long training) differ:
> MMDiT and single-stream DiTs power most current production systems.


## The toy task: captioned shapes

Every image is one coloured shape on a dark background, rendered procedurally at 32×32:

```
"a {small|large} {red|green|blue|yellow|purple|cyan} {circle|square|triangle|cross}
 at the {top left|top|top right|left|center|right|bottom left|bottom|bottom right}"
```

That gives 432 prompts with 4 independent attributes, plus random jitter in position and scale.
It is simple enough to learn on a CPU in minutes, yet a model only scores well if it binds
**every** word to the right visual property.

**Evaluation without humans.** Because the ground truth is known, a small CNN "oracle"
(`tinyprism/evaluate.py`) is trained to read colour, shape, position and size from real images
(≈100% accurate). We run it on generated images and report per-attribute accuracy and the
fraction of images with **all four attributes correct**: a toy analogue of GenEval.

## Quick start

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
pytest -q                      # 30 tests: shapes, gradients, sampling, tokenizers, overfitting

# full pipeline: oracle -> tokenizers -> 9 models -> evaluation -> results/
bash scripts/run_all.sh
```

Or one model at a time:

```bash
tinyprism-train oracle --out runs/oracle
tinyprism-train mmdit  --out runs/mmdit
tinyprism-train vq     --out runs/vq
tinyprism-train maskgit --tokenizer runs/vq/model.pt --out runs/maskgit

tinyprism-sample --ckpt runs/mmdit/model.pt -n 8 --out red.png \
    --prompt "a large red circle at the top left"
tinyprism-eval --ckpt runs/mmdit/model.pt --oracle runs/oracle/model.pt --n 256
```

Model sizes are a CPU preset (~1M parameters). On a GPU, scale up any model with
`--set`, e.g. `tinyprism-train dit --set dim=384 depth=12 heads=6 --steps 50000 --batch-size 256`.

## Repository layout

```
src/tinyprism/
├── data/shapes.py          synthetic captioned-shapes dataset (pre-rendered jitter bank)
├── text/                   word tokenizer + small transformer text encoder (CLIP/T5 stand-in)
├── nn/layers.py            attention, adaLN modulation, timestep embedding, patchify, ResBlock
├── generation/             DDPM schedule + DDIM sampler, rectified-flow utilities + Euler sampler
├── tokenizers/vq.py        VQ-VAE with single-scale or VAR multi-scale residual quantizer
├── models/
│   ├── base.py             T2IModel / FlowModel / VQModel interfaces, classifier-free guidance
│   ├── unet.py             U-Net + DDPM (Stable Diffusion lineage)
│   ├── dit.py              DiT (PixArt-α cross-attention) and JiT (x-prediction)
│   ├── mmdit.py            MMDiT (SD3 / FLUX double-stream)
│   ├── single_stream.py    single-stream DiT (FLUX single blocks / Z-Image)
│   ├── autoregressive.py   next-token AR over VQ codes
│   ├── maskgit.py          masked generative transformer
│   ├── var.py              next-scale prediction
│   └── mar.py              masked AR with continuous tokens + flow head
├── registry.py             model zoo: default sizes, training recipes, checkpoint I/O
├── train.py / sample.py / evaluate.py   CLIs
scripts/run_all.sh          end-to-end benchmark, scripts/make_report.py builds results/
tests/                      unit tests for every component
docs/architectures.md       literature survey and design notes
```

Every model implements the same two methods, which keeps the code easy to compare:

```python
loss = model.loss(images, tokens)                  # training objective
images = model.generate(tokens, cfg_scale=3.0)     # sampling with classifier-free guidance
```

## License

MIT
