# text2image

**Twelve modern text-to-image architectures implemented side by side in PyTorch, small enough to train and compare on a laptop CPU.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

`text2image` covers the main ways images are generated from text today: from the Stable Diffusion
U-Net to FLUX-style MMDiT, one-step MeanFlow, Transfusion-style unified models, VAR's next-scale
prediction and MAR's continuous-token autoregression. Every model shares the same data, text
tokenizer, training loop and automatic evaluation, so the **only** thing that changes between
them is the architecture.

<p align="center"><img src="results/comparison.png" width="760" alt="Samples from every model for the same 8 prompts"></p>

## Contents

1. [Architectures](#architectures)
2. [Benchmark: captioned shapes](#benchmark-captioned-shapes)
3. [Results](#results)
4. [Key findings](#key-findings)
5. [Quick start](#quick-start)
6. [Repository layout](#repository-layout)
7. [Documentation](#documentation)
8. [License](#license)

## Architectures

| family | key | architecture | used in / inspired by | paper |
|---|---|---|---|---|
| Diffusion | `unet` | U-Net + DDPM, v-prediction | Stable Diffusion 2.x, SDXL | [Rombach et al. 2022](https://arxiv.org/abs/2112.10752) |
| Diffusion | `unet-eps` | same U-Net, ε-prediction *(ablation)* | Stable Diffusion 1.x | [Ho et al. 2020](https://arxiv.org/abs/2006.11239) |
| Flow matching | `dit` | DiT with cross-attention | PixArt-α | [Peebles & Xie 2023](https://arxiv.org/abs/2212.09748) |
| Flow matching | `jit` | DiT predicting the clean image (pixel space) | JiT | [Li & He 2025](https://arxiv.org/abs/2511.13720) |
| Flow matching | `mmdit` | double-stream MMDiT with joint attention | SD3, FLUX.1 | [Esser et al. 2024](https://arxiv.org/abs/2403.03206) |
| Flow matching | `single-stream` | single-stream DiT | FLUX single blocks, Z-Image | [Z-Image 2025](https://arxiv.org/abs/2511.22699) |
| Flow matching | `sana` | linear-attention DiT, Mix-FFN, no positional embedding | SANA | [Xie et al. 2024](https://arxiv.org/abs/2410.10629) |
| One-step | `meanflow` | average-velocity flow, one network evaluation | MeanFlow | [Geng et al. 2025](https://arxiv.org/abs/2505.13447) |
| Unified | `transfusion` | one transformer as caption LM + image denoiser | Transfusion, HunyuanImage 3.0 | [Zhou et al. 2024](https://arxiv.org/abs/2408.11039) |
| Autoregressive | `ar` | next-token prediction over VQ codes | DALL-E, Parti, LlamaGen | [Sun et al. 2024](https://arxiv.org/abs/2406.06525) |
| Masked tokens | `maskgit` | masked generative transformer | MaskGIT, Muse | [Chang et al. 2022](https://arxiv.org/abs/2202.04200) |
| Autoregressive | `var` | next-scale prediction | VAR, Infinity | [Tian et al. 2024](https://arxiv.org/abs/2404.02905) |
| Masked AR | `mar` | continuous tokens + per-token flow head | MAR, NextStep-1 | [Li et al. 2024](https://arxiv.org/abs/2406.11838) |

[docs/architectures.md](docs/architectures.md) surveys the field (2025-2026), explains why these
models were chosen, and points to where each idea lives in the code.

## Benchmark: captioned shapes

### The task

Every image is one coloured shape on a dark background, rendered procedurally at 32×32:

```
"a {small|large} {red|green|blue|yellow|purple|cyan} {circle|square|triangle|cross}
 at the {top left|top|top right|left|center|right|bottom left|bottom|bottom right}"
```

That gives **432 prompts** with 4 independent attributes, plus random jitter in position and
scale. It is simple enough to learn on a CPU in minutes, but a model only scores well if it binds
**every** word to the right visual property.

### Automatic evaluation

Because the ground truth is known, a small CNN "oracle" (`src/text2image/evaluate.py`) is trained
to read colour, shape, position and size from real images, and it is ≈100% accurate. It then
scores generated images: per-attribute accuracy, plus the fraction of images with **all four
attributes correct**. This is a toy version of GenEval.

## Results

Every model was trained for **2,000 steps at batch size 32 on a CPU** (no GPU), then generated
images for 128 random prompts, which the oracle scored:

| model | family | trainable params | color | shape | position | size | all 4 correct | train time (CPU) | sampling / image |
|---|---|---|---|---|---|---|---|---|---|
| `unet` | diffusion (DDPM) | 1.17M | 100% | 100% | 100% | 100% | **100%** | 77 min | 781 ms |
| `unet-eps` | diffusion (DDPM) | 1.17M | 62% | 38% | 11% | 60% | **3%** | 72 min | 2659 ms |
| `dit` | flow matching | 0.95M | 100% | 32% | 100% | 90% | **25%** | 37 min | 113 ms |
| `jit` | flow matching (x-prediction) | 0.95M | 100% | 50% | 100% | 100% | **50%** | 23 min | 111 ms |
| `mmdit` | flow matching | 1.34M | 100% | 34% | 100% | 98% | **34%** | 37 min | 119 ms |
| `single-stream` | flow matching | 0.68M | 100% | 30% | 100% | 88% | **23%** | 20 min | 121 ms |
| `sana` | flow matching (linear-attention DiT) | 0.76M | 87% | 39% | 11% | 89% | **4%** | 27 min | 435 ms |
| `meanflow` | flow matching, one-step (average velocity) | 0.98M | 100% | 31% | 100% | 89% | **25%** | 64 min | 9 ms |
| `transfusion` | unified LM + flow matching (one transformer) | 0.64M | 100% | 41% | 100% | 100% | **41%** | 40 min | 153 ms |
| `ar` | autoregressive (discrete, next-token) | 0.54M | 83% | 47% | 80% | 86% | **30%** | 14 min | 101 ms |
| `maskgit` | masked generative (discrete) | 0.69M | 91% | 84% | 100% | 100% | **77%** | 15 min | 40 ms |
| `var` | autoregressive (discrete, next-scale) | 0.69M | 92% | 72% | 97% | 97% | **65%** | 23 min | 15 ms |
| `mar` | masked autoregressive (continuous, diffusion/flow head) | 0.87M | 100% | 73% | 100% | 98% | **72%** | 46 min | 241 ms |
| *real images (oracle ceiling)* | - | - | 100% | 100% | 100% | 100% | **100%** | - | - |

The comparison grid at the top uses these 8 prompts, left to right:

1. "a large red circle at the center"
2. "a small blue square at the top left"
3. "a large yellow triangle at the bottom"
4. "a small green cross at the right"
5. "a large purple square at the bottom right"
6. "a small cyan circle at the top"
7. "a large green triangle at the left"
8. "a small red cross at the bottom left"

Per-model sample grids are in [`results/`](results/). VQ tokenizer reconstructions are in
[`reconstructions_vq.png`](results/reconstructions_vq.png) and
[`reconstructions_msvq.png`](results/reconstructions_msvq.png).

> **Caveats.** Train and sampling times are wall-clock on a CPU shared with other heavy jobs, so
> only compare them within this table. Scores at n=128 carry about ±4% standard error, and most
> models have not converged. Rankings at real scale (billions of parameters, VAE latents, long
> training) differ: MMDiT and single-stream DiTs power most current production systems.
> "Trainable params" excludes the frozen VQ tokenizer of `ar`, `maskgit` and `var`. All
> multi-step models sample with guidance scale 3.0. MeanFlow learns guidance during training
> and samples with 1.0.

## Key findings

These findings hold at this small scale and training budget.

**1. Every architecture learns text conditioning.** All models except SANA get colour and
position right 80-100% of the time. Random guessing would give 17% and 11%.

**2. The convolutional U-Net with v-prediction wins at this scale (100%).** Convolutions have
exactly the right inductive bias for small, pixel-exact shapes, and it is the only model that
draws the small crosses crisply. Transformers trade that bias for scalability, which pays off far
beyond this toy task.

**3. The prediction target is decisive.**
- `unet` and `unet-eps` are the *same network* with the same training; only the regression target
  differs. ε-prediction (SD 1.x) scores 3%. Near pure noise, recovering x₀ from ε divides by
  √ᾱ ≈ 0, so small errors explode exactly where the layout is decided, and position falls to
  chance (11%). Lowering the guidance scale makes it worse (0-3%), which rules out over-guidance.
  v-prediction (SD 2.x) scores 100%.
- JiT vs DiT is the same comparison for transformers: predicting the clean image doubles the
  score from 25% to 50%, as *Back to Basics* (Li & He, 2025) predicts.

**4. MeanFlow samples with one network evaluation instead of 30.** It uses the same DiT backbone
but learns the *average* velocity. One step matches DiT's 30-step score, and two steps beat it:

| MeanFlow steps | 1 | 2 | 4 | 8 | *DiT, 30 steps* |
|---|---|---|---|---|---|
| all 4 correct | 25% | 35% | 34% | 37% | *25%* |

**5. Transfusion is both a caption language model and a denoiser (41%).** It completes captions
grammatically: `"a small blue"` → `"a small blue square at the top right"`. How images are
patchified matters. With a plain linear patchify, images fell apart into independent patches.
The paper's small U-Net down/up blocks, now the default, fixed it.

**6. Token models are the strongest transformers at equal training steps.** The VQ decoder handles
fine detail, so the transformer only has to pick 64 codes: MaskGIT reaches 77%, MAR 72% and VAR
65%. VAR samples fastest of the three.

**7. SANA's no-positional-embedding design needs depth.** With no positional embedding, tokens
must infer absolute position from the zero-padding of the depthwise convolutions. With only 4
blocks on an 8×8 grid, position stays at chance (11%); SANA relies on 20+ blocks. The linear
attention itself is checked against the exact formula in the tests.

**8. Pixel-space flow transformers get the layout right but blur the shapes** after 2,000 steps
(DiT, MMDiT, single-stream).

## Quick start

### Install

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
pytest -q        # 44 tests: shapes, gradients, sampling, tokenizers, JVP, masks, overfitting
```

### Run the full benchmark

```bash
# oracle -> tokenizers -> all 13 models -> evaluation -> results/
bash scripts/run_all.sh
```

### Train, sample and evaluate one model

```bash
text2image-train oracle --out runs/oracle
text2image-train mmdit  --out runs/mmdit
text2image-train vq     --out runs/vq
text2image-train maskgit --tokenizer runs/vq/model.pt --out runs/maskgit

text2image-sample --ckpt runs/mmdit/model.pt -n 8 --out red.png \
    --prompt "a large red circle at the top left"
text2image-eval --ckpt runs/mmdit/model.pt --oracle runs/oracle/model.pt --n 256
```

### Scale up on a GPU

Default model sizes are a CPU preset (~1M parameters). Override any hyperparameter with `--set`:

```bash
text2image-train dit --set dim=384 depth=12 heads=6 --steps 50000 --batch-size 256
```

### Python API

Every model implements the same two methods, which keeps the code easy to compare:

```python
loss = model.loss(images, tokens)                  # training objective
images = model.generate(tokens, cfg_scale=3.0)     # sampling with classifier-free guidance
```

## Repository layout

```
text2image/
├── src/text2image/
│   ├── data/shapes.py          synthetic captioned-shapes dataset (pre-rendered jitter bank)
│   ├── text/                   word tokenizer + small transformer text encoder (CLIP/T5 stand-in)
│   ├── nn/layers.py            attention, adaLN modulation, timestep embedding, patchify, ResBlock
│   ├── generation/             DDPM schedule + DDIM sampler, rectified-flow utilities + Euler sampler
│   ├── tokenizers/vq.py        VQ-VAE with single-scale or VAR multi-scale residual quantizer
│   ├── models/
│   │   ├── base.py             T2IModel / FlowModel / VQModel interfaces, classifier-free guidance
│   │   ├── unet.py             U-Net + DDPM (Stable Diffusion lineage)
│   │   ├── dit.py              DiT (PixArt-α cross-attention) and JiT (x-prediction)
│   │   ├── mmdit.py            MMDiT (SD3 / FLUX double-stream)
│   │   ├── single_stream.py    single-stream DiT (FLUX single blocks / Z-Image)
│   │   ├── sana.py             linear-attention DiT with Mix-FFN and no positional embedding (SANA)
│   │   ├── meanflow.py         one-step generation via the average velocity (MeanFlow)
│   │   ├── transfusion.py      one transformer as both caption LM and image denoiser
│   │   ├── autoregressive.py   next-token AR over VQ codes
│   │   ├── maskgit.py          masked generative transformer
│   │   ├── var.py              next-scale prediction
│   │   └── mar.py              masked AR with continuous tokens + flow head
│   ├── registry.py             model zoo: default sizes, training recipes, checkpoint I/O
│   └── train.py, sample.py, evaluate.py   command-line tools
├── scripts/
│   ├── run_all.sh              end-to-end benchmark
│   └── make_report.py          builds results/ from the trained runs
├── tests/                      unit tests for every component
├── docs/architectures.md       literature survey and design notes
└── results/                    benchmark table, comparison grid, per-model samples
```

## Documentation

- [docs/architectures.md](docs/architectures.md): state of the field, how each architecture works,
  what is simplified compared with production systems, and the full reference list.
- [results/results.md](results/results.md): the benchmark table generated by `scripts/make_report.py`.

## License

[MIT](LICENSE) © Ismail Ben Alla
