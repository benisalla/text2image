# tiny-prism

**Nine modern text-to-image architectures, written from scratch, small enough to train and compare on a laptop CPU.**

A prism turns one beam of light into a full spectrum. `tiny-prism` takes one prompt and
sends it through the full spectrum of today's generative architectures, from the
Stable Diffusion U-Net to FLUX-style MMDiT, VAR's next-scale prediction and MAR's
continuous-token autoregression. All nine share the same text encoder, data,
training loop and automatic evaluation, so the *only* thing that differs is the architecture.

| family | models |
|---|---|
| **Diffusion / flow matching** | `unet` (SD 1.x DDPM U-Net) · `dit` (DiT / PixArt-α) · `jit` (JiT pixel x-prediction) · `mmdit` (SD3 / FLUX double-stream) · `single-stream` (FLUX single blocks / Z-Image S3-DiT) |
| **Autoregressive / masked tokens** | `ar` (DALL-E / Parti / LlamaGen) · `maskgit` (MaskGIT / Muse) · `var` (VAR next-scale) · `mar` (MAR + flow head, NextStep-1 style) |

See [docs/architectures.md](docs/architectures.md) for the survey of the field (2025-2026),
the reasoning behind this selection, and exactly where each idea lives in the code.

## Results

<!-- RESULTS -->

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
pytest -q                      # 28 tests: shapes, gradients, sampling, tokenizers, overfitting

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
