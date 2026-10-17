# Text-to-image architectures: a field guide (October 2026)

This page surveys the architecture families behind today's text-to-image systems
and maps each one to the file in `tiny-prism` that implements it. The goal of the
repo is **to make the differences tangible**: every family is implemented from
scratch with the same text encoder, data and evaluation, so you can read, train
and compare them side by side.

## 1. Where the field is (2025-2026)

Two broad paradigms dominate, and they are converging:

| paradigm | how an image is produced | flagship systems |
|---|---|---|
| **Diffusion / flow matching** | start from noise and iteratively denoise the *whole* image | Stable Diffusion 1-3.5, SDXL, FLUX.1 / FLUX.2, Qwen-Image, Z-Image, SANA, PixArt |
| **Autoregressive / masked token modelling** | predict image tokens (discrete codes or continuous vectors) one step at a time | DALL-E 1, Parti, Muse, LlamaGen, VAR / Infinity, MAR, NextStep-1, HunyuanImage 3.0 |

Trends that most current systems share:

* **Transformers replaced U-Nets** as the denoising backbone (DiT, then MMDiT).
  SD3, FLUX, Qwen-Image and Z-Image all use transformer backbones over latent patches.
* **Rectified flow / flow matching replaced DDPM** as the training objective
  (straight noise-to-data paths, velocity prediction, logit-normal timestep sampling, timestep shift).
* **Joint text-image attention** (MMDiT and single-stream DiT) replaced plain
  cross-attention: text tokens are part of the sequence, not just a side input.
  DiT-Air (Apple, 2025) found a plain single-stream DiT is the most parameter-efficient.
* **Bigger language models as text encoders** (T5-XXL, then LLM / VLM encoders:
  Mistral-Small in FLUX.2, Qwen2.5-VL in Qwen-Image, Qwen3 in Z-Image).
* **Autoregressive models dropped vector quantization** (MAR, NextStep-1) or moved
  to coarse-to-fine scales (VAR, Infinity), closing much of the gap with diffusion.
* **Pixel-space is back**: JiT ("Back to Basics", Li & He, CVPR 2026) shows that a
  plain transformer on raw pixel patches works if the network predicts the *clean
  image* (x-prediction) rather than noise or velocity.
* **Unified multimodal models** (HunyuanImage 3.0, Janus-Pro, BLIP3-o, Transfusion-style
  MoE) put understanding and generation in one LLM. They are out of scope for a tiny
  repo, but their image heads are exactly the families below.

## 2. The nine architectures implemented here

### Diffusion & flow family (continuous pixels)

| key | file | what it is | key ideas you can see in the code |
|---|---|---|---|
| `unet` | `models/unet.py` | **U-Net + DDPM**: the Stable Diffusion 1.x / 2.x / SDXL lineage | conv ResBlocks with time embedding, spatial transformers that cross-attend to text, epsilon prediction, cosine schedule, DDIM sampling |
| `dit` | `models/dit.py` | **DiT with cross-attention** (PixArt-alpha) | patch tokens, adaLN-Zero modulation from timestep + pooled text, cross-attention to the text sequence, QK-norm |
| `jit` | `models/dit.py` | **JiT**: same backbone, but predicts the clean image | `prediction="x"` turns the network output into a velocity: `v = (x_t - x̂) / t` |
| `mmdit` | `models/mmdit.py` | **MMDiT** (SD3, FLUX.1 double-stream blocks) | separate weights per modality, *joint* attention over `[text; image]`, the last block's text stream is "pre-only" |
| `single-stream` | `models/single_stream.py` | **single-stream DiT** (FLUX single blocks, Z-Image S3-DiT) | one shared transformer over `[text; image]`, parallel attention + MLP from a fused projection |

All flow models share `FlowModel` in `models/base.py`: `x_t = (1-t)·x0 + t·ε`,
target `v = ε - x0`, logit-normal t, Euler sampling with optional timestep shift.

### Token family (autoregressive / masked)

| key | file | what it is | key ideas you can see in the code |
|---|---|---|---|
| `ar` | `models/autoregressive.py` | **decoder-only next-token AR** (DALL-E 1, Parti, LlamaGen) | VQ codes in raster order, text as a prefix, causal attention, top-k sampling |
| `maskgit` | `models/maskgit.py` | **masked generative transformer** (MaskGIT, Muse) | cosine masking schedule, bidirectional attention, confidence-based parallel decoding in ~12 steps |
| `var` | `models/var.py` | **next-scale prediction** (VAR, NeurIPS 2024 best paper; Infinity) | multi-scale residual VQ, block-causal attention across scales 1→2→3→4→6→8 |
| `mar` | `models/mar.py` | **masked AR with continuous tokens** (MAR; NextStep-1-style flow head) | no quantizer, a per-token MLP flow-matching head, random-order generation |

The discrete models use the VQ-VAE in `tokenizers/vq.py`. One class covers both
a standard single-scale VQ (`scales=(8,)`) and VAR's multi-scale residual quantizer
(`scales=(1,2,3,4,6,8)`), including VAR's `phi` refinement convs and dead-code restarts.

## 3. Shared pieces (so the comparison is fair)

* `text/encoder.py`: a small transformer text encoder producing a token sequence
  (for cross / joint attention) and a pooled vector (for adaLN). This stands in for
  CLIP / T5 / an LLM; it is trained jointly because the vocabulary is tiny.
* **Classifier-free guidance everywhere**: 10% of training prompts are replaced by the
  all-padding null prompt, and every `generate` combines conditional and unconditional
  outputs (noise, velocity, logits or flow-head velocity, depending on the family).
* **EMA weights**, AdamW, warmup + cosine LR, gradient clipping: one training loop for all.

## 4. Deliberate simplifications (what changes at scale)

| here | real systems |
|---|---|
| 32×32 pixels (pixel space) | a VAE / autoencoder latent (8× or 16× downsampling, 16-128 channels), except JiT-style pixel models |
| ~1M-parameter models | 1B-80B parameters |
| tiny word tokenizer + 1-layer text encoder | frozen CLIP / T5-XXL / LLM or VLM encoders |
| learned / sin-cos positions | 2-D or 3-D RoPE (FLUX, Qwen-Image's MSRoPE) |
| no KV cache in AR / VAR sampling | KV caching, speculative / parallel decoding |
| VQ-VAE without GAN or perceptual loss | VQGAN-style adversarial + LPIPS losses, very large codebooks (or bitwise codes, Infinity) |
| no distillation | consistency / adversarial / rectified-flow distillation for 1-4 step sampling |

## 5. References

Implemented here:
* Ho et al., *Denoising Diffusion Probabilistic Models*, 2020. Song et al., *DDIM*, 2021.
* Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion), 2022.
* Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT), 2023.
* Chen et al., *PixArt-α*, 2023.
* Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3 / MMDiT), 2024. https://arxiv.org/abs/2403.03206
* Black Forest Labs, FLUX.1 (double- and single-stream blocks), 2024; FLUX.2, 2025. See also *Demystifying Flux Architecture*, https://arxiv.org/abs/2507.09595
* Z-Image team, *Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer*, 2025. https://arxiv.org/abs/2511.22699
* Li & He, *Back to Basics: Let Denoising Generative Models Denoise* (JiT), CVPR 2026. https://arxiv.org/abs/2511.13720
* Ramesh et al., *Zero-Shot Text-to-Image Generation* (DALL-E), 2021. Yu et al., *Parti*, 2022. Sun et al., *LlamaGen*, 2024.
* Chang et al., *MaskGIT*, 2022; *Muse*, 2023.
* Tian et al., *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction* (VAR), NeurIPS 2024.
* Han et al., *Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis*, CVPR 2025. https://arxiv.org/abs/2412.04431
* Li et al., *Autoregressive Image Generation without Vector Quantization* (MAR), NeurIPS 2024. https://arxiv.org/abs/2406.11838
* NextStep-1 (AR model with a flow-matching head), ICLR 2026.

Context:
* Apple, *DiT-Air: Revisiting the Efficiency of Diffusion Model Architecture Design in Text to Image Generation*, 2025. https://arxiv.org/abs/2503.10618
* Tencent, HunyuanImage 3.0 (80B MoE, unified autoregressive), 2025. https://arxiv.org/abs/2509.23951
* Qwen-Image (MMDiT with a VLM text encoder), 2025.
