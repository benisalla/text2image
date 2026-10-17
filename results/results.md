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
