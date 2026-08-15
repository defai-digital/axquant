---
license: apache-2.0
library_name: mlx
tags:
  - mlx
  - mlx-vlm
  - axquant
  - axq
  - muse
  - glimmer
  - multimodal
  - agent
  - development
base_model: meta-models/Muse-Glimmer-30B
pipeline_tag: image-text-to-text
---

# AX-Muse-Glimmer-30B-MLX-AXQ-6bit

**Development** AXQuant (AXQ) **6-bit language** MLX pack of
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
@ `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.

Language attention/MLP at **6-bit**; vision tower / adapter / projection **BF16-preserved**.

Converted via MLX-VLM `muse_glimmer`.

> **Not certified** for AXQuant checkpoint Tier 1 on `df-macstudio-m2` (2026-08-15).
> MLX-VLM load and generate smoke passed. Dual-suite quality vs BF16 cannot run because
> `evaluate-quality` uses the mlx-lm backend, which rejects `model_type=muse_glimmer`.
> See the [evaluation record](https://github.com/defai-digital/axquant/blob/main/docs/certifications/muse-glimmer-30b-axq6-tier1.md).

## Claims

| Claim | Status |
| --- | --- |
| AXQuant architecture-prior / development quant | **Yes** |
| Checkpoint Tier 1 | **Not certified** |
| Dual-suite quality vs BF16 | **Not measured** (mlx-lm backend gap) |
| MLX-VLM load + generate smoke | Passed on `df-macstudio-m2` |
| Vision / multimodal quality | **Not claimed** — vision BF16-protected only |
| Certified release | **No** |

## Attribution

Base weights © Meta (Apache-2.0). Quantization by AXQuant (development).
