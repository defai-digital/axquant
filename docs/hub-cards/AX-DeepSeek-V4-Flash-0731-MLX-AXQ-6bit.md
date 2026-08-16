---
license: mit
library_name: mlx
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- quantized
- axquant
- axq
- deepseek
---

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-6bit

AXQuant 6-bit group-128 pack of DeepSeek-V4-Flash-0731.
Recipe `examples/deepseek-v4-experimental-6bit-g128-v0.1.yaml`.

**Checkpoint Tier 1 cannot be run on `df-macstudio-m2` (192 GB).** Dual-suite
generate is memory-blocked. Recert on a larger Mac. The SKU stays listed.
