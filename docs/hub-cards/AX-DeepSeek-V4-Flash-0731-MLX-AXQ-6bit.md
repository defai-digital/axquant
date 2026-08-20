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

**Stub SKU. Hub weights are not uploaded.**

AXQuant 6-bit group-128 pack of DeepSeek-V4-Flash-0731 is listed for
catalog completeness. Recipe `examples/deepseek-v4-experimental-6bit-g128-v0.1.yaml`.
The source *can* do MTP; this repo does **not** ship `mtp.safetensors`, so
the leaf does **not** end in `-MTP`. The suffix is added only when a sidecar
ships.

**Checkpoint Tier 1 cannot be run on `df-macstudio-m2` (192 GB).** Dual-suite
generate is memory-blocked. Recert on a larger Mac. The SKU stays listed.
