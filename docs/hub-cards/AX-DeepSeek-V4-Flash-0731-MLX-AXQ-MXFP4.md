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
- mxfp4
- deepseek
---

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-MXFP4

**Stub SKU. Hub weights are not uploaded.**

AXQuant MXFP4 pack of DeepSeek-V4-Flash-0731 (`--q-mode mxfp4`) is listed
for catalog completeness. Recipe `examples/deepseek-v4-experimental-mxfp4-v0.1.yaml`.
The source *can* do MTP; this repo does **not** ship `mtp.safetensors`, so
the leaf does **not** end in `-MTP`. The suffix is added only when a sidecar
ships.

**Checkpoint Tier 1 is not certified** on `df-macstudio-m2` (192 GB). If the
pack is in the 170 GB+ class, factory generate/cert is memory-blocked.

This is an MLX Apple MXFP4 pack, not an OCP or NVIDIA checkpoint.
