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
- mixed-precision
- axquant
- axq
- development
- deepseek
- deepseek-v4
- experimental
---

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP

**Development / experimental** AXQuant 2-bit pack of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
@ `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Converted on `df-macstudio-m2` (Apple M2 Ultra, 192 GB) from the native FP8
0731 source (`quant_method=fp8`). Product class `2bit-experimental`.

> This is **not** the older `DeepSeek-V4-Flash` Hub pack. Do not treat
> [AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP)
> certificates as evidence for this 0731 revision.

## Measured precision

| Property | Value |
| --- | --- |
| Target class | `2bit-experimental` |
| Measured main BPW | `3.1328993873020314` |
| Measured total BPW | `3.2142055528774454` |
| Weight bytes | `122,212,298,775` |
| Source | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Convert host | `df-macstudio-m2` |
| AXQuant | `1.8.1` |
| mlx / mlx-lm | `0.32.0` / `0.31.3` (vendored `deepseek_v4` + FP8 load hook) |

## Claims

| Claim | Status |
| --- | --- |
| Converted on Studio from the pinned 0731 revision | **Yes** |
| mlx-lm load + generate smoke | **Passed** on `df-macstudio-m2` |
| Official DSV4 `chat_template.jinja` | **In pack** |
| Checkpoint Tier 1 (generation viability suite) | **Not certified** on this record; Studio 15+15 factory QA with official DSV4 chat is combined **0.633** on both mlx-lm and AX Engine 7.0.2 native |
| AX Engine native manifest | **Generated** on `df-macstudio-m2` — split `switch_mlp.gate_proj` + `up_proj` remapped to `ffn_gate_exps` / `ffn_up_exps` (`packed=0`, 43/43). Stock 7.0.2 server loaded and served `/v1/chat/completions` |
| MTP assets (`mtp.safetensors`) | **Packaged** — Hub name uses `-MTP` |
| MTP acceleration | **Not certified** (direct decode is the default) |

Requires `AX_ENGINE_2BIT_EXPERIMENTAL=1` for AX Engine 7.0.2 native serve.
mlx-lm generate does not need that env. Comparison:
[optiq2-vs-axq2-v190](../deepseek-v4-flash-0731-optiq2-vs-axq2-v190.md).

## Attribution

Base weights © DeepSeek. Quantization by AXQuant (development).
