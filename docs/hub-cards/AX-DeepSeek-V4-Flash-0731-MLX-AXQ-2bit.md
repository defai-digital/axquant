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

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit

**Development / experimental** AXQuant 2-bit pack of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
@ `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Converted on `df-macstudio-m2` (Apple M2 Ultra, 192 GB) from the native FP8
0731 source (`quant_method=fp8`). Product class `2bit-experimental`.

> This is **not** the older `DeepSeek-V4-Flash` Hub pack. Do not treat
> [AX-DeepSeek-V4-Flash-MLX-AXQ-2bit](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit)
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
| Checkpoint Tier 1 (generation viability suite) | **Not certified** on this record; Studio 15+15 factory QA with chat is 0.633 combined |
| AX Engine native manifest | **Not generated** — `generate-manifest --validate` rejected split `switch_mlp.gate_proj` / `up_proj` (`[256, 2048, 256]` vs fused `[256, 4096, 256]`) |
| MTP acceleration | **Not certified** |

Requires `AX_ENGINE_2BIT_EXPERIMENTAL=1` if served with AX Engine after a future
manifest fix. mlx-lm generate does not need that env.

## Attribution

Base weights © DeepSeek. Quantization by AXQuant (development).
