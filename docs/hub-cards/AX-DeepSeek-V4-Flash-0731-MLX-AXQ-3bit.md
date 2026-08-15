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

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-3bit

**Development / experimental** AXQuant 3-bit pack of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
@ `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Converted on `df-macstudio-m2`. Product class `3bit-experimental`.

> Not the older `DeepSeek-V4-Flash` 3-bit Hub pack.

## Measured precision

| Property | Value |
| --- | --- |
| Target class | `3bit-experimental` |
| Measured main BPW | `4.110998966099255` |
| Measured total BPW | `4.128490312221629` |
| Weight bytes | `156,975,738,865` |
| Convert host | `df-macstudio-m2` |
| AXQuant | `1.8.1` |

## Claims

| Claim | Status |
| --- | --- |
| Converted on Studio from the pinned 0731 revision | **Yes** |
| mlx-lm load + generate smoke | **Passed** on `df-macstudio-m2` |
| Checkpoint Tier 1 | **Not certified** this record |
| AX Engine native manifest | **Not generated** (same split gate/up miss as 2-bit) |
| MTP acceleration | **Not certified** |

Requires `AX_ENGINE_3BIT_EXPERIMENTAL=1` for a future engine serve path.

## Attribution

Base weights © DeepSeek. Quantization by AXQuant (development).
