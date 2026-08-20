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
| AXQuant | `1.9.0` |

## Claims

| Claim | Status |
| --- | --- |
| Converted on Studio from the pinned 0731 revision | **Yes** |
| Official DSV4 `chat_template.jinja` | **In pack** |
| Checkpoint Tier 1 (generation viability suite) | **Not certified** — factory 15+15 combined **0.633** on AX Engine **7.1.5** native (floor 0.90). Same score as the prior 7.0.2 native run. |
| AX Engine 7.1.5 native load | **Passed** on `df-macstudio-m2` (Hub commit `cb1a34b4`, `--stream-experts off`, chat smoke `Okay.`) |
| Decode-128 (informational) | 15.535 tok/s on 7.1.5; not a Tier 1 claim |
| MTP assets (`mtp.safetensors`) | **Packaged** — Hub name uses `-MTP` |
| MTP acceleration | **Not certified** (7.1.5 default is direct decode for uncertified MTP) |

Requires `AX_ENGINE_2BIT_EXPERIMENTAL=1` for AX Engine native serve.
Certificate:
[deepseek-v4-flash-0731-axq2-tier1.md](https://github.com/defai-digital/axquant/blob/main/docs/certifications/deepseek-v4-flash-0731-axq2-tier1.md).
Comparison vs OptiQ 2-bit:
[optiq2-vs-axq2-v190](https://github.com/defai-digital/axquant/blob/main/docs/deepseek-v4-flash-0731-optiq2-vs-axq2-v190.md).

## Attribution

Base weights © DeepSeek. Quantization by AXQuant (development).
