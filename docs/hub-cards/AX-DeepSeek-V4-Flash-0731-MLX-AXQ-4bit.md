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
- deepseek
- deepseek-v4
---

# AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit

AXQuant 4-bit group-128 pack of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
@ `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Attention is 6-bit DWQ clip then affine pack; fused experts are 4-bit affine.
Official DSV4 `chat_template.jinja` is in the repo.

Converted on `df-macstudio-m2`. Recipe
`examples/deepseek-v4-experimental-4bit-g128-dwq-v0.1.yaml`.

## Measured

| Property | Value |
| --- | --- |
| Measured total BPW | `4.371` |
| Disk | ~155 GB |
| Factory decode-128 | ~26.5 tok/s (resident mlx-lm, 192 GB Studio) |
| Checkpoint Tier 1 | **Not certified** on this host (formal 76+44 suite reserved for a later Mac) |

Group-32 4-bit (179 GB) OOMed generate on 192 GB and is not published.

## Runtime

mlx-lm resident. AX Engine 2-bit env is not required. MTP sidecar is present;
acceleration is not claimed.
