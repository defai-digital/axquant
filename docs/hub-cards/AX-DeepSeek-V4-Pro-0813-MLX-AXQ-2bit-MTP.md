---
license: mit
language:
- en
- zh
base_model: deepseek-ai/DeepSeek-V4-Pro-0813
tags:
- axquant
- mlx
- moe
- deepseek
- experimental
- not-certified
- mtp
pipeline_tag: text-generation
library_name: mlx
---

# AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP

Experimental **AXQ 2-bit** MLX pack of
[`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813).

**Not certified. Will not be certified in this revision.** Layer-stack
SSD expert paging is too slow for practical serving, so this pack is a
hobby / curiosity artifact: a 1.6T-class DeepSeek V4 Pro MoE that can exist
on a Mac only because experts are paged from disk. If that sounds fun, enjoy.
If you need something you can actually work with, use a smaller certified AXQ
pack (Qwen 3.6, Flash, Coder-Next, GPT-OSS). **No AXQ 4-bit sibling will be
published** for this base.

This card is convert evidence, not a quality or speed claim. Quality vs
BF16 / FP8 was not measured.

This is an **AXQuant** pack (`deepseek-v4-v1`), not mlx-optiq. Do not load
OptiQ DeepSeek repos in AX Engine.

Full convert notes:
[docs/reports/deepseek-v4-pro-0813-axq-2bit.md](https://github.com/defai-digital/axquant/blob/main/docs/reports/deepseek-v4-pro-0813-axq-2bit.md).

## Why it is slow

The full expert table does not fit in unified memory on any shipping Mac
(512 GB is still too small). AX Engine pages one fused expert layer at a
time (`ax_expert_stream.json`, `required=true`). Every token waits on SSD
I/O for routed experts. That is why this revision is not a product path.

You still need `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

Do not `mlx_lm.load` this pack as a fully resident model.

## Recipe

Affine, group size 32
([`deepseek-v4-pro-0813-experimental-2bit-v0.1.yaml`](https://github.com/defai-digital/axquant/blob/main/examples/deepseek-v4-pro-0813-experimental-2bit-v0.1.yaml)):

| Role | Bits |
| --- | --- |
| Expert, attention, shared MLP | 2 |
| Embedding, router (`ffn.gate`) | 8 |
| Norms, LM head | 16 (BF16) |
| DSpark / MTP | 16, byte-preserved into `mtp.safetensors` |

Source: official mixed FP4+FP8 revision `72e1d3230f6c080a530b0a1d46f8eb4602340597`.
Convert uses the AXQuant stream backend because `mlx_lm.load` cannot ingest
that snapshot. DSpark speculative decode is packaged, not enabled. Stock
MLX-LM can see the text backbone; AX Engine keeps uncertified MTP on direct
decode.

## Status

| Item | Status |
| --- | --- |
| Convert + `ax_expert_stream.json` | Factory job on `df-macstudio-m2` |
| Convert git SHA | (stamped at Hub upload) |
| License | Upstream DeepSeek LICENSE copied into the pack |
| Hub weights | Uploaded after convert |
| Quality vs BF16 / FP8 | Not measured |
| AX Engine cert | **Will not certify this revision** (too slow to be practical) |
| MTP present | Native `mtp.safetensors` sidecar (DSpark under `mtp.*`) |
| MTP / DSpark acceleration | Not claimed |

Hobby use only. Not a supported product pack.
