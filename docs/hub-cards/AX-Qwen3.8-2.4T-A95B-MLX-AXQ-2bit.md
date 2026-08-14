---
license: apache-2.0
language:
- en
- zh
base_model: Qwen/Qwen3.8-2.4T-A95B-FP8
tags:
- axquant
- mlx
- moe
- qwen
- experimental
- not-certified
pipeline_tag: text-generation
library_name: mlx
---

# AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit

Experimental **AXQ 2-bit** MLX pack of
[`Qwen/Qwen3.8-2.4T-A95B-FP8`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8).

**Not certified. Will not be certified in this revision.** Layer-stack
SSD expert paging is too slow for practical serving, so this pack is a
hobby / curiosity artifact: a 2.4T-class Qwen MoE that can exist on a Mac
only because experts are paged from disk. If that sounds fun, enjoy. If you
need something you can actually work with, use a smaller certified AXQ pack
(Qwen 3.6, Flash, Coder-Next, GPT-OSS). **No AXQ 4-bit sibling will be
published** for this base.

This card is convert evidence, not a quality or speed claim. Quality vs
BF16 / FP8 was not measured.

This is an **AXQuant** pack (`qwen38-moe-v1`), not mlx-optiq. Do not load
the OptiQ Qwen 3.8 repos in AX Engine.

Full convert notes:
[docs/qwen38-axq-2bit.md](https://github.com/defai-digital/axquant/blob/main/docs/qwen38-axq-2bit.md).

## Why it is slow

The full table is **~1.13 TiB**. No shipping Mac can resident-load it
(512 GB unified memory is still too small). AX Engine pages one fused
expert layer at a time (`ax_expert_stream.json`, `required=true`). Every
token waits on SSD I/O for routed experts. That is why this revision is
not a product path.

Estimated streamed peak: ~56 GiB resident trunk + ~12 GiB one layer + KV
(about 100 GiB with headroom). You still need
`AX_ENGINE_2BIT_EXPERIMENTAL=1`.

Do not `mlx_lm.load` this pack as a fully resident model.

## Recipe

Affine, group size 32
([`qwen38-experimental-2bit-v0.1.yaml`](https://github.com/defai-digital/axquant/blob/main/examples/qwen38-experimental-2bit-v0.1.yaml)):

| Role | Bits |
| --- | --- |
| Expert, attention, shared MLP | 2 |
| Embedding, router (`mlp.gate`) | 8 |
| Norms, LM head | 16 (BF16) |
| MTP | 16, byte-preserved into `mtp.safetensors` |

Source: official FP8 revision `d2dc35658bcf77e66643428cb52e774cc3b5bd29`
(128×128 `weight_scale_inv`, unfused per-expert tensors). Convert uses the
AXQuant stream backend because `mlx_lm.load` cannot ingest that snapshot.

## Measured artifact

From `axquant_manifest.json` after weight verification (2026-08-13):

| Quantity | Value |
| --- | --- |
| Plan effective BPW | 3.157 |
| Measured total BPW | 4.074 |
| Measured main BPW | 4.030 |
| Logical parameters | 2,446,182,725,504 |
| Weight files | 1,245,853,341,088 bytes (1.13 TiB) |
| MTP sidecar | 26,989,794,744 bytes |
| Product class | `2bit-experimental` |

Stream contract: 512 experts, 10 per token, 828 streamed tensors, layer-stack
mode. Converted `model_type` is `qwen3_5_moe` (92 hybrid-attention layers,
hidden 8192, MoE intermediate 2048, one MTP layer).

## Status

| Item | Status |
| --- | --- |
| Convert + `ax_expert_stream.json` | Done on `df-macstudio-m2` |
| Hub weights | Uploaded |
| Quality vs BF16 / FP8 | Not measured |
| AX Engine cert | **Will not certify this revision** (too slow to be practical) |
| MTP acceleration | Not claimed |

Hobby use only. Not a supported product pack.
