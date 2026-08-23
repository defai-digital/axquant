# AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP

Technical report for the experimental AXQuant 2-bit Super-class pack of
DeepSeek-V4-Pro-0813. Hub:

[`AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP)

**This revision will not be certified.** SSD layer-stack paging is too slow
for practical serving. The pack stays on the Hub as convert evidence and a
hobby / curiosity artifact. It is not a Tier 1 / Tier 2 product pack.
**No AXQ 4-bit sibling will be released** for this base.

Related: [experimental pack index](deepseek-v4-pro-0813-axq-experimental.md).

## What this pack is

DeepSeek-V4-Pro-0813 is a hybrid-attention text MoE (61 layers, 384 routed
experts, 6 experts per token, hidden 7168, MoE intermediate 3072, one DSpark
stage under `mtp.*`). Official `model_type` is `deepseek_v4`. Total ~1.65T
parameters (49B active). Context 1M.

The 2-bit pack is an AXQuant (`deepseek-v4-v1`) affine convert of the official
mixed FP4+FP8 snapshot, not an mlx-optiq artifact. Routed experts are fused
into `switch_mlp` stacks and paged one layer at a time. Embeddings, attention,
routers, shared experts, norms, the LM head, and DSpark/MTP stay in the usual
resident roles.

Product class: `2bit-experimental`. Profile: `general`.

## Source

| Item | Value |
| --- | --- |
| Upstream | [`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |
| Revision | `72e1d3230f6c080a530b0a1d46f8eb4602340597` |
| Format | Official mixed FP4 experts (I8 e2m1) + FP8 + BF16 |
| Expert layout | Unfused `layers.*.ffn.experts.{i}.{w1,w2,w3}` |
| Disk | ~1.78 TB — downloaded with Hugging Face **Xet** high performance on Ext12T |

`mlx_lm.load` cannot ingest this snapshot on a 192 GB Studio. Convert therefore
uses the AXQuant stream backend (dequant one layer’s 384 FP4 experts, stack,
affine pack, write, then emit metadata). Factory host: `df-macstudio-m2`.

## Recipe

Manual recipe
[`examples/deepseek-v4-pro-0813-experimental-2bit-v0.1.yaml`](../examples/deepseek-v4-pro-0813-experimental-2bit-v0.1.yaml)
(`--allow-unmeasured`). Affine, group size 32:

| Role | Bits | Notes |
| --- | --- | --- |
| Expert, attention, shared MLP | 2 | Fused experts pack as one `switch_mlp` module per projection |
| Embedding, router (`ffn.gate`) | 8 | Protection floors |
| Norms, LM head | 16 (BF16) | Protection floors |
| DSpark / MTP | 16 | Byte-preserved into `mtp.safetensors` |

Convert flags: `--expert-stream required --allow-unmeasured --ax-engine-manifest skip`.

## Expert stream contract

`ax_expert_stream.json` (`axquant.expert-stream.v1`), mode `layer-stack`,
`required=true`.

A 512 GB Mac still cannot resident-load the full table. Peak RAM for a
streamed serve is roughly resident trunk + one layer + KV. Runtime also
needs `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

`RuntimeMetadata.memory_policy.expert_stream` is `required`. Do not pass
`--expert-stream off` at convert time for this pack.

## Measured artifact

Values from `axquant_manifest.json` on `df-macstudio-m2` after convert
verification (2026-08-22). Hub listing is confirmed only after upload.

| Quantity | Value |
| --- | --- |
| Plan effective BPW | 3.655 |
| Measured total BPW | 4.109 |
| Measured main BPW | 4.097 |
| Logical parameters | 1,650,497,936,906 |
| Main logical parameters | 1,572,999,528,803 |
| Weight files | 847,647,914,360 bytes (789.3 GiB) |
| Main weight files | 805,667,764,532 bytes |
| MTP sidecar | 41,980,149,828 bytes (`mtp.safetensors`) |
| Files in the artifact tree | 216 |

## What is not claimed

| Claim | Status |
| --- | --- |
| Convert + stream metadata | Done on `df-macstudio-m2` (`required=true`) |
| Hub upload | Live: 192 shards + `mtp.safetensors` (Hub commit `2d0657484482dbb371dbb187e465dfec234f8858`) |
| AX Engine generation smoke | Pending stream-capable engine on the verify host |
| Checkpoint Tier 1 / Tier 2 | **Will not certify this revision** (too slow to be practical) |
| Quality vs BF16 / FP8 | Not measured |
| DSpark / MTP acceleration | Not claimed (sidecar is byte-preserved) |

Serve only with AX Engine layer-stack streaming. Do not `mlx_lm.load` the
full pack as a resident model.

## Reproduce the convert

On the factory convert host with the Ext12T Hugging Face cache, Xet HP, and
the `feat/expert-ssd-stream` tree. Set `HF_HOME` to that cache, export
`HUGGINGFACE_HUB_CACHE` / `HF_HUB_CACHE` / `HF_XET_CACHE` under it, set
`HF_XET_HIGH_PERFORMANCE=1`, and unset `HF_HUB_ENABLE_HF_TRANSFER`. Then:

```bash
scripts/run_deepseek_v4_pro_0813_axq2_stream.sh
```
