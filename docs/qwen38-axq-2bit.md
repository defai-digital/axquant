# AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP

Technical report for the experimental AXQuant 2-bit Super-class pack of
Qwen3.8-2.4T-A95B. Hub:

[`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP)

**This revision will not be certified.** SSD layer-stack paging is too slow
for practical serving. The pack stays on the Hub as convert evidence and a
hobby / curiosity artifact. It is not a Tier 1 / Tier 2 product pack.
**No AXQ 4-bit sibling will be released** for this base.

Related: [experimental pack index](qwen38-axq-experimental.md). OptiQ Hub repos
for this model are a different path and are not AX Engine artifacts
([qwen38-optiq-experimental.md](qwen38-optiq-experimental.md)).

## What this pack is

Qwen3.8-2.4T-A95B is a hybrid-attention text MoE (92 layers, 512 routed
experts, 10 experts per token, hidden 8192, MoE intermediate 2048, one MTP
layer). Official `model_type` is `qwen3_5_moe_text`; the converted MLX config
uses `qwen3_5_moe` so public `mlx_lm` can resolve the architecture class.

The 2-bit pack is an AXQuant (`qwen38-moe-v1`) affine convert of the official
FP8 snapshot, not an mlx-optiq artifact. Routed experts are fused into
`switch_mlp` stacks and paged one layer at a time. Embeddings, attention,
routers, shared experts, norms, the LM head, and MTP stay in the usual
resident roles.

Product class: `2bit-experimental`. Profile: `general`.

## Source

| Item | Value |
| --- | --- |
| Upstream | [`Qwen/Qwen3.8-2.4T-A95B-FP8`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) |
| Revision | `d2dc35658bcf77e66643428cb52e774cc3b5bd29` |
| Format | Official FP8, 128×128 `weight_scale_inv` blocks |
| Expert layout | Unfused `model.layers.*.mlp.experts.{i}.{gate,up,down}_proj` |
| BF16 dump | 4.45 TiB — not used (does not fit Ext4T) |

`mlx_lm.load` cannot ingest this snapshot: sanitize expects packed
`experts.gate_up_proj`, does not fuse indexed experts, leaves
`weight_scale_inv` extras, and drops `mtp.*`. Convert therefore uses the
AXQuant stream backend (dequant one layer’s 512 FP8 experts, stack, affine
pack, write, then emit metadata).

## Recipe

Manual recipe
[`examples/qwen38-experimental-2bit-v0.1.yaml`](../examples/qwen38-experimental-2bit-v0.1.yaml)
(`--allow-unmeasured`). Affine, group size 32:

| Role | Bits | Notes |
| --- | --- | --- |
| Expert, attention, shared MLP | 2 | Fused experts pack as one `switch_mlp` module per projection |
| Embedding, router (`mlp.gate`) | 8 | Protection floors |
| Norms, LM head | 16 (BF16) | Protection floors |
| MTP | 16 | Byte-preserved into `mtp.safetensors` |

Convert flags: `--expert-stream required --allow-unmeasured --ax-engine-manifest skip`.

## Measured artifact

Values from `axquant_manifest.json` on the convert host after weight
verification (2026-08-13).

| Quantity | Value |
| --- | --- |
| Plan effective BPW | 3.157 |
| Measured total BPW | 4.074 |
| Measured main BPW | 4.030 |
| Logical parameters | 2,446,182,725,504 |
| Main logical parameters | 2,419,804,697,984 |
| Weight files | 1,245,853,341,088 bytes (1.13 TiB) |
| Main weight files | 1,218,863,546,344 bytes |
| MTP sidecar | 26,989,794,744 bytes (`mtp.safetensors`) |
| Files in the artifact tree | 298 |

Measured BPW is above the plan figure because affine sidecars (`.scales` /
`.biases`) and the protected MTP/head share are included in storage.

## Expert stream contract

`ax_expert_stream.json` (`axquant.expert-stream.v1`), mode `layer-stack`,
`required=true`.

| Field | Value |
| --- | --- |
| Experts | 512 |
| Experts per token | 10 |
| Stream tensors | 828 |
| Estimated full-resident size | 1,245,852,449,280 bytes (~1.13 TiB) |
| Estimated resident (non-expert) | 60,441,475,584 bytes (~56.3 GiB) |
| Estimated max one-layer expert | 12,884,901,888 bytes (~12.0 GiB) |

A 512 GB Mac still cannot resident-load the full table. Peak RAM for a
streamed serve is roughly resident trunk + one layer + KV, on the order of
~100 GiB with a 32 GiB headroom reserve — within a 192 GB Studio if the
engine pages experts. Runtime also needs `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

`RuntimeMetadata.memory_policy.expert_stream` is `required`. Do not pass
`--expert-stream off` at convert time (rejected for this adapter).

## What is not claimed yet

| Claim | Status |
| --- | --- |
| Convert + stream metadata | Done on `df-macstudio-m2` |
| Hub upload | In progress at the time of this report |
| AX Engine generation smoke | Pending stream-capable engine on the verify host |
| Checkpoint Tier 1 / Tier 2 | **Will not certify this revision** (too slow to be practical) |
| Quality vs BF16 / FP8 | Not measured |
| MTP acceleration | Not claimed (sidecar is byte-preserved) |

Serve only with AX Engine layer-stack streaming. Do not `mlx_lm.load` the
full pack as a resident model. Do not load the OptiQ Qwen 3.8 repos in
AX Engine.

## Reproduce the convert

On `df-macstudio-m2` with the Ext4T FP8 snapshot and the
`feat/expert-ssd-stream` tree:

```bash
export PYTHONPATH=/path/to/axquant/src
python -m axquant inspect \
  --model /path/to/Qwen3.8-2.4T-A95B-FP8 \
  --model-id Qwen/Qwen3.8-2.4T-A95B \
  --revision d2dc35658bcf77e66643428cb52e774cc3b5bd29 \
  --allow-quantized \
  --output inventory.json

python -m axquant plan-manual \
  --inventory inventory.json \
  --recipe examples/qwen38-experimental-2bit-v0.1.yaml \
  --output plan.json

python -m axquant convert \
  --model /path/to/Qwen3.8-2.4T-A95B-FP8 \
  --revision d2dc35658bcf77e66643428cb52e774cc3b5bd29 \
  --plan plan.json \
  --allow-unmeasured \
  --expert-stream required \
  --ax-engine-manifest skip \
  --output ./AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP
```

The historical convert used `--output ./AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit`
(no `-MTP`) even though convert packaged `mtp.safetensors`. That name is a
defect: Hub identity is `...-2bit-MTP`. The operator move is
`scripts/hnc_hub_move_and_refresh.py` (dry-run default).
