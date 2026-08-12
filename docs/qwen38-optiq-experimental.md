# Qwen3.8-2.4T-A95B — experimental OptiQ MLX packs

**These published packs are not AXQ packs and they are not supported by AX Engine.**

OptiQ remains the current published experimental serve path. A separate native AXQ + AX Engine
layer-stack streaming path is in progress; it has not yet loaded a real Qwen 3.8 pack and does not
make these OptiQ repositories AX Engine-compatible.

Hub repos (weights land after the `df-macstudio-m2` convert/upload job finishes):

| Pack | Hub | Recipe |
| --- | --- | --- |
| Experimental OptiQ 2-bit | [`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-2bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-2bit) | `optiq convert --method static --candidate-bits 2,4 --target-bpw 2.5` |
| Experimental OptiQ 4-bit | [`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-4bit) | `optiq convert --method static --candidate-bits 4,8 --target-bpw 4.5` |

## Why the current published path uses OptiQ

Qwen3.8-2.4T-A95B is 2.4T total / 95B active (`model_type=qwen3_5_moe_text`). A 192 GB Mac
Studio cannot hold the expert table. The currently published OptiQ path pages routed experts from
SSD (`optiq serve --stream-experts`). The in-progress native path instead defines an AXQuant
`ax_expert_stream.json` contract for AX Engine to page one fused layer stack at a time; see
[AX Expert SSD Stream v1](expert-ssd-stream.md).

Convert input is the official FP8 dump
[`Qwen/Qwen3.8-2.4T-A95B-FP8`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8)
(the BF16 dump is 4.45 TiB and does not fit Ext4T).

## Status

| Claim | Status |
| --- | --- |
| Experimental OptiQ MLX artifact | Intended |
| AX Engine supported | **No** for these OptiQ packs; native real-pack validation is pending |
| AXQuant inspect/convert/cert track | Thin native development track; **no certification track** |
| Quality evidence | **Not measured or claimed** |
| MTP acceleration | **Not claimed** (OptiQ's large-FP8 streaming convert may drop the MTP head) |

Serve these published packs only with their documented OptiQ runtime. Do not load them in
`ax-engine`. Keep this warning until AX Engine loads and validates a real native AXQ Qwen 3.8 pack.

Each Hub card retains the experimental / not-AX-Engine disclaimer. Internal conversion jobs and
logs are operational state, not a public reproduction contract.
