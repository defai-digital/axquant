# Qwen3.8-2.4T-A95B — experimental OptiQ MLX packs

**These are not AXQ packs and they are not supported by AX Engine.**

Hub repos (weights land after the `df-macstudio-m2` convert/upload job finishes):

| Pack | Hub | Recipe |
| --- | --- | --- |
| Experimental OptiQ 2-bit | [`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-2bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-2bit) | `optiq convert --method static --candidate-bits 2,4 --target-bpw 2.5` |
| Experimental OptiQ 4-bit | [`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-OptiQ-4bit) | `optiq convert --method static --candidate-bits 4,8 --target-bpw 4.5` |

## Why OptiQ, not AXQuant

Qwen3.8-2.4T-A95B is 2.4T total / 95B active (`model_type=qwen3_5_moe_text`). A 192 GB Mac
Studio cannot hold the expert table. AXQuant / mlx-lm / AX Engine keep all experts resident,
so an AXQ 2-bit pack (~0.8 TB) still cannot *run* on this host. OptiQ can page routed
experts from SSD (`optiq serve --stream-experts`).

Convert input is the official FP8 dump
[`Qwen/Qwen3.8-2.4T-A95B-FP8`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8)
(the BF16 dump is 4.45 TiB and does not fit Ext4T).

## Status

| Claim | Status |
| --- | --- |
| Experimental OptiQ MLX artifact | Intended |
| AX Engine supported | **No** |
| AXQuant inspect/convert/cert track | **No** — not in the public certification matrix |
| Quality vs BF16 | **Not measured** |
| MTP acceleration | **Not claimed** (OptiQ's large-FP8 streaming convert may drop the MTP head) |

Serve only with `mlx-optiq` ≥ 0.4.19. Do not load these checkpoints in `ax-engine`.

## Operator job (macstudio)

Orchestrator (already launched on `df-macstudio-m2`):

```text
/Volumes/Ext4T/axquant/scripts/run_qwen38_optiq_experimental.sh
```

Logs:

```text
/Volumes/Ext4T/axquant/logs/qwen38-optiq-orchestrator.log
/Volumes/Ext4T/axquant/logs/qwen38-optiq-2bit-convert.log
/Volumes/Ext4T/axquant/logs/qwen38-optiq-4bit-convert.log
```

The job converts 2-bit, uploads, deletes the local 2-bit tree, then converts and uploads 4-bit.
Each Hub card states the experimental / not-AX-Engine disclaimer.
