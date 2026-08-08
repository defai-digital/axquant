# Gemma 4 26B-A4B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit) commit `main-pending`.

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `4.900118671944353` |
| Size ratio vs uniform | `1.0126613143612873` (max `1.15`; pass=True) |
| Quality agent retention | `1.0078740157480315` |
| Quality general retention | `1.0` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference.
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- MTP acceleration Tier 2 **not** claimed.

Machine-readable: [gemma4-26b-a4b-axq4-tier1.json](gemma4-26b-a4b-axq4-tier1.json).
