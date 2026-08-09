# Gemma 4 31B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit) commit `a0a0fe3ad14f366646df00251c22377326d7512f`.

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `6.000026926153772` |
| Size ratio vs uniform | `0.8990842384855067` (max `1.1`; pass=True) |
| Quality agent retention | `0.9922480620155038` |
| Quality general retention | `1.0` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference.
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- MTP acceleration Tier 2 **not** claimed.
- Hub pack is **quality-only AXQ** (no fused `assistant/` MTP). A 2026-08-09 experimental
  assistant fuse was **reverted** on the Hub; history remains in git commits.

Machine-readable: [gemma4-31b-axq6-tier1.json](gemma4-31b-axq6-tier1.json).
