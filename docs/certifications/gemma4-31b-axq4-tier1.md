# Gemma 4 31B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit) commit `5d8d37b8e7105eab1ac843945454a6bdbd986e35`.

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `4.899922859286156` |
| Size ratio vs uniform | `1.0403241383095085` (max `1.15`; pass=True) |
| Quality agent retention | `0.9923664122137404` |
| Quality general retention | `1.0` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference.
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- MTP acceleration Tier 2 **not** claimed.
- Hub pack is **quality-only AXQ** (no fused `assistant/` MTP). A 2026-08-09 experimental
  assistant fuse was **reverted** on the Hub; history remains in git commits.

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
