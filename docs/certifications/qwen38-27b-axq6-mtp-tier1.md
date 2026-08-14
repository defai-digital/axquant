# Qwen3.8-27B AXQ 6-bit MTP — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/tree/a5a0b700ea7c5c529c66ca3005b79425ab2f7ea6) |
| Source | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host | `df-macbookpro-m3` |
| Size ratio vs uniform-6 | `0.956628` (≤ 1.15) |
| Agent-coding retention vs BF16 | `0.992754` (≥ 0.98) |
| General retention vs BF16 | `1.000000` |
| MTP acceleration | `not-certified` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform-6 | ≤ `1.15` | `0.956628` | Pass |
| Agent-coding retention | ≥ `0.98` | `0.992754` | Pass |
| General retention | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Adapter `qwen38-dense-v1` (dense hybrid VLM `model_type=qwen3_5`); not Qwen 3.6 flagship track.
- Quality measured against same-pin BF16; size against local `mlx_lm.convert` uniform-6 from the same BF16 pin.
- Vision remains BF16-protected; no VLM quality claim.
- MTP acceleration is **not** certified on this record.

Machine-readable: [qwen38-27b-axq6-mtp-tier1.json](qwen38-27b-axq6-mtp-tier1.json).
