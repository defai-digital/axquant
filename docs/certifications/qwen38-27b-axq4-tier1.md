# Qwen3.8-27B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit/tree/a8c56f941eafc5d5078177ea03e173d84e30b977) |
| Source | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host | `df-macbookpro-m3` |
| Product class | `4bit` (recovery layout; measured ~5.067 main BPW) |
| Plan | `plan-manual` + recovery recipe (lm_head **8-bit**, attention/MLP **4-bit**) |
| Size ratio vs uniform-4 | `1.144908` (≤ 1.15) |
| Agent-coding retention vs BF16 | `1.007246` (≥ 0.98) |
| General retention vs BF16 | `1.000000` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform-4 | ≤ `1.15` | `1.144908` | Pass |
| Agent-coding retention | ≥ `0.98` | `1.007246` | Pass |
| General retention | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Architecture-prior AXQ-4bit raised the budget to ~5.42 BPW (lm_head BF16) and failed size (~1.22×).
- Recovery uses AXQ-026 **lm_head 8-bit** with measured dual-suite quality retention.
- Adapter `qwen38-dense-v1`; not the Qwen 3.6 flagship track.
- Vision remains BF16-protected; no VLM quality claim.
- MTP acceleration is **not** certified on this record.

Machine-readable: [qwen38-27b-axq4-tier1.json](qwen38-27b-axq4-tier1.json).
