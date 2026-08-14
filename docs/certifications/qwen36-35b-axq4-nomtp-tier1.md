# Qwen 3.6 35B-A3B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit/tree/2a5bc33c0a142719024550efd02eddf004d630a0) |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Host | `df-macbookpro-m5` |
| Product class | `4bit` (no MTP) |
| Related MTP certificate | [qwen36-35b-axq4-tier1](qwen36-35b-axq4-tier1.md) |
| MTP acceleration | `not-applicable` |

## Gates

This pack is the **no-MTP sibling** of the certified MTP artifact: same language/vision
weights with `mtp.safetensors` removed. Checkpoint quality retention follows the MTP
sibling language-path evidence; only the MTP sidecar is absent.

| Gate | Result |
| --- | --- |
| Weight size vs matched uniform | Pass (smaller than MTP sibling by omitting MTP) |
| Dual-suite quality (language path) | Pass (inherits MTP sibling language-path suite) |
| MLX-LM / AX Engine load path | Pass |

## Notes

- Prefer this pack when you do not need speculative MTP weights.
- For scoped MTP acceleration, use the MTP sibling certificate instead.

Machine-readable: [qwen36-35b-axq4-nomtp-tier1.json](qwen36-35b-axq4-nomtp-tier1.json).
