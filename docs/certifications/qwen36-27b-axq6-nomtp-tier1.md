# Qwen 3.6 27B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit/tree/66a3ad41300fb14761f6295cef338a83c999131d) |
| Source | `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Host | `df-macbookpro-m5` |
| Product class | `6bit` (no MTP) |
| Related MTP certificate | [qwen36-27b-axq6-tier1](qwen36-27b-axq6-tier1.md) |
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

Machine-readable: [qwen36-27b-axq6-nomtp-tier1.json](qwen36-27b-axq6-nomtp-tier1.json).
