# Qwen 3.6 35B-A3B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit/tree/8519cd1d15277f20deb899d6ea588c2e91815234) |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Host | `df-macbookpro-m5` |
| Product class | `6bit` (no MTP) |
| Related MTP certificate | [qwen36-35b-axq6-tier1](qwen36-35b-axq6-tier1.md) |
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

Machine-readable: [qwen36-35b-axq6-nomtp-tier1.json](qwen36-35b-axq6-nomtp-tier1.json).
