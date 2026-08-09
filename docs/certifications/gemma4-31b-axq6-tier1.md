# Gemma 4 31B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `6.000026926153772` |
| Size ratio vs uniform | `0.8990842384855067` (max `1.1`; pass=True) |
| Quality agent retention | `0.9922480620155038` |
| Quality general retention | `1.0` |
| Tier 1 weight commit / Hub tag **`v1`** | `a0a0fe3ad14f366646df00251c22377326d7512f` (target-only; no assistant MTP) |
| Hub tag **`v2`** (same repo name + assistant-MTP) | pin `v2` on Hub; target digests unchanged from `v1` |

**Stable Hub id:** the repository name is **not** renamed so users can track history. `v1` lacked
MTP because the early publish pipeline did not fuse the Gemma assistant drafter into this package;
`v2` adds `assistant/` + contract under the same id.

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (AXQ **target** weights on **`v1`**).
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- Hub **`v2`** includes Gemma **assistant-MTP**; target weight digests match the Tier 1 bind above.
- MTP acceleration Tier 2 **not** claimed (present ≠ certified speed).

Machine-readable: [gemma4-31b-axq6-tier1.json](gemma4-31b-axq6-tier1.json).
