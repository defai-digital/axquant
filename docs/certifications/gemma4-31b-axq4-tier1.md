# Gemma 4 31B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `4.899922859286156` |
| Size ratio vs uniform | `1.0403241383095085` (max `1.15`; pass=True) |
| Quality agent retention | `0.9923664122137404` |
| Quality general retention | `1.0` |
| Tier 1 weight commit / Hub tag **`v1`** | `5d8d37b8e7105eab1ac843945454a6bdbd986e35` (target-only; no assistant MTP) |
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

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
