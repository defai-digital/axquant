# Gemma 4 26B-A4B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `4.900118671944353` |
| Size ratio vs uniform | `1.0126613143612873` (max `1.15`; pass=True) |
| Quality agent retention | `1.0078740157480315` |
| Quality general retention | `1.0` |
| Tier 1 weight commit / Hub tag **`v1`** | `7a5198b1ae1903187b15bfb5f079d352a139ccc3` (target-only; no assistant MTP) |
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

Machine-readable: [gemma4-26b-a4b-axq4-tier1.json](gemma4-26b-a4b-axq4-tier1.json).
