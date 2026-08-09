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
| Tier 1 weight commit (size/quality bound) | `7a5198b1ae1903187b15bfb5f079d352a139ccc3` |
| Hub head with assistant-MTP (2026-08-09) | `e15687f110f210ea6b35b3c18ba92208dccf4176` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (AXQ **target** weights).
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- Hub pack now includes Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json`);
  target weight digests are unchanged from the Tier 1 bind above.
- MTP acceleration Tier 2 **not** claimed (present ≠ certified speed).

Machine-readable: [gemma4-26b-a4b-axq4-tier1.json](gemma4-26b-a4b-axq4-tier1.json).
