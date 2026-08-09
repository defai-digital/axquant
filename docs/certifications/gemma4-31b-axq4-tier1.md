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
| Tier 1 weight commit (size/quality bound) | `5d8d37b8e7105eab1ac843945454a6bdbd986e35` |
| Hub head with assistant-MTP (2026-08-09) | `7534128e0cb04a560a64653a66c6d6ba73092cf3` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (AXQ **target** weights).
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- Hub pack now includes Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json`);
  target weight digests are unchanged from the Tier 1 bind above.
- MTP acceleration Tier 2 **not** claimed (present ≠ certified speed).

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
