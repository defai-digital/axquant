# Gemma 4 31B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08 for **AXQ target** size/quality on `df-macbookpro-m5`.

Hub (live, **with assistant-MTP**): [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP) commit `bc2de70bf2bc6b03da1d50801a4f95894d32eec4` (renamed to Qwen-style -MTP 2026-08-09).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW (target) | `4.899922859286156` |
| Size ratio vs uniform | `1.0403241383095085` (max `1.15`; pass=True) |
| Quality agent retention | `0.9923664122137404` |
| Quality general retention | `1.0` |
| Quality-bound target revision (pre-MTP fuse) | `5d8d37b8e7105eab1ac843945454a6bdbd986e35` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- MLX-LM runtime check on the quality-bound target revision above.
- Vision/multimodal quality **not** claimed.
- Hub pack **includes** Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json` +
  composition provenance). Target digests match the quality-bound revision; assistant is attached
  for product completeness.
- MTP acceleration Tier 2 **not** claimed (present ≠ ≥1.20× / ≥1.10× certified speed).

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
