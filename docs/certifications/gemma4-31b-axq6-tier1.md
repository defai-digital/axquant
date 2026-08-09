# Gemma 4 31B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08 for **AXQ target** size/quality on `df-macbookpro-m5`.

Hub (live, **with assistant-MTP**): [`AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit) commit `2025e3aec78eb370dda19dcbc88dfae639f4bb5f` (republished 2026-08-09).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-31B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW (target) | `6.000026926153772` |
| Size ratio vs uniform | `0.8990842384855067` (max `1.1`; pass=True) |
| Quality agent retention | `0.9922480620155038` |
| Quality general retention | `1.0` |
| Quality-bound target revision (pre-MTP fuse) | `a0a0fe3ad14f366646df00251c22377326d7512f` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- MLX-LM runtime check on the quality-bound target revision above.
- Vision/multimodal quality **not** claimed.
- Hub pack **includes** Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json` +
  composition provenance). Target digests match the quality-bound revision; assistant is attached
  for product completeness.
- MTP acceleration Tier 2 **not** claimed (present ≠ ≥1.20× / ≥1.10× certified speed).

Machine-readable: [gemma4-31b-axq6-tier1.json](gemma4-31b-axq6-tier1.json).
