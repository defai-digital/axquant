# Gemma 4 26B-A4B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08 for **AXQ target** size/quality on `df-macbookpro-m5`.

Hub (live, **with assistant-MTP**): [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP) commit `85b0a78a14843a818d403f9a2525efa2f081c2a4` (renamed to Qwen-style -MTP 2026-08-09).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `4bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW (target) | `4.900118671944353` |
| Size ratio vs uniform | `1.0126613143612873` (max `1.15`; pass=True) |
| Quality agent retention | `1.0078740157480315` |
| Quality general retention | `1.0` |
| Quality-bound target revision (pre-MTP fuse) | `7a5198b1ae1903187b15bfb5f079d352a139ccc3` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- MLX-LM runtime check on the quality-bound target revision above.
- Vision/multimodal quality **not** claimed.
- Hub pack **includes** Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json` +
  composition provenance). Target digests match the quality-bound revision; assistant is attached
  for product completeness.
- MTP acceleration Tier 2 **not** claimed (present ≠ ≥1.20× / ≥1.10× certified speed).

Machine-readable: [gemma4-26b-a4b-axq4-tier1.json](gemma4-26b-a4b-axq4-tier1.json).
