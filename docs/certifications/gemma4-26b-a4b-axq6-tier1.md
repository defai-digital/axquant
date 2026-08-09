# Gemma 4 26B-A4B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08 for **AXQ target** size/quality on `df-macbookpro-m5`.

Hub (live, **with assistant-MTP**): [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP) commit `4a62bf66aa74f2063bbbf3ad0ff2cbdc72dc5bcb` (renamed to Qwen-style -MTP 2026-08-09).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW (target) | `6.000119253417119` |
| Size ratio vs uniform | `0.8885983382306534` (max `1.1`; pass=True) |
| Quality agent retention | `1.0` |
| Quality general retention | `1.0` |
| Quality-bound target revision (pre-MTP fuse) | `539f7fbd303621007d307be6139f325cbc12f7ee` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- MLX-LM runtime check on the quality-bound target revision above.
- Vision/multimodal quality **not** claimed.
- Hub pack **includes** Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json` +
  composition provenance). Target digests match the quality-bound revision; assistant is attached
  for product completeness.
- MTP acceleration Tier 2 **not** claimed (present ≠ ≥1.20× / ≥1.10× certified speed).

Machine-readable: [gemma4-26b-a4b-axq6-tier1.json](gemma4-26b-a4b-axq6-tier1.json).
