# Gemma 4 26B-A4B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit).

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `6.000119253417119` |
| Size ratio vs uniform | `0.8885983382306534` (max `1.1`; pass=True) |
| Quality agent retention | `1.0` |
| Quality general retention | `1.0` |
| Tier 1 weight commit (size/quality bound) | `539f7fbd303621007d307be6139f325cbc12f7ee` |
| Hub head with assistant-MTP (2026-08-09) | `545112ef22e7bd09174c71875c53923ca73b1ff0` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (AXQ **target** weights).
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- Hub pack now includes Gemma **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json`);
  target weight digests are unchanged from the Tier 1 bind above.
- MTP acceleration Tier 2 **not** claimed (present ≠ certified speed).

Machine-readable: [gemma4-26b-a4b-axq6-tier1.json](gemma4-26b-a4b-axq6-tier1.json).
