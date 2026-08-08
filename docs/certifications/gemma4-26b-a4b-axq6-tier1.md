# Gemma 4 26B-A4B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified on 2026-08-08.

Hub: [`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit) commit `539f7fbd303621007d307be6139f325cbc12f7ee`.

| Property | Value |
| --- | --- |
| Source | `google/gemma-4-26B-A4B-it` |
| Product class | `6bit` |
| Host | `df-macbookpro-m5` |
| Measured BPW | `6.000119253417119` |
| Size ratio vs uniform | `0.8885983382306534` (max `1.1`; pass=True) |
| Quality agent retention | `1.0` |
| Quality general retention | `1.0` |

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference.
- MLX-LM runtime check.
- Vision/multimodal quality **not** claimed.
- MTP acceleration Tier 2 **not** claimed.

Machine-readable: [gemma4-26b-a4b-axq6-tier1.json](gemma4-26b-a4b-axq6-tier1.json).
