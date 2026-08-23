---
license: other
language:
- en
- zh
base_model: MiniMaxAI/MiniMax-M3
tags:
- axquant
- mlx
- moe
- minimax
- experimental
- not-certified
pipeline_tag: text-generation
library_name: mlx
---

# AX-MiniMax-M3-MLX-AXQ-2bit

Experimental **AXQ 2-bit** MLX pack of
[`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) (BF16 source).

**Not certified. Will not be certified in this revision.** Layer-stack SSD
expert paging is too slow for practical serving. Hobby / curiosity only.
**No AXQ 4-bit affine sibling.** An MXFP4 SKU is a separate pack.
Config `num_mtp_modules` is **not** packaged MTP, so this leaf has no `-MTP`.

Language path only. Vision stays BF16; vision generate is not claimed.
Quality vs BF16 was not measured.

| Item | Status |
| --- | --- |
| Convert + `ax_expert_stream.json` | Done on `df-macstudio-m2`; measured main BPW 4.143 |
| Convert git SHA | (stamped at Hub upload) |
| License | Upstream MiniMax LICENSE copied into the pack |
| Stream | `--expert-stream required` |
