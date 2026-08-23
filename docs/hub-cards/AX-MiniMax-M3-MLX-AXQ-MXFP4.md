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
- mxfp4
pipeline_tag: text-generation
library_name: mlx
---

# AX-MiniMax-M3-MLX-AXQ-MXFP4

Experimental **AXQ MXFP4** MLX pack of
[`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) (BF16 source).

**Not certified. Will not be certified in this revision.** Stream required.
Hobby / curiosity only. Default Hub name has no `-MTP`.

Language path only. Vision stays BF16.

| Item | Status |
| --- | --- |
| Convert + `ax_expert_stream.json` | Done on `df-macstudio-m2`; measured main BPW 4.366 |
| Convert git SHA | (stamped at Hub upload) |
| License | Upstream MiniMax LICENSE copied into the pack |
