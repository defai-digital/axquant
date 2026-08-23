---
license: other
language:
- en
- zh
base_model: moonshotai/Kimi-K3
tags:
- axquant
- mlx
- moe
- kimi
- experimental
- not-certified
pipeline_tag: text-generation
library_name: mlx
---

# AX-Kimi-K3-MLX-AXQ-2bit

Experimental **AXQ 2-bit** MLX pack of
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3).

**Not certified. Will not be certified in this revision.** Stream required.
Hobby / curiosity only. **2-bit only** — no AXQ MXFP4 sibling (source is already
native MXFP4 QAT). Official card has no MTP; this leaf has no `-MTP`.

Language path only. Vision (MoonViT-V2) stays BF16. Convert requires public
mlx-vlm Kimi Delta Attention.

| Item | Status |
| --- | --- |
| Convert + `ax_expert_stream.json` | Done on `df-macstudio-m2`; measured main BPW 4.018 |
| Convert git SHA | (stamped at Hub upload) |
| License | Kimi K3 License copied into the pack |
