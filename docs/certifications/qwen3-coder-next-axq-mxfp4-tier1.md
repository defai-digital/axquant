# Qwen3-Coder-Next AXQ-MXFP4 — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on `df-macstudio-m2`.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4)
commit
[`5ccdfb385b05cba3d8610c771185afd203fe174c`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4/tree/5ccdfb385b05cba3d8610c771185afd203fe174c).

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4) |
| Source | `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| Host | `df-macstudio-m2` |
| Product class | `MXFP4` |
| Architecture | `Qwen3NextForCausalLM` (hybrid MoE, no MTP) |
| Size vs reference | `1.014693` (≤ 1.2) |
| Agent-coding | `1.0` |
| General | `1.0` |
| MTP acceleration | `not-applicable` |

## Notes

- Trunk attention and fused expert/MLP tensors are native MXFP4 (group 32).
- Embeddings and routers stay 8-bit affine; norms/lm_head are BF16.
- Adapter `qwen3-next-v1`.
- Quality is measured against a matched uniform quantized reference (not BF16).
- MTP acceleration is **not** certified on this record.

## Related

- Sibling 4-bit: [qwen3-coder-next-axq4-tier1.md](qwen3-coder-next-axq4-tier1.md)
- Sibling 6-bit: [qwen3-coder-next-axq6-tier1.md](qwen3-coder-next-axq6-tier1.md)

Machine-readable: [qwen3-coder-next-axq-mxfp4-tier1.json](qwen3-coder-next-axq-mxfp4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `not-applicable` | `false` | vision not supported on this pack |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

