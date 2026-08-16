# Qwen3-VL-32B-Thinking MLX AXQ MXFP4 — checkpoint Tier 1

**Verdict:** **not certified** on `df-macstudio-m2`.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-MXFP4) |
| Source | `Qwen/Qwen3-VL-32B-Thinking@7edd10ffd1196091948fb245ff63e406ccb2d4d1` |
| Product class | `MXFP4` |
| Quality backend | mlx-vlm (`qwen3_vl`) vs same-pin BF16 |
| MTP | not-applicable |

Machine-readable: [qwen3-vl-32b-thinking-axq-mxfp4-tier1.json](qwen3-vl-32b-thinking-axq-mxfp4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision tower BF16-protected; VL quality not certified |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

