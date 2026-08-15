# Qwen3.8-27B AXQ-MXFP4 — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on `df-macstudio-m2`.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4/tree/106526955b0eb3278573bcfae10f29bb74268bbd) |
| Source | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host | `df-macstudio-m2` |
| Product class | `MXFP4` (attention+MLP trunk; embed/lm_head 8-bit affine) |
| Size vs uniform-MXFP4 | `1.158989` (≤ 1.2) |
| Agent-coding vs BF16 | `1.000000` |
| General vs BF16 | `1.000000` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Size vs uniform-MXFP4 | ≤ `1.2` | `1.158989` | Pass |
| Agent-coding | ≥ `0.98` | `1.000000` | Pass |
| General | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Trunk attention and MLP are native MXFP4 (group 32).
- Embeddings and lm_head stay 8-bit affine; vision/norms are BF16.
- Adapter `qwen38-dense-v1`; not the Qwen 3.6 flagship track.
- Vision remains BF16-protected; no VLM quality claim.
- MTP acceleration is **not** certified on this record.

Machine-readable: [qwen38-27b-axq-mxfp4-tier1.json](qwen38-27b-axq-mxfp4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

