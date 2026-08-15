# Qwen3.8-27B AXQ 8-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on df-macstudio-m2.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit/tree/36f9d25c4b1ea2282774b9acf84fdad0241a8a54) |
| Source | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host | `df-macstudio-m2` |
| Product class | `8bit` |
| Size vs uniform-8 | `0.957217` (≤ 1.15) |
| Agent-coding vs BF16 | `1.000000` |
| General vs BF16 | `1.000000` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Size vs uniform-8 | ≤ `1.15` | `0.957217` | Pass |
| Agent-coding | ≥ `0.98` | `1.000000` | Pass |
| General | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Adapter `qwen38-dense-v1`; not the Qwen 3.6 flagship track.
- Vision remains BF16-protected; no VLM quality claim.
- MTP acceleration is **not** certified on this record.

Machine-readable: [qwen38-27b-axq8-tier1.json](qwen38-27b-axq8-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

