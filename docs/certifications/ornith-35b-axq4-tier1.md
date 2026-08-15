# Ornith-1.0-35B AXQ 4bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on `df-macstudio-m2`.
**MTP acceleration Tier 2 is not applicable** (no MTP weights).

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit/tree/d7416c665cd8ae6e5fbebc3f17bd547b78cf11fc) |
| Source | `deepreinforce-ai/Ornith-1.0-35B@5df2ed3f675c7beaa490328cc70bb573b65fb660` |
| Host | `df-macstudio-m2` |
| Product class | `4bit` (architecture-prior) |
| Size vs uniform-4 | `1.097731` (≤ 1.15) |
| Agent-coding vs uniform-4bit | `0.985507` |
| General vs uniform-4bit | `1.000000` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Size vs uniform-4bit | ≤ `1.15` | `1.097731` | Pass |
| Agent-coding | ≥ `0.98` | `0.985507` | Pass |
| General | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Adapter `qwen35-moe-v1`; not the official Qwen 3.6 certification track.
- Vision remains BF16-protected; no VLM quality claim.
- Config-only MTP (if present) is cleared; Hub names omit `-MTP`.
- 4bit pack is the published architecture-prior convert.

## Tier 2 status

Not applicable. No MTP weights are packaged.

Machine-readable: [ornith-35b-axq4-tier1.json](ornith-35b-axq4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

