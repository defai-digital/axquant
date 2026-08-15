# Ornith-1.0-35B AXQ 6bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on `df-macstudio-m2`.
**MTP acceleration Tier 2 is not applicable** (no MTP weights).

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit/tree/37361076641d7b7487d1b5ce1b68243ffbdbffe0) |
| Source | `deepreinforce-ai/Ornith-1.0-35B@5df2ed3f675c7beaa490328cc70bb573b65fb660` |
| Host | `df-macstudio-m2` |
| Product class | `6bit` (architecture-prior) |
| Size vs uniform-6 | `0.934751` (≤ 1.15) |
| Agent-coding vs uniform-6bit | `1.000000` |
| General vs uniform-6bit | `1.011494` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Size vs uniform-6bit | ≤ `1.15` | `0.934751` | Pass |
| Agent-coding | ≥ `0.98` | `1.000000` | Pass |
| General | ≥ `0.98` | `1.011494` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Adapter `qwen35-moe-v1`; not the official Qwen 3.6 certification track.
- Vision remains BF16-protected; no VLM quality claim.
- Config-only MTP (if present) is cleared; Hub names omit `-MTP`.
- 6bit pack is the published architecture-prior convert.

## Tier 2 status

Not applicable. No MTP weights are packaged.

Machine-readable: [ornith-35b-axq6-tier1.json](ornith-35b-axq6-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

