# Holo-3.1-35B-A3B MLX AXQ 8-bit — checkpoint Tier 1

**Verdict:** **not certified** on `df-macstudio-m2`.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit) |
| Source | `Hcompany/Holo-3.1-35B-A3B@2bdb92851a8cd9d72cdd891fdf38cfcc7fefae2c` |
| Product class | `8bit` |
| Quality backend | mlx-lm vs same-pin BF16 |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |
| MTP | not-applicable |

Machine-readable: [holo31-35b-axq8-tier1.json](holo31-35b-axq8-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

