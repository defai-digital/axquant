# Qwen3.8-27B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit/tree/edfedb5c1976ffd796ebcecdbff5d1aba3b50f5b) |
| Source | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host | `df-macbookpro-m3` |
| Size ratio vs uniform-6 | `0.938756` (≤ 1.15) |
| Agent-coding retention vs BF16 | `0.992754` (≥ 0.98) |
| General retention vs BF16 | `1.000000` |
| MTP acceleration | `not-applicable` |

## Gates

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform-6 | ≤ `1.15` | `0.938756` | Pass |
| Agent-coding retention | ≥ `0.98` | `0.992754` | Pass |
| General retention | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | pass | pass | Pass |
| AX Engine doctor | ready | ready | Pass |

## Notes

- Adapter `qwen38-dense-v1` (dense hybrid VLM `model_type=qwen3_5`); not Qwen 3.6 flagship track.
- Quality measured against same-pin BF16; size against local `mlx_lm.convert` uniform-6 from the same BF16 pin.
- Vision remains BF16-protected; no VLM quality claim.
- MTP acceleration is **not** certified on this record.

Machine-readable: [qwen38-27b-axq6-tier1.json](qwen38-27b-axq6-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision present sidecar=['vision.safetensors'] keys=['model.visual']; mlx-vlm smoke failed on df-macbookpro-m3 (Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "site-packages/mlx_vlm/generate/__main__.py). Text Tier 1 unchanged. Evidence: docs/certifications/evidence/modality-recert-capability-gated/results/qwen38-27b-axq6-tier1.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
