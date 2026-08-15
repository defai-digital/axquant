# Qwen 3.6 35B-A3B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14 for the published Hub revision.

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit/tree/2a5bc33c0a142719024550efd02eddf004d630a0) |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Host | `df-macbookpro-m5` |
| Product class | `4bit` (no MTP) |
| Related MTP certificate | [qwen36-35b-axq4-tier1](qwen36-35b-axq4-tier1.md) |
| MTP acceleration | `not-applicable` |

## Gates

This pack is the **no-MTP sibling** of the certified MTP artifact: same language/vision
weights with `mtp.safetensors` removed. Checkpoint quality retention follows the MTP
sibling language-path evidence; only the MTP sidecar is absent.

| Gate | Result |
| --- | --- |
| Weight size vs matched uniform | Pass (smaller than MTP sibling by omitting MTP) |
| Dual-suite quality (language path) | Pass (inherits MTP sibling language-path suite) |
| MLX-LM / AX Engine load path | Pass |

## Notes

- Prefer this pack when you do not need speculative MTP weights.
- For scoped MTP acceleration, use the MTP sibling certificate instead.

Machine-readable: [qwen36-35b-axq4-nomtp-tier1.json](qwen36-35b-axq4-nomtp-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision present sidecar=['vision.safetensors'] keys=['model.visual']; mlx-vlm smoke failed on df-macbookpro-m3 (Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/akiralam/code/axquant/.venv/lib/python3.12/site-packages/mlx_vlm/generate/__main__.py). Text Tier 1 unchanged. Evidence: /Users/akiralam/code/axquant/docs/certifications/evidence/modality-recert-capability-gated/results/qwen36-35b-axq4-nomtp-tier1.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
