# Qwen3-VL 30B-A3B Instruct AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-11.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit)
commit
[`71f90ad5aa72911c4829944e8e26b8a296af3996`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit/tree/71f90ad5aa72911c4829944e8e26b8a296af3996).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3VLMoeForConditionalGeneration` (vision MoE Instruct; no MTP) |
| Product class | `6bit` |
| Source | `Qwen/Qwen3-VL-30B-A3B-Instruct@9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Candidate manifest SHA-256 | `7f24438444b0c2dcc07310f7946bdcece00d2f1286e6ca5e4cd9ea0643d612fa` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `6.0000538003036` |
| Certification host | `df-macbookpro-m5` |
| Primary runtime | AX Engine `6.15.0` |
| Compatible runtime | MLX-VLM (vision smoke) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `6.0` | `6.0000538003036` | Pass |
| Weight-size ratio vs uniform | ≤ `1.15` | `0.900259` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| AX Engine runtime | generate-manifest + doctor | Pass | Pass |
| MLX-VLM runtime | vision generation smoke | Pass | Pass |

Candidate weight bytes `23,303,274,476` vs uniform reference
`25,885,081,647` (`mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit`) → **0.9003×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.894737` | `0.894737` | `1.000000` | `1.163503` |
| General (44) | `0.943182` | `0.943182` | `1.000000` | `1.120794` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.2`, AX Engine `6.15.0`.
Quality is measured against the matched uniform quantized reference (not BF16).

## Tier 2 status

**Not applicable.** Qwen3-VL-30B-A3B-Instruct has no declared MTP weights; this
certificate is non-MTP checkpoint Tier 1 only (AX Engine primary decode + MLX-VLM
vision compatibility). No speculative-decode speedup claim is authorized.

## Related

- Sibling: see certifications index for the matching AXQ pack
- Certification index: [README.md](README.md)

Machine-readable: [qwen3-vl-30b-axq6-tier1.json](qwen3-vl-30b-axq6-tier1.json).

## Multimodal modalities (1.8.1)

| Modality | Status | Notes |
| --- | --- | --- |
| Vision | `smoke-certified` | MLX-VLM generation smoke re-validated on `df-macstudio-m2` |
| Audio | `not-applicable` | Not supported on this pack |

Vision quality retention is **not** certified. Evidence: [modality-recert-macstudio-m2](evidence/modality-recert-macstudio-m2/).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `smoke-certified` | `true` | vision runtime smoke passed on df-macstudio-m2 (mlx-vlm); quality suite not certified. Evidence: docs/certifications/evidence/modality-recert-macstudio-m2/results/qwen3-vl-30b-axq6.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
