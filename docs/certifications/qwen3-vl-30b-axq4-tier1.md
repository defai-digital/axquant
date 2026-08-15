# Qwen3-VL 30B-A3B Instruct AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-11.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit)
commit
[`ffcad97ec6102d08e3556fc292b26616813fdc81`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit/tree/ffcad97ec6102d08e3556fc292b26616813fdc81).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3VLMoeForConditionalGeneration` (vision MoE Instruct; no MTP) |
| Product class | `4bit` |
| Source | `Qwen/Qwen3-VL-30B-A3B-Instruct@9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Candidate manifest SHA-256 | `114298ada6770cde054f92f2e6b1d7834832418d4f045512b25384860241b3b1` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `4.860054505934367` |
| Certification host | `df-macbookpro-m5` |
| Primary runtime | AX Engine `6.15.0` |
| Compatible runtime | MLX-VLM (vision smoke) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `4.8` | `4.860054505934367` | Pass |
| Weight-size ratio vs uniform | ≤ `1.15` | `1.034165` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| AX Engine runtime | generate-manifest + doctor | Pass | Pass |
| MLX-VLM runtime | vision generation smoke | Pass | Pass |

Candidate weight bytes `18,875,694,767` vs uniform reference
`18,252,103,673` (`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`) → **1.0342×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.894737` | `0.894737` | `1.000000` | `0.951154` |
| General (44) | `0.943182` | `0.943182` | `1.000000` | `0.934715` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.2`, AX Engine `6.15.0`.
Quality is measured against the matched uniform quantized reference (not BF16).

## Tier 2 status

**Not applicable.** Qwen3-VL-30B-A3B-Instruct has no declared MTP weights; this
certificate is non-MTP checkpoint Tier 1 only (AX Engine primary decode + MLX-VLM
vision compatibility). No speculative-decode speedup claim is authorized.

## Related

- Sibling: see certifications index for the matching AXQ pack
- Certification index: [README.md](README.md)

Machine-readable: [qwen3-vl-30b-axq4-tier1.json](qwen3-vl-30b-axq4-tier1.json).

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
| Vision | `smoke-certified` | `true` | vision runtime smoke passed on df-macstudio-m2 (mlx-vlm); quality suite not certified. Evidence: /Users/akiralam/code/axquant/docs/certifications/evidence/modality-recert-macstudio-m2/results/qwen3-vl-30b-axq4.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
