# Gemma 4 31B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified** (present ≠ certified speed).

This certificate covers
[`AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP)
commit
[`f024707acb0d123642a1ed21c75fc2b7b337bd1e`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP/tree/f024707acb0d123642a1ed21c75fc2b7b337bd1e).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `6bit` |
| Source | `google/gemma-4-31B-it` |
| Candidate manifest SHA-256 | `a0baa8cac423696b946c28777407b90b714aeddd0b85d518d6707f900137a476` |
| Measured total BPW | `6.000026926153772` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`); target digests match quality-bound pack |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.1` | `0.899084` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.992248` | Pass |
| MLX-LM / AX Engine runtime | load + smoke | Pass | Pass |

Candidate weight bytes `23,454,921,915` vs uniform reference
`26,087,568,785` → **0.8991×**
(`mlx-community/gemma-4-31b-it-6bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `1.296861` |
| Agent-coding | `0.848684` | `0.842105` | `0.992248` | `1.146214` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Tier 2 status

**Not certified** on the released engine.

Formal pilot A/B on `df-macbookpro-m5` with AX Engine `6.14.0`,
`--gemma4-assistant-exact-profile`, and the complete Gemma exact confidence
gates (`AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE=0.0001`,
`AX_MLX_GEMMA4_ASSISTANT_MTP_DEEP_DRAFT_MIN_CONFIDENCE=0.0001`) shows:

| Config | Exactness | Weighted speedup | Prompt-median | `release_ready` |
| --- | --- | ---: | ---: | --- |
| depth 2, conf 0.0001 | **Fail** | ~1.38× (clears ≥1.20) | ~1.13× (clears ≥1.10) | false |
| depth 1, conf 0.0001 | **Fail** | ~1.32× | ~1.11× | false |
| depth 2, prod 0.85/0.999 | **Fail** | ~1.24× | ~1.07× | false |

When MTP is active and accepts draft tokens, **greedy outputs diverge** from the
direct arm. Exactness is fail-closed: speed alone cannot authorize Tier 2.
Earlier incomplete-env smokes also failed (MTP often inactive / slower).

Assistant-MTP remains in the pack for product completeness; product default
stays direct fallback until a released engine path holds exactness and both
speed gates on formal authorizing profiles.


## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- Vision/multimodal quality **not** claimed.
- Short-answer / universal prompt acceleration **not** claimed.

Machine-readable: [gemma4-31b-axq6-tier1.json](gemma4-31b-axq6-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision present sidecar=['vision.safetensors']; mlx-vlm smoke failed on df-macstudio-m2 (mlx-vlm expects vision_tower.*; sidecar/layout mismatch). Text Tier 1 unchanged. Evidence: docs/certifications/evidence/modality-recert-capability-gated/results/AX-gemma-4-31b-MLX-AXQ-6bit-MTP.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
