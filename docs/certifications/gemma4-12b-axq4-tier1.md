# Gemma 4 12B AXQ 4-bit — historical checkpoint record (not active)

> **Withdrawn from the public certification index (2026-08-10).**
> Gemma 4 public certification is **AXQ 6-bit only**. Do not cite this file as an active
> certificate. Use [gemma4-12b-axq6-tier1.md](gemma4-12b-axq6-tier1.md). This record is
> retained only as an audit trail of a prior 2026-08-09 measurement.

**Historical verdict (2026-08-09):** met checkpoint Tier 1 gates for the fused
assistant-MTP Hub revision on `df-macbookpro-m5`. **Not an active certificate.**
**MTP acceleration Tier 2 was never certified** for this pack.

This certificate covers
[`AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP)
commit
[`6d124af8f40f79d3e45ddfbdb50d721e63ab5dc2`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP/tree/6d124af8f40f79d3e45ddfbdb50d721e63ab5dc2)
rebuilt from **`google/gemma-4-12b-it`** (prior non-IT `google/gemma-4-12b` packs
failed quality with multimodal placeholder generation).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `4bit` |
| Source | `google/gemma-4-12b-it` |
| Candidate manifest SHA-256 | `bc3fde05ba2af82c303b3ef13f1082586fb15e0735bccb050e9768c9d19dcc75` |
| Measured total BPW | `4.900060746386949` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.15` | `0.666689` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.032520` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `7,325,425,576` vs uniform reference
`10,987,772,576` → **0.6667×** (`mlx-community/gemma-4-12B-it-4bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `0.994181` |
| Agent-coding | `0.809211` | `0.835526` | `1.032520` | `1.079593` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`.

## Repair notes

1. Previous Hub packs sourced from `google/gemma-4-12b` produced `<image|>` /
   `<audio|>` loops under the formal suite.
2. Rebuild uses `google/gemma-4-12b-it` + `gemma4_unified` text-path prep that
   drops multimodal tensors **and** strips `vision_config` / `audio_config` so
   MLX-LM does not emit empty `vision_embedder` biases during convert.
3. Assistant-MTP is re-composed without mutating target weight digests.

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


Machine-readable: [gemma4-12b-axq4-tier1.json](gemma4-12b-axq4-tier1.json).
