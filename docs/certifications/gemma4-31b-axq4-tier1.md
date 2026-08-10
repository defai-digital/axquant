# Gemma 4 31B AXQ 4-bit — historical checkpoint record (not active)

> **Withdrawn from the public certification index (2026-08-10).**
> Gemma 4 public certification is **AXQ 6-bit only**. Do not cite this file as an active
> certificate. Use [gemma4-31b-axq6-tier1.md](gemma4-31b-axq6-tier1.md). This record is
> retained only as an audit trail of a prior 2026-08-09 measurement.

**Historical verdict (2026-08-09):** met checkpoint Tier 1 gates for the fused
assistant-MTP Hub revision on `df-macbookpro-m5`. **Not an active certificate.**
**MTP acceleration Tier 2 was never certified** for this pack.

This certificate covers
[`AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP)
commit
[`bc2de70bf2bc6b03da1d50801a4f95894d32eec4`](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP/tree/bc2de70bf2bc6b03da1d50801a4f95894d32eec4).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `4bit` |
| Source | `google/gemma-4-31B-it` |
| Candidate manifest SHA-256 | `989f7c7e0ce9a7550e239b4eadb2167f6ca495522d200081b4899e64f04d00f7` |
| Measured total BPW | `4.899922859286156` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`); target digests match quality-bound pack |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.15` | `1.040324` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.992366` | Pass |
| MLX-LM / AX Engine runtime | load + smoke | Pass | Pass |

Candidate weight bytes `19,154,465,383` vs uniform reference
`18,412,016,676` → **1.0403×**
(`mlx-community/gemma-4-31b-it-4bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `0.766606` |
| Agent-coding | `0.861842` | `0.855263` | `0.992366` | `0.796989` |

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

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
