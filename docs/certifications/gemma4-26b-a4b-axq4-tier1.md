# Gemma 4 26B-A4B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified** (present ≠ certified speed).

This certificate covers
[`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP)
commit
[`85b0a78a14843a818d403f9a2525efa2f081c2a4`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP/tree/85b0a78a14843a818d403f9a2525efa2f081c2a4).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `4bit` |
| Source | `google/gemma-4-26B-A4B-it` |
| Candidate manifest SHA-256 | `2bc1a334bf43f7509eea171a36bcbebe5457ac48211657f585b138e306f7b231` |
| Measured total BPW | `4.900118671944353` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`); target digests match quality-bound pack |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.15` | `1.012661` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.007874` | Pass |
| MLX-LM / AX Engine runtime | load + smoke | Pass | Pass |

Candidate weight bytes `15,806,466,986` vs uniform reference
`15,608,838,574` → **1.0127×**
(`mlx-community/gemma-4-26b-a4b-it-4bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `1.310694` |
| Agent-coding | `0.835526` | `0.842105` | `1.007874` | `1.330682` |

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

Machine-readable: [gemma4-26b-a4b-axq4-tier1.json](gemma4-26b-a4b-axq4-tier1.json).
