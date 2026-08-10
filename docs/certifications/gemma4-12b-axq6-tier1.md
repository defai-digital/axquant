# Gemma 4 12B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified**.

**Certification track:** Gemma 4 public certification is **AXQ 6-bit only**. Further
Tier 2 work for this family targets this pack class, not 4-bit.

This certificate covers
[`AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP)
commit
[`d0a1a932a177c24acf87204ce46886723673f934`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP/tree/d0a1a932a177c24acf87204ce46886723673f934)
rebuilt from **`google/gemma-4-12b-it`** (prior non-IT `google/gemma-4-12b` packs
failed quality with multimodal placeholder generation).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `6bit` |
| Source | `google/gemma-4-12b-it` |
| Candidate manifest SHA-256 | `dd328239bd1eebf5643642849b251d7ff8f4a33aa25815e9a29313d25c04a85d` |
| Measured total BPW | `6.000087809338533` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.1` | `0.756829` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.992126` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `8,969,928,940` vs uniform reference
`11,851,987,609` → **0.7568×** (`mlx-community/gemma-4-12B-it-6bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `1.271654` |
| Agent-coding | `0.835526` | `0.828947` | `0.992126` | `1.105415` |

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


Machine-readable: [gemma4-12b-axq6-tier1.json](gemma4-12b-axq6-tier1.json).
