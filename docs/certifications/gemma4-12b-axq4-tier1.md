# Gemma 4 12B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified**.

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

**Not certified.** Assistant-MTP present only.

Machine-readable: [gemma4-12b-axq4-tier1.json](gemma4-12b-axq4-tier1.json).
