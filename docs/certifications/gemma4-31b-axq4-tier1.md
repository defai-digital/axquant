# Gemma 4 31B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified** (present ≠ certified speed).

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

**Not certified.** The Hub pack ships Gemma **assistant-MTP** for product
completeness. Formal decode-heavy A/B gates (exactness 100%, weighted speedup
≥1.20×, prompt-median ≥1.10×) are **not** claimed for this revision. Default
product route remains standard direct decode.

## Scope

- Checkpoint size/quality vs matched mlx-community uniform reference (**AXQ target** weights).
- Vision/multimodal quality **not** claimed.
- Short-answer / universal prompt acceleration **not** claimed.

Machine-readable: [gemma4-31b-axq4-tier1.json](gemma4-31b-axq4-tier1.json).
