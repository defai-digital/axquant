# Gemma 4 26B-A4B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-09 for the
**fused assistant-MTP Hub revision** on `df-macbookpro-m5`. **MTP acceleration
Tier 2 is not certified** (present ≠ certified speed).

This certificate covers
[`AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP)
commit
[`4a62bf66aa74f2063bbbf3ad0ff2cbdc72dc5bcb`](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP/tree/4a62bf66aa74f2063bbbf3ad0ff2cbdc72dc5bcb).

## Bound artifact

| Property | Value |
| --- | --- |
| Product class | `6bit` |
| Source | `google/gemma-4-26B-A4B-it` |
| Candidate manifest SHA-256 | `7cbb9fe85439a53c3da909962a2322b5ba2ee509e606d050947bb7fa3ddccb53` |
| Measured total BPW | `6.000119253417119` |
| Certification host | `df-macbookpro-m5` |
| Assistant-MTP | fused (`assistant/` + `ax_gemma4_assistant_mtp.json`); target digests match quality-bound pack |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Weight-size ratio vs uniform | ≤ `1.1` | `0.888598` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM / AX Engine runtime | load + smoke | Pass | Pass |

Candidate weight bytes `19,354,773,474` vs uniform reference
`21,781,239,781` → **0.8886×**
(`mlx-community/gemma-4-26b-a4b-it-6bit`).

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General | `1.000000` | `1.000000` | `1.000000` | `1.106344` |
| Agent-coding | `0.835526` | `0.835526` | `1.000000` | `1.275455` |

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

Machine-readable: [gemma4-26b-a4b-axq6-tier1.json](gemma4-26b-a4b-axq6-tier1.json).
