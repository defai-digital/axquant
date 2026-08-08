# Qwen 3.6 35B-A3B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-08. **MTP
acceleration is not certified** on this revision (see Tier 2 status).

This certificate covers
[`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP)
commit
[`7b9ff47abfb8be01e636f516edb0226aa25ea1cc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP/tree/7b9ff47abfb8be01e636f516edb0226aa25ea1cc).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (MoE A3B) |
| Product class | `6bit` |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Candidate manifest SHA-256 | `1279fac7d32a267bfd8b106488ed0af5c5ecedf5f1fae23a9be5b408d3ac2035` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `6.000061` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `6.0` | `6.000061` | Pass |
| Weight-size ratio vs uniform-6 | ≤ `1.10` | `0.927821` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| Scorer errors | `0 / 0` | `0 / 0` | Pass |

Candidate weight bytes `26,964,142,055` vs uniform-6 reference
`29,061,803,049` → about **7.2% smaller** than the matched uniform-6 pack.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General (44) | `1.000` | `1.000` | `1.000` | `0.999802` |
| Agent-coding (76) | `0.888158` | `0.888158` | `1.000` | `1.003816` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Tier 2 status

**Not certified.** MoE MTP sidecar packing is loadable after the AX Engine
`experts.gate_up_proj` fix, and greedy exactness passes, but formal decode-heavy
A/B speedups do not clear ≥1.20× / ≥1.10× gates:

| Profile | Exactness | Weighted speedup | Prompt-median | `release_ready` |
| --- | --- | ---: | ---: | --- |
| agent-coding | Pass | **1.091×** | **0.882×** | false |
| long-form general | Pass | **1.007×** | **1.010×** | false |

Default product route remains direct fallback.

## Related

- Sibling 35B 4-bit: [Tier 1](qwen36-35b-axq4-tier1.md)
- Dense 27B flagship: [6-bit Tier 1](qwen36-27b-axq6-tier1.md)
