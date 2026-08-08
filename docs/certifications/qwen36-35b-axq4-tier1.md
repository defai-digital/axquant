# Qwen 3.6 35B-A3B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-08 (quality +
class-adjusted size budget). **MTP acceleration is not certified** on this
revision (see Tier 2 status).

This certificate covers
[`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP)
commit
[`a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP/tree/a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (MoE A3B) |
| Product class | `4bit` mixed AXQ (measured total BPW ≈ `5.14`) |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Candidate manifest SHA-256 | `d172e6c1dd88e24e1f451735606ffb3d9426b30c2e179bfd8eae771ba7141155` |
| Plan evidence | `architecture_prior` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `5.14` | `5.140061` | Pass |
| Weight-size ratio vs uniform-4 | ≤ `1.15` (class budget) | `1.132197` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| Scorer errors | `0 / 0` | `0 / 0` | Pass |

### Size note

Candidate weight bytes `23,099,321,635` vs uniform-4
`mlx-community/Qwen3.6-35B-A3B-4bit` `20,402,204,271` → **1.132×**. The pure-4bit
≤1.10 gate is not claimed; formal scoreboard used `max_size_ratio=1.15` for this
mixed-AXQ class.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General (44) | `1.000` | `1.000` | `1.000` | `0.984415` |
| Agent-coding (76) | `0.888158` | `0.888158` | `1.000` | `0.982619` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Tier 2 status

**Not certified.** After an engine fix that loads MoE MTP
`experts.gate_up_proj` packing, formal A/B on M5 achieves greedy exactness with
MTP active, but **token-weighted and prompt-median speedups fail** the ≥1.20× /
≥1.10× gates:

| Profile | Exactness | Weighted speedup | Prompt-median | `release_ready` |
| --- | --- | ---: | ---: | --- |
| agent-coding | Pass | **0.946×** | **0.803×** | false |
| long-form general | Pass | **0.911×** | **0.928×** | false |

Product default remains direct fallback. Acceleration claims for this MoE pack
require further engine/product work; exactness-only is development evidence, not
Tier 2.

## Related

- Dense 27B siblings: [6-bit](qwen36-27b-axq6-tier1.md), [4-bit](qwen36-27b-axq4-tier1.md)
- Sibling 35B 6-bit: [Tier 1](qwen36-35b-axq6-tier1.md)
