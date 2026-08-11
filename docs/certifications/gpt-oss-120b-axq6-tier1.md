# GPT-OSS 120B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on 2026-08-11.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This record covers
[`AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit)
commit
[`6b2b4c1be8b00db91000565024bacf861d09546c`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit/tree/6b2b4c1be8b00db91000565024bacf861d09546c).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `6bit` |
| Source (convert input) | `mlx-community/gpt-oss-120b-MXFP4-Q4@bce781bef0f2fc85ed4e575af74054f5aad73ddd` |
| Upstream lineage | OpenAI `gpt-oss-120b` |
| Measured total BPW | `6.000009063970246` |
| Evaluation host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `6.0` | `6.000009` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.55` | `1.405774` | Pass |
| General quality retention | ≥ `0.98` | `1.002438` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.956000` | **Fail** |
| MLX-LM runtime | load + smoke | Pass | Pass |

### Quality suites

| Profile | Retention | Gate |
| --- | ---: | --- |
| Agent-coding | `0.956` | Fail (&lt; 0.98) |
| General | `1.002` | Pass |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Why not certified

Agent-coding retention fell below the 0.98 checkpoint gate under an
`architecture_prior` plan re-packed from native MXFP4. Size, general quality,
and MLX-LM load pass. The Hub pack remains **development evidence** only.

## Tier 2 status

**Not applicable.** No MTP.

## Related

- Sibling 4-bit: [gpt-oss-120b-axq4-tier1.md](gpt-oss-120b-axq4-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-120b-axq6-tier1.json](gpt-oss-120b-axq6-tier1.json).
