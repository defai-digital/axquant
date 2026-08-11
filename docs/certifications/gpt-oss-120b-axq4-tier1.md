# GPT-OSS 120B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on 2026-08-10.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This record covers
[`AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit)
commit
[`7e0f77ed63c0fb83d0fcc57d84b3018f269ec8f3`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit/tree/7e0f77ed63c0fb83d0fcc57d84b3018f269ec8f3).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `4bit` |
| Source (convert input) | `mlx-community/gpt-oss-120b-MXFP4-Q4@bce781bef0f2fc85ed4e575af74054f5aad73ddd` |
| Upstream lineage | OpenAI `gpt-oss-120b` |
| Measured total BPW | `4.800009864578611` |
| Evaluation host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `4.8` | `4.800010` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.20` | `1.124620` | Pass |
| General quality retention | ≥ `0.98` | `1.002438` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.952000` | **Fail** |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `70,097,638,062` vs MXFP4-Q4 reference `62,330,057,589` → **1.125×**.

### Quality suites

| Profile | Retention | Gate |
| --- | ---: | --- |
| Agent-coding | `0.952` | Fail (&lt; 0.98) |
| General | `1.002` | Pass |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Why not certified

Agent-coding retention fell below the 0.98 checkpoint gate under an
`architecture_prior` plan re-packed from native MXFP4. Size, general quality,
and MLX-LM load pass. The Hub pack remains **development evidence** only.

## Tier 2 status

**Not applicable.** No MTP.

## Related

- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-120b-axq4-tier1.json](gpt-oss-120b-axq4-tier1.json).
