# GPT-OSS 20B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on 2026-08-10.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This record covers
[`AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit)
commit
[`8123294fe643f8213bebbcbfe27376c5c1dd4ad0`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit/tree/8123294fe643f8213bebbcbfe27376c5c1dd4ad0).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `4bit` |
| Source (convert input) | `mlx-community/gpt-oss-20b-MXFP4-Q4@f356f2747216d7e98fee755df25987459fc19089` |
| Upstream lineage | OpenAI `gpt-oss-20b` (`model_type=gpt_oss`) |
| Candidate manifest SHA-256 | `b752ae93facfe4abe4c9590f0537729128b485dcf0a9c965a004a6dd7fc4a40f` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `4.940036776666028` |
| Evaluation host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `4.8` (raised for floors) | `4.940036776666028` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.20` | `1.155332` | Pass |
| General quality retention | ≥ `0.98` | `0.893162` | **Fail** |
| Agent-coding quality retention | ≥ `0.98` | `1.279503` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `12,914,958,708` vs MXFP4-Q4 uniform reference
`11,178,569,159` → **1.155×**.

### Quality suites

| Profile | Reference | Candidate | Retention |
| --- | ---: | ---: | ---: |
| Agent-coding (76) | `0.353070` | `0.451754` | `1.279503` |
| General (44) | `0.638182` | `0.570000` | `0.893162` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.
Quality is measured against the matched `mlx-community` MXFP4-Q4 pack (not BF16).

## Why not certified

General-suite retention fell below the 0.98 checkpoint gate under an
`architecture_prior` mixed-precision plan re-packed from native MXFP4. Size,
agent-coding retention, and MLX-LM load all pass. The Hub pack remains valid
**development evidence** only until a revision clears every Tier 1 gate.

## Tier 2 status

**Not applicable.** No MTP; no acceleration claim.

## Related

- Sibling 6-bit (certified): [gpt-oss-20b-axq6-tier1.md](gpt-oss-20b-axq6-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-20b-axq4-tier1.json](gpt-oss-20b-axq4-tier1.json).
