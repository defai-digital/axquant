# GPT-OSS 20B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** **certified** for AXQuant checkpoint Tier 1 on 2026-08-11.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit)
commit
[`0c1806bf56eeb2fb8a55ea3d47fcae63705b9828`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit/tree/0c1806bf56eeb2fb8a55ea3d47fcae63705b9828).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `4bit` (manual recovery layout; measured ~5.055 BPW) |
| Source (convert input) | `mlx-community/gpt-oss-20b-MXFP4-Q4@f356f2747216d7e98fee755df25987459fc19089` |
| Upstream lineage | OpenAI `gpt-oss-20b` |
| Candidate manifest SHA-256 | `17cbf5252799486af3b64061018df212633a29a1804019ebe837c11d9e6548d0` |
| Plan | `plan-manual` recipe (attention 8-bit; experts/MLP 4-bit) |
| Measured total BPW | `5.055320757387761` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | recipe target `5.2` | `5.055321` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.2` | `1.182294` | Pass |
| General quality retention | ≥ `0.98` | `1.073797` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.088050` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `13,216,350,766` vs MXFP4-Q4 reference `11,178,569,159` → **1.182×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.348684` | `0.379386` | `1.088050` | `0.978972` |
| General (44) | `0.615455` | `0.660873` | `1.073797` | `0.947374` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.2`.
Quality is measured against the matched `mlx-community` MXFP4-Q4 pack (not BF16).

## Plan evidence notes

The first `architecture_prior` 4-bit re-pack failed **general** retention (`0.893`).
This certified pack uses
[`examples/gpt-oss-20b-axq4-agent-v0.1.yaml`](../../examples/gpt-oss-20b-axq4-agent-v0.1.yaml):

- experts / MLP at **4-bit**
- attention raised to **8-bit**
- target BPW **5.2** (measured ~**5.055**)

## Tier 2 status

**Not applicable.** GPT-OSS has no declared MTP weights.

## Related

- Sibling 6-bit (certified): [gpt-oss-20b-axq6-tier1.md](gpt-oss-20b-axq6-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-20b-axq4-tier1.json](gpt-oss-20b-axq4-tier1.json).
