# GPT-OSS 120B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** **certified** for AXQuant checkpoint Tier 1 on 2026-08-11.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit)
commit
[`50537a80011ed2a2d51ffe43fe9b14b864d4d7c1`](https://huggingface.co/AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit/tree/50537a80011ed2a2d51ffe43fe9b14b864d4d7c1).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `6bit` (manual higher-fidelity layout; measured ~6.58 BPW) |
| Source (convert input) | `mlx-community/gpt-oss-120b-MXFP4-Q4@bce781bef0f2fc85ed4e575af74054f5aad73ddd` |
| Upstream lineage | OpenAI `gpt-oss-120b` |
| Candidate manifest SHA-256 | `0d821fa1ddc002bad95ea29bbd1c0f0209b69ee1503cffa8fd968bff0e2d1de4` |
| Plan | `plan-manual` agent-coding recipe (no 4-bit trunk; attention 8-bit; experts 6-bit) |
| Measured total BPW | `6.576879801034742` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | recipe target `6.6` | `6.576880` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.55` | `1.540933` | Pass |
| General quality retention | ≥ `0.98` | `1.000138` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.004000` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `96,046,415,086` vs MXFP4-Q4 uniform reference
`62,330,057,589` → **1.541×** (within the 6-bit ≤1.55 gate).

### Quality suites

| Profile | Reference | Candidate | Retention |
| --- | ---: | ---: | ---: |
| Agent-coding (76) | `0.548246` | `0.550439` | `1.004000` |
| General (44) | `0.774476` | `0.774583` | `1.000138` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.
Quality is measured against the matched `mlx-community` MXFP4-Q4 pack (not BF16).

## Plan evidence notes

The first `architecture_prior` 6.0 BPW re-pack failed agent-coding retention (`0.956`).
This certified pack uses the backup manual recipe
[`examples/gpt-oss-120b-axq6-agent-v0.1.yaml`](../../examples/gpt-oss-120b-axq6-agent-v0.1.yaml):

- experts / MLP at **6-bit only** (no 4-bit trunk)
- attention raised to **8-bit**
- target BPW **6.6** (storage-adjusted measured ~**6.58**)

Conversion used `AXQUANT_FORCE_CPU=1` after Metal GPU timeouts on the large re-pack.
No measured sensitivity campaign was required for this Tier 1 close.

## Tier 2 status

**Not applicable.** GPT-OSS has no declared MTP weights; this certificate is
non-MTP direct-decode checkpoint Tier 1 only. No speculative-decode speedup
claim is authorized. AX Engine work is not in scope for this certificate.

## Related

- Sibling 4-bit: [gpt-oss-120b-axq4-tier1.md](gpt-oss-120b-axq4-tier1.md) (**not certified** — agent-coding quality; not listed)
- Sibling 20B 6-bit: [gpt-oss-20b-axq6-tier1.md](gpt-oss-20b-axq6-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-120b-axq6-tier1.json](gpt-oss-120b-axq6-tier1.json).
