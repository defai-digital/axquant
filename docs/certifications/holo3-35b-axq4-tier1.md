# Holo3-35B-A3B AXQ 4-bit — checkpoint Tier 1 evaluation

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on 2026-08-14.
**MTP acceleration Tier 2 is not applicable** (no MTP weights).

This record covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit)
commit
[`41c052c2906cee88ec2d282fbfe179272add3047`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit/tree/41c052c2906cee88ec2d282fbfe179272add3047).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (35B-A3B MoE) |
| Product class | `4bit` |
| Source | `Hcompany/Holo3-35B-A3B@208d5ae3a03f99d561f32ab5e606f73397a390ea` |
| Candidate manifest SHA-256 | `bae5f5659ccfbb2e33f41c7ab2af24ecc3c521ff78892087696cd92efddc8981` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `5.605047665481184` (includes BF16 vision) |
| Evaluation host | `df-macstudio-m2` |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |

## Evaluation results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | product class `4bit` | `5.605048` | Pass (class budget; vision floors) |
| Weight-size ratio vs uniform-4 | ≤ `1.15` | `1.134684` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.979310` | **Fail** |
| AX Engine 6.15.0 runtime | load + chat smoke | Pass | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `24,597,178,519` vs uniform-4 reference `21,677,556,069` → **1.1347×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.953947` | `0.934211` | `0.979310` | `0.976004` |
| General (44) | `0.931818` | `0.931818` | `1.000000` | `0.971457` |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.2`.

## Recovery note

A recovery convert with `--target-bpw 5.2` reached agent-coding retention
**0.9862** but measured total BPW **~6.17** (collapsed out of the 4-bit product
class). That artifact was **not** published as the 4-bit Hub SKU.

## Tier 2 status

**Not applicable.** No MTP weights.

## Related

- Certified sibling 6-bit: [holo3-35b-axq6-tier1.md](holo3-35b-axq6-tier1.md)
- Development runbook: [../holo3-35b-axq-dev-runbook.md](../holo3-35b-axq-dev-runbook.md)

Machine-readable: [holo3-35b-axq4-tier1.json](holo3-35b-axq4-tier1.json).
