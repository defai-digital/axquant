# Holo3-35B-A3B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14.
**MTP acceleration Tier 2 is not applicable** (no MTP weights).

This certificate covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit)
commit
[`e6cc340b04bfcec57544e462ec756e48dd248cf9`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit/tree/e6cc340b04bfcec57544e462ec756e48dd248cf9).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (35B-A3B MoE, GUI-agent VLM fine-tune) |
| Product class | `6bit` |
| Source | `Hcompany/Holo3-35B-A3B@208d5ae3a03f99d561f32ab5e606f73397a390ea` |
| Candidate manifest SHA-256 | `85e8abdb85e3c292b415e26ecfc351fe91242ebece7db81ba0567b61766c908f` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `7.006492923995311` (includes BF16 vision) |
| Certification host | `df-macstudio-m2` |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | product class `6bit` | `7.006493` | Pass (class budget; vision floors) |
| Weight-size ratio vs uniform-6 | ≤ `1.15` | `1.013519` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.007042` | Pass |
| AX Engine 6.15.0 runtime | load + chat smoke | Pass | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `30,747,277,727` vs uniform-6 reference `30,337,155,264` → **1.0135×**.
Uniform reference built with `mlx_lm convert -q --q-bits 6 --q-group-size 64` from the same BF16 pin.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.934211` | `0.940789` | `1.007042` | `1.011342` |
| General (44) | `0.977273` | `0.977273` | `1.000000` | `1.003764` |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.2`.
Quality is measured against the matched uniform quantized reference (not BF16).

## Tier 2 status

**Not applicable.** No MTP weights are present; this certificate is non-MTP
direct-decode checkpoint Tier 1 only. No speculative-decode speedup claim is
authorized.

## Product boundaries

- **Not** the official Qwen 3.6 35B-A3B certificate family (different source + adapter).
- Vision remains BF16-protected; GUI / VLM quality is **not** claimed.
- AX Engine serves the language path (product id `holo3-35b` after engine catalog patch).

## Related

- Sibling 4-bit evaluation record: [holo3-35b-axq4-tier1.md](holo3-35b-axq4-tier1.md)
- Development runbook: [../holo3-35b-axq-dev-runbook.md](../holo3-35b-axq-dev-runbook.md)
- Certification index: [README.md](README.md)

Machine-readable: [holo3-35b-axq6-tier1.json](holo3-35b-axq6-tier1.json).
