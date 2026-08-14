# Holo3-35B-A3B AXQ 4-bit-MTP — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14.
**MTP acceleration Tier 2 is not certified** (grafted parent MTP; exactness/speedup not claimed).

This certificate covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP)
commit
[`c048f577843225ac0545be5674b4d68b9a51dcf0`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP/tree/c048f577843225ac0545be5674b4d68b9a51dcf0).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (35B-A3B MoE) |
| Product class | `4bit` (attention **6-bit** / experts **4-bit** recovery layout) |
| Source trunk | `Hcompany/Holo3-35B-A3B@208d5ae3a03f99d561f32ab5e606f73397a390ea` |
| MTP donor | `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` (grafted; not co-trained) |
| Candidate manifest SHA-256 | `1d0b7e7b29cea505e5cdd5a3a3769937231b247f47859c7ffb1d89c67fbfeb87` |
| Plan | `plan-manual` + [`examples/holo3-35b-axq4-agent-v0.1.yaml`](../../examples/holo3-35b-axq4-agent-v0.1.yaml) + `compose-grafted-mtp` |
| Measured main BPW | `5.665439451180904` (includes BF16 vision; excludes MTP sidecar from size gate) |
| MTP sidecar | `mtp.safetensors` BF16, 19 packed tensors, 1 689 283 610 bytes |
| Certification host | `df-macstudio-m2` |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW (main) | product class `4bit` | `5.665439` | Pass |
| Weight-size ratio vs uniform-4 | ≤ `1.15` | `1.146910` | Pass |
| General quality retention | ≥ `0.98` | `1.048780` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.006897` | Pass |
| AX Engine / MLX-LM runtime | load + eval; MTP present | Pass | Pass |

Candidate **main** weight bytes `24,862,201,695` vs uniform-4 reference `21,677,556,069` → **1.1469×**.
Quality and size evidence are bound to the certified non-MTP 4-bit trunk; this `-MTP` edition attaches the sidecar without mutating main shards.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.953947` | `0.960526` | `1.006897` | `0.922347` |
| General (44) | `0.931818` | `0.977273` | `1.048780` | `0.951576` |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.2`.

### Graft honesty

Holo3 BF16 declares config-only MTP and ships **no** `mtp.*` tensors. The published
sidecar is **extracted and packed** from the fine-tune parent
`Qwen/Qwen3.5-35B-A3B` (immutable pin above) into the 19-tensor layout used by
Qwen3.6 35B AXQ MTP packs. See pack file `axquant_mtp_graft.json`.

## Tier 2 status

**Not certified.** MTP assets are included for experimentation and runtime attach,
but speculative-decode exactness and speedup are **not** product claims. See
[holo3-35b-axq4-mtp-tier2.md](holo3-35b-axq4-mtp-tier2.md).

## Related

- Non-MTP sibling: [holo3-35b-axq4-tier1.md](holo3-35b-axq4-tier1.md)
- 6-bit-MTP: [holo3-35b-axq6-mtp-tier1.md](holo3-35b-axq6-mtp-tier1.md)
- Runbook: [../holo3-35b-axq-dev-runbook.md](../holo3-35b-axq-dev-runbook.md)

Machine-readable: [holo3-35b-axq4-mtp-tier1.json](holo3-35b-axq4-mtp-tier1.json).
