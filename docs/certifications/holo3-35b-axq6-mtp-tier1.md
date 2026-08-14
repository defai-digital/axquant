# Holo3-35B-A3B AXQ 6-bit-MTP — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14.
**MTP acceleration Tier 2 is not certified** (grafted parent MTP; exactness/speedup not claimed).

This certificate covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP)
commit
[`f474549461817cafb73909847af43af2431d4a0d`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP/tree/f474549461817cafb73909847af43af2431d4a0d).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (35B-A3B MoE) |
| Product class | `6bit` |
| Source trunk | `Hcompany/Holo3-35B-A3B@208d5ae3a03f99d561f32ab5e606f73397a390ea` |
| MTP donor | `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` (grafted; not co-trained) |
| Candidate manifest SHA-256 | `c12621c9c3156e1c0965f15a814eda83468bb5c5d0901b9168c9a3cd530a0bfd` |
| Measured main BPW | `7.006492923995311` (includes BF16 vision; excludes MTP sidecar from size gate) |
| MTP sidecar | `mtp.safetensors` BF16, 19 packed tensors, 1 689 283 610 bytes |
| Certification host | `df-macstudio-m2` |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW (main) | product class `6bit` | `7.006493` | Pass |
| Weight-size ratio vs uniform-6 | ≤ `1.15` | `1.013519` | Pass |
| General quality retention | ≥ `0.98` | `1.000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.007042` | Pass |
| AX Engine / MLX-LM runtime | load + eval; MTP present | Pass | Pass |

Candidate **main** weight bytes `30,747,277,727` vs uniform-6 reference `30,337,155,264` → **1.0135×**.
Quality and size evidence are bound to the certified non-MTP 6-bit trunk; this `-MTP` edition attaches the sidecar without mutating main shards.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.934211` | `0.940789` | `1.007042` | `1.011342` |
| General (44) | `0.977273` | `0.977273` | `1.000` | `1.003764` |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.2`.

### Graft honesty

Holo3 BF16 declares config-only MTP and ships **no** `mtp.*` tensors. The published
sidecar is **extracted and packed** from the fine-tune parent
`Qwen/Qwen3.5-35B-A3B` (immutable pin above) into the 19-tensor layout used by
Qwen3.6 35B AXQ MTP packs. See pack file `axquant_mtp_graft.json`.

## Tier 2 status

**Not certified.** Soft probe on this revision saw exactness pass but decode
speedup ~0.43× (fail). Speculative-decode acceleration is **not** a product claim.
See [holo3-35b-axq6-mtp-tier2.md](holo3-35b-axq6-mtp-tier2.md).

## Related

- Non-MTP sibling: [holo3-35b-axq6-tier1.md](holo3-35b-axq6-tier1.md)
- 4-bit-MTP: [holo3-35b-axq4-mtp-tier1.md](holo3-35b-axq4-mtp-tier1.md)
- Runbook: [../holo3-35b-axq-dev-runbook.md](../holo3-35b-axq-dev-runbook.md)

Machine-readable: [holo3-35b-axq6-mtp-tier1.json](holo3-35b-axq6-mtp-tier1.json).
