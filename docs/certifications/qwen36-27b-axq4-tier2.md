# Qwen 3.6 27B AXQ 4-bit (5.6 BPW) — MTP acceleration Tier 2 certification

**Verdict:** certified for AXQuant **Tier 2 MTP acceleration** on 2026-08-08
(scoped; see claim boundaries below).

This certificate applies to the **same exact weight set** already certified for
[checkpoint Tier 1](qwen36-27b-axq4-tier1.md):

[`AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP)
commit
[`f44a9eeebec0c488d0f42201c8763db770a1c0a8`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP/tree/f44a9eeebec0c488d0f42201c8763db770a1c0a8).

Candidate manifest SHA-256:
`fa58908d73a8f7046d9f8d86cff278326fc97632e71f0d6675cbc3e8b13439b7`.

The companion [machine-readable record](qwen36-27b-axq4-tier2.json) pins
thresholds, digests, and A/B results.

## What Tier 2 certifies

On host **`df-macbookpro-m5`** (Apple M5 Max), with AX Engine **`6.14.0`**
(binary SHA-256
`02c2f0b378499408918f87b476bc0e9aa922c8a6db6a64fc7d1cb106cbac2989`) and AXQuant
**`1.6.1`**, the candidate passes the three independent MTP acceleration gates
on **authorizing decode-heavy workloads**:

| Gate | Requirement | Agent-coding | Long-form general |
| --- | ---: | ---: | ---: |
| Greedy exactness | 100% identical MTP-off/on tokens | Pass (0 divergent / 5 trials) | Pass (0 divergent / 5 trials) |
| Token-weighted decode speedup | ≥ **1.20×** | **1.301310×** | **1.222678×** |
| Prompt-median speedup | ≥ **1.10×** | **1.104172×** | **1.248500×** |
| Comparison `release_ready` | fully bound evidence | **true** | **true** |

Harness: greedy, temperature `0`, seed `20260728`, draft depth `1`, warmup `2`,
measured trials `5`, max tokens `512`, metric `token-weighted-decode-tps`.
Formal env is the Qwen 3.6 exact MTP profile (`--qwen36-exact-profile`),
including `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1` and
`AX_MLX_MTP_LINEAR_EXACT_REPLAY=0` (exact lazy checkpoint path).

| Profile | Dataset SHA-256 | Comparison SHA-256 |
| --- | --- | --- |
| agent-coding | `440eea149a2fe5306a7068684a662b9957fe2fa90f8ce5a1914187ea9b3c7adc` | `93262f4c6c45e3e469af39a2bef8066522f48571f0b2ada74261bfc3dec535fc` |
| long-form general | `c43e076e3f78a9e2fa69047d7e8eb2527988ce214a427e04a95a999f93dd8f07` | `3601628edd0dfda45ef4d88ea90ade3bc7a75f8d3e01201d068f74d29acfb52d` |

## Claim boundaries (read carefully)

| Claim | Status |
| --- | --- |
| Decode-heavy agent-coding MTP speedup + exactness on formal host | **Certified** |
| Long-form general MTP speedup + exactness on formal host | **Certified** |
| Checkpoint size/quality (Tier 1) | Certified with `5p6bpw` size budget |
| **Product default** Qwen linear MTP enabled without opt-in | **Not claimed** — safe default remains direct fallback |
| Short-answer / chat-length generations always faster with MTP | **Not claimed** |
| Vision / multimodal quality | **Not claimed** |
| Other revisions / MoE 35B siblings | **Not claimed** by this certificate |
| Full flagship M0–M8 publication campaign closure | Separate process |

Public language must not say “always 1.30× faster for every prompt.” Correct form:

> On Apple M5 Max (`df-macbookpro-m5`), with AX Engine 6.14.0 under the formal
> Qwen linear MTP exact contract, this 27B AXQ 4-bit (5.6 BPW) checkpoint
> achieves ≥1.20× token-weighted and ≥1.10× prompt-median decode speedup with
> greedy-identical outputs on the authorizing agent-coding and long-form
> general suites.

## Runtime how-to (acceleration route)

```bash
export AX_MLX_QWEN_LINEAR_MTP_EXACT=1
export AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1
export AX_MLX_MTP_LINEAR_EXACT_REPLAY=0
# optional product short-budget skip (default 16); formal cert used 0
export AX_MLX_MTP_MIN_REMAINING_TOKENS=16
```

## Integrity

- Tier 1 remains authoritative for weights, size class, and quality.
- Tier 2 is invalid if weights, thresholds, host, or engine binary diverge from
  the bound digests.
