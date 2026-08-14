# Qwen3.8-27B AXQ 4-bit MTP — MTP acceleration Tier 2 certification

**Verdict:** certified for AXQuant **Tier 2 MTP acceleration** on 2026-08-14
(scoped; see claim boundaries below).

This certificate applies to the **same exact weight set** already certified for
[checkpoint Tier 1](qwen38-27b-axq4-mtp-tier1.md):

[`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP)
commit
[`32f448461caf4aedcc3c16a77a63b6a94bf0667c`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP/tree/32f448461caf4aedcc3c16a77a63b6a94bf0667c).

Candidate manifest SHA-256:
`f92dda812fe79978a719f2f1fbb54502fdf4eb022c744731111389e4d17aa41d`.

The companion [machine-readable record](qwen38-27b-axq4-mtp-tier2.json) pins
thresholds, digests, and A/B results.

## What Tier 2 certifies

On host **`df-macbookpro-m3`** (Apple M3 Max), with AX Engine **`6.16.1`**
(binary SHA-256
`c39b69a18cb8cf41c020dd52c32c6011de52f841212a5237fef9ac1e9f78dec7`) and AXQuant
**`1.6.2`**, the candidate passes the three independent MTP acceleration gates
on **authorizing decode-heavy workloads**:

| Gate | Requirement | Agent-coding | Long-form general |
| --- | ---: | ---: | ---: |
| Greedy exactness | 100% identical MTP-off/on tokens | Pass (0 divergent / 2 trials) | Pass (0 divergent / 2 trials) |
| Token-weighted decode speedup | ≥ **1.20×** | **1.346687×** | **1.294289×** |
| Prompt-median speedup | ≥ **1.10×** | **1.352577×** | **1.301133×** |
| Comparison `release_ready` | fully bound evidence | **true** | **true** |

Harness: greedy, temperature `0`, seed `20260728`, draft depth `1`, warmup `1`,
measured trials `2`, max tokens `64`, `ignore_eos=true`, metric
`token-weighted-decode-tps`. Formal env is the Qwen3.8 dense exact MTP profile
(`QWEN38_EXACT_MTP_PROFILE_ENV`): Qwen linear MTP exact + certification candidate
with **async draft**, verify-submit layers `8`, pipeline granularity `layer`
(same exactness contract as certified Qwen 3.6 dense linear, with async draft
enabled for decode-heavy net speedup on M3 Max).

| Profile | Dataset SHA-256 | Comparison SHA-256 |
| --- | --- | --- |
| agent-coding | `ebecd0157c601287ab64216998e212f1a0935c167e4e53ea1f4661f833edff60` | `b1d2fa7459c344f907e6a5192691293e90d74d7b4c497d53c6ae8ea34ce34105` |
| long-form general | `55af1c786f4c994d1e65135954e6c2a5fadfded0d2446255fab780c0b87738b7` | `cc4b5e5446339949e53e46c98c3c7890a0ef1dc60a13d89bf04d9bc6e563092b` |

Evidence package: [evidence/qwen38-27b-axq4-mtp-tier2/](evidence/qwen38-27b-axq4-mtp-tier2/).

## Claim boundaries (read carefully)

| Claim | Status |
| --- | --- |
| Decode-heavy agent-coding MTP speedup + exactness on formal host | **Certified** |
| Long-form general MTP speedup + exactness on formal host | **Certified** |
| Checkpoint size/quality (Tier 1) | Certified separately for this `4bit` pack |
| **Product default** Qwen linear MTP enabled without opt-in | **Not claimed** — safe default remains direct fallback |
| Short-answer / chat-length generations always faster with MTP | **Not claimed** (cert harness uses fixed token budgets with `ignore_eos`) |
| Vision / multimodal quality | **Not claimed** |
| Other revisions / non-MTP siblings | **Not claimed** by this certificate |

Public language must not say “always 1.35× faster for every prompt.” Correct form:

> On Apple M3 Max (`df-macbookpro-m3`), with AX Engine 6.16.1 under the formal
> Qwen3.8 dense exact MTP contract (async draft + chunked verify-submit), this 27B
> AXQ 4bit MTP checkpoint achieves ≥1.20× token-weighted and ≥1.10× prompt-median
> decode speedup with greedy-identical outputs on the authorizing agent-coding and
> long-form general suites.

## Runtime how-to (acceleration route)

```bash
export AX_MLX_QWEN_LINEAR_MTP_EXACT=1
export AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1
export AX_MLX_MTP_LINEAR_EXACT_REPLAY=0
export AX_MLX_MTP_ASYNC_DRAFT=1
export AX_MLX_MTP_VERIFY_SUBMIT_LAYERS=8
export AX_MLX_PIPELINE_GRANULARITY=layer
# optional product short-budget skip (default 16); formal cert used 0
export AX_MLX_MTP_MIN_REMAINING_TOKENS=0
```

Or use AXQuant's profile constant `QWEN38_EXACT_MTP_PROFILE_ENV` /
`axquant benchmark-ab --qwen38-exact-profile`.

## Integrity

- Tier 1 remains authoritative for weights, size class, and quality.
- Tier 2 is invalid if weights, thresholds, host, or engine binary diverge from
  the bound digests.
