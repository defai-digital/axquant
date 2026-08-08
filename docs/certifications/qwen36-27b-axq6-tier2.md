# Qwen 3.6 27B AXQ 6-bit — MTP acceleration Tier 2 certification

**Verdict:** certified for AXQuant **Tier 2 MTP acceleration** on 2026-08-08
(scoped; see claim boundaries below).

This certificate applies to the **same exact v3 weight set** already certified for
[checkpoint Tier 1](qwen36-27b-axq6-tier1.md):

[`AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP)
tag `v3` / commit
[`cdd13bf81cf21818a01cf59a31fc116ef84326bc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP/tree/cdd13bf81cf21818a01cf59a31fc116ef84326bc).

Candidate manifest SHA-256:
`88269a83d0fd7bf70381745e1163f2b919e09798bcc92e352757d0f00e112c43`.

The companion [machine-readable record](qwen36-27b-axq6-tier2.json) and
[evidence package](evidence/qwen36-27b-axq6-tier2/) pin thresholds, digests, and A/B results.

## What Tier 2 certifies

On host **`df-macbookpro-m5`** (Apple M5 Max), with AX Engine **`6.14.0`** (binary SHA-256
`02c2f0b378499408918f87b476bc0e9aa922c8a6db6a64fc7d1cb106cbac2989`) and AXQuant **`1.6.1`**,
the candidate passes the three independent MTP acceleration gates on **authorizing decode-heavy
workloads**:

| Gate | Requirement | Agent-coding | Long-form general |
| --- | ---: | ---: | ---: |
| Greedy exactness | 100% identical MTP-off/on tokens | Pass (0 divergent / 5 trials) | Pass (0 divergent / 5 trials) |
| Token-weighted decode speedup | ≥ **1.20×** | **1.257580×** | **1.232980×** |
| Prompt-median speedup | ≥ **1.10×** | **1.111899×** | **1.249950×** |
| Comparison `release_ready` | fully bound evidence | **true** | **true** |

Harness: greedy, temperature `0`, seed `20260728`, draft depth `1`, warmup `2`, measured trials
`5`, max tokens `512`, metric `token-weighted-decode-tps`. Formal env is the Qwen 3.6 exact MTP
profile (`--qwen36-exact-profile`), including
`AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1` and
`AX_MLX_MTP_LINEAR_EXACT_REPLAY=0` (exact lazy checkpoint path).

| Profile | Dataset SHA-256 | Comparison SHA-256 |
| --- | --- | --- |
| agent-coding | `440eea149a2fe5306a7068684a662b9957fe2fa90f8ce5a1914187ea9b3c7adc` | `1c773b4808d8e86db80467233920fca4277739f43a84ecec1ee52c2a774cc945` |
| long-form general | `c43e076e3f78a9e2fa69047d7e8eb2527988ce214a427e04a95a999f93dd8f07` | `25f7290e864a7cfc46d1dd739e414fa381b6f80928d814739cc3ed8f4072b5e5` |

## Claim boundaries (read carefully)

| Claim | Status |
| --- | --- |
| Decode-heavy agent-coding MTP speedup + exactness on formal host | **Certified** |
| Long-form general MTP speedup + exactness on formal host | **Certified** |
| Checkpoint size/quality/default-route (Tier 1) | Still certified; unchanged |
| **Product default** Qwen linear MTP enabled without opt-in | **Not claimed** — safe default remains direct fallback |
| Short-answer / chat-length generations always faster with MTP | **Not claimed** (forced MTP can be slower; product should skip MTP for short remaining budgets) |
| Vision / multimodal quality | **Not claimed** |
| Sibling packs / other revisions | **Not claimed** |
| Full flagship M0–M8 publication campaign closure | Separate process; this certificate closes the **Tier 2 acceleration metric gates** for this artifact |

Public language must not say “always 1.25× faster for every prompt.” Correct form:

> On Apple M5 Max (`df-macbookpro-m5`), with AX Engine 6.14.0 under the formal Qwen linear MTP
> exact contract, this v3 checkpoint achieves ≥1.20× token-weighted and ≥1.10× prompt-median
> decode speedup with greedy-identical outputs on the authorizing agent-coding and long-form
> general suites.

## Runtime how-to (acceleration route)

Default install remains fail-closed for uncertified product promotion. To exercise the certified
route (matching formal evidence):

```bash
export AX_MLX_QWEN_LINEAR_MTP_EXACT=1
export AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1
export AX_MLX_MTP_LINEAR_EXACT_REPLAY=0
# optional product short-budget skip (default 16); formal cert used 0
export AX_MLX_MTP_MIN_REMAINING_TOKENS=16
```

AX Engine builds must include the exact-profile lazy checkpoint path (not always-on singleton
recompute). The formal binary is versioned at `6.14.0` with the SHA above.

## Relation to historical failures

Earlier exactness failures (route-dependent arithmetic; always-singleton recompute) are archived
development evidence. They no longer block this Tier 2 certificate for the v3 weights + AX Engine
6.14.0 formal path documented here.

## Integrity

- Tier 1 certificate remains authoritative for weights, size, and quality.
- Tier 2 is invalid if weights, plan, source revision, thresholds, host, or engine binary diverge
  from the bound digests.
- Evidence copies: [evidence/qwen36-27b-axq6-tier2/](evidence/qwen36-27b-axq6-tier2/).
