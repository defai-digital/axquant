# Qwen 3.6 35B-A3B AXQ 4bit — MTP acceleration Tier 2 certification

**Verdict:** certified for AXQuant **Tier 2 MTP acceleration** on 2026-08-09
(scoped; see claim boundaries below).

This certificate applies to the **same exact weight set** already certified for
[checkpoint Tier 1](qwen36-35b-axq4-tier1.md):

[`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP)
commit
[`a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP/tree/a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc).

Candidate manifest SHA-256:
`d172e6c1dd88e24e1f451735606ffb3d9426b30c2e179bfd8eae771ba7141155`.

The companion [machine-readable record](qwen36-35b-axq4-tier2.json) pins
thresholds, digests, and A/B results.

## What Tier 2 certifies

On host **`df-macbookpro-m5`** (Apple M5 Max), with AX Engine **`6.14.1`**
(binary SHA-256
`854647ab53b2e3e3f0336b24aa1f3fda5492282d03388f647901ded980580cd1`) under the formal Qwen 3.6 **MoE exact** MTP profile
(`--qwen36-moe-exact-profile`: certification candidate, async draft, exact lazy
replay off, `AX_MLX_MTP_VERIFY_SUBMIT_LAYERS=8`,
`AX_MLX_PIPELINE_GRANULARITY=layer`), the candidate passes the three
independent MTP acceleration gates on **authorizing decode-heavy workloads**:

| Gate | Requirement | Agent-coding | Long-form general |
| --- | ---: | ---: | ---: |
| Greedy exactness | 100% identical MTP-off/on tokens | Pass (0 divergent / 5 trials) | Pass (0 divergent / 5 trials) |
| Token-weighted decode speedup | ≥ **1.20×** | **1.316212×** | **1.277165×** |
| Prompt-median speedup | ≥ **1.10×** | **1.100291×** | **1.304732×** |
| Comparison `release_ready` | fully bound evidence | **true** | **true** |

Harness: greedy, temperature `0`, seed `20260728`, draft depth `1`, warmup `2`,
measured trials `5`, max tokens `512`, metric `token-weighted-decode-tps`.

| Profile | Dataset SHA-256 | Comparison SHA-256 |
| --- | --- | --- |
| agent-coding | `440eea149a2fe5306a7068684a662b9957fe2fa90f8ce5a1914187ea9b3c7adc` | `9348b0422989f90acd5136f788ff86018ee5647b464af11ec7566a7b02f17c7a` |
| long-form general | `c43e076e3f78a9e2fa69047d7e8eb2527988ce214a427e04a95a999f93dd8f07` | `30ab3853cd0efa44624c068a1732e633018ba715bd989151ba1f653967f4c99c` |

## Claim boundaries (read carefully)

| Claim | Status |
| --- | --- |
| Decode-heavy agent-coding MTP speedup + exactness on formal host | **Certified** |
| Long-form general MTP speedup + exactness on formal host | **Certified** |
| Checkpoint size/quality (Tier 1) | Certified (see Tier 1 record) |
| **Product default** Qwen linear MTP enabled without opt-in | **Not claimed** — safe default remains direct fallback |
| Short-answer / chat-length generations always faster with MTP | **Not claimed** |
| Vision / multimodal quality | **Not claimed** |
| Dense 27B siblings | **Not claimed** by this certificate |
| Full flagship M0–M8 publication campaign closure | Separate process |

Public language must not say “always N× faster for every prompt.” Correct form:

> On Apple M5 Max (`df-macbookpro-m5`), with AX Engine 6.14.1 under the formal
> Qwen MoE exact MTP contract (async draft, verify-submit interval 8, pipeline
> granularity layer), this 35B-A3B AXQ 4bit checkpoint achieves
> ≥1.20× token-weighted and ≥1.10× prompt-median decode speedup with
> greedy-identical outputs on the authorizing agent-coding and long-form
> general suites.

## Runtime how-to (acceleration route)

```bash
export AX_MLX_QWEN_LINEAR_MTP_EXACT=1
export AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1
export AX_MLX_MTP_LINEAR_EXACT_REPLAY=0
export AX_MLX_MTP_ASYNC_DRAFT=1
export AX_MLX_MTP_VERIFY_SUBMIT_LAYERS=8
export AX_MLX_PIPELINE_GRANULARITY=layer
# optional product short-budget skip (default 16); formal cert used 0
export AX_MLX_MTP_MIN_REMAINING_TOKENS=16
```

## Integrity

- Tier 1 remains authoritative for weights, size class, and quality.
- Tier 2 is invalid if weights, thresholds, host, or engine binary diverge from
  the bound digests.
