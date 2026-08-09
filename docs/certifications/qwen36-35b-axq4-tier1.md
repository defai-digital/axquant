# Qwen 3.6 35B-A3B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-08 (quality +
class-adjusted size budget). **MTP acceleration is not certified** on this
revision (see Tier 2 status).

This certificate covers
[`AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP)
commit
[`a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP/tree/a549387d5b812c6f6cbdb0ebde37adb3b3f4a2bc).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (MoE A3B) |
| Product class | `4bit` mixed AXQ (measured total BPW ≈ `5.14`) |
| Source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Candidate manifest SHA-256 | `d172e6c1dd88e24e1f451735606ffb3d9426b30c2e179bfd8eae771ba7141155` |
| Plan evidence | `architecture_prior` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `5.14` | `5.140061` | Pass |
| Weight-size ratio vs uniform-4 | ≤ `1.15` (class budget) | `1.132197` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| Scorer errors | `0 / 0` | `0 / 0` | Pass |

### Size note

Candidate weight bytes `23,099,321,635` vs uniform-4
`mlx-community/Qwen3.6-35B-A3B-4bit` `20,402,204,271` → **1.132×**. The pure-4bit
≤1.10 gate is not claimed; formal scoreboard used `max_size_ratio=1.15` for this
mixed-AXQ class.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| General (44) | `1.000` | `1.000` | `1.000` | `0.984415` |
| Agent-coding (76) | `0.888158` | `0.888158` | `1.000` | `0.982619` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Tier 2 status

**Not certified.** Greedy exactness passes with MTP active, but the speed gates
did not clear on the released engine:

| Profile | Exactness | Weighted speedup | Prompt-median | `release_ready` |
| --- | --- | ---: | ---: | --- |
| agent-coding | Pass | **0.946×** | **0.803×** | false |
| long-form general | Pass | **0.911×** | **0.928×** | false |

The cause is now identified and is **not** sparse-expert weight bandwidth.
Per-step phase timing on this pack (`df-macbookpro-m5`, AX Engine
`6.14.0-moe-mtp`, agent-coding, depth 1) attributes 4.0 ms of every 17.4 ms
speculative step to `verify_forward_wall_us` — building the verify graph on the
host, serially, with the GPU idle — and a further 2.7 ms to a synchronous draft.
That fixed cost is roughly the same in absolute terms on the dense 27B siblings,
where it is ~15% of a much longer step; a MoE step reads only its routed experts
and is several times shorter, so the same cost becomes ~45% of a step. That
single asymmetry is why the dense siblings certified and this pack did not.

Two changes address it, and both preserve greedy exactness by construction —
neither alters an operand, shape, or reduction order:

| Configuration | Weighted | Prompt-median | Both gates |
| --- | ---: | ---: | --- |
| Released `6.14.0-moe-mtp`, dense profile | 0.946× / 0.911× | 0.803× / 0.928× | no |
| Released engine, `--qwen36-moe-exact-profile` | 1.111× / 1.064× | 0.924× / 1.089× | no |
| Plus chunked verify submit (unreleased) | 1.265× / 1.216× | 1.035× / 1.246× | long-form only |

(agent-coding / long-form general; measured 2026-08-09 on `df-macbookpro-m5`,
`divergent_trial_count = 0` throughout.)

The first row-to-row gain needs no engine change: it comes from
`AX_MLX_MTP_ASYNC_DRAFT`, which the dense certification profile never set. The
second needs `AX_MLX_MTP_VERIFY_SUBMIT_LAYERS`
([ax-engine#77](https://github.com/defai-digital/ax-engine/pull/77)), which is
**not in a released engine build**.

Product default remains direct fallback, and this pack stays Tier 1 only. Tier 2
here needs three things that do not exist yet: a released engine carrying the
verify-submit change, a certification-grade binary built from a clean tree at a
distinct version, and a long-form-only scope decision or a prompt-median result
that clears 1.10× on agent-coding. Pre-release measurements are development
evidence and do not authorize an acceleration claim.

## Related

- Dense 27B siblings: [6-bit](qwen36-27b-axq6-tier1.md), [4-bit](qwen36-27b-axq4-tier1.md)
- Sibling 35B 6-bit: [Tier 1](qwen36-35b-axq6-tier1.md)
