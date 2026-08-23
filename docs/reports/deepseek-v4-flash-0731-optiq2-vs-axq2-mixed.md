# DeepSeek V4 Flash-0731 — mixed AXQ 2-bit vs OptiQ 2-bit

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, factory development suites |

This pack tries the first clean-room improvement from the OptiQ review:
a **finer 2-bit manual recipe**, not `plan-joint` and not mlx-optiq.

Recipe: [`examples/deepseek-v4-experimental-2bit-mixed-v0.1.yaml`](../examples/deepseek-v4-experimental-2bit-mixed-v0.1.yaml).

- Shared experts + first two / last two decoder FFN stacks: **4-bit**
- Routed expert down-proj (`w2`): **3-bit**
- Remaining routed gate/up: **2-bit**
- Attention still 4-bit; floors unchanged

Fused switch modules still share one bit-width per projection per layer.

**Result:** mixed AXQ mean **0.300** vs OptiQ **0.724**. Versus the uniform
2-bit AXQ pack (mean **0.310**), general got **worse** (0.400 vs 0.487) and
coding only slightly better (0.200 vs 0.133). Heuristic bit lifts did **not**
close the OptiQ gap.

## Bound artifacts

| Pack | Origin | Size | Path |
| --- | --- | --- | --- |
| AXQ mixed 2-bit | `axquant==1.9.0` local convert | 130 GB, 3.666 BPW | Studio Ext12T `AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-mixed` |
| AXQ uniform 2-bit (prior) | `axquant==1.9.0` same source | 114 GB, 3.214 BPW | Studio Ext12T `AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0` |
| OptiQ 2-bit | [`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) | 86 GB streamed | Studio Ext12T `DeepSeek-V4-Flash-0731-OptiQ-2bit` |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Quality (same suite as the uniform pack)

Seed `20260728`, max new tokens `64`.

| Suite | Uniform AXQ 2-bit | Mixed AXQ 2-bit | OptiQ 2-bit |
| --- | --- | --- | --- |
| agent-coding | 0/15 mean 0.133 | 0/15 mean **0.200** | 0/15 mean 0.500 |
| general | 7/15 mean **0.487** | 6/15 mean 0.400 | 14/15 mean **0.948** |
| mean | **0.310** | 0.300 | **0.724** |

## Speed (this run)

OptiQ decode was slower than the earlier v1.9.0 run (1.2 vs 3.3 tok/s);
treat speed here as noisy. AXQ mixed decode **27.8** tok/s, RSS **112 GB**.

| Case | Mixed AXQ | OptiQ (this run) |
| --- | --- | --- |
| decode-128 | 27.80 tok/s | 1.23 tok/s |
| prefill-512 | 2.90 tok/s | 0.11 tok/s |
| prefill-2k | 0.44 tok/s | 0.06 tok/s |

## What this means

- Spending more bits by **hand** (shared / edges / down-proj) is not enough.
- Next shipped lever is [`plan-experimental-mix`](experimental-trunk-mix.md):
  measured 2/3/4 on the robust trunk, fused switch modules as one unit.
  Fused Flash experts still cannot take AWQ/GPTQ in convert today.
- `plan-joint` still does not apply (no 2-bit cell).

Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-mixed-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq2-mixed-macstudio-m2/).
