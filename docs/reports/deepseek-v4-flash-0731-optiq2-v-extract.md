# DeepSeek V4 Flash-0731 — OptiQ 2-bit on AXQ `v-extract`

| Field | Value |
| --- | --- |
| Status | Measured protocol baseline; **not** an AXQ certificate |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-20 |
| Protocol | Factory `v-extract` (same as AXQ T1): seed `20260728`, coding 384 / general 64, suite system prompts, DSV4 non-thinking, greedy |

Question: on the **same** `v-extract` suite that scores AXQ uniform v0.1 at combined **0.887**, does the public market 2-bit pack clear the 0.90 floor?

**Short answer:** yes. `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit` on its native mlx-optiq runtime scores combined **0.967** (coding **1.000**, general **0.933**). AXQ uniform v0.1 remains **0.887** / not certified. This is not a same-kernel A/B and not an AXQ cert of the OptiQ pack.

## Bound artifacts

| Pack | Runtime | Path |
| --- | --- | --- |
| OptiQ 2-bit | mlx-optiq stream (`mlx-optiq==0.4.25`, experts resident on 192 GB) | Ext12T `DeepSeek-V4-Flash-0731-OptiQ-2bit` |
| AXQ uniform v0.1 | AX Engine `80f2a3e6` native, `--stream-experts off` | Hub `…-2bit-MTP` @ `cb1a34b4` |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b`.

## Quality (`v-extract`)

Pass = every check on the task scores 1.0.

| Suite | N | AXQ uniform v0.1 | OptiQ 2-bit |
| --- | ---: | --- | --- |
| agent-coding | 15 | 12/15 mean **0.900** | **15/15 mean 1.000** |
| general | 15 | 13/15 mean **0.874** | 14/15 mean **0.933** |
| combined | 30 | **0.887** | **0.967** |
| vs floor 0.90 | | miss | **clear** |

OptiQ's only miss: `instruction-009` (`banana, apple, cherry` unsorted; scorer wants `apple, banana, cherry`).

## Speed (this run)

Not a same-kernel A/B. OptiQ decode-128 **1.94** tok/s; AXQ Engine uniform v0.1 was ~15 tok/s resident.

| Case | OptiQ tok/s |
| --- | ---: |
| decode-128 | 1.944 |
| prefill-512-decode-8 | 0.552 |
| prefill-2k-decode-8 | 0.137 |

Load 81.3 s; process RSS ~23 GB (expert scales/biases resident).

## What this does and does not mean

- The 0.90 floor is **reachable in the 2-bit product class** on this checkpoint and protocol. It is not reachable with AXQ uniform v0.1.
- Isolated AXQ raises (attention 6-bit, shared 4-bit) scored **worse** than uniform. The market pack is a **different complete assignment** (attention 6 / shared 8 / LM-head 8 / routed 2), not those one-axis YAML tweaks.
- Do not copy mlx-optiq code, calibration, or metadata. A follow-up AXQ-owned convert would have to re-derive bit assignment from public facts + AXQ measurement.
- This record does **not** certify the AXQ Hub pack and does **not** certify OptiQ as an AXQ artifact.

Runner: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`](../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py) `market-v-extract`.
Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-v-extract-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-v-extract-macstudio-m2/).
