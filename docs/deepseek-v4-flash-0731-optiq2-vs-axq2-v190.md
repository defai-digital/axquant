# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 2-bit (v1.9.0)

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, factory development suites |

Product question: on the same Mac Studio, how does the mlx-community OptiQ 2-bit streaming pack compare to the AXQuant 1.9.0 AXQ 2-bit resident pack for short QA and decode speed?

**Short answer:** factory-suite mean score AXQ `0.310` vs OptiQ `0.724`. Speed is native-runtime: AXQ resident mlx-lm vs OptiQ SSD expert streaming.

## Bound artifacts

| Pack | Hub | Runtime | Local path |
| --- | --- | --- | --- |
| DeepSeek V4 Flash-0731 AXQ 2-bit (v1.9.0) | local convert on Studio (`axquant==1.9.0`, not Hub-published) | resident mlx-lm | `/Volumes/Ext12T/models/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0` |
| DeepSeek V4 Flash-0731 OptiQ 2-bit | [`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) | mlx-optiq stream | `DeepSeek-V4-Flash-0731-OptiQ-2bit` |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Quality (factory development suites)

Seed `20260728`, max new tokens `64`. Pass = every check on the task scores 1.0.

| Suite | N | AXQ 2-bit | OptiQ 2-bit |
| --- | ---: | --- | --- |
| agent-coding | 15 | 0 / 15 (0.0%) mean 0.133 | 0 / 15 (0.0%) mean 0.500 |
| general | 15 | 7 / 15 (46.7%) mean 0.487 | 14 / 15 (93.3%) mean 0.948 |

## Speed (native runtime, greedy)

Columns: prompt tokens / generated tokens / wall / tok/s. Ratio is OptiQ / AXQ.

| Case | AXQ 2-bit | OptiQ 2-bit | OptiQ / AXQ |
| --- | --- | --- | ---: |
| decode-128 | 16 / 128 / 4.77s / 26.81 | 20 / 128 / 38.48s / 3.33 | 0.12x |
| prefill-512-decode-8 | 481 / 8 / 2.69s / 2.98 | 485 / 8 / 15.86s / 0.50 | 0.17x |
| prefill-2k-decode-8 | 3841 / 8 / 19.85s / 0.40 | 3845 / 8 / 63.38s / 0.13 | 0.31x |

Load time: AXQ `46.9` s, OptiQ `43.2` s. Peak RSS (process): AXQ `99.9` GB, OptiQ `21.4` GB.

## Notes

- This is **not** checkpoint Tier 1 and **not** a retention-vs-BF16 claim.
- Both packs are **DeepSeek-V4-Flash-0731** (`7872f01b…`). The older non-0731 Flash packs were not used.
- AXQ 0731 2-bit remains **not certified** (dual-suite viability was previously skipped; AX Engine manifest fails on fused gate+up).
- 1.9.0 used the same experimental 2-bit manual recipe as the prior 0731 pack (`mlp`/`expert` 2-bit, attention 4-bit, floors 8/16). Mean scores match that earlier pack (`0.310` vs OptiQ `0.724`); this is not a task-score win from `plan-joint`.
- OptiQ streams routed experts from SSD; AXQ keeps the expert table resident. Speed is not a same-kernel A/B.
- Suites: `development-agent-coding` and `development-general` on Ext12T.

Runner: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`](../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py).
Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-v190-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq2-v190-macstudio-m2/).

