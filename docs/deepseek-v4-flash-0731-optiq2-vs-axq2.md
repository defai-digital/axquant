# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 2-bit

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, factory development suites |

Product question: on the same Mac Studio, how does the mlx-community OptiQ 2-bit streaming pack compare to the AutomatosX AXQ 2-bit resident pack for short QA and decode speed?

**Short answer:** on this short greedy suite OptiQ is **much more usable** (mean `0.724` vs AXQ `0.310`), especially on general instruction items (`14/15` vs `7/15`). AXQ is **much faster** at decode (`26.5` vs `3.6` tok/s) because the expert table stays resident. Neither pack fully passes the 15 coding tasks at a 64-token cap — outputs are truncated mid-function, so `python-syntax` fails.

## Bound artifacts

| Pack | Hub | Runtime | Local path |
| --- | --- | --- | --- |
| DeepSeek V4 Flash-0731 AXQ 2-bit | [`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit) @ `408c0ab335f6211812645ca44071301c20a55957` | resident mlx-lm | `AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit` |
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
| decode-128 | 16 / 128 / 4.83s / 26.53 | 20 / 128 / 35.63s / 3.59 | 0.14x |
| prefill-512-decode-8 | 481 / 8 / 2.94s / 2.72 | 485 / 8 / 12.47s / 0.64 | 0.24x |
| prefill-2k-decode-8 | 3841 / 8 / 18.67s / 0.43 | 3845 / 8 / 55.61s / 0.14 | 0.34x |

Load time: AXQ `192.5` s, OptiQ `34.8` s. Peak RSS (process): AXQ `104.1` GB, OptiQ `23.4` GB.

## Notes

- This is **not** checkpoint Tier 1 and **not** a retention-vs-BF16 claim.
- AXQ 0731 2-bit remains **not certified** (dual-suite viability was previously skipped; AX Engine manifest fails on fused gate+up).
- OptiQ streams routed experts from SSD; AXQ keeps the expert table resident. Speed is not a same-kernel A/B.
- Factory development suites on the Studio host. Coding items are truncated at 64 new tokens, so strict `python-syntax` almost never passes; treat agent-coding as a generation-viability signal, not a unit-test pass.

Runner: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`](../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py).
Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq2-macstudio-m2/).

