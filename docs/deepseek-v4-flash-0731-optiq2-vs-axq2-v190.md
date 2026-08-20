# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 2-bit (v1.9.0)

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, official DSV4 chat, factory development suites |

Product question: on the same Mac Studio, how does the mlx-community OptiQ 2-bit
streaming pack compare to the AXQuant 1.9.0 AXQ 2-bit pack when both use the
official DeepSeek V4 chat template — and does native AX Engine 7.0.2 match
mlx-lm on that pack?

**Short answer:** with the official chat template, AXQ quality is the same on
**mlx-lm** and **AX Engine 7.0.2 native** (combined mean **0.633**). OptiQ is
still better on this short suite (**0.724**). Decode: mlx-lm **27.837** tok/s,
AX Engine 7.0.2 **15.196** tok/s, OptiQ **3.471** tok/s. AX Engine is native
MLX, not mlx-lm-delegated.

## Bound artifacts

| Pack | Hub | Inference engine | Local path |
| --- | --- | --- | --- |
| DeepSeek V4 Flash-0731 AXQ 2-bit (v1.9.0) | local convert on Studio (`axquant==1.9.0`) | **mlx-lm** resident (`0.31.3`) | Ext12T `AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0` |
| same AXQ 2-bit pack | same | **ax-engine-server 7.0.2** native (`mlx-preview`, `--stream-experts off`) | same pack + remapped `model-manifest.json` |
| DeepSeek V4 Flash-0731 OptiQ 2-bit | [`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) | **mlx-optiq** expert stream | `DeepSeek-V4-Flash-0731-OptiQ-2bit` |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

AX Engine 7.0.2 originally rejected this pack (`ffn_gate_up_exps_packed` expected
`[256, 4096, 256]`, got `[256, 2048, 256]`). Convert now remaps split
`switch_mlp.gate_proj` + `up_proj` to `ffn_gate_exps` / `ffn_up_exps`. Manifest
on Studio: `packed=0`, `gate_exps=43`, `up_exps=43`. Server used was the stock
`ax-engine-server` 7.0.2 binary (mlx-preview tier).

## Quality

Seed `20260728`, max new tokens `64`. Pass = every check on the task scores 1.0.
All three columns use official DSV4 chat (thinking off).

| Suite | N | AXQ 2-bit / mlx-lm | AXQ 2-bit / AX Engine 7.0.2 | OptiQ 2-bit / mlx-optiq |
| --- | ---: | --- | --- | --- |
| agent-coding | 15 | 0 / 15 (0.0%) mean **0.500** | 0 / 15 (0.0%) mean **0.500** | 0 / 15 (0.0%) mean **0.500** |
| general | 15 | 11 / 15 (73.3%) mean **0.767** | 11 / 15 (73.3%) mean **0.767** | 14 / 15 (93.3%) mean **0.948** |
| combined | 30 | **0.633** | **0.633** | **0.724** |

AX Engine smoke was `Okay.` General misses on Engine: `instruction-005`,
`007`, `011` (score 0) and `015` (0.5). Coding stays at 0.5 on every item at
the 64-token cap (`python-syntax` fails on truncated functions).

An earlier mlx-lm row without the official chat template scored combined
**0.310**. That number is not used in this table.

## Speed

Columns: prompt tokens / generated tokens / wall / tok/s.

| Case | AXQ / mlx-lm | AXQ / AX Engine 7.0.2 | OptiQ / mlx-optiq |
| --- | --- | --- | --- |
| decode-128 | 20 / 128 / 4.60s / **27.837** | 20 / 128 / 8.42s / **15.196** | 20 / 128 / 36.88s / **3.471** |
| prefill-512-decode-8 | 485 / 8 / 2.86s / 2.799 | 485 / 8 / 3.04s / 2.633 | 485 / 8 / 13.57s / 0.590 |
| prefill-2k-decode-8 | 3845 / 8 / 18.61s / 0.430 | 3845 / 8 / 16.52s / 0.484 | 3845 / 8 / 61.71s / 0.130 |

Load: mlx-lm `198.8` s (RSS `104.1` GB); AX Engine `335.7` s; OptiQ `35.2` s
(RSS `23.4` GB). Engine decode is slower than mlx-lm because this pack is
**split** gate/up (two `gather_qmm`s), not fused packed experts. Prefill-2k
is within noise of mlx-lm. Speed is not a same-kernel A/B versus OptiQ.

## Notes

- This is **not** checkpoint Tier 1 and **not** a retention-vs-BF16 claim.
- Combined 0.633 is below the experimental generation-viability floor (0.90).
  The 0731 2-bit SKU stays **listed, not certified**.
- AX Engine path is **native** 7.0.2 MLX (`--support-tier mlx-preview`). It is
  not `--support-tier mlx-lm-delegated`.
- `plan-joint` is not the allocator on this 2-bit recipe.

Runners: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`](../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py)
(mlx-lm / OptiQ) and
[`scripts/run_deepseek_v4_0731_axq2_axengine.py`](../scripts/run_deepseek_v4_0731_axq2_axengine.py)
(AX Engine 7.0.2 native).
Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-axengine-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq2-axengine-macstudio-m2/).
