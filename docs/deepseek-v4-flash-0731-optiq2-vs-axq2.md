# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 2-bit

| Field | Value |
| --- | --- |
| Status | **Scheduled** on `df-macstudio-m2`; results not yet measured |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB unified memory) |
| First attempt | 3 hours after schedule, then every hour until the factory GPU/memory is idle |
| Protocol | Greedy, temperature `0`, thinking off; factory development QA suites + native-runtime speed |

This report answers a product question, not a certification question:

> On the same Mac Studio, how does [`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) compare to [`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit) for short QA and decode speed?

**Short answer:** not measured yet. The factory job waits for Holo-3.1 convert/cert to finish (and any other large Studio job), then runs both packs under their native runtimes.

## Bound artifacts

| Pack | Hub | Runtime | Local path on host |
| --- | --- | --- | --- |
| AXQ 2-bit (exp.) | [`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit) @ `408c0ab3` | resident mlx-lm | `/Volumes/Ext12T/models/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit` (~114 GB) |
| OptiQ 2-bit | [`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit) | mlx-optiq expert streaming | `/Volumes/Ext12T/models/DeepSeek-V4-Flash-0731-OptiQ-2bit` (~92.5 GB on disk, ~6.5 GB resident) |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Protocol

| Setting | Value | Why |
| --- | --- | --- |
| Decoding | temperature `0`, greedy | Reproducible A/B |
| Thinking | off | Matched instruct-mode cost |
| QA suites | `development-agent-coding` + `development-general` | Same factory tasks used for other Studio evals |
| QA max tokens | 64 | Factory T1 generation length |
| Speed | decode-128; prefill ~512 and ~2k then 8 new tokens | Separate from QA |
| AXQ runtime | mlx-lm resident load | AX Engine 6.16.1 manifest fails on this 0731 pack (split `up_proj`) |
| OptiQ runtime | isolated `mlx-optiq` venv + `moe_stream.load_streaming` | Intended OptiQ path; not AX Engine |
| Isolation | one large model at a time; start only when convert/quality/T2 jobs are gone | AXQ 2-bit is ~114 GB resident on a 192 GB host |

This is **not** checkpoint Tier 1. The 0731 AXQ 2-bit pack stays [not certified](certifications/deepseek-v4-flash-0731-axq2-tier1.md). Speed is not a same-kernel A/B: OptiQ streams routed experts from SSD; AXQ keeps them in memory.

## Results

Pending factory measurement. Raw JSON will land in
[`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq2-macstudio-m2/).
Runner: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`](../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py).

Studio log: `/Volumes/Ext12T/logs/dsv4-optiq-vs-axq2.log`.
