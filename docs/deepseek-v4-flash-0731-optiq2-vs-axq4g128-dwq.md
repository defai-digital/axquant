# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 4-bit g128 DWQ

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, factory development suites, official DSV4 chat |

Same-run factory bench of the 4-bit group-128 pack with **attention 6-bit DWQ
clip** (fused trunk 4-bit affine) against
`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`.

**Result:** AXQ combined mean **0.750** vs OptiQ **0.724**. AXQ decode-128
**26.469** tok/s vs OptiQ **2.604**. AXQ wins quality and speed.

## Bound artifacts

| Pack | Origin | Size |
| --- | --- | --- |
| AXQ 4-bit g128 DWQ-attn | `axquant` local convert, recipe `examples/deepseek-v4-experimental-4bit-g128-dwq-v0.1.yaml` | 155 GB, 4.371 BPW |
| OptiQ 2-bit | mlx-community OptiQ-2bit | streamed |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Quality

Seed `20260728`, max new tokens `64`.

| Suite | AXQ 4-bit g128 DWQ | OptiQ 2-bit |
| --- | --- | --- |
| agent-coding | 0/15 mean **0.500** | 0/15 mean **0.500** |
| general | **15/15** mean **1.000** | 14/15 mean **0.948** |
| mean | **0.750** | 0.724 |

AXQ general includes a pass on `instruction-009` (`apple, banana, cherry`).
Without DWQ clip the same 4-bit g128 + chat pack was combined 0.717
(failed that sort item).

## Speed

| Case | AXQ 4-bit g128 DWQ | OptiQ 2-bit |
| --- | --- | --- |
| decode-128 | **26.469** tok/s | 2.604 tok/s |
| prefill-512-decode-8 | 2.899 tok/s | 0.318 tok/s |
| prefill-2k-decode-8 | 0.432 tok/s | 0.103 tok/s |

AXQ RSS ~142 GB; OptiQ RSS ~23 GB. The prefill-* rows are generate-8 over
full wall time, not clean prefill-only rates.

`plan-joint` was not the allocator. Not mlx-optiq.

Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq4g128-dwq-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq4g128-dwq-macstudio-m2/).
