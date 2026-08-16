# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 4-bit g128

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) |
| Date | 2026-08-16 |
| Protocol | Greedy, temperature `0`, thinking off, factory development suites |

Same-run factory bench of the 4-bit group-128 AXQ pack against
`mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit`.

**Result:** AXQ combined mean **0.339** vs OptiQ **0.724**. AXQ decode-128
**28.04** tok/s vs OptiQ **2.53**. Speed wins; quality does not.

## Bound artifacts

| Pack | Origin | Size | Path |
| --- | --- | --- | --- |
| AXQ 4-bit g128 | `axquant` local convert, recipe `examples/deepseek-v4-experimental-4bit-g128-v0.1.yaml` | 154 GB, 4.35 BPW | Studio Ext12T `AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-g128` |
| OptiQ 2-bit | mlx-community OptiQ-2bit | streamed | Studio Ext12T `DeepSeek-V4-Flash-0731-OptiQ-2bit` |

Common source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Quality

Seed `20260728`, max new tokens `64`.

| Suite | AXQ 4-bit g128 | OptiQ 2-bit |
| --- | --- | --- |
| agent-coding | 0/15 mean **0.200** | 0/15 mean **0.500** |
| general | 7/15 mean **0.478** | 14/15 mean **0.948** |
| mean | 0.339 | **0.724** |

## Speed

| Case | AXQ 4-bit g128 | OptiQ 2-bit |
| --- | --- | --- |
| decode-128 | **28.04** tok/s | 2.53 tok/s |
| prefill-512 | 2.95 tok/s | 0.30 tok/s |
| prefill-2k | 0.44 tok/s | 0.10 tok/s |

AXQ RSS 144 GB; OptiQ RSS ~23 GB.

A group-32 4-bit pack (179 GB) matched OptiQ coding (0.500) in log scores but
OOMed on Metal before speed; general scores there were 0.000 and are not
trusted. 3-bit same-run mean was 0.300 at 27.08 tok/s.

Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq4g128-macstudio-m2/`](eval/deepseek-v4-flash-0731-optiq2-vs-axq4g128-macstudio-m2/).
