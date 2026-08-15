# Qwen3.8-27B AXQ-MXFP4-MTP — scoped Tier 2 evaluation

**Verdict:** **not certified** for MTP acceleration on `df-macstudio-m2` (2026-08-15).

Measured with AX Engine **6.16.1** (`c39b69a1…`) under the Qwen3.8 exact-async
profile. In-repo development suites (8 agent-coding + 6 general prompts) produced
`exactness_pass=false` and token-weighted / prompt-median speedups of `0.0`.

| Profile | Exactness | Weighted decode | Prompt median | Result |
| --- | ---: | ---: | ---: | --- |
| agent-coding | fail | 0.00× | 0.00× | Fail |
| general-long | fail | 0.00× | 0.00× | Fail |

Tier 1 checkpoint certification on this host is unchanged. Product default remains
direct fallback.

Machine-readable: [qwen38-27b-axq-mxfp4-mtp-tier2.json](qwen38-27b-axq-mxfp4-mtp-tier2.json).
