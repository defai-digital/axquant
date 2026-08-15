# Qwen3.8-27B AXQ 8-bit-MTP — scoped Tier 2 evaluation

**Verdict:** **not certified** for public MTP acceleration on `df-macstudio-m2`
(2026-08-15). A **short** AX Engine 6.16.1 probe (`full=false`) cleared
exactness and speedup gates; that is not the authorizing `--full` suite.

| Profile | Exactness | Weighted decode | Prompt median | Probe |
| --- | ---: | ---: | ---: | --- |
| agent-coding | pass | 1.225× | 1.225× | 8 prompts / 1 trial |
| general-long | pass | 1.227× | 1.227× | 6 prompts / 1 trial |

Product default remains direct fallback until `--full` is measured.

Machine-readable: [qwen38-27b-axq8-mtp-tier2.json](qwen38-27b-axq8-mtp-tier2.json).
