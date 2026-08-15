# Muse Glimmer 30B AXQ 4-bit — checkpoint Tier 1 evaluation

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on `df-macstudio-m2`
(2026-08-15). MLX-VLM load and generate smoke passed; dual-suite quality
retention vs BF16 could not be measured because `evaluate-quality` uses the
MLX-LM backend, which rejects `model_type=muse_glimmer`.

Hub pack:
[`AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit)
@ `bcfb0b748fc44487c1657fb6ae190592d515398b`.

| Property | Value |
| --- | --- |
| Host | `df-macstudio-m2` |
| Adapter | `muse-glimmer-v1` |
| Measured BPW | 5.95007 (includes BF16 vision) |
| Weight bytes | 22,146,628,188 |
| MLX-VLM load | Pass (1452 modules) |
| Generate smoke | Pass (nonempty, marker present, 3.28s) |
| Dual-suite quality | **Fail / not run** — backend gap |
| Tier 2 | N/A (no MTP) |

Machine-readable: [muse-glimmer-30b-axq4-tier1.json](muse-glimmer-30b-axq4-tier1.json).
