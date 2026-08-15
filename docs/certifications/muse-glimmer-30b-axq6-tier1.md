# Muse Glimmer 30B AXQ 6-bit — checkpoint Tier 1 evaluation

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 on `df-macstudio-m2`
(2026-08-15). MLX-VLM generate smoke passed; dual-suite quality vs BF16 cannot
run (`muse_glimmer` is not an MLX-LM model type).

Hub pack:
[`AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit)
@ `f1cfad2d2aa2fb0572786d63f7420fdb4321bed5`.

| Property | Value |
| --- | --- |
| Host | `df-macstudio-m2` |
| Measured BPW | 7.69022 (includes BF16 vision) |
| Weight bytes | 28,623,608,770 |
| Generate smoke | Pass (2.61s) |
| Dual-suite quality | **Fail / not run** |
| Tier 2 | N/A |

Machine-readable: [muse-glimmer-30b-axq6-tier1.json](muse-glimmer-30b-axq6-tier1.json).
