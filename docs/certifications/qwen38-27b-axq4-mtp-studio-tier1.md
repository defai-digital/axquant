# Qwen3.8-27B AXQ 4-bit MTP — Studio recert evaluation

**Verdict:** **not certified** as a replacement checkpoint Tier 1 on `df-macstudio-m2`
(2026-08-15). Historical record
[qwen38-27b-axq4-mtp-tier1.md](qwen38-27b-axq4-mtp-tier1.md) (`df-macbookpro-m3`)
is unchanged.

mlx-lm `evaluate-quality` ran on the same Hub pack
(`32f448461caf4aedcc3c16a77a63b6a94bf0667c`) against `prepare-suite` v2.

| Suite | Samples | Candidate mean | Retention |
| --- | ---: | ---: | --- |
| agent-coding | 52 | 0.9647 | not computed (no BF16 on Ext12T) |
| general | 16 | 0.8750 | not computed |

Machine-readable: [qwen38-27b-axq4-mtp-studio-tier1.json](qwen38-27b-axq4-mtp-studio-tier1.json).
