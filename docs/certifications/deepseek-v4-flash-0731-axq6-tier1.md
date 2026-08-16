# DeepSeek V4 Flash-0731 AXQ 6-bit — listed, not certified (host memory)

**Verdict:** **not certified** on `df-macstudio-m2` (2026-08-16).

Recipe: `examples/deepseek-v4-experimental-6bit-g128-v0.1.yaml`.
Estimated ~200 GB+ resident. The 179 GB 4-bit g32 pack already required 172 GB
Metal and OOMed generate. **df-macstudio-m2 (192 GB) cannot run the dual-suite
generate required for checkpoint Tier 1.** Recert needs a larger Studio.

The SKU stays **listed**. Machine-readable:
[deepseek-v4-flash-0731-axq6-tier1.json](deepseek-v4-flash-0731-axq6-tier1.json).
