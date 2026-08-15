# DeepSeek V4 Flash-0731 AXQ 2-bit (experimental) — Studio evaluation

**Verdict:** **not certified** for checkpoint Tier 1 on `df-macstudio-m2` (2026-08-15).

Converted from
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
to
[`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit)
@ `408c0ab335f6211812645ca44071301c20a55957`.

This is **not** the older Flash source (`60d8d707`) certified in
[deepseek-v4-flash-axq2-tier1.md](deepseek-v4-flash-axq2-tier1.md).

| Gate | Result |
| --- | --- |
| Measured main BPW | `3.1328993873020314` |
| mlx-lm generate smoke | Pass |
| Dual-suite generation viability | **Not run** |
| AX Engine manifest | **Fail** (split `up_proj` vs fused gate+up) |

Machine-readable: [deepseek-v4-flash-0731-axq2-tier1.json](deepseek-v4-flash-0731-axq2-tier1.json).
