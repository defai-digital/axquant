# DeepSeek V4 Flash-0731 AXQ 3-bit (experimental) — Studio evaluation

**Verdict:** **not certified** for checkpoint Tier 1 on `df-macstudio-m2` (2026-08-15).

Converted from
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
to
[`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-3bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-3bit)
@ `bc10f5413050a18950c101cb260c0fb2016e01e3`.

This is **not** the older Flash source certified in
[deepseek-v4-flash-axq3-tier1.md](deepseek-v4-flash-axq3-tier1.md).

| Gate | Result |
| --- | --- |
| Measured main BPW | `4.110998966099255` |
| mlx-lm generate smoke | Pass |
| Dual-suite generation viability | **Not run** |
| AX Engine manifest | **Fail** (split `up_proj` vs fused gate+up) |

Machine-readable: [deepseek-v4-flash-0731-axq3-tier1.json](deepseek-v4-flash-0731-axq3-tier1.json).
