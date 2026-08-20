# DeepSeek V4 Flash-0731 AXQ 2-bit (experimental) — Studio evaluation

**Verdict:** **not certified** for checkpoint Tier 1 on `df-macstudio-m2`
(2026-08-20) with **AX Engine 7.1.5**.

Converted from
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
to
[`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP)
@ `cb1a34b4dc8e182cc94689fce013d85766c5f3d1`.

This is **not** the older Flash source (`60d8d707`) certified in
[deepseek-v4-flash-axq2-tier1.md](deepseek-v4-flash-axq2-tier1.md).

| Gate | Result |
| --- | --- |
| Measured main BPW | `3.1328993873020314` |
| AX Engine 7.1.5 native load | **Pass** — Hub snapshot loaded with `--stream-experts off`; chat smoke `Okay.` |
| Dual-suite generation viability | **Not certified** — factory 15+15 combined **0.633** (agent-coding **0.500**, general **0.767**; floor 0.90) |
| Decode-128 | 15.535 tok/s (native 7.1.5, not a Tier 1 claim) |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.9.0`.
Requires `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

Machine-readable: [deepseek-v4-flash-0731-axq2-tier1.json](deepseek-v4-flash-0731-axq2-tier1.json).
Native 7.1.5 evidence: [axq2-axengine.json](../eval/deepseek-v4-flash-0731-axq2-axengine-7.1.5-macstudio-m2/axq2-axengine.json).

Prior 7.0.2 native QA was the same combined **0.633**. Practical QA/speed vs
the mlx-community OptiQ 2-bit pack:
[deepseek-v4-flash-0731-optiq2-vs-axq2-v190.md](../deepseek-v4-flash-0731-optiq2-vs-axq2-v190.md).

## Tier 2 status

**Not certified.** MTP weights are packaged (`mtp.safetensors`; Hub leaf uses
`-MTP`). AX Engine 7.1.5 keeps uncertified MTP on direct decode unless an
operator opts into a certification run. This record does not authorize
speculative-decode speedup.
