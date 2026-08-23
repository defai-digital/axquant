# DeepSeek-V4-Pro-0813 — experimental AXQ stream pack

| Pack | Hub | Notes |
| --- | --- | --- |
| AXQ 2-bit | [`AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP) | Super-class stream convert. Native DSpark sidecar packaged as `mtp.safetensors`; acceleration not claimed. **[Technical report](deepseek-v4-pro-0813-axq-2bit.md).** **Will not certify** — SSD paging is too slow for practical use (hobby / curiosity pack). |

**No AXQ 4-bit pack will be released** for this base. A 4-bit sibling would
be even larger and still require the same layer-stack SSD path that made
2-bit too slow to certify.

This is an AXQuant Super-class pack (`deepseek-v4-v1`) with
`ax_expert_stream.json` `required=true`. It cannot resident-load on any
shipping Mac. 2-bit also needs `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

The 2-bit revision **will not be certified** (streamed decode is too slow
for practical serving). Treat it as a hobby / curiosity pack, not a product
path.
