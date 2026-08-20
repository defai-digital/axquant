# Qwen3.8-2.4T-A95B — experimental AXQ stream pack

| Pack | Hub | Notes |
| --- | --- | --- |
| AXQ 2-bit | [`AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP) | Convert complete. Native MTP sidecar packaged; acceleration not claimed. **[Technical report](qwen38-axq-2bit.md).** **Will not certify** — SSD paging is too slow for practical use (hobby / curiosity pack). |

**No AXQ 4-bit pack will be released** for this base. A 4-bit sibling would
be even larger (~1.8 TB class) and still require the same layer-stack SSD
path that made 2-bit too slow to certify.

This is an AXQuant Super-class pack (`qwen38-moe-v1`) with
`ax_expert_stream.json` `required=true`. It cannot resident-load on any
shipping Mac. 2-bit also needs `AX_ENGINE_2BIT_EXPERIMENTAL=1`.

The 2-bit revision **will not be certified** (streamed decode is too slow
for practical serving). Treat it as a hobby / curiosity pack, not a product
path.

OptiQ Qwen 3.8 Hub repos are not AX Engine artifacts. See
[qwen38-optiq-experimental.md](qwen38-optiq-experimental.md).
