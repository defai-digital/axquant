# MiniMax-M3 MLX AXQ MXFP4 (exp.) — checkpoint Tier 1

**Verdict:** **not certified** on `tn-macstudio-m3` with AX Engine `7.2.0`.

This certificate covers [`AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4) commit [`228f7b75249a92ed1abf43bc5b7b7d8c388c55b3`](https://huggingface.co/AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4/tree/228f7b75249a92ed1abf43bc5b7b7d8c388c55b3).

| Field | Value |
| --- | --- |
| Hub | [`AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4) |
| Source | `MiniMaxAI/MiniMax-M3@f0e1c1e04d40177e4673a22097036854f536e9c0` |
| Host | `tn-macstudio-m3` |
| Product class | `MXFP4` |
| Architecture | `MiniMaxM3SparseForConditionalGeneration` |
| Measured main BPW | `4.365766982835565` |
| Weight bytes | `233044718032` |
| Agent-coding viability | `0.0` (need ≥ 0.9) |
| General viability | `0.0` (need ≥ 0.9) |
| MTP acceleration | `not-applicable` (no packaged MTP) |
| Stream | `ax_expert_stream.json` required |

## Notes

- Experimental Super-class track: generation viability, not BF16 retention.
- Language path only. Vision is present/protected and **not** certified.
- Seed `20260728`, max gen 64, AX Engine `7.2.0`.
- Native convert, load, and chat smoke now succeed on 7.2.0, but greedy text is incoherent (viability 0.0). MiniMax chat framing from `chat_template.jinja` is not implemented yet (`PlainRolePrefix`).

## Related

- Sibling 2-bit: [minimax-m3-axq2-tier1.md](minimax-m3-axq2-tier1.md)

Machine-readable: [minimax-m3-axq-mxfp4-tier1.json](minimax-m3-axq-mxfp4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | Vision tower is BF16-protected; language-path cert does not claim image/video quality. |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

