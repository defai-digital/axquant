# DeepSeek V4 Flash AXQ 2-bit (experimental) — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 (**experimental low-bit**) on 2026-08-10
on host **`df-macstudio-m2`** (Apple M2 Ultra, 192 GB).

This certificate covers
[`AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit)
commit
[`e22b117aa812b29943b160bb0fbf0b962d0d3819`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit/tree/e22b117aa812b29943b160bb0fbf0b962d0d3819).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `DeepseekV4ForCausalLM` (MoE Flash) |
| Product class | `2bit-experimental` |
| Source | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` |
| Candidate manifest SHA-256 | `9466fb3ddbaf2d286bac5474af63dca1aa4b6ed959059bdea12cdc013437dedd` |
| Measured main BPW | `3.132898927449397` |
| Weight bytes | `114,942,890,815` |
| Certification host | `df-macstudio-m2` |
| MTP in pack | `True` (acceleration **not** claimed) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured main BPW | recorded | `3.132898927449397` | Pass |
| Weight bytes | recorded | `114,942,890,815` | Pass |
| Agent-coding generation viability | ≥ `0.90` | `0.9737` (76 tasks) | Pass |
| General generation viability | ≥ `0.90` | `0.9545` (44 tasks) | Pass |
| MLX generation | suite load + decode | Pass | Pass |
| AX Engine 6.15.0 runtime | load + chat + stream + context | Pass | Pass |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.1`.

### AX Engine 6.15.0 runtime (2026-08-11)

On `df-macstudio-m2` with **ax-engine 6.15.0**
(`28dbcd252331f8a0eca9829609f2975a1b4be6a8`), support tier `mlx_preview`, and
`AX_ENGINE_2BIT_EXPERIMENTAL=1` / `AX_ENGINE_3BIT_EXPERIMENTAL=1`:

| Probe | Result |
| --- | --- |
| Model load + `/v1/models` | Pass (`deepseek_v4`, buffers bound) |
| Chat non-stream | Exact `AXQ2 OK` |
| Chat stream | `STREAM OK` + `[DONE]` |
| Context retrieval (~1.3k tokens) | Exact `PURPLE-ELEPHANT-42` |

Evidence package: durable host certification archive
`deepseek-v4-flash-axq-axengine-v6150/2bit` (operator Ext4T tree; not in-repo).

### Scope and limits

- **Experimental** 2/3-bit track: not interchangeable with 4/6-bit certified product SKUs.
- Measured BPW exceeds the nominal class because of protected tensors and MTP.
- Quality scores measure **generation viability** on the development suites (non-empty /
  expected-overlap scorer), not BF16 retention or formal coding-suite unit tests.
- AX Engine admission for 2/3-bit remains behind experimental env gates even after
  the 6.15.0 runtime smoke pass.
- Chat API is the authoritative instruct surface; bare `/v1/completions` is untemplated.

## Tier 2 status

**Not certified.** MTP weights are present in the pack, but this certificate does **not**
authorize speculative-decode speedup or greedy exactness. No formal MTP A/B scoreboard on
`df-macstudio-m2` (or any host) has closed the acceleration gates for this revision.

Product default remains direct decode until a revision-bound Tier 2 certificate exists.

## Related

- Sibling 3-bit: [deepseek-v4-flash-axq3-tier1.md](deepseek-v4-flash-axq3-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [deepseek-v4-flash-axq2-tier1.json](deepseek-v4-flash-axq2-tier1.json).
