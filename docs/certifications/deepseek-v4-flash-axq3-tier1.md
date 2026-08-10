# DeepSeek V4 Flash AXQ 3-bit (experimental) — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 (**experimental low-bit**) on 2026-08-10
on host **`df-macstudio-m2`** (Apple M2 Ultra, 192 GB).

This certificate covers
[`AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-3bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-3bit)
commit
[`5f00e2dff2f7f16cb1607109538749527a4ee836`](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-3bit/tree/5f00e2dff2f7f16cb1607109538749527a4ee836).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `DeepseekV4ForCausalLM` (MoE Flash) |
| Product class | `3bit-experimental` |
| Source | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` |
| Candidate manifest SHA-256 | `700034a28c0fd13182abac926ca9bc5e8a83de3a9dbe5ff33e70ed5421281545` |
| Measured main BPW | `4.110998463480101` |
| Weight bytes | `149,706,329,385` |
| Certification host | `df-macstudio-m2` |
| MTP in pack | `True` (acceleration **not** claimed) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured main BPW | recorded | `4.110998463480101` | Pass |
| Weight bytes | recorded | `149,706,329,385` | Pass |
| Agent-coding generation viability | ≥ `0.90` | `1.0000` (76 tasks) | Pass |
| General generation viability | ≥ `0.90` | `0.9318` (44 tasks) | Pass |
| MLX generation | suite load + decode | Pass | Pass |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.1`.

### Scope and limits

- **Experimental** 2/3-bit track: not interchangeable with 4/6-bit certified product SKUs.
- Measured BPW exceeds the nominal class because of protected tensors and MTP.
- Quality scores measure **generation viability** on the development suites (non-empty /
  expected-overlap scorer), not BF16 retention or formal coding-suite unit tests.
- **MTP Tier 2 / speculative speedup is not certified.**
- AX Engine admission for 2/3-bit remains behind experimental env gates.

## Related

- Sibling pack under this directory
- Certification index: [README.md](README.md)

Machine-readable: [deepseek-v4-flash-axq3-tier1.json](deepseek-v4-flash-axq3-tier1.json).
