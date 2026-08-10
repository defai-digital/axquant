# AXQuant checkpoint certifications

This directory records public, revision-bound AXQuant certificates. A checkpoint-tier certificate
covers the exact artifact named by its record; it does not promote sibling models, other
revisions, or unscoped runtime claims.

| Checkpoint | Edition | Tier 1 (checkpoint) | Tier 2 (MTP acceleration) |
| --- | --- | --- | --- |
| [Qwen 3.6 27B AXQ 6-bit](qwen36-27b-axq6-tier1.md) | v3 | [Certified](qwen36-27b-axq6-tier1.md) | [Certified](qwen36-27b-axq6-tier2.md) |
| [Qwen 3.6 27B AXQ 4-bit (5.6 BPW)](qwen36-27b-axq4-tier1.md) | main@`f44a9eee` | [Certified](qwen36-27b-axq4-tier1.md) | [Certified](qwen36-27b-axq4-tier2.md) |
| [Qwen 3.6 35B-A3B AXQ 4-bit](qwen36-35b-axq4-tier1.md) | main@`a549387d` | [Certified](qwen36-35b-axq4-tier1.md) | [Certified](qwen36-35b-axq4-tier2.md) |
| [Qwen 3.6 35B-A3B AXQ 6-bit](qwen36-35b-axq6-tier1.md) | main@`7b9ff47a` | [Certified](qwen36-35b-axq6-tier1.md) | [Certified](qwen36-35b-axq6-tier2.md) |
| [Gemma 4 12B AXQ 4-bit](gemma4-12b-axq4-tier1.md) | main@`6d124af8` (IT rebuild) | [Certified](gemma4-12b-axq4-tier1.md) | [Not Certified](gemma4-12b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 12B AXQ 6-bit](gemma4-12b-axq6-tier1.md) | main@`d0a1a932` (IT rebuild) | [Certified](gemma4-12b-axq6-tier1.md) | [Not Certified](gemma4-12b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 26B-A4B AXQ 4-bit](gemma4-26b-a4b-axq4-tier1.md) | main@`85b0a78a` | [Certified](gemma4-26b-a4b-axq4-tier1.md) | [Not Certified](gemma4-26b-a4b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 26B-A4B AXQ 6-bit](gemma4-26b-a4b-axq6-tier1.md) | main@`4a62bf66` | [Certified](gemma4-26b-a4b-axq6-tier1.md) | [Not Certified](gemma4-26b-a4b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 31B AXQ 4-bit](gemma4-31b-axq4-tier1.md) | main@`bc2de70b` | [Certified](gemma4-31b-axq4-tier1.md) | [Not Certified](gemma4-31b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 31B AXQ 6-bit](gemma4-31b-axq6-tier1.md) | main@`f024707a` | [Certified](gemma4-31b-axq6-tier1.md) | [Not Certified](gemma4-31b-axq6-tier1.md#tier-2-status) |
| [Qwen3-Coder-Next AXQ 4-bit](qwen3-coder-next-axq4-tier1.md) | main@`53dce509` | [Certified](qwen3-coder-next-axq4-tier1.md) | N/A (no MTP) |
| [Qwen3-Coder-Next AXQ 6-bit](qwen3-coder-next-axq6-tier1.md) | main@`c6f3ae55` | [Certified](qwen3-coder-next-axq6-tier1.md) | N/A (no MTP) |
| [DeepSeek V4 Flash AXQ 2-bit (exp.)](deepseek-v4-flash-axq2-tier1.md) | main@`e22b117a` | [Certified](deepseek-v4-flash-axq2-tier1.md) | [Not Certified](deepseek-v4-flash-axq2-tier1.md) |
| [DeepSeek V4 Flash AXQ 3-bit (exp.)](deepseek-v4-flash-axq3-tier1.md) | main@`5f00e2df` | [Certified](deepseek-v4-flash-axq3-tier1.md) | [Not Certified](deepseek-v4-flash-axq3-tier1.md) |

**Gemma 4:** checkpoint **Tier 1** is certified for the AXQ 4-bit and 6-bit fused assistant-MTP
Hub packs (12B / 26B-A4B / 31B). **Tier 2 (MTP acceleration) is not certified** on any Gemma pack
while formal assistant-MTP exactness remains open on a released engine.

**Qwen3-Coder-Next:** hybrid MoE coding checkpoint with **no declared MTP**. Public certificates
are non-MTP direct-decode checkpoint Tier 1 only (size, matched uniform quality, MLX-LM load).

**DeepSeek V4 Flash:** experimental **2/3-bit** AXQ packs are checkpoint Tier 1 on
`df-macstudio-m2` (generation viability on development suites). Product classes remain
`*-experimental`; measured BPW exceeds the nominal class; MTP acceleration is not certified.

Machine-readable companions:

- [27B 6-bit Tier 1 JSON](qwen36-27b-axq6-tier1.json)
- [27B 6-bit Tier 2 JSON](qwen36-27b-axq6-tier2.json)
- [27B 6-bit Tier 2 evidence package](evidence/qwen36-27b-axq6-tier2/)
- [27B 4-bit Tier 1 JSON](qwen36-27b-axq4-tier1.json)
- [27B 4-bit Tier 2 JSON](qwen36-27b-axq4-tier2.json)
- [35B 4-bit Tier 1 JSON](qwen36-35b-axq4-tier1.json)
- [35B 4-bit Tier 2 JSON](qwen36-35b-axq4-tier2.json)
- [35B 6-bit Tier 1 JSON](qwen36-35b-axq6-tier1.json)
- [35B 6-bit Tier 2 JSON](qwen36-35b-axq6-tier2.json)
- [12B 4-bit Tier 1 JSON](gemma4-12b-axq4-tier1.json)
- [12B 6-bit Tier 1 JSON](gemma4-12b-axq6-tier1.json)
- [26B-A4B 4-bit Tier 1 JSON](gemma4-26b-a4b-axq4-tier1.json)
- [26B-A4B 6-bit Tier 1 JSON](gemma4-26b-a4b-axq6-tier1.json)
- [31B 4-bit Tier 1 JSON](gemma4-31b-axq4-tier1.json)
- [31B 6-bit Tier 1 JSON](gemma4-31b-axq6-tier1.json)
- [Coder-Next 4-bit Tier 1 JSON](qwen3-coder-next-axq4-tier1.json)
- [Coder-Next 6-bit Tier 1 JSON](qwen3-coder-next-axq6-tier1.json)
- [DeepSeek V4 Flash 2-bit Tier 1 JSON](deepseek-v4-flash-axq2-tier1.json)
- [DeepSeek V4 Flash 3-bit Tier 1 JSON](deepseek-v4-flash-axq3-tier1.json)

See [flagship certification](../flagship-certification.md) for the two-tier policy and claim
boundaries (default route vs formal acceleration route; decode-heavy vs short-answer).
