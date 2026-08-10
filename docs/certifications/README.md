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
| [Gemma 4 12B AXQ 6-bit](gemma4-12b-axq6-tier1.md) | main@`d0a1a932` (IT rebuild) | [Certified](gemma4-12b-axq6-tier1.md) | [Not Certified](gemma4-12b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 26B-A4B AXQ 6-bit](gemma4-26b-a4b-axq6-tier1.md) | main@`4a62bf66` | [Certified](gemma4-26b-a4b-axq6-tier1.md) | [Not Certified](gemma4-26b-a4b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 31B AXQ 6-bit](gemma4-31b-axq6-tier1.md) | main@`f024707a` | [Certified](gemma4-31b-axq6-tier1.md) | [Not Certified](gemma4-31b-axq6-tier1.md#tier-2-status) |

**Gemma 4 policy:** public certification is **AXQ 6-bit only** (Tier 1 listed above; Tier 2
in progress on 6-bit). Gemma AXQ 4-bit is **out of the public certification index** — historical
records remain under this directory for audit trail only and must not be cited as active
certificates.

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
- [12B 6-bit Tier 1 JSON](gemma4-12b-axq6-tier1.json)
- [26B-A4B 6-bit Tier 1 JSON](gemma4-26b-a4b-axq6-tier1.json)
- [31B 6-bit Tier 1 JSON](gemma4-31b-axq6-tier1.json)

See [flagship certification](../flagship-certification.md) for the two-tier policy and claim
boundaries (default route vs formal acceleration route; decode-heavy vs short-answer).
