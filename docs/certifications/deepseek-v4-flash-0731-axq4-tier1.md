# DeepSeek V4 Flash-0731 AXQ 4-bit — Studio evaluation

**Verdict:** **not certified** for checkpoint Tier 1 on `df-macstudio-m2` (2026-08-16).

Local pack: group-128 affine, 154 GB, measured 4.347 BPW. Recipe
`examples/deepseek-v4-experimental-4bit-g128-v0.1.yaml`.

Factory 15+15 chat QA: coding 0.500 / general 0.933 / decode 27.889 tok/s.
Formal 76+44 generation-viability suite not closed. Group-32 4-bit (179 GB)
OOMed generate (172 GB Metal) and **cannot be certified** on this 192 GB host.

Hub publish pending. Machine-readable:
[deepseek-v4-flash-0731-axq4-tier1.json](deepseek-v4-flash-0731-axq4-tier1.json).
