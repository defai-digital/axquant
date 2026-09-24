# Cyber-Tiel Coder 35B-A3B AXQ-MXFP4-MTP — scoped Tier 2 evaluation

**Verdict:** **not certified** for MTP acceleration on `df-macstudio-m2` (2026-09-24).

Measured with AX Engine **7.5.4** (`eeb40413…`) under the Qwen 3.6 MoE exact-MTP
certification profile on the in-repo development suites (15 agent-coding + 15
general prompts, the same `data/eval` files bound by the checkpoint Tier 1
record). Agent-coding clears all three gates; general-long fails exactness and
both speed floors. Tier 2 requires every authorizing profile to pass.

| Profile | Exactness | Weighted decode | Prompt median | Result |
| --- | ---: | ---: | ---: | --- |
| agent-coding | pass | 1.21× | 1.21× | Pass |
| general-long | fail (2/2 divergent trials) | 1.09× | 1.09× | Fail |

A `df-macbookpro-m5` comparison run of the **same engine binary** shows the same
pattern with a larger agent-coding speedup (1.3778×) and the same general-long
inexactness — the general-long gap is a pack property (the MXFP4 multi-token
verify path is not row-exact against direct decode for this pack), not a
factory-host defect. Tier 1 checkpoint certification on this host is unchanged.
Product default remains direct fallback.

Machine-readable: [cyber-tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json](cyber-tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json).
Evidence: [evidence/tiel-mxfp4-tier2-20260924](evidence/tiel-mxfp4-tier2-20260924).
