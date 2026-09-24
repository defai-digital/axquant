# Tiel Coder 35B-A3B AXQ-MXFP4-MTP — scoped Tier 2 evaluation

**Verdict:** **not certified** for MTP acceleration on `df-macstudio-m2` (2026-09-24).

Measured with AX Engine **7.5.4** (`eeb40413…`) under the Qwen 3.6 MoE exact-MTP
certification profile on the in-repo development suites (15 agent-coding + 15
general prompts, the same `data/eval` files bound by the checkpoint Tier 1
record). Greedy MTP-vs-direct exactness failed on both authorizing profiles, and
no profile clears both speed floors.

| Profile | Exactness | Weighted decode | Prompt median | Result |
| --- | ---: | ---: | ---: | --- |
| agent-coding | fail (1/2 divergent trials; 1 trial lost to a transient Metal GPU OOM) | 0.67× | 0.67× | Fail |
| general-long | fail (2/2 divergent trials) | 1.01× | 1.02× | Fail |

A `df-macbookpro-m5` comparison run of the **same engine binary** reaches
1.3729× (agent-coding) / 1.1986× (general-long) but still fails greedy
exactness on both profiles — the inexactness is a pack property (the MXFP4
multi-token verify path is not row-exact against direct decode for this pack),
not a factory-host defect. Tier 1 checkpoint certification on this host is
unchanged. Product default remains direct fallback.

Machine-readable: [tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json](tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json).
Evidence: [evidence/tiel-mxfp4-tier2-20260924](evidence/tiel-mxfp4-tier2-20260924).
