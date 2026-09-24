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
exactness on both profiles, ruling out a factory-host defect. An
engine-version bisect (`v7.1.1`–`v7.5.4`, evidence
`comparison-df-macbookpro-m5/engine-version-bisect/`) shows why: engines
≤ v7.4.0 never load this pack's MTP head (its sidecar declares
`raw_hf_delta` while the norms are already converted; only the 7.5.x
hash-bound compatibility shim loads it), and every 7.5.x build that loads it
is inexact for this pack on both profiles — including agent-coding, where the
certified AXQ control pack stays exact. The inexactness is pack-specific on
top of the engine-wide general-long regression. The documented path to
certification is a metadata-corrected revision (`mtp_norm_layout
mlx_multiplier`, byte-identical weights) plus an engine row-exactness fix.
Tier 1 checkpoint certification on this host is unchanged. Product default
remains direct fallback.

Machine-readable: [tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json](tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json).
Evidence: [evidence/tiel-mxfp4-tier2-20260924](evidence/tiel-mxfp4-tier2-20260924).
