# Qwen 3.6 27B development evidence status

**Name:** Qwen 3.6 27B development evidence status  
**Not:** “E5 complete” or “certification blocked only by MTP.”

Project phase E5 (see `expansion-implementation-plan.md`) also covers other official Qwen 3.6
sizes and first head-to-head publication; E6 is the first non-Qwen family certification wave.
This note only describes **Qwen 3.6 27B development smoke evidence** gathered on Apple Silicon
(M5 Max host `mbp-m5` / `e5-evidence-m5`, 2026-08-02).

## Bottom line

E5-style **development smoke** for 27B largely ran. The **release evidence chain is not closed**.
Do **not** merge refinement, conversion, quality, and MTP into one release lineage.

## Separate candidates (do not cross-bind)

| Slice | Artifact / value | Notes |
| --- | --- | --- |
| Plan → convert (M5) | plan effective **5.9999855** BPW → converted measured **6.0000659** BPW | 19 GiB, 4 main shards, quantizer records all success / no fallback; manifests present for AXQuant paths |
| Proxy refinement | selected **6.761280** BPW (`cand-0000-004`) | `selection_basis=proxy`, `evidence_label=proxy-development`; history `measured_bpw` / `measured_loss` all null. “Converged” = proxy stop, not measured refine |
| Candidate-only quality (6.0 BPW convert) | Coding **0.9667** (14 full + one 0.5; syntax-valid **0.9333**); Reasoning / JSON-tool / Instruction **1.0** | No BF16 reference comparison → **not** a formal quality gate pass. Do not claim “coding 15/15 score 1.0” |
| AXQ-026 MTP speed | **1.1911828×** on **5.3 BPW** AXQ-026 candidate | **Not** the 6.000066 BPW M5 convert. Formal audit requires MTP on the **same** candidate |
| AX Engine | `ax_engine_runtime_ready=false` | Convert log: skipped AX Engine manifest (`ax-engine-bench` missing); no `model-manifest.json` |
| Feasibility | `ready-for-conversion` | Useful static readiness only |
| Local toolkit gates | 443 tests, Ruff lint, mypy | Format check must stay clean |

## Confirmed development evidence

- Sensitivity: 1199 tensors, `measured_development`
- Plan: 5.9999855 effective BPW (target 6.0)
- Conversion: 6.0000659 measured total BPW; integrity of produced AXQuant files / shards OK
- MLX-LM generation smoke: pass (`mlx_lm generate`)
- Wheel package present in that run’s `dist/`

## Still required for certification

- Measured refine-measure (not proxy-only selection)
- Convert of the **selected** measured candidate (or re-bind selection to the 6.0 BPW plan deliberately)
- Dual-profile **reference vs candidate** quality comparison on that same convert
- AX Engine doctor / runtime / MTP A/B on that same convert (speed ≥ 1.20× if claiming MTP)
- Pareto, hardware registry, compatibility matrix
- Full M0–M8 release audit on a supported host

## Suggested status sentence

> Qwen 3.6 27B **development smoke** is largely complete; **formal certification is not**.
> AXQ-026’s measured MTP **1.1912× &lt; 1.20×** is a known failing number on the **5.3 BPW**
> lineage only. Refine-measure, same-candidate quality comparison, AX Engine evidence, Pareto,
> hardware registry, compatibility matrix, and M0–M8 audit still need to be produced for one
> bound release candidate.
