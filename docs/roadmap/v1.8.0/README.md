# AXQuant 1.8.0 planning suite

Internal product and engineering documents for the 1.8.0 release. Nothing
here is release evidence, a quality claim, or a certification statement.

1.8.0 makes AXQ the **certified Apple Silicon deployment standard**. It does
not reopen the 2.x method-federation proposal, and it does not ship CUDA.

| Document | Purpose |
| --- | --- |
| [`prd.md`](prd.md) | Product requirements, naming decision, acceptance |
| [`tech-spec.md`](tech-spec.md) | Workstreams, modules, schema impact |
| [`adr/0007-axq-standard-layers.md`](adr/0007-axq-standard-layers.md) | B→D→C→A layers; Apple-only boundary |
| [`adr/0008-public-pack-identity.md`](adr/0008-public-pack-identity.md) | `4bit`/`6bit` SKUs vs measured-BPW claims |
| [`adr/0009-one-deployment-budget.md`](adr/0009-one-deployment-budget.md) | Weights + KV share one memory budget |
| [`../../certification-spec-v1.0.md`](../../certification-spec-v1.0.md) | Public Certification Specification v1.0 |

## Reading order

1. This README.
2. `prd.md` — what 1.8.0 is and is not.
3. ADR-0008 — naming (the contested decision).
4. ADR-0007 and ADR-0009 — architecture boundaries.
5. `tech-spec.md` — how to implement.

## Inherited ground rules

- ADR-0001 freeze discipline still governs every schema and campaign.
- ADR-0006 still excludes NVIDIA formats from the MLX certification narrative.
- AXQ-001 still forbids implementation-level dependence on competing toolkits.
- Formal performance evidence remains authorizing only on `df-macbookpro-m5`.

The existing completion-program suite (`docs/roadmap/prd.md` and ADRs
0001–0006) stays in force for flagship closure and algorithm work. 1.8.0 is
a **release program** layered on that suite, not a replacement.
