# AXQuant completion & improvement roadmap (internal planning docs)

This directory holds the planning suite for closing the items README lists as
"Still incomplete" and for the algorithm/runtime improvements identified in the
2026-08 competitive review. These are internal development documents — nothing
here is release evidence, a quality claim, or a certification statement.

| Document | Purpose |
| --- | --- |
| [`prd.md`](prd.md) | Product requirements: what we are closing, why, and the acceptance criteria |
| [`tech-spec.md`](tech-spec.md) | Engineering design per workstream, with module touch points and schema impact |
| [`implementation-plan.md`](implementation-plan.md) | Phased delivery plan (Phase 0–4) with entry/exit criteria and evidence impact |
| [`adr/`](adr/) | Architecture decision records governing the contested design choices |

## Reading order

1. `prd.md` — scope and acceptance criteria.
2. `adr/0001` — the freeze discipline that constrains everything else.
3. `implementation-plan.md` — sequencing.
4. `tech-spec.md` + remaining ADRs — per-workstream detail when starting that workstream.

## Ground rules inherited from the project

- Competitive comparison with other MLX quantization toolkits stays at the
  feature level only; no reading or porting of their implementations (AXQ-001).
- Formal performance evidence is authorizing only on the frozen formal host
  (`mbp-m5`). Everything else proves reproduction, not performance.
- Architecture-prior analysis, smoke probes, and manual plans remain
  non-release development evidence.
