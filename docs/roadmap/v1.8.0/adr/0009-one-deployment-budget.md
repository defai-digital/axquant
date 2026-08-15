# ADR-0009 — One deployment budget for weights and KV

- **Status:** accepted
- **Date:** 2026-08-14
- **Release:** AXQuant 1.8.0
- **Supersedes (for 1.8.0 scope):** the v2.x / CUDA framing in
  `docs/prd/weight-kv-joint-optimization.md` and
  `docs/adr/ADR-002-weight-kv-joint-optimization.md` Decision 7
- **References:** ADR-0003, ADR-0007

## Context

`axquant plan` currently allocates weight precision first, then may attach a
KV plan (`--kv-cache`). The two decisions do not share one user-facing
memory budget. Long-context serving memory is approximately

```text
M_total ≈ M_weights + M_KV + M_activation + M_runtime_reserve
```

A 4-bit weight plan that still needs BF16 KV can miss an 18 GB MacBook
target. Independently compressing KV can also be the better quality trade.

`docs/prd/weight-kv-joint-optimization.md` already describes the full
joint optimizer as a 2.x product. 1.8.0 does not implement that entire
design. It extracts the part that makes AXQ a **deployment** standard:
one constraint, one breakdown, fail-closed.

## Decision

Add a thin **deployment planner** above the existing weight and KV
allocators. Do not rewrite `plan_quantization` or the KV allocators.

```text
                 axquant optimize
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     existing weight planner    existing KV planner
            │                         │
            └────────────┬────────────┘
                         ▼
              joint memory accounting
                         │
                         ▼
         fail closed if weights + KV + reserve
              exceed the requested budget
```

Normative constraint:

```text
weight_bytes + kv_bytes + explicit_runtime_reserve  <=  requested_budget
```

Bound inputs for 1.8.0: hardware profile, runtime, context length, batch
size, workload profile, quality floor, MTP policy. KV remains static
per-layer precision from the existing measured or prior path.

Evidence levels stay distinct. A plan built from architecture priors cannot
be labeled measured or certified.

`target_mode` (`balanced` / `quality` / `low-memory` / `speed`) must change
objective weights, not only be copied into the result.

Out of 1.8.0 for this ADR:

- per-layer interaction-aware joint search (weight-KV PRD Phase 4)
- W4A4 / activation quantization
- dynamic per-token KV
- CUDA / ROCm runtime adapters

## Consequences

- New schema kind for a deployment plan / memory breakdown (new
  `schema_version`, ADR-0001).
- New CLI `axquant optimize` orchestrates inspect / existing plan / KV /
  accounting. It does not replace `plan` or `convert`.
- Peak resident memory is still validated on a real build when the user
  asks for a measured or certified result. The static byte sum is a
  planning gate, not a substitute for runtime-check.

## Alternatives rejected

- **Keep weight and KV independent.** Fails the 1.8.0 product sentence
  (“one memory budget, not just bits”).
- **Merge KV into `planner.py` option ladders.** Couples two memory models
  and makes later activation work harder.
- **Ship the full 2.x joint optimizer.** Too large for 1.8.0; the
  allocators already exist and should be reused.
