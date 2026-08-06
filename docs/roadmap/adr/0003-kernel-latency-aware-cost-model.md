# ADR-0003 — Measured kernel-latency cost model for the planner

- **Status:** accepted
- **Date:** 2026-08-06

## Context

The planner's objective (via `profiles.objective_for`) trades measured quality
sensitivity against abstract bit cost. On Apple Silicon, decode is usually
memory-bandwidth-bound, so bits are a decent proxy — but not a perfect one:
dequant path, group size, and layout determine which configurations have fast
kernels in MLX-LM and AX Engine. The competitive lesson from hardware-native
GPU formats (e.g. Blackwell NVFP4) is that formats must align with what the
runtime executes fast; on Apple, that alignment must come from measurement,
because we do not control the kernel roster.

Group-size candidates are already runtime-constrained
(`AX_ENGINE_EXECUTABLE_GROUP_SIZES`, `candidate_group_sizes` probing) — this
ADR extends alignment from *feasible* to *fast*.

## Decision

1. Add a kernel-latency measurement harness that times decode-shaped and
   prefill-shaped matmuls per (runtime, bits, group size, packing/method
   layout) on a named host, emitting a checksummed, hardware-scoped
   `kernel_latency` artifact bound to a hardware-registry entry — the same
   scoping rule as all performance evidence (authorizing only on `mbp-m5`;
   other hosts inform development).
2. The planner accepts an optional latency table. When present, the cost term
   for a candidate uses measured relative latency instead of abstract bits;
   when absent, behavior is bit-identical to today. The plan artifact records
   which cost model produced it.
3. Latency tables are inputs to planning, not quality evidence. Quality gates
   are unchanged; the table only re-ranks candidates *within* the
   quality-feasible set.

## Alternatives rejected

- **Analytical cost model** (bytes moved × bandwidth): cannot capture kernel
  quality differences between group sizes/layouts, which is the entire point.
- **Hard-coding "fast" configurations**: rots as runtimes evolve; a measured
  table regenerates per runtime release.
- **Inventing a new dtype/layout for speed**: without a shipped kernel in
  MLX-LM or AX Engine it accelerates nothing (the NVFP4 fallacy).

## Consequences

- Plans become host-scoped when latency-driven; a plan built with an `mbp-m5`
  table must record that scope, consistent with existing claims discipline.
- The harness gives AX Engine a concrete kernel wishlist: configurations that
  are quality-optimal but kernel-slow are visible in one report.
- Deterministic planning is preserved: the table is a frozen input artifact,
  not a live benchmark at plan time.
