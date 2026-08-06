# ADR-0006 — Out-of-scope tracks: NVIDIA formats and weight-mutating recovery

- **Status:** accepted
- **Date:** 2026-08-06

## Context

Two directions keep resurfacing and need a recorded "no":

1. **NVIDIA hardware formats.** Competitors gain throughput from
   hardware-native low-precision formats (NVFP4 on Blackwell tensor cores,
   FP8 on Hopper). AXQuant targets Apple Silicon through MLX and AX Engine
   with portable affine packing; there is no NVFP4 execution path on this
   stack, and a format the runtime cannot execute natively accelerates
   nothing.
2. **Weight-mutating recovery.** `recover` records identity-copy provenance
   only (AXQ-029 QP2 scope: calibration-only, no domain SFT/DPO, never implied
   by convert). Extending it to mutate weights would blur the line between
   quantization evidence and post-hoc repair.

## Decision

1. **NVIDIA formats are a separate product track, if ever.** They get their
   own artifacts, naming, evidence chain, and documentation, and never appear
   in the MLX certification narrative or share its manifests. Within this
   roadmap they are limited to feature-level competitive analysis. What we
   *do* adopt from that world is the principle of hardware-aligned formats,
   implemented for Apple via the measured kernel-latency cost model
   (ADR-0003).
2. **`recover` stays provenance-only.** Anything that changes packed weights
   after conversion goes through the refinement path (`refine-awq-dwq`
   ladder, convert-time refinement), which is already calibration-bound and
   evidence-versioned. Documentation and CLI help must not imply `recover`
   repairs weights.

## Consequences

- No engineering effort is budgeted for CUDA/TensorRT paths in Phases 0–4.
- A future NVIDIA track requires its own PRD and would not reuse this
  program's gates.
- Users wanting post-conversion quality repair are pointed at refinement;
  the `recovery-rank` output remains useful as its targeting input.
