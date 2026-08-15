# ADR-0007 — AXQ standard layers and Apple Silicon boundary

- **Status:** accepted
- **Date:** 2026-08-14
- **Release:** AXQuant 1.8.0
- **Supersedes:** none
- **References:** ADR-0001, ADR-0003, ADR-0006

## Context

A reviewer proposed turning the planner into a method-agnostic multi-backend
optimizer (AXQ, MLX, AutoRound, later GPTQ/AWQ/GGUF/FP8/NVFP4) with
heterogeneous per-module formats. Independent reviews (Codex `gpt-5.6-sol`
and Qoder `Qwen3.8-Max`) rejected that as a 1.x/2.x architecture.

The current product already chooses `(bits, group, method)` inside one
portable affine packing contract (`planner.py`, `converter.py`). Crossing to
CUDA, or treating AutoRound as a second physical format, would change the
artifact, the certificate, and the runtime story.

The maintainer accepted a narrower north star for 1.8.0:

> AXQuant finds the best certified, workload-aware quantization strategy
> within a runtime-native, portable artifact contract on Apple Silicon.

## Decision

1.8.0 sequences four layers. Later layers may not ship before earlier ones
are specified and testable.

| Layer | Role | 1.8.0 meaning |
| --- | --- | --- |
| **B. Evidence protocol** | Defines the standard | Public Certification Spec v1.0; Tier 1 / scoped Tier 2; measured BPW; fail-closed provenance; `axquant verify-cert` |
| **D. Runtime contract** | Locks the standard | Frozen portable affine U32 pack that AX Engine and stock MLX-LM both load without repack |
| **C. Planner UX** | Acquires users | `axquant optimize` spends one memory budget across weights + KV under quality, context, and workload constraints |
| **A. Pack catalog** | Follows trust | AutomatosX Hub packs named and certified under ADR-0008 |

Apple Silicon / MLX / AX Engine remains the only shipping platform in 1.8.0.
ADR-0006 stands: NVIDIA formats, CUDA, TensorRT, vLLM adapters, GGUF export,
and NVFP4/FP8 packs are a **separate product track** with their own artifacts,
names, and certificates. They do not appear in 1.8.0 launch scope.

The planner should stay free of MLX object types in new 1.8 modules so a
future NVIDIA track is not structurally blocked. That is hygiene, not a
commitment to staff `runtime/cuda` in this release.

## Consequences

- 1.8.0 success is a **quotable standard** (spec + verify + interchange), not
  a new quantizer brand or a wider method menu.
- Existing algorithm work (act-order GPTQ, interaction optimization,
  2/3-bit hardening) stays on the completion-program roadmap and is not a
  1.8.0 launch gate.
- Proposed `docs/adr/ADR-002-weight-kv-joint-optimization.md` Decision 7
  (`runtime/cuda`, `runtime/rocm`) is not adopted for 1.8.0. ADR-0009 covers
  the weight+KV budget extract.

## Alternatives rejected

- **Method-agnostic backend federation.** Breaks the one-pack contract,
  explodes certification, and contradicts ADR-0002/0003/0006.
- **Same AXQ affine packs on CUDA.** A layout the runtime cannot execute
  natively accelerates nothing (ADR-0006).
- **Runtime-neutral core + CUDA adapter in 1.8.0.** Today's
  `HardwareProfile` / `PlanRequest` are AX Engine shaped
  (`schema/planning.py`). Neutralizing them is a schema rewrite, not a flag.
