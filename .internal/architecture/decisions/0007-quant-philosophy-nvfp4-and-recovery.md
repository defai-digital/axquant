# ADR 0007: Quantization Philosophy — NVFP4 Lessons and Fine-Tune Boundary

**Status:** Accepted  
**Date:** 2026-08-01  
**Decisions:** AXQ-028, AXQ-029  
**PRD:** `.internal/product/quant-philosophy-prd.md`  
**Tech spec:** `.internal/engineering/quant-philosophy-technical-specification.md`

## Context

Three external references shape pressure on AXQuant’s design:

1. **mlx-optiq** — MLX-native mixed-precision PTQ plus fine-tune and serve.
2. **NVFP4** — NVIDIA Blackwell 4-bit float (E2M1) with 16-element micro-blocks, FP8
   (E4M3) per-block scales, and a tensor-level FP32 scale; native Tensor Core acceleration.
3. **AXQuant today** — Tensor-level bit allocation (primarily bit width), affine/AWQ/DWQ under
   the public MLX-LM contract, hard role floors, fail-closed conversion, no training path.

The product question is whether AXQuant should (a) adopt FP4-like formats, (b) add fine-tuning
as a first-class product, and (c) how to improve PTQ quality without abandoning auditability.

## Decision summary

| ID | Decision |
| --- | --- |
| **AXQ-028** | Learn NVFP4’s **scale / block / dynamic-range philosophy**; do **not** adopt NVFP4 as a weight format. Expand planning to **bits × group_size × method** with first-class strategy metadata. |
| **AXQ-029** | Fine-tune / domain SFT / full QAT are **out of product scope**. Optional **quantization recovery** may be added later under strict provenance; the default convert path stays pure PTQ. |

## AXQ-028: NVFP4 lessons without NVFP4 format

### Decision

**Import as philosophy, not as format:**

| NVFP4 idea | AXQuant translation |
| --- | --- |
| Smaller scale blocks (16 vs 32) | Prefer smaller **group_size** (32 before 64/128) for sensitive tensors when budget allows |
| Two-level scaling | Keep and strengthen **group affine scales** + **AWQ channel scales** + optional tensor-level normalize/clip (DWQ); record strategy on each allocation |
| Near-FP8 accuracy at ~4-bit | Achieve via **mixed precision + fine groups + method selection**, not uniform E2M1 |
| Hardware FP4 matmul | **Out of scope** on Apple Silicon / MLX portable weights |

**Configuration space (normative for planner/probe):**

```text
candidate = (bits, method, group_size)
storage_bpw = bits + 32/group_size   # BF16 => 16
```

Uniqueness of sensitivity candidates is `(bits, method, group_size)` (BF16: no group).

**Rejected alternatives:**

- Emulate NVFP4 packing in MLX without native kernels (slow, non-portable, breaks AXQ-004).
- Collapse candidates by bits only (current behavior discards useful group/method diversity).
- Per-channel arbitrary bit widths outside hardware support (still deferred under AXQ-007).

### Consequences

- Probe cost grows with the Cartesian product of bits × methods × group sizes; defaults stay
  conservative; users opt into wider grids.
- Plans become more expressive at fixed schema major version via additive fields.
- Competitive narrative: “MLX-portable mixed precision with NVFP4-inspired scale granularity,”
  never “NVFP4 on Mac.”

## AXQ-029: Fine-tune boundary and optional recovery

### Decision

**Default path remains pure PTQ.** AXQuant does not train new capabilities.

| Category | Status |
| --- | --- |
| Domain SFT / DPO / chat or coding fine-tune | **Forbidden** in AXQuant product scope |
| Full-network QAT as default | **Forbidden** |
| Optional post-PTQ **recovery** (scale/bias, calibration-only, optional LoRA-merge) | **Allowed in QP2** with recovery manifest |
| LoRA rank guidance from sensitivity | Deferred; must not block QP0–QP1 |

Recovery, when implemented:

1. is **opt-in** (`recover` or explicit flag — never implied by `quantize`);
2. uses **calibration / recovery datasets with digests**, not open-ended user chat logs by default;
3. may only claim **retention restoration** relative to BF16 / pre-recovery quantized baseline;
4. fails closed if provenance is incomplete.

**Rejected alternatives:**

- Match mlx-optiq’s fine-tune + Lab as a mainline feature (duplicates product identity; provenance
  explosion; conflicts with “does not add learned capabilities”).
- Silent weight updates during convert without a recovery artifact.

### Consequences

- Marketing and README keep the non-training claim.
- QP2 is the earliest recovery ship window; QP0–QP1 must not invent partial training APIs.
- External training tools remain the path for domain adaptation; AXQuant consumes
  revision-pinned BF16 checkpoints.

## Supersession / interaction

- **Does not supersede** AXQ-007 floors; may only choose among legal candidates.
- **Does not supersede** AXQ-008 evidence kinds; multi-group priors remain `architecture_prior`.
- **Amends deferred list** in the decision register: group/method granularity and recovery move
  from “deferred decisions” into phased program QP0–QP2.
- Extends clean-room provenance (AXQ-027 / ADR 0006) as new algorithms land (Pareto ladder,
  recovery update rules).

## Implementation authority

Normative contracts live in
`.internal/engineering/quant-philosophy-technical-specification.md`.
Product milestones live in `.internal/product/quant-philosophy-prd.md`.
