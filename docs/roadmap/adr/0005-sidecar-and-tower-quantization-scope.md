# ADR-0005 — MTP sidecar, vision tower, and per-expert MoE scope

- **Status:** accepted
- **Date:** 2026-08-06

## Context

Three deferred items share a shape: tensors currently protected (byte-preserved
or BF16) because the runtime story or the evaluation story was missing.

- **MTP sidecars** are byte-preserved external artifacts; quantizing them
  shrinks packs but the quantized layout must be executable by AX Engine's MTP
  path.
- **Vision towers** (Qwen3-VL) are BF16-protected; text-only evals cannot
  certify vision degradation.
- **Per-expert MoE precision**: packed expert stacks quantize as fused switch
  modules with one precision per group; finer splits need MLX-LM-side support.

## Decision

1. **MTP sidecar quantization** proceeds as a *separate artifact*: quantized
   sidecar with its own manifest, checksum, and bits metadata
   (`mtp_sidecar_bits` already exists as a metadata channel). Byte-preserved
   stays the default; the quantized sidecar is opt-in and gated on an AX
   Engine runtime check proving the layout executes with MTP enabled. The
   primary pack's evidence chain is unchanged either way.
2. **Vision-tower quantization** is gated on evidence, not implementation:
   before any tower tensor drops below BF16, the eval suite must include
   vision tasks whose scores the release gates consume, and the probe must
   cover tower tensors with the same per-tensor measured discipline as text.
   Patch-embedding and merger tensors get protection floors by default.
3. **Per-expert MoE precision** is explicitly *not built* in AXQuant. We track
   the MLX-LM capability and revisit when upstream can execute unfused
   per-expert layouts. AXQuant does not fork or patch MLX-LM at runtime.

## Consequences

- Sidecar quantization can land as soon as AX Engine's check passes, without
  touching flagship evidence.
- Vision quantization is deliberately slow: eval-suite authorship
  (clean-room, like the existing 60-task suite) is the long pole, and that is
  accepted — an unmeasured vision regression is worse than a big BF16 tower.
- MoE granularity stays coarse until upstream moves; this is recorded so the
  gap is a known dependency, not a forgotten TODO.
