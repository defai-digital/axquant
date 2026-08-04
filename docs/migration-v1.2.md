# Migration guide: v1.1.x → v1.2.0

v1.2.0 is a hardening release: it closes the calibration evidence chain end to end, fixes
Qwen3-Next fused-expert classification, and tightens capture and Hub-source provenance.
This page lists every breaking or behavior-changing item and what to do about it.

## Evidence-chain hardening

- AWQ/GPTQ analysis records the activation-capture manifest digest, tokenized-cache manifest
  digest, cache key, and calibration dataset ID in `CalibrationEvidence.metadata`.
- The planner carries those bindings unchanged. Conversion rejects a measured plan if the loaded
  capture differs, and packages `activation_capture_manifest.json` with the checkpoint.
- Publication preparation and release audit independently revalidate the packaged capture
  manifest. A capture from the right model but a different cache is no longer interchangeable.

## Qwen3-Next fused experts

`switch_mlp` and `switch_glu` 3-D weights now classify as `expert` in the registered family
adapter, matching the generic inspector and MLX-LM fused-module path. **Artifacts made before
this fix may have retained most expert weights at BF16 and must be regenerated.** Inspection now
downgrades a supported MoE checkpoint to inventory-only if fused-expert classification coverage
ever drifts again.

Packed gate/up source tensors that split into two MLX runtime modules now remain unmatched until
both runtime modules are visited. Packed/fused expert paths are affine-only; AWQ/GPTQ refinement
on these 3-D modules is rejected before conversion.

## Hub BF16 source preparation

`scripts/hf_to_mlx_bf16.py` now requires `--revision <40-character commit SHA>`, passes that
revision to `snapshot_download`, converts into a sibling staging directory, and atomically
renames only after a usable checkpoint exists. It refuses to overwrite an existing output and
writes `axquant_source.json` with the immutable source identity. Update automation that previously
relied on a floating Hub branch or silent output reuse.

## Capture immutability and resume hardening

- Completed capture directories are immutable: a capture will not overwrite a directory
  containing `completion.json`; choose a new output path.
- Every committed resume chunk is validated against checkpoint row accounting; only uncommitted
  crash output is discarded. Inconsistent checksums across entries sharing one shard are
  rejected. Resume or recapture into a new output directory when one of these gates fires.

## New and changed CLI surface

- `analyze` gains `--capture-points`: plain dense backbones expose logits only, so measured
  probing on non-multimodal layouts (MiniCPM5, Qwen3 dense, Mistral, …) must pass
  `--capture-points output`. The default `("output", "hidden")` fails closed with a hint on
  those layouts.
- `refine` gains `--lm-head-floor`, matching `plan`; the governed `--lm-head-floor 8bit` path
  still requires measured support.
- Runtime metadata for copied (byte-preserved) MTP sidecars now emits a structured
  `mtp_sidecar_bits` contract instead of an implied precision.

## Plan naming

Mixed-precision plans are labeled by their target BPW instead of inheriting a flat storage-class
name. A `4bit` pack name remains a storage-class label, not a claim that every tensor is 4-bit;
manifests carry the exact per-tensor distribution.

## No action needed

- Plan/sensitivity/inventory schema versions are unchanged.
- `--allow-unmeasured` and evidence-gating semantics are unchanged.
- Conversion remains fail-closed on uncovered modules; fused MoE expert groups remain
  affine-only.
