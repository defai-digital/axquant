# Migration guide: v1.0.x → v1.1.x

v1.1.0 added GPTQ Hessian error compensation and completed the AWQ calibration path.
v1.1.1 hardens that surface (capture resume, compressed/sharded artifacts, GPTQ memory).
This page lists every breaking or behavior-changing item and what to do about it.

## Python API renames

- `axquant.converter.convert_model(awq_activations=...)` → `calibration_activations=`.
- `axquant.predicate.PlanPredicate(awq_activations=...)` → `calibration_activations=`.
- `PlanPredicate.awq_metadata` → `method_metadata` (now also carries GPTQ metadata).

The mapping shape is unchanged — `{module_path: activation array (rows, in_features)}` — and is
now shared by AWQ and GPTQ allocations. `load_capture_activations` now returns a
mapping-compatible `LoadedActivationCapture` carrying the verified manifest identity. High-level
AWQ/GPTQ probe and conversion paths reject a raw, unbound `dict`; load the artifact through the
public loader instead. Update call sites and any monkeypatching in downstream tests.

## Probe resume state invalidated

The measured-probe backend version moved from `axquant-mlx-isolated-probe-v4` through `v5` to
`v6`. `v6` includes the activation-capture digest in resume identity and calibration evidence.
Saved `--state` probe progress from older versions is rejected; rerun `analyze` from scratch
(reference passes are the only real cost).

## New and changed CLI surface

- New command: `capture-activations` — records checksum-bound per-module Linear input
  activations from a verified tokenized cache. Required input for AWQ/GPTQ conversion and
  measured AWQ/GPTQ probing.
- `analyze` gains `--calibration-activations <dir>`; `--methods` now accepts `awq` and `gptq`
  alongside `affine` and `dwq`.
- `convert` gains `--calibration-activations <dir>`. Plans containing AWQ or GPTQ allocations
  fail closed without it.
- `capture-activations` gains `--segment-batches` (default 8, resume granularity) and
  `--modules-per-shard` (default 1, artifact grouping) in v1.1.1.

## Capture artifact layout (v1.1.1)

- Capture is resumable: interrupted runs leave `capture_progress.json` plus per-segment chunks
  under `activations/.partial/`; rerunning the identical command resumes. A changed invocation
  (`--max-rows`, cache, model, budget, segments) against an existing progress file is rejected —
  use a fresh output directory.
- Final npz files are deflate-compressed. `load_capture_activations` handles both layouts.
- A capture is only loadable once `completion.json` exists; `load_capture_activations` fails
  closed on partial captures. New completion markers are atomically written and bind the semantic
  manifest digest; the loader validates every available legacy marker field during migration.
- `ActivationCaptureEntry` gained an optional `array_key` field (used when
  `--modules-per-shard N > 1` groups modules into `shard-NNNN.npz` archives). Old manifests
  without the field still validate; the schema version is unchanged.
- Post-v1.1.1 hardening makes completed capture directories immutable, validates every committed
  resume chunk against checkpoint row accounting, discards only uncommitted crash output, and
  rejects inconsistent checksums when several entries share one shard. Resume or recapture into
  a new output directory when one of these gates fires.
- `capture-activations --revision <SHA>` is resolved at the exact revision bound by the tokenized
  cache. When an MLX BF16 source contains `axquant_source.json`, its model ID and immutable
  revision must also match that cache.

## Evidence-chain hardening (next patch after v1.1.1)

- AWQ/GPTQ analysis records the activation-capture manifest digest, tokenized-cache manifest
  digest, cache key, and calibration dataset ID in `CalibrationEvidence.metadata`.
- The planner carries those bindings unchanged. Conversion rejects a measured plan if the loaded
  capture differs, and packages `activation_capture_manifest.json` with the checkpoint.
- Publication preparation and release audit independently revalidate the packaged capture
  manifest. A capture from the right model but a different cache is no longer interchangeable.

## Qwen3-Next fused experts (next patch after v1.1.1)

`switch_mlp` and `switch_glu` 3-D weights now classify as `expert` in the registered family
adapter, matching the generic inspector and MLX-LM fused-module path. Artifacts made before this
fix may have retained most expert weights at BF16 and must be regenerated. Inspection now
downgrades a supported MoE checkpoint to inventory-only if fused-expert classification coverage
ever drifts again.

Packed gate/up source tensors that split into two MLX runtime modules now remain unmatched until
both runtime modules are visited. Packed/fused expert paths are affine-only; AWQ/GPTQ refinement
on these 3-D modules is rejected before conversion.

## Hub BF16 source preparation (next patch after v1.1.1)

`scripts/hf_to_mlx_bf16.py` now requires `--revision <40-character commit SHA>`, passes that
revision to `snapshot_download`, converts into a sibling staging directory, and atomically
renames only after a usable checkpoint exists. It refuses to overwrite an existing output and
writes `axquant_source.json` with the immutable source identity. Update automation that previously
relied on a floating Hub branch or silent output reuse.

## Planner / policy defaults

- `HardwareProfile.supported_methods` now includes GPTQ by default. Manual recipes that pin an
  explicit hardware profile are unaffected.
- Measured GPTQ candidates carry `scale_strategy: "gptq-hessian"`.
- The `refine-awq-dwq` ladder includes GPTQ candidates.

## No action needed

- Plan/sensitivity/inventory schema versions are unchanged.
- `--allow-unmeasured` and evidence-gating semantics are unchanged.
- Conversion remains fail-closed on uncovered modules; fused MoE expert groups remain
  affine-only.
