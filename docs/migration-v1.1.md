# Migration guide: v1.0.x → v1.1.x

v1.1.0 added GPTQ Hessian error compensation and completed the AWQ calibration path.
v1.1.1 hardens that surface (capture resume, compressed/sharded artifacts, GPTQ memory).
This page lists every breaking or behavior-changing item and what to do about it. Changes that
landed after v1.1.1 are covered by the [v1.2.0 migration guide](migration-v1.2.md).

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
- Capture immutability and resume-chunk validation landed in v1.2.0; see the
  [v1.2.0 migration guide](migration-v1.2.md).
- `capture-activations --revision <SHA>` is resolved at the exact revision bound by the tokenized
  cache. When an MLX BF16 source contains `axquant_source.json`, its model ID and immutable
  revision must also match that cache.

## Evidence-chain hardening, fused experts, Hub source prep

These items landed in v1.2.0; see the [v1.2.0 migration guide](migration-v1.2.md).

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
