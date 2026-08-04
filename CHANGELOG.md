# Changelog

All notable changes to AXQuant are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The release workflow extracts the
matching section from this file as the curated GitHub Release notes and fails the release when
the section is missing — add an entry in the same change as any user-facing modification.

## [1.2.0] - 2026-08-04

### Added

- `analyze --capture-points`: measured probing on plain dense backbones (MiniCPM5, Qwen3 dense,
  Mistral, …) via `--capture-points output`; the default `("output", "hidden")` fails closed
  with a hint on those layouts.
- `refine --lm-head-floor`, matching `plan`; the governed `8bit` path still requires measured
  support.
- Structured `mtp_sidecar_bits` runtime contracts in copied (byte-preserved) MTP sidecar
  metadata.
- New evaluation task categories: JSON, tool-use, multilingual, and long-context.

### Changed

- Calibration evidence is checksum-bound end to end: AWQ/GPTQ analysis records the
  activation-capture manifest digest, tokenized-cache manifest digest, cache key, and dataset
  ID; the planner carries the bindings; conversion rejects a mismatched capture and packages
  `activation_capture_manifest.json` with the checkpoint; publication preparation and release
  audit independently revalidate it.
- Completed capture directories are immutable (`completion.json` present → no overwrite), and
  every committed resume chunk is validated against checkpoint row accounting.
- `scripts/hf_to_mlx_bf16.py` requires `--revision <40-char SHA>`, converts into a staging
  directory, atomically renames on success, and writes `axquant_source.json` with the immutable
  source identity.
- Mixed-precision plans are labeled by target BPW instead of inheriting a flat storage-class
  name.

### Fixed

- Qwen3-Next adapter: fused `switch_mlp`/`switch_glu` 3-D weights now classify as `expert`.
  **Pre-fix artifacts may have retained most expert weights at BF16 and must be regenerated.**
- Capture resume now fails closed on binding drift (model, cache, `--max-rows`, segments, token
  budget) instead of silently mixing runs.

See the [v1.1.x → v1.2.0 migration guide](docs/migration-v1.2.md) for action items.

## [1.1.1] - 2026-08-03

### Added

- Resumable, compressed, shardable activation capture: `capture_progress.json` +
  `activations/.partial/` resume, deflate-compressed npz output, `--segment-batches` and
  `--modules-per-shard` controls.
- Real-hardware end-to-end integration tests (`pytest -m integration` on Apple Silicon).
- CI workflow (lint on Ubuntu, full MLX suite on `macos-14`) and a signed release workflow
  (SHA256SUMS + keyless Sigstore attestation).
- Migration guide, environment compatibility matrix, and known-issues list under `docs/`.

### Changed

- GPTQ peak memory cut ~30% at `in_features = 8192` (6.7 GB → 4.7 GB).

### Fixed

- BF16 activations are cast inside MLX before numpy export in capture.
- `cosine_distance` is clamped to non-negative against float rounding.

## [1.1.0] - 2026-08-03

### Added

- GPTQ Hessian error-compensated weight refinement, wired through the conversion predicate and
  the `convert` CLI.
- End-to-end AWQ: `capture-activations` records checksum-bound per-module Linear input
  activations; measured AWQ/GPTQ probing via `analyze --calibration-activations`; planner
  selection with the `gptq-hessian` scale strategy; convert-time refinement via
  `convert --calibration-activations`.
- Development model card generation for published checkpoints.

### Changed

- Python API renames: `convert_model(awq_activations=...)` → `calibration_activations=`,
  `PlanPredicate.awq_metadata` → `method_metadata`; `load_capture_activations` returns a
  mapping-compatible `LoadedActivationCapture`.
- Measured-probe backend moved to `axquant-mlx-isolated-probe-v6`; saved probe resume state
  from older versions is rejected.

## [1.0.2] - 2026-08-03

### Added

- Qwen3-Embedding and Qwen3-Next / Coder-Next conversion support.
- HF → MLX BF16 helper script and an expanded Hub catalog in the README.

## [1.0.1] - 2026-08-02

### Fixed

- Protection-floor and path-safety gaps found on re-audit; six high-severity and three
  medium-severity release-safety bugs.

## [1.0.0] - 2026-08-01

### Added

- Initial public release: inspect → calibrate → analyze → plan → convert → validate → publish
  pipeline with checksum-bound artifacts, evidence gating, fail-closed conversion, family
  adapters (Qwen 3.6 primary track, Qwen 3.5, Gemma-4, MiniCPM5, Mistral/Devstral, Mistral3,
  Nemotron 3 Nano), and AX Engine runtime manifests.

[1.2.0]: https://github.com/defai-digital/axquant/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/defai-digital/axquant/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/defai-digital/axquant/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/defai-digital/axquant/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/defai-digital/axquant/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/defai-digital/axquant/releases/tag/v1.0.0
