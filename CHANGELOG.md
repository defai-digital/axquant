# Changelog

All notable changes to AXQuant are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The release workflow extracts the
matching section from this file as the curated GitHub Release notes and fails the release when
the section is missing — add an entry in the same change as any user-facing modification.

## [Unreleased]

## [1.5.1] - 2026-08-07

### Fixed

- Ext4T layout planner no longer schedules `rename_or_move` /
  `rename_to_canonical` when a basename is a name-conflict (same name,
  different content) or when another action already claims the destination
  path, so executing a plan cannot clobber a distinct package.
- Ext4T `preferred_location` classifies `axquant/logs` and `axquant/scripts`
  before name heuristics (`smoke`/`tmp`/`candidate`), and `host_local` also
  follows the source logs category, so operational log trees stay
  non-fleet-synced and non-deleteable.
- Ext4T package indexing skips package-root symlinks (and marks a direct
  symlink fingerprint incomplete) so content hashing cannot collide with
  the real tree and schedule deleting the target while keeping a dangling
  link.
- Coding-suite and general-holdout overlap now use
  `axquant-token-5gram-v2` (same CJK-aware tokenization as
  `campaign-overlap`), so pure non-ASCII records no longer collapse to
  empty strings and false-positive exact matches. Coding suite manifests
  must be regenerated.
- GPTQ column encoding uses the joint `round(w/s + z)` form shared with
  AWQ/portable affine (fixes banker's-rounding half-integer disagreements
  with `round(w/s)+z`), and the standalone GPTQ plugin packs codes at
  float16 storage scale.

## [1.5.0] - 2026-08-07

### Changed

- **The flagship formal-host identifier is now `df-macbookpro-m5`** (was
  `mbp-m5`), following the owner's canonical DNS naming for the certification
  machine (`df-macbookpro-m5.defai.digital`, Apple M5 Max MacBook Pro). The
  identifier is a schema literal on the formal host contract, host evidence,
  preflight, reproduction review, and hardware authorization records; it is
  the mandatory certified-claim hardware scope id, the required
  `hardware-df-macbookpro-m5` archive record name, and the exact string
  `campaign-preflight` compares against the machine's live hostname. No
  campaign was ever frozen and no claim was ever rendered under the old id,
  so nothing published or bound is invalidated. Documents and audits citing
  `mbp-m5` as the formal host now read `df-macbookpro-m5`.

## [1.4.1] - 2026-08-07

### Fixed

- `campaign-overlap --id-field` is now repeatable and tried in order per
  record (default `id`, then `task_id`), so one overlap run can span
  mixed-schema campaign datasets — calibration corpora keyed by `id`
  alongside strict `QualityTask` suites keyed by `task_id`. Previously a
  single id field applied to every file, and no field exists in both
  shapes: `QualityTask` (`extra="forbid"`) rejects an added `id`, and the
  frozen calibration dataset cannot gain a `task_id` without invalidating
  its digest bindings. Reports are unchanged (record ids never leave the
  loader), so existing overlap reports stay reproducible.

## [1.4.0] - 2026-08-06

### Fixed

- `campaign-overlap` normalization now tokenizes non-ASCII scripts
  (`axquant-token-5gram-v2`): ASCII word runs stay whole tokens and every
  other letter (CJK, Kana, Hangul, accented Latin) becomes a
  single-character token, so shingles exist for unspaced scripts. The v1
  normalizer dropped every non-`[a-z0-9_]` character, which made any
  CJK-only record — including the shipped reference calibration
  dataset's multilingual rows — fail closed with "normalizes to empty
  text" and blocked flagship campaign freezes. ASCII-only inputs produce
  byte-identical reports under v2; `CampaignOverlapReport` now records
  the v2 algorithm id. The Qwen3-Next coding-suite normalizer is
  unchanged.

### Changed

- **Quantized MTP sidecars now emit AX Engine's executable MLX-packed layout**
  (`mlx-affine-packed-u32`): `mx.quantize` uint32-packed codes plus BF16 group
  scales/biases under the engine's `<base>.scales` / `<base>.biases` key
  convention, each tensor verified by an `mx.dequantize` round trip against
  the BF16-cast source. This replaces the 1.3.0 `axquant-portable-affine-u8`
  format, which no runtime could execute (the capability gate always refused,
  so no artifact was ever produced in it). Reading the engine's
  `mtp_take_weight` loader showed the executable contract already exists —
  AXQuant now targets it instead of inventing a second layout.

### Added

- `quantize-mtp-sidecar --runtime-json` stamps `mtp_sidecar_bits` into
  `mtplx_runtime.json` (via `annotate_mtp_runtime_sidecar_bits`) so the engine
  dequantizes packed projections at the declared width instead of the 4-bit
  default; sidecar bits are restricted to the engine runtime contract
  (2/4/6/8).
- The capability probe records the engine's reported `supported_bits` and
  `packing` when present (`ax-engine mtp-capability` output), and sidecar
  quantization fails closed when the requested bits or emitted packing fall
  outside the reported capability.
- `benchmark-kernels --from-ax-engine` ingests the engine's
  `ax-engine.kernel-latency-raw.v1` document (emitted by the new
  `axquant-kernel-latency-probe` microbench) into a host-scoped
  `axquant.kernel-latency.v1` table with `runtime=ax-engine` entries; the
  planner's latency provider now infers the runtime from single-runtime
  tables, so engine tables plug into `plan --latency-table` directly.

### Fixed

- Quantized MTP sidecar bits are restricted to {4, 6, 8} — the intersection
  of the engine loader's tolerance with AXQuant's own `mtp_sidecar_bits`
  runtime contract. Previously a 2-bit sidecar could be produced that
  AXQuant's inspector and runtime validators would then reject.
- `benchmark-kernels --from-ax-engine` refuses documents reporting packing
  methods the toolkit does not recognize instead of silently relabeling them
  as affine measurements.
- `annotate_mtp_runtime_sidecar_bits` (and `--runtime-json` help) now states
  that stamping rewrites the file and therefore invalidates any recorded
  sha256 binding to the original — stamp packaging copies, not validated
  bundles.
- KV serving-quality reports reject a boolean `quantized_layers_active` in
  the execution summary instead of counting `true` as one active layer.
- The flagship M7 MTP-admissibility helper guards its own evidence loads
  (unreadable files become named gate issues if reached), and a regression
  test pins the audit-level contract: a damaged benchmark bundle aborts the
  whole audit with a named checksum error rather than emitting gate verdicts
  against a tampered evidence set.

## [1.3.0] - 2026-08-06

### Added

- Group-preserving GPTQ act-order as the new `gptq-act` method (ADR-0002): groups process in
  descending Hessian-mass order and columns by Hessian diagonal, while group membership and the
  portable affine packed layout stay byte-compatible with static-grid GPTQ. Registered as a
  distinct probe/plan/convert method label so the measured frontier — not the literature —
  decides adoption per tensor.
- Measured-holdout-safe interaction optimization (`optimize_candidate_interactions`, ADR-0004):
  complete-candidate selection driven by measured evaluations on development dataset roles,
  with a fail-closed guard that rejects formal-holdout or missing `dataset_role` provenance and
  a separate `interaction_measurement_set_sha256` binding so the formal holdout binding stays
  free for final selection.
- Kernel-latency-aware planning (ADR-0003): `benchmark-kernels` measures decode/prefill GEMM
  latency per (bits, group size) on the current host into a host-scoped
  `axquant.kernel-latency.v1` table; `plan --latency-table` re-ranks candidates by measured
  kernel speed strictly inside the quality near-tie window and records
  `cost_model` / `kernel_latency_sha256` / host provenance on the plan. Without a table,
  planning is bit-identical to before.
- Method near-tie surfacing (RM-14): plans record `method_near_ties` (capped, most fragile
  first, with an explicit omitted count) wherever a losing packing method lands within the
  configurable `method_near_tie_epsilon` of the winner at the same storage key.
- Report-only measured-KV serving-quality artifact (`axquant.kv-serving-quality.v1`, RM-21):
  binds the executed per-layer KV plan digest to dual-profile short/long-context quality
  retention versus BF16 KV, and fails closed unless the mlx-lm-kv runtime check proved exact
  per-layer execution.
- Formal MTP A/B admissibility (`formal_mtp_bundle_issues`, RM-20): an MTP off/on evaluation
  pair is authorizing only when both halves bind the frozen formal-host contract (device, chip,
  OS, power mode), share identical controls, datasets, seeds, and checkpoint, and report zero
  kernel fallbacks.
- Opt-in quantized MTP sidecar (`quantize_qwen36_mtp_sidecar`, ADR-0005): a separate
  `axquant-portable-affine-u8` artifact emitted alongside the untouched byte-preserved default,
  gated on a recorded passing AX Engine capability check for the quantized MTP layout.
- End-to-end CLI and audit wiring for the new evidence paths: `refine-select --interaction`
  runs the holdout-safe interaction selection; `quantize-mtp-sidecar` executes a live AX Engine
  capability probe (`--capability-command`, subprocess JSON contract) or consumes a recorded
  check (`--capability-result`) before emitting the quantized sidecar; `kv-serving-quality`
  builds the report-only KV artifact from an executed plan, kv_exec summary, and measured
  results; and the flagship M7 gate now loads both MTP A/B bundles from the release validation
  chain and enforces `formal_mtp_bundle_issues` against the frozen `mbp-m5` contract and the
  digest-bound hardware-registry device identity.

### Changed

- Experimental 2/3-bit assignment is restricted to robust-trunk tensor classes (MLP and expert
  projections, RM-42); attention and all protected roles keep 4-bit-and-up candidates, and
  experimental plans now list every low-bit tensor with its predicted loss in the plan warnings.
- The default hardware profile is `ax-engine-apple-silicon-affine-dwq-v3`, adding `gptq-act`
  to the supported method set. Previously serialized plans keep their recorded v2 profile.

- Qwen3-ASR 1.7B and Qwen3-VL 8B Instruct text-path conversion through public MLX-Audio and
  MLX-VLM APIs, protected BF16 modality towers, architecture-specific runtime metadata, media
  generation smokes, and four stable-name AXQ v2 development repositories.
- Revision-pinned Qwen3-ASR BF16 normalization in `scripts/hf_to_mlx_bf16.py`; the helper forces
  the public MLX-Audio STT backend and records the key/layout remap in `axquant_source.json`.
- Explicit development-artifact edition metadata and the completed 28-repository AXQ v2
  migration from 14 immutable sources. Stable repository names now serve receipt-bound v2
  revisions on `main` and tag `v2`, while `legacy-pre-v2` preserves each replaced artifact.
  Every promoted revision passed pre/post-upload MLX-LM checks, reverified Hub trees and hashes,
  evidence-safe sibling links/model-card metadata, and a verified stable-name catalog entry.
- Additive `qwen36-mtp-v2` flagship certification contracts: path-neutral `CheckpointKey` and
  `CandidateKey`, frozen campaigns with disjoint dataset roles, exact-`mbp-m5` preflight,
  formal-cycle/holdout consumption state, durable archive proof, independent review, clean-host
  reproduction review, a separate final publication-claim review, and final M0–M8 dispatch.
- Append-only artifact lifecycle events and semantic impact scans. Certified candidates become
  ineligible for certified rendering/publication after `superseded` or `revoked`.
- Deterministic measured-BPW certified naming, bound public metric claims, and generated
  certified model cards.
- CLI commands `campaign-freeze`, `campaign-preflight`, `campaign-start-formal`,
  `campaign-complete-formal`, `campaign-close-no-go`, `campaign-overlap`,
  `campaign-frontier`, `campaign-record-publication`, `artifact-lifecycle`, and `claim-render`.
- Checksum-bound cheapest-failure-first frontier/no-go records and formal raw-evidence/custodian
  attestations, so candidate selection, formal failure, and no-feasible-candidate closure are
  evidence-backed rather than boolean declarations.
- Downloaded publication inventory and runtime-verification records; `published` campaign state is
  unreachable until the exact Hub revision, checkpoint bytes, final audit, claim, lifecycle,
  AX Engine/MLX-LM smokes, and zero-fallback result are reverified.
- A fail-closed flagship publication privacy scan for credentials, private host/path data,
  oversized or invalid text evidence, sensitive filenames, and formal raw holdout files. It
  rechecks the exact post-packaging tree and scans legitimate large `tokenizer.json` assets
  without treating them as oversized evidence.
- A non-symlinked durable-root boundary for the complete campaign/state-transition tree and
  six-part formal-host evidence whose results bind the exact frozen `mbp-m5` contract.

### Fixed

- Qwen3-ASR inspection now rejects the unnormalized upstream `thinker.*` layout for planning and
  conversion instead of counting its duplicated tied LM head as an independently convertible
  tensor. The error points to the pinned BF16 normalization helper.
- Multimodal runtime smokes require real transcript/image-generation output: ASR no longer passes
  on log text without a transcript, VLM runs non-verbose, and simple conversion rejects a smoke
  from the wrong architecture backend before conversion.
- Qwen3-ASR and Qwen3-VL model-card commands now use MLX-Audio's required `--output-path` and
  MLX-VLM's supported `--temperature` option.
- Converted-checkpoint verification now accepts only documented MLX-LM output identities:
  one-to-one wrapper aliases, Qwen packed gate/up one-to-many splits, contiguous indexed-expert
  many-to-one stacks, and the exact Qwen 3.5/Qwen3-Next/Nemotron-H `Conv1d` sanitize-axis
  transforms. Membership, precision, parameter count, metadata, and shapes remain fail-closed.
- Budget allocation preserves the deterministic marginal-gain order with a priority queue instead
  of rescanning every tensor after every upgrade, removing quadratic planning time on
  74,000-tensor Qwen3-Coder-Next checkpoints.
- The AXQ v2 migration uses immutable Mistral-published BF16 sources and materializes only
  `model.safetensors.index.json`-bound shards when upstream also ships a redundant consolidated
  checkpoint; AXQuant's unindexed-Safetensors rejection remains strict.
- Development cards no longer advertise an AX Engine serve path when the package has no validated
  native `model-manifest.json`; MLX-LM remains the documented standard text/backbone path.
- Qwen3 Embedding development cards link their real stable-name 4-bit/8-bit siblings instead of
  a nonexistent 6-bit variant.
- Release-bound identity comparisons ignore absolute `local_path` while preserving it as
  execution provenance; checkpoint/candidate byte digests still fail closed on content drift.
- Generation-smoke no longer requires the `mlx_lm` Python package; advisory KV / runtime
  metadata is validated before install gates so Ubuntu non-MLX CI matches Mac developer paths.
- Gemma4 sharded source prep validates index shard paths and types before importing MLX.
- mypy on the Ubuntu lint job ignores missing optional `mlx` / `mlx_lm` / `transformers`
  modules (they are only installed via `axquant[mlx]`).

### Changed

- Existing `4bit`/`6bit` repository names remain development identifiers. A certified flagship
  must use generated `MP-<measured-main-BPW>bpw[-MTP]` naming and cannot downgrade to the
  historical Qwen 3.6 v4 audit/publisher path.
- CI and release workflows use current Node 24 GitHub Actions majors and grant permissions
  per job, reducing deprecated-runtime warnings while keeping PyPI OIDC and build provenance
  least-privilege.
- Documented historical Actions red-run root causes and added `scripts/ci-local.sh` to mirror
  CI gates (including PATH-isolated non-MLX pytest) before push.
- Release `pypi` job is gated on repository variable `ENABLE_PYPI_PUBLISH=true` so tag
  Releases succeed with GitHub assets when Trusted Publishing is not yet configured on
  pypi.org (set the variable only after registering a publisher).

### Notes

- Real PyPI upload still requires a Trusted Publisher on pypi.org plus
  `ENABLE_PYPI_PUBLISH=true` (operator steps in docs/ci-root-causes.md).
- Local-only engineering/product notes stay out of the published tree: force-added
  certification docs were removed from git (still ignored on developer machines).

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
- AWQ/GPTQ convert-time refinement no longer crashes on bfloat16 module weights (the default
  dtype of real checkpoints).
- Affine/DWQ/AWQ quantizers build the per-group grid from the float16-stored scale (rounded
  up, never down), restoring the documented ≤ scale/2 per-element round-trip bound.
- `plan`/`refine` inject BF16 for policy-preserved tensors (norms, LM head, vision) when 16 is
  omitted from `--bits`, matching manual planning, instead of failing per tensor; manual
  tied-weight harmonization now refreshes scale/outlier strategies and stored BPW.
- Feasibility audits require MTP evidence only from MTP-declaring architectures, making the
  Qwen3-Next direct-track N0 gate satisfiable with tool-produced reports.
- Measured probe/KV-probe runs no longer abort when the calibration token count leaves a
  single-token trailing replay batch.
- The `kv-exec` runtime check now fails when executed per-layer KV bits differ from the plan
  (previously reported but not enforced), and benchmark trial timeouts kill the whole
  AX Engine process group instead of orphaning the generation.
- MTP benchmark aggregates pair numerators and denominators over counter-reporting trials
  only, instead of inflating `effective_tokens_per_forward` when some trials lack counters.
- `prepare_development_model_card` sanitizes the manifest's calibration reference along with
  the plan's (local-path leak that also broke the release-audit M1 equality invariant).
- Release audit records issues instead of crashing on wheels missing `axquant/__init__.py`,
  corrupt measurement-set or calibration files, and non-scalar benchmark metadata; hardware
  registry checks record all-failed raw benchmarks instead of raising.
- `--kv-default-bits` rejects sub-policy-floor values (2/3) at parse time instead of failing
  late in planning.
- `scripts/hf_to_mlx_bf16.py` writes the lm_head-synthesized shard aside and swaps atomically;
  saving onto the mmap-backed source silently corrupted every tensor in the shard.
- Ext4T scripts: union-sync verification compares content only (mtime-only differences no
  longer deadlock syncs), empty-array expansions no longer crash stock macOS bash 3.2 under
  `set -u`, shell-config marker blocks no longer corrupt an rc file lacking a trailing
  newline, live log/publish directories are moved aside before migrate-and-delete, and
  `--status`/argument handling exit correctly.
- Architecture-gate errors and inventory warnings name the adapter registry instead of
  hardcoding Qwen 3.6 (the checks were already registry-wide).
- Ext4T planning treats `axquant/logs` packages as host-local: layout moves only — no
  cross-host mirroring, duplicate deletion, or fingerprint-driven renames of mutable log
  trees; index loading reports the exact file and line for corrupt or incomplete records.

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

[1.3.0]: https://github.com/defai-digital/axquant/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/defai-digital/axquant/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/defai-digital/axquant/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/defai-digital/axquant/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/defai-digital/axquant/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/defai-digital/axquant/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/defai-digital/axquant/releases/tag/v1.0.0
