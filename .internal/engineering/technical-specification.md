# AXQuant Technical Specification

**Document status:** Normative v0.x implementation baseline  
**Specification version:** 0.1  
**Applies to:** AXQuant v0.x and v1 foundations  
**Primary model:** Qwen3.6-27B  
**Primary runtime:** AX Engine  
**Compatibility runtime:** MLX-LM  
**Last reviewed:** 2026-07-28

This document defines the engineering contracts for AXQuant. Product intent and release claims
are defined in the [product requirements](../product/requirements.md). Binding architectural
choices and rejected alternatives are defined in the
[decision register](../architecture/decision-register.md).

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative. A feature
labelled **Planned** remains a requirement but is not a claim about the current implementation.

## 1. Scope and implementation status

### 1.1 v0.x system boundary

AXQuant v0.x is a Python 3.11+ command-line toolkit that:

1. resolves and inspects local or cached MLX checkpoints;
2. recognizes the supported Qwen3.6-27B language path;
3. inventories backbone, protected, vision, and MTP tensors;
4. records calibration provenance;
5. represents per-tensor 4/6/8/BF16 sensitivity evidence;
6. creates deterministic manual or sensitivity-driven mixed-precision plans;
7. executes supported plans through public MLX-LM conversion interfaces;
8. preserves an external MTP sidecar byte-for-byte;
9. emits portable MLX weights plus AXQuant and AX Engine metadata;
10. validates externally measured quality, MTP, hardware, and integrity evidence;
11. prepares or publishes an artifact only after release gates pass.

The optimized scope is the Qwen 3.6 dense language path. Vision tensors are preserved at BF16.
Generic architectures may be inventoried, but MUST NOT be converted as supported AXQuant v0.x
artifacts.

### 1.2 Capability truth table

| Capability | Current status | Release significance |
| --- | --- | --- |
| Checkpoint resolution and inspection | Implemented | Required |
| Qwen3.6-27B adapter and text-path boundary | Implemented | Required |
| MTP declaration, tensor, and sidecar detection | Implemented | Required |
| Calibration JSONL validation and manifest | Implemented | Development foundation |
| Tokenized calibration cache | Implemented | Required for measured analysis |
| Architecture-prior sensitivity map | Implemented | Development evidence only |
| Measured MLX forward sensitivity probes | Implemented | Required for release planning |
| Manual mixed-precision recipe and plan | Implemented | v0.1 development only |
| Single-candidate greedy planner | Implemented | Requires measured evidence for release |
| Top-N refinement and evidence-bound selection | Implemented foundation | Candidate conversion/evaluation orchestration remains |
| 4/6/8/BF16 affine execution | Partially implemented | Backend and kernel validation required |
| DWQ refinement | Implemented | Complete-candidate validation required |
| AWQ refinement | Implemented (portable scale search + convert packing) | Complete-candidate and host validation still required |
| External MTP byte preservation | Implemented | v0.1 requirement |
| AX Engine manifest generation and doctor check | Implemented | Required |
| MLX-LM load and generation check | Implemented | Required compatibility evidence |
| MTP benchmark harness | Implemented | Required before MTP-aware release claims |
| Deterministic benchmark suites | Implemented | Required for reproducibility |
| Real MLX-LM generation check | Implemented | Required |
| Evaluation-bundle validation | Implemented | Required |
| Atomic artifact conversion | Implemented | Required |
| Publication preparation and Hub upload | Implemented | Release-gated |

No architecture-prior report, manual recipe, static runtime check, or synthetic unit test MAY be
described as measured model quality or production performance.

## 2. System context

```text
Qwen 3.6 source checkpoint
        |
        v
Feasibility audit and model inspection
        |
        +--> architecture_report.json
        +--> feasibility_report.json/.md
        |
        v
Calibration provenance and activation capture
        |
        +--> calibration_manifest.json
        +--> tokenized cache with checksummed shards
        |
        v
Per-tensor sensitivity analysis
        |
        +--> sensitivity_map.json
        +--> resumable probe progress
        |
        v
Mixed-precision planning
        |
        +--> plan-01.json
        +--> candidate history                         [top-N history: planned]
        |
        v
Atomic MLX conversion
        |
        +--> standard MLX checkpoint
        +--> byte-preserved MTP sidecar
        +--> AXQuant plan/runtime/artifact manifests
        +--> AX Engine model-manifest.json
        |
        v
Quality, MTP, integrity, and hardware evaluation
        |
        +--> evaluation bundles
        +--> benchmark_report.json/.md
        |
        v
Release gate and publication preparation
        |
        +--> reproducible Hugging Face-ready directory
```

The portable artifact is a standard MLX checkpoint. AX Engine consumes additional metadata for
optimized execution. MLX-LM is expected to perform standard backbone inference and may ignore
AXQuant-specific runtime metadata.

## 3. Runtime and dependency requirements

### 3.1 Supported environment

- Host OS: macOS for conversion and hardware measurements.
- Hardware: Apple Silicon for release benchmarks.
- Python: 3.11 or newer.
- Weight container: Safetensors.
- Primary runtime: AX Engine.
- Compatibility runtime: MLX-LM.
- Package and CLI name: `axquant`.

Schema-only inspection, planning, report generation, and most tests MAY run without MLX.
Conversion and measured forward probes require the optional MLX dependencies. Hub resolution
requires a locally cached snapshot unless the user explicitly supplies `--allow-download`.

### 3.2 Dependency boundary

AXQuant MUST use public and documented interfaces. It MUST NOT import or vendor implementation
code from mlx-optiq. The supported execution backend in v0.x is the public `mlx_lm.convert`
quantization predicate path.

The converter currently accepts:

- `affine` for standard quantized tensors;
- `dwq` for deterministic distribution-clipped weights packed through portable MLX-LM affine
  kernels;
- `bf16` for unquantized protected tensors.

The schema also includes `awq` and `gptq` so artifacts remain extensible. A plan containing either
method MUST fail before conversion until a portable execution backend is implemented. Registering
an analysis plugin alone is insufficient to make a method conversion-capable.

## 4. Repository component map

| Component | Module | Responsibility |
| --- | --- | --- |
| CLI | `axquant.cli` | Argument parsing, command dispatch, stable exit behavior |
| Schemas | `axquant.schema` | Strict Pydantic artifact contracts |
| Serialization | `axquant.serde` | Canonical writes, model loading, SHA-256 fingerprints |
| Inspection | `axquant.inspector` | Checkpoint resolution, index parsing, tensor inventory |
| Architecture registry | `axquant.architectures` | Explicit adapter selection and support boundary |
| Qwen adapter | `axquant.architectures.qwen36` | Qwen3.6 matching, scope, tensor roles |
| Calibration | `axquant.calibration` | Dataset validation and provenance manifest |
| Analysis | `axquant.analyzer`, `axquant.probe` | Architecture priors and measured MLX probes |
| Calibration cache | `axquant.activation_cache` | Tokenization, sharding, checksums, resumption |
| Profiles | `axquant.profiles` | Workload objective weights and validation thresholds |
| Planner | `axquant.planner` | Constraint filtering and deterministic greedy allocation |
| Refinement runner | `axquant.refinement_runner` | Resumable complete-candidate execution and selection |
| Manual planner | `axquant.manual` | Reviewed v0.1 precision recipes |
| Predicate | `axquant.predicate` | Exact plan-to-MLX module mapping |
| Converter | `axquant.converter` | Preflight, conversion, staging, MTP copy, artifact manifest |
| Runtime | `axquant.runtime` | AX Engine contracts and MLX-LM compatibility checks |
| Benchmark | `axquant.benchmark` | AX Engine trials and identical-checkpoint MTP A/B |
| Benchmark evidence | `axquant.benchmark_evidence` | Complete baseline index and omission guard |
| Release validation | `axquant.release_validation` | Dual-profile validation and dataset-separation gate |
| Quality | `axquant.quality` | Perplexity and deterministic task evaluation |
| Suites | `axquant.suites` | Reproducible disjoint benchmark inputs |
| Validator | `axquant.validator` | Cross-bundle invariants and release comparisons |
| Reporting | `axquant.reporting` | Markdown reports and publication preparation |
| Reproduction | `axquant.reproduction` | Bound recipe and regenerated-weight verification |
| Compatibility | `axquant.compatibility` | Family-wide artifact/runtime/validation matrix |
| Publisher | `axquant.publisher` | Preview and explicit Hugging Face upload |
| Naming | `axquant.naming` | Canonical model repository names |

Cross-component data MUST cross a Pydantic schema boundary. Unversioned dictionaries MUST NOT
become persistent core artifacts, except explicitly open metadata maps within a versioned schema.

## 5. CLI contract

### 5.1 General behavior

The CLI entry point is:

```bash
axquant <command> [options]
```

All commands MUST emit structured logs to stderr. Machine-readable command output MUST be written
to the requested file or directory, not mixed with logs. `axquant name` is the sole command whose
primary artifact is stdout.

Exit codes:

| Code | Meaning |
| ---: | --- |
| `0` | Command completed and its requested gate passed |
| `1` | A completed feasibility, validation, or runtime check did not satisfy the requested gate |
| `2` | Invalid input, schema failure, unsupported operation, backend failure, or I/O failure |

An error MUST NOT leave a partially published final conversion directory.

### 5.2 Commands

| Command | Principal input | Principal output | Status |
| --- | --- | --- | --- |
| `feasibility` | 4-bit/6-bit baselines; optional BF16/mixed baselines | `feasibility_report.json`, `.md` | Implemented |
| `inspect` | MLX model directory or Hub ID | `architecture_report.json` | Implemented |
| `calibrate` | JSONL dataset and profile | calibration manifest and tokenized cache | Implemented |
| `analyze` | model, candidate bits, optional calibration | `sensitivity_map.json` | Implemented |
| `plan` | sensitivity report and constraints | `plan-01.json` | Single candidate implemented |
| `plan-manual` | inventory and reviewed recipe | plan JSON and optional Markdown | Implemented |
| `convert` | source model and plan | final MLX artifact directory | Implemented for affine/BF16 |
| `validate` | reference, MTP-off, and MTP-on evaluation bundles | `benchmark_report.json` | Implemented |
| `report` | plan and optional validation | Markdown report | Implemented |
| `runtime-check` | model and runtime name | `runtime_check.json` | Implemented |
| `evaluate-quality` | model and quality-task JSONL | quality evaluation JSON | Implemented |
| `benchmark` | model and prompt JSONL | AX Engine evaluation bundle | Implemented |
| `benchmark-ab` | model and prompt JSONL | MTP-off/on evaluation bundles | Implemented |
| `benchmark-index` | complete baseline request | checksum-bound benchmark evidence index | Implemented |
| `validation-index` | agent-coding/general validation requests | release validation index | Implemented |
| `prepare-suite` | output directory | deterministic quality and prompt suites | Implemented |
| `refine-run` | execution request and embedded candidate plans | execution manifest, measurements, selection, Pareto report | Implemented |
| `hardware-registry` | raw A/B logs, evaluations, plan, sensitivity, validation, and conversion evidence | measured hardware profile registry | Implemented |
| `release-audit` | toolkit wheel and complete M0–M8 evidence graph | machine-readable release proof and blockers | Implemented |
| `publish-prepare` | model directory, repo ID, validation index, hardware registry, and Pareto report | release files in model directory | Implemented |
| `publish` | prepared model and complete release evidence | preview or Hub upload | Implemented |
| `verify-reproduction` | recipe and regenerated model directory | reproduction verification JSON | Implemented |
| `compatibility-matrix` | compatibility request and evidence files | compatibility matrix JSON | Implemented |
| `name` | base model and target class | canonical `owner/name` | Implemented |

### 5.3 Safety flags

- `inspect --allow-download` explicitly permits Hub download. Without it, resolution is
  cache-only.
- `inspect --allow-quantized` permits inventory of a quantized checkpoint. Such inventory is not
  authorization to use that checkpoint as a conversion source.
- `plan --allow-unmeasured` and `convert --allow-unmeasured` are development escape hatches.
  Outputs produced with them MUST NOT pass publication preparation.
- `convert --ax-engine-manifest required` is the production default.
- `publish` is a preview unless `--yes` is present. Upload is the only remote mutation in the CLI.
- `feasibility --require-ready` returns `1` for `baseline-ready`; without this flag,
  `baseline-ready` is a successful audit result.

## 6. Model resolution and checkpoint discovery

### 6.1 Resolution

`resolve_model_dir` MUST:

1. accept an existing local directory after `~` expansion; or
2. resolve a Hub model and revision using `snapshot_download`;
3. set `local_files_only=True` unless download was explicitly allowed;
4. return an absolute local directory;
5. fail with an artifact error when the model cannot be resolved.

A release input MUST record an immutable source revision. A mutable model name alone is
insufficient for manual release planning or publication.

### 6.2 Checkpoint membership

When `model.safetensors.index.json` exists:

- `weight_map` MUST be a non-empty JSON object;
- each shard reference MUST be a relative path;
- `..`, absolute paths, and non-Safetensors targets MUST be rejected;
- every referenced shard MUST exist;
- duplicate shard paths MAY occur in the map and are de-duplicated for reading;
- a root `mtp.safetensors`, when present, is added explicitly.

Unindexed auxiliary Safetensors files MUST NOT silently become checkpoint members.

When no index exists, all root `*.safetensors` files are checkpoint members. This fallback is
accepted for unsharded checkpoints and SHOULD be avoided for release artifacts that contain
unrelated root Safetensors files.

### 6.3 Inspection invariants

The inspector MUST:

- require `config.json` containing a JSON object;
- open tensor metadata lazily through Safetensors;
- reject duplicate tensor names across checkpoint files;
- record physical shape, dtype, storage bytes, source file, role, and module path;
- preserve deterministic tensor order by `(file, name)`;
- hash the canonical configuration;
- record tied embedding and LM-head weights when `tie_word_embeddings` is true.

A tensor is eligible for v0.x quantization only when it:

- is two-dimensional;
- has a floating dtype in BF16, F16, F32, or F64;
- is not a normalization tensor;
- is not scale/bias quantization metadata.

Eligibility does not override role protection or architecture scope.

### 6.4 Logical parameter accounting

For an unpacked tensor:

```text
logical_parameters = product(shape)
```

For a packed `U32` weight tensor with known precision `b`:

```text
logical_parameters = physical_elements × 32 / b
```

The division MUST yield the same integer reconstruction used by the inspector. Quantization
metadata tensors ending in `.scales` or `.biases` contribute zero logical model parameters.

Actual file bytes and logical parameter-derived BPW are separate measurements. Neither may be
silently substituted for the other.

## 7. Qwen 3.6 architecture contract

### 7.1 Adapter match

The v0.x adapter ID is `qwen36-v1`. It matches only when:

1. `config.model_type == "qwen3_5"`; and
2. either the model reference or `_name_or_path` identifies Qwen 3.6, or the validated structural
   signature matches the reference architecture.

The initial supported conversion signature is a dense 27B language model with:

```text
num_hidden_layers = 64
hidden_size       = 5120
intermediate_size = 17408
```

The structural fallback additionally uses the known vocabulary and MTP declaration where
available. A Qwen 3.6 checkpoint outside the validated signature is inventory-only until an
explicit adapter update and test matrix accepts it.

### 7.2 Optimization scope

The adapter returns one of:

- `supported` + `text-path` for validated dense Qwen3.6-27B;
- `inventory-only` + `inventory-only` for other matches;
- generic inventory-only behavior for unmatched models.

Conversion MUST require:

```text
product_family == "qwen3.6"
optimization_scope == "text-path"
```

### 7.3 Tensor roles

The adapter classifies:

- embeddings;
- attention projections and linear-attention components;
- MLP projections;
- norms;
- LM head;
- router and expert tensors;
- MTP projections, blocks, and output heads;
- vision modules;
- other tensors.

Role detection considers both tensor name and source filename so an external MTP sidecar is
classified even when its internal names do not include a complete model prefix.

### 7.4 Default protection floors

| Role | Minimum or treatment |
| --- | --- |
| Normalization | BF16 |
| LM head | BF16 |
| Vision | BF16 |
| Embedding | 8-bit minimum |
| Router | 8-bit minimum |
| Protected MTP | Configured MTP minimum, default 8-bit |
| External MTP sidecar | Byte-preserved; no v0.x requantization |
| Non-quantizable tensor | Preserved |

Tied weights MUST be assigned one compatible precision. Manual planning harmonizes a tied group
at the maximum selected precision.

## 8. Persistent artifact schemas

### 8.1 General schema rules

All core JSON artifacts use strict Pydantic models:

- unknown fields are rejected;
- assignment remains validated;
- schema versions are literal strings;
- invalid enum values, ranges, or cross-field combinations fail closed;
- timestamps use UTC;
- JSON serialization MUST be deterministic enough to support stable SHA-256 fingerprints.

The current schema registry is:

| Schema ID | Model | Purpose |
| --- | --- | --- |
| `axquant.inventory.v1` | `Inventory` | Model and tensor inventory |
| `axquant.calibration.v1` | `CalibrationManifest` | Calibration provenance |
| `axquant.sensitivity.v1` | `SensitivityReport` | Per-tensor candidate evidence |
| `axquant.runtime.v1` | `RuntimeMetadata` | Runtime tiers and recommendations |
| `axquant.runtime-check.v2` | `RuntimeCheck` | Model-bound runtime diagnostic result |
| `axquant.feasibility.v1` | `FeasibilityReport` | Baseline readiness audit |
| `axquant.manual-recipe.v1` | `ManualPlanRecipe` | Reviewed manual allocation input |
| `axquant.plan-request.v1` | `PlanRequest` | Automated planning request |
| `axquant.plan.v1` | `QuantizationPlan` | Executable per-tensor allocation |
| `axquant.artifact.v2` | `ArtifactManifest` | Converted checkpoint manifest and measured weight accounting |
| `axquant.evaluation.v1` | `EvaluationBundle` | External benchmark measurements |
| `axquant.validation.v1` | `ValidationReport` | Release-gate comparisons |
| `axquant.tokenized-cache.v1` | `TokenizedCacheManifest` | Verified token cache |
| `axquant.probe-progress.v1` | `ProbeProgress` | Resumable tensor probe state |
| `axquant.quality-evaluation.v2` | `QualityEvaluationResult` | Quality task results and prompt-rendering provenance |
| `axquant.quality-comparison.v1` | `QualityComparisonReport` | Per-task and aggregate quality deltas |
| `axquant.benchmark-suite.v1` | `BenchmarkSuiteManifest` | Benchmark input digests |
| `axquant.benchmark-evidence-request.v1` | `BenchmarkEvidenceRequest` | Explicit status of every comparison baseline |
| `axquant.benchmark-evidence-index.v1` | `BenchmarkEvidenceIndex` | Checksum-bound available and unavailable baselines |
| `axquant.release-validation-request.v1` | `ReleaseValidationRequest` | Agent-coding/general validation inputs |
| `axquant.release-validation-index.v1` | `ReleaseValidationIndex` | Dual-profile publication gate |
| `axquant.refinement.v2` | `RefinementResult` | Executable candidate plans, selected plan, and immutable history |
| `axquant.refinement-measurements.v5` | `RefinementMeasurementSet` | Stable-ID complete-candidate and named-host results with profile-exact validation, MLX-LM, power-mode, and fallback evidence |
| `axquant.refinement-execution-request.v2` | `RefinementExecutionRequest` | Complete-candidate execution inputs, power mode, and raw-log controls |
| `axquant.refinement-execution.v1` | `RefinementExecutionManifest` | Resumable exact commands and output checksums |
| `axquant.pareto.v3` | `ParetoReport` | Deterministic non-dominated stable-ID measurements on complete hardware evidence |
| `axquant.hardware-registry-request.v3` | `HardwareRegistryRequest` | Measurement-ID-specific paths to checksum-bound candidate and raw hardware evidence |
| `axquant.hardware-registry.v3` | `HardwareProfileRegistry` | Named-host measurement, kernel, version, power-mode, shape coverage, and semantic evidence bindings |
| `axquant.release-audit-request.v4` | `ReleaseAuditRequest` | Complete evidence graph, including catalog-bound compatibility inputs and optional governed size exception, required to prove M0–M8 |
| `axquant.release-audit.v4` | `ReleaseAudit` | Request-bound fail-closed milestone checks and v1 blockers |
| `axquant.quantizer-execution.v1` | `QuantizerExecutionManifest` | Per-module executed method and metadata |
| `axquant.artifact-size-evidence.v1` | `ArtifactSizeEvidence` | Candidate/uniform-4 size gate input |
| `axquant.reproduction.v3` | `ReproductionRecipe` | Checksummed inputs, complete MTP bundle bindings, and executable regeneration commands |
| `axquant.reproduction-verification.v1` | `ReproductionVerification` | Regenerated weight and provenance verification |
| `axquant.compatibility-request.v2` | `CompatibilityMatrixRequest` | Release-time official dense scope plus dual-profile artifact/runtime/validation evidence |
| `axquant.compatibility-matrix.v2` | `CompatibilityMatrix` | Catalog-bound family compatibility and exact-scope gate |

### 8.2 Evidence taxonomy

`SensitivityReport.evidence_kind` is one of:

- `measured`: produced by an AXQuant measurement backend with calibration provenance;
- `measured_development`: real forward measurements below release corpus or token thresholds;
- `imported`: externally produced measurements accepted through a documented adapter, also with
  provenance;
- `architecture_prior`: deterministic role-based development estimates.

Only `measured` and `imported` are release-quality at the schema level. Imported evidence MUST
still pass source attribution and governance review. `measured_development` and
`architecture_prior` require an explicit development override to plan or convert and are rejected
by publication preparation.

### 8.3 Model identity

`ModelIdentity` contains:

```json
{
  "model_id": "Qwen/Qwen3.6-27B",
  "revision": "<immutable revision>",
  "format": "mlx",
  "architecture": "<reported architecture>",
  "local_path": "<optional local resolution>"
}
```

Release manifests MUST retain the public model identity and revision. A local path is diagnostic
metadata, not a portable identity.

### 8.4 Sensitivity candidate

Each tensor candidate records:

- precision and quantization method;
- group size for any precision below 16;
- a `MetricVector`;
- backend support status;
- an explanatory note.

The `(bits, method)` pair MUST be unique within one tensor entry. A 16-bit candidate MUST use
`bf16` and have no group size. A sub-16-bit candidate MUST specify a group size.

### 8.5 Plan

A plan MUST contain:

- the exact source identity and architecture profile;
- profile, target mode, target and achieved proxy BPW;
- normalized objective inputs;
- hardware and MTP policies;
- hard constraints;
- software versions and seed;
- evidence kind, analysis digest, and calibration provenance;
- one allocation for every inventoried tensor;
- whole-model and MTP precision distributions;
- `global_validation_required = true`.

The plan is executable intent, not proof of achieved model quality, actual artifact size, or
runtime speed.

### 8.6 Artifact manifest

`axquant_manifest.json` binds:

- AXQuant version and source identity;
- SHA-256 of the exact plan;
- profile and calibration provenance;
- target class, proxy effective BPW, and authoritative measured main/total BPW;
- reconstructed main/total logical parameters and inspected main/MTP/protected weight-file bytes;
- precision distributions and MTP policy;
- runtime metadata and software versions;
- every artifact file other than the manifest itself, with size and SHA-256.

Publication preparation MUST revalidate every listed path, byte size, and checksum.
Measured conversion MUST receive the calibration manifest named by the plan, verify its checksum,
source model, profile, dataset, sample/domain coverage, sequence length, and evaluation-separation
attestation, then byte-copy it into the staged artifact before the atomic rename.

### 8.7 Evaluation bundle

An evaluation bundle is an externally populated, strict measurement envelope. It identifies:

- the exact model and baseline kind;
- runtime and MTP-enabled state;
- workload and evaluation-dataset digest;
- quality, MTP, hardware, and integrity metrics;
- software versions and random seed.

The current CLI validates bundles but does not generate all measurements. Benchmark producers
MUST use the same schema and MUST NOT omit required release metrics.

## 9. Feasibility audit

### 9.1 Inputs

The audit requires:

- uniform 4-bit baseline;
- uniform 6-bit baseline.

It optionally accepts:

- BF16 source;
- mixed-precision baseline;
- AX Engine and MLX-LM runtime checks.

Each artifact target has an explicit `BaselineKind`, optional public model ID, and optional
revision. The local path is never inferred as an immutable revision.

### 9.2 Integrity checks

The audit checks, as applicable:

- valid `config.json`;
- presence and completeness of the Safetensors index;
- Safetensors weight presence;
- tokenizer/config assets;
- native AX Engine manifest presence and validity;
- MTP sidecar, runtime contract, and provenance files;
- architecture adapter and optimization scope;
- expected quantization state for the baseline kind;
- logical parameter and byte accounting;
- runtime diagnostics when requested.

### 9.3 Status state machine

```text
invalid or contradictory required baseline
        -> blocked

valid 4-bit and 6-bit baselines, but no valid BF16 source
        -> baseline-ready

valid required baselines + valid revision-pinned BF16 source
        -> ready-for-conversion
```

Warnings identify non-blocking limitations. Blockers MUST be explicit and machine-readable.
`baseline-ready` proves that baseline comparison inputs are usable; it does not authorize
source conversion.

## 10. Calibration pipeline

### 10.1 Implemented calibration and token cache

`calibrate` currently accepts a local JSONL dataset. Every non-empty line MUST parse as a JSON
object. The command records:

- absolute dataset identifier;
- SHA-256 of exact bytes;
- non-empty sample count;
- profile and per-sample observed domains, with declarations labelled separately;
- maximum sequence length;
- seed and tokenizer revision;
- whether calibration/evaluation separation was attested.

If a calibration directory already contains a manifest:

- an identical semantic input returns the existing manifest;
- a changed input fails rather than mutating the cache in place.

The command tokenizes examples through the pinned Transformers tokenizer and writes:

```text
calibration-cache/
  calibration_manifest.json
  tokenized/
  completion.json
```

The cache key MUST include at least:

```text
source model revision
config digest
tokenizer revision/digest
dataset digest
profile
sequence length
sample selection
random seed
AXQuant measurement backend version
MLX and MLX-LM versions
capture-point definition
```

Token shards are content-addressed, carry sample-order and tokenizer digests, and are verified by
SHA-256 and array shape before reuse. A completion marker is written only after all expected
shards pass verification. Release evidence requires at least 128 recorded samples, 8,192 replayed
tokens, all profile domains from sample records, and at least one declared long-context sequence.

### 10.3 Calibration/evaluation separation

Release runs MUST use disjoint calibration and evaluation datasets. Dataset IDs and digests MUST
be published, while private sample contents MAY remain private when licensing or confidentiality
requires it. A false or missing separation attestation MUST block a production release workflow.

## 11. Sensitivity engine

### 11.1 Current architecture-prior backend

The current `analyze` command emits deterministic role-based estimates. Its report:

- contains all inventory tensors;
- tests requested precision candidates only for quantizable tensors;
- assigns BF16 only to non-quantizable tensors;
- records `evidence_kind = architecture_prior`;
- warns that the metrics are not measured.

When `--calibration` is supplied, `analyze` invokes the measured backend. Without calibration it
emits only architecture priors.

### 11.2 Measured probe

For each eligible tensor `t` and candidate configuration `c`, the measured backend:

1. restore the same source-model state;
2. quantize only the intended tensor or capture-equivalent module for the isolated probe;
3. replay the fixed calibration sample/order;
   equal-length sequences may be grouped into an explicitly recorded replay batch without
   changing their order or the measured-token budget;
4. capture metrics at declared model points;
5. restore or discard the mutation completely;
6. persist the result with calibration and software provenance;
7. support resumption without accepting partial/corrupt measurements.

The initial metric vector is:

```text
output KL divergence
hidden-state mean squared error
cosine distance
output token disagreement
task-loss delta
MTP acceptance loss
long-context loss
peak-memory cost
prefill-latency cost
decode-latency cost
```

Metric normalization MUST be declared per analysis version. Measurements with incompatible
normalization MUST NOT be merged into one plan.

### 11.2a Probe cost model and efficiency strategy

Isolated per-tensor probing is the most expensive milestone in the roadmap. For a 27B dense
model with roughly one thousand quantizable tensors, three quantized candidates each, and a
calibration replay per candidate, a naive implementation requires thousands of partial forward
passes over a 55 GB source. The measured backend design MUST address this before M3 execution:

1. **Module-group probing.** Where tensors share a transformer block, the backend SHOULD support
   probing at module-group granularity (for example one attention block or MLP) as a cheaper
   first pass, refining only ambiguous groups at tensor granularity. Group results MUST be
   recorded as group-level evidence and MUST NOT be relabelled as tensor measurements.
2. **Prefix and KV reuse.** When the runtime permits, calibration prefix computation SHOULD be
   cached across candidates that mutate only modules downstream of the captured point.
3. **Sample budget.** The backend MUST declare its calibration token budget per candidate and
   record it in backend metadata. The budget MUST be identical across candidates of one tensor
   so comparisons remain internally valid.
4. **Early termination.** Candidates whose proxy loss exceeds a declared dominance bound relative
   to a cheaper candidate for the same tensor MAY be skipped, provided the skip is recorded as
   `supported = false` with a note rather than silently omitted.

A probe run that cannot complete within the declared wall-clock or energy budget MUST persist
verified partial results and resume, never restart from scratch.

### 11.3 Metric definitions

For reference distribution `P` and candidate distribution `Q`:

```text
KL(P || Q) = sum_i P_i × log(P_i / Q_i)
```

Implementations MUST use numerically stable log probabilities and define the token positions
included in aggregation.

For reference hidden tensor `H` and candidate tensor `Hq`:

```text
hidden_state_error = mean((H - Hq)^2)
cosine_distance    = 1 - cosine(H, Hq)
```

Token disagreement is the fraction of evaluated positions where reference and candidate argmax
tokens differ. Task-loss, MTP-loss, and long-context terms MUST identify their dataset slice and
aggregation rule in backend metadata.

### 11.4 Measured output

`sensitivity_map.json` contains the strict report and scalar measurements.
`sensitivity_map.safetensors` MAY contain larger diagnostic tensors or aggregated activation
statistics. The JSON report MUST identify and checksum any companion binary artifact before it
can be used as release evidence.

## 12. Quantization planning

### 12.1 Storage model

The planner estimates storage-adjusted bits per logical weight as:

```text
storage_bpw(16, none) = 16
storage_bpw(b, g)     = b + 32 / g, where b < 16
```

The `32/g` term models one FP16 scale plus one FP16 bias per group. This is a planning proxy.
Actual output size MUST be measured after conversion for a release claim.

For allocations `i` with logical parameter count `n_i`:

```text
nominal_bpw   = sum(n_i × selected_bits_i) / sum(n_i)
effective_bpw = sum(n_i × storage_bpw_i) / sum(n_i)
```

### 12.2 Objective

For a candidate metric vector `m` and normalized profile weights `w`:

```text
predicted_loss(t, c) = sum_k w_k × m_k(t, c)
```

Hard compatibility and protection constraints are applied before objective optimization.
Quality, MTP retention, actual size, and runtime thresholds stored in a plan remain release gates;
the current proxy planner cannot prove them.

### 12.3 Automated single-candidate algorithm

The implemented algorithm:

1. validates profile, runtime, evidence, architecture, and hardware compatibility;
2. filters candidates by allowed bits, methods, group sizes, role floors, and MTP policy;
3. selects the minimum-storage candidate for every tensor;
4. rejects the target when this policy minimum exceeds the BPW budget;
5. repeatedly considers the next precision upgrade for every tensor;
6. calculates marginal predicted-loss reduction per added storage bit;
7. applies the best positive upgrade that remains inside the budget;
8. stops when no beneficial feasible upgrade remains;
9. emits deterministic assignments and distributions.

The planner emits one executable plan per call. Top-N generation is implemented by the refinement
module as deterministic budget perturbations; complete-model measured selection remains required
before a refined result is release evidence.

Tie-breaking is deterministic under the current tensor ordering and candidate ordering. Future
randomized or beam search MUST use and record the plan seed.

The plan-level `group_size` field records the planning request default. When a sensitivity
report contains candidates measured at different group sizes, the per-allocation `group_size`
is authoritative for conversion: the predicate emits each allocation's own value, and the
plan-level field is only the backend default for modules without an explicit override.
Consumers MUST NOT assume all allocations share the plan-level group size.

### 12.4 Manual plan algorithm

Manual planning is allowed only for a revision-pinned, unquantized, supported source inventory.
Rules select tensors by:

- exact case-sensitive tensor glob;
- module glob;
- role set;
- or their conjunction.

Rules are evaluated in list order and the first matching rule wins. A rule that matches no tensor
MUST fail unless `allow_unmatched_rules` is true. A rule MUST NOT lower a tensor below its
protection floor or choose a method/group unsupported by the hardware profile.

Manual plans:

- use zero proxy metrics;
- carry `architecture_prior` evidence;
- require `--allow-unmeasured` for development conversion;
- cannot pass publication preparation.

The reviewed foundation recipe is maintained at
[`examples/qwen36-27b-manual-v0.1.yaml`](../../examples/qwen36-27b-manual-v0.1.yaml).

### 12.5 Planned global refinement

The v1 planner MUST:

1. generate top-N initial plans;
2. convert and evaluate complete candidates;
3. compare proxy predictions with observed regressions;
4. identify precision upgrades or same-budget swaps;
5. run bounded coordinate descent or beam refinement;
6. retain an immutable candidate history;
7. stop on convergence, evaluation budget, or wall-clock budget;
8. select only from complete-model results.

The history MUST record parent candidate, exact change, reason, measured delta, budget impact,
and rejection/selection state.

The refinement result embeds every executable candidate plan plus the selected plan and its
candidate ID. Proxy-only selection is explicitly marked `selection_basis = proxy` and cannot be
used as complete-model release evidence.

`refine-select` requires a measurement set bound to the exact refinement-result SHA-256. Each
record has a stable measurement ID and binds the plan, converted artifact, quality comparison,
validation report, measured BPW, quality/MTP/memory metrics, and normalized objective loss.
Multiple named-host records may identify the same candidate. A candidate is ineligible if any of
its complete records fails validation; selection is deterministic by worst-host objective loss,
worst-host measured BPW, then candidate ID and sets `selection_basis = complete-model`.

`refine-run` turns every embedded plan into an exact, resumable candidate pipeline:

```text
convert
→ evaluate-quality
→ compare-quality
→ benchmark-ab
→ size-evidence
→ validate
→ refine-measure
→ merge measurements
→ refine-select
→ pareto
```

The command is a dry run unless `--execute` is supplied. It binds every input by SHA-256, copies
the refinement result into the run, writes exact argument arrays, and verifies completed output
hashes on resume. A conversion/evaluation failure skips dependent work for that candidate.
Validation exit `1` is retained as complete failed-gate evidence and still proceeds to
`refine-measure`; process/schema failure exit `2` fails the candidate. Final selection succeeds
only when at least one complete candidate passed validation.

## 13. Quantization executor

### 13.1 Plan-to-module predicate

Each plan allocation maps a tensor `module_path` to one MLX module. A trailing `.weight` is
normalized away. Suffix matching MAY accommodate one model-root prefix, but:

- duplicate normalized module paths are rejected;
- an ambiguous suffix match is rejected;
- every sub-16-bit allocation MUST be visited;
- unplanned modules are not quantized;
- BF16 allocations return `False` to the backend predicate.

For a quantized affine allocation, the predicate returns:

```json
{
  "group_size": 64,
  "bits": 6,
  "mode": "affine"
}
```

For a DWQ allocation, AXQuant deterministically samples the module's weight distribution, clips
to the recorded 0.1/99.9-percentile bounds, then returns the same portable affine packing
configuration. The exact sample count, stride, and clip bounds are written to
`axquant_quantizer_execution.json`. Preflight disables the mutation; only the conversion
predicate executes it.

For an AWQ allocation, AXQuant requires per-module calibration activations at conversion time.
It searches activation-aware channel scales (same objective as the AWQ plugin), quantizes in the
scaled domain, unscales the reconstruction into refined float weights, then returns the same
portable affine packing configuration. Channel scales, chosen alpha, and reconstruction MSE are
written to `axquant_quantizer_execution.json`. Preflight admits AWQ without mutating weights;
missing activations fail closed only when refinement execution is enabled.

### 13.1a Verified MLX-LM conversion API contract

The executor depends on the public `mlx_lm.convert` interface (verified against mlx-lm ≥ 0.31,
July 2026):

```python
convert(
    hf_path,
    mlx_path,
    quantize=True,
    q_group_size=...,
    q_bits=...,
    quant_predicate=callable,
    revision=...,
)
```

Binding contract points:

- `quant_predicate` is a callable receiving `(path, module)` and returning `False` (leave the
  module unquantized) or a dict with exactly `group_size`, `bits`, and `mode` keys;
- predicate dicts are only honored for `mode = "affine"`; MLX-LM raises for other modes when a
  predicate is active;
- `mlx_path` must not already exist (MLX-LM raises `ValueError`); AXQuant's staging directory
  satisfies this by construction;
- non-quantized modules are cast to the source config `torch_dtype` by MLX-LM default; AXQuant
  does not override dtype in v0.x;
- MLX-LM's own `mixed_4_6` recipe confirms per-module mixed precision (including 6-bit) is an
  established, supported pattern.

An MLX-LM release that changes the predicate signature, dict keys, or affine-only restriction
is a breaking backend change requiring an adapter update and regression tests before use.

### 13.2 Preflight

Before writing output, conversion MUST:

1. reject non-release evidence unless a development override is explicit;
2. enforce the supported Qwen 3.6 text-path scope;
3. require at least one quantized assignment;
4. reject a pre-existing final output directory;
5. load the model structure lazily;
6. exercise the predicate against named modules;
7. reject unmatched or ambiguous planned quantized modules;
8. reject unsupported quantization methods.

### 13.3 Atomic conversion

Conversion MUST occur in a temporary sibling directory:

```text
<output-parent>/.<output-name>.<random>/artifact/
```

The executor calls the MLX-LM converter with:

- quantization enabled;
- plan group size;
- minimum selected quantized precision as the backend default;
- the per-module plan predicate;
- requested source revision.

After conversion, it MUST:

- confirm every planned quantized module was visited;
- preserve any supplied external MTP bundle;
- re-inspect the complete staging checkpoint and require exact total, MTP, and protected-vision
  logical parameter equivalence with the plan;
- derive measured main and total BPW from the inspected Safetensors files;
- write the exact plan;
- generate and validate AX Engine metadata according to policy;
- write runtime and artifact manifests;
- rename the complete staging artifact to the final output path.

On any failure, temporary output is removed and the final path remains absent. Existing final
output is never overwritten.

### 13.4 MTP sidecar preservation

The v0.x executor supports:

```text
mtp.safetensors
mtplx_runtime.json                 optional companion
ax_mtp_sidecar_manifest.json       optional companion
```

The source argument MAY be the sidecar file or its containing directory. Files are copied with
metadata preservation and verified by SHA-256. If the staging destination already contains the
same file, its checksum MUST match. A different file is a hard error.

External MTP weights are not requantized in v0.x. A plan that disables byte preservation while
supplying an external sidecar MUST fail.

## 14. Runtime contracts

### 14.1 Compatibility levels

| Level | Runtime | Contract |
| --- | --- | --- |
| A | AX Engine | Optimized runtime, native MTP path, runtime-specific manifests and claims |
| B | MLX-LM | Standard backbone inference using portable MLX weights |

Level B does not promise identical acceleration, MTP support, kernel selection, or memory policy.

### 14.2 AX Engine manifest generation

Production conversion runs:

```bash
ax-engine-bench generate-manifest \
  --json \
  --validate \
  <model-directory>
```

The check passes only when:

- the executable exists;
- the command exits `0`;
- `model-manifest.json` exists afterward.
- stdout/stderr contain no AX Engine validation-failure diagnostic.

The default policy is `required`. `if-available` is allowed for development and still fails when
an available backend reports an error. `skip` produces no AX Engine manifest and is not
publishable.

### 14.3 AX Engine readiness

`runtime-check --runtime ax-engine` runs:

```bash
ax-engine doctor \
  --json \
  --mlx-model-artifacts-dir <model-directory>
```

It passes only on exit code `0` and JSON field:

```json
{"result": "ready"}
```

Raw non-JSON output is retained diagnostically and does not satisfy the readiness contract.

### 14.4 MLX-LM compatibility

The production MLX-LM check performs an actual model load and deterministic two-token generation.
It first requires:

- an importable MLX-LM package or recognized MLX-LM executable;
- `config.json`;
- at least one root Safetensors file.

`--static-only` remains a development diagnostic. The generation smoke is compatibility evidence,
not a quality or performance result.

### 14.5 Runtime metadata

`axquant_runtime.json` declares:

- AX Engine as Level A primary runtime;
- MLX-LM as Level B compatibility runtime;
- architecture optimization scope;
- MTP detection and sidecar information;
- preferred group size;
- whether kernel evidence is measured;
- runtime-managed memory recommendations.

Fields such as fused MTP, decode kernel, acceptance retention, and measured speedup MUST remain
unset or `unmeasured` until supported by a benchmark report. Metadata MUST NOT imply that a
detected sidecar was optimized.

### 14.6 Runtime evidence identity

Every `axquant.runtime-check.v2` result contains a `ModelIdentity`. Release checks MUST provide
`--model-id` and an immutable `--revision`; local execution records the resolved artifact path.
Family compatibility evidence is rejected unless the declared model ID and revision equal the
validated candidate and both the explicit model path and runtime command target resolve to the
candidate artifact directory.

## 15. Benchmark and evaluation pipeline

### 15.1 Required comparison set

Every production reference release MUST retain evaluation bundles for:

1. source BF16 or highest available precision;
2. uniform MLX 4-bit;
3. uniform MLX 6-bit;
4. available attributed mixed-precision baseline;
5. available AWQ baseline;
6. available DWQ baseline;
7. the identical AXQuant candidate with MTP disabled;
8. the identical AXQuant candidate with MTP enabled.

Unavailable optional baselines MUST be reported as unavailable, not silently omitted from the
human report.

### 15.2 MTP A/B invariant

The MTP speed comparison MUST use:

- the identical AXQuant checkpoint;
- the identical AX Engine build;
- the identical hardware and power mode;
- the identical prompt set and dataset digest;
- the identical workload and generation controls;
- MTP disabled for the direct bundle;
- MTP enabled for the candidate bundle.

Changing weights, runtime, prompts, or decoding settings invalidates the speedup comparison.

### 15.3 Required measurements

Quality:

```text
perplexity
task scores by named suite
JSON validity
syntax validity
```

MTP:

```text
draft-position token accuracy
average accepted tokens
acceptance rate
rejection rate
effective tokens per forward
repetition rate
divergence rate
```

Hardware:

```text
load time
peak unified memory
prefill tokens/second
direct decode tokens/second
MTP effective tokens/second
kernel fallbacks
device, chip, memory, and OS
optional energy
```

Integrity:

```text
Safetensors validity
index completeness
configuration validity
MTP layout validity
pinned source revision
```

Latency measurements SHOULD also retain warmup count, trial count, prompt lengths, generated
lengths, and distribution summaries in benchmark backend metadata.

### 15.4 Metric formulas

MTP acceptance:

```text
acceptance_rate =
  accepted_draft_tokens / proposed_draft_tokens
```

Acceptance retention relative to a high-precision or 6-bit MTP reference:

```text
acceptance_retention =
  candidate_acceptance_rate / reference_acceptance_rate
```

Effective MTP speedup:

```text
effective_speedup =
  candidate_mtp_effective_tokens_per_second
  / same_candidate_direct_decode_tokens_per_second
```

Peak-memory ratio:

```text
peak_memory_ratio =
  candidate_peak_memory_bytes / reference_peak_memory_bytes
```

Aggregate quality retention MUST be defined by the benchmark suite before public use. Individual
task scores MUST remain visible; a weighted aggregate MUST NOT conceal a critical structured
output, coding, multilingual, or long-context failure.

### 15.5 Harness requirements

The implemented AX Engine harness:

- pin prompt ordering and random seed;
- record temperature, top-p, top-k, max tokens, stop conditions, and draft depth;
- separate warmup from measured trials;
- support deterministic correctness checks where the runtime permits;
- record failed, timed-out, and prematurely terminated requests;
- compare multiple contexts and prompt categories;
- compute distributions, not only one average;
- emits strict `EvaluationBundle` files;
- preserves raw logs or checksummed summaries sufficient for audit.

It calls `ax-engine-bench generate` with tokenizer-produced token IDs. MTP-off trials set
`AX_NO_SPEC=1`; an explicit draft depth is passed to both arms as
`AX_MLX_MTP_MAX_DEPTH=<depth>` and retained in the raw command evidence. Greedy A/B evidence is
rejected unless output token IDs are identical and AX Engine reports MTP active.
`/usr/bin/time -l` supplies peak resident memory, and the raw AX Engine JSON is retained per trial.

## 16. Validation

### 16.1 Cross-bundle invariants

The validator requires:

- reference and MTP-on candidate to use the same evaluation dataset and workload;
- MTP-off and MTP-on candidate bundles to use the same dataset and workload;
- MTP-off and MTP-on bundles to identify the same checkpoint and runtime;
- production MTP validation to use AX Engine;
- the direct candidate to declare `mtp_enabled = false`;
- the MTP candidate to declare `mtp_enabled = true`.
- a calibration manifest whose profile matches, source revision is pinned, and separation is
  attested;
- calibration bytes to differ from both the quality suite and AX Engine prompt suite.

### 16.2 Default profile thresholds

The schema defaults are:

| Metric | Default limit |
| --- | ---: |
| Perplexity relative increase | `0.03` |
| Per-task score drop | `0.02` |
| MTP acceptance absolute drop | `0.03` |
| MTP acceptance retention | `0.95` minimum |
| Draft-position token accuracy drop | `0.04` |
| JSON/syntax validity drop | `0.01` |
| Repetition increase | `0.01` |
| Divergence increase | `0.01` |
| Effective MTP speedup | `1.0` minimum at schema default |
| Peak-memory ratio | `1.0` maximum at schema default |

Product release thresholds can be stricter. The v1 product gate requires at least `1.20×` MTP
speedup and the size/memory targets in the PRD. Both v1 release profiles (`agent-coding` and
`general`) therefore set `min_effective_speedup = 1.20`; profile-specific thresholds from
`axquant.profiles` are authoritative for the invoked validation command.

### 16.3 Missing metrics

When `require_complete_metrics = true`, a missing required reference/candidate pair is an error.
When false, it becomes a warning but cannot support a public claim for that metric.

`ValidationReport.passed` is true only when no issue has severity `error`. Warnings remain in the
report.

## 17. Reporting and publication

### 17.1 Plan and benchmark reports

Human-readable reports MUST present:

- source and revision;
- profile and target class;
- target, nominal, and storage-adjusted BPW;
- evidence kind;
- MTP policy;
- whole-model and MTP precision distributions;
- release constraints;
- validation comparisons and every issue.

Reports MUST explicitly say that proxy planning does not satisfy complete-model quality, MTP, or
speed gates.

### 17.2 Publication preparation gate

`publish-prepare` MUST reject the artifact unless:

- validation passed;
- plan evidence is release-quality;
- calibration provenance exists;
- source revision is pinned;
- plan hash equals the manifest binding;
- all manifest paths are safe and files match size/checksum;
- AX Engine is the primary runtime;
- `model-manifest.json` exists;
- `axquant_runtime.json` exists.

After the gate, preparation:

- incorporates measured acceptance retention and speedup into manifests;
- marks MTP optimized only when MTP is present and validation passed;
- writes benchmark JSON and Markdown;
- packages a release-ready benchmark evidence index and every available evaluation bundle;
- packages passing, checksum-bound `agent-coding` and `general` validations using distinct
  benchmark datasets;
- packages a release-ready hardware registry, every checksum-bound raw/supporting host evidence
  file, and a Pareto frontier from the identical complete-measurement set;
- writes a canonical quantization plan;
- writes a reproduction recipe with exact argument arrays;
- binds the plan, calibration manifest, immutable conversion manifest, and optional MTP sidecar
  by SHA-256;
- records every expected Safetensors path, size, and SHA-256;
- preserves an existing upstream README as `UPSTREAM_README.md`;
- generates a model card that states AX Engine and MLX-LM support levels.

The benchmark index MUST explicitly contain BF16, uniform 4-bit, uniform 6-bit, attributed mixed,
AWQ, DWQ, AXQuant MTP-off, and AXQuant MTP-on entries. BF16, both uniform baselines, and the
identical AXQuant pair are mandatory. Mixed, AWQ, and DWQ MAY be unavailable only with a
non-empty reason. Available entries MUST share profile, dataset digest, random seed, hardware,
power mode, benchmark controls, and complete software versions; name their quantizer and
quantizer version; complete all trials without fallbacks; use the AX Engine runtime and immutable
model revisions; and be bound and packaged by SHA-256.

Publication MUST consume an `axquant.release-validation-index.v1`. It contains exactly the
`agent-coding` and `general` profiles. Both validations and both benchmark indexes must pass,
candidate and uniform-6 identities must be identical across profiles, and the profile dataset
digests must differ. Each source file is checksum-bound before publication and rewritten to its
packaged artifact-relative path.

### 17.3 Publication mutation

`publish` MUST validate `owner/name` syntax. An executed upload MUST first load a release-ready
`axquant.release-audit.v4` and its original `axquant.release-audit-request.v4`, rerun the complete
M0–M8 audit from current evidence, and require the result (excluding only its creation timestamp)
to equal the authorizing audit. The audit MUST record the exact request-file SHA-256. Publication
MUST also require exact toolkit version 1.0.0 and matching candidate
repository/revision/artifact path, then re-hash the audit's M1 artifact/plan, M2 validation index,
and M7 hardware/Pareto inputs before publication preparation. It MUST repeat that binding check
after preparation and before any remote mutation.

- Without `--yes`, it returns the file list and performs no Hub mutation.
- With `--yes`, both `--release-audit` and `--release-audit-request` are required. Only after the
  fresh audit rerun and current evidence bindings pass may the command create the model repository
  and upload the folder through the public Hugging Face API.
- The exact authorizing audit MUST be packaged as `release_audit.json`. Publication MUST reject a
  different existing file rather than overwrite it.
- Private publication is explicit with `--private`.

Credentials MUST come from the normal Hugging Face credential mechanism and MUST NOT be written
to an AXQuant artifact or `.internal`.

### 17.4 Qwen family compatibility matrix

`compatibility-matrix` consumes a strict request that records:

- the `all-official-dense-sizes-at-release` scope policy;
- the official Qwen 3.6 collection URL and a timezone-qualified verification time;
- the complete enumerated official dense model IDs and parameter sizes;
- exactly the `agent-coding` and `general` required profiles.

Its candidates each identify:

- an AXQuant artifact directory;
- an AX Engine doctor result;
- an MLX-LM generation-smoke result;
- a passing release validation report.

For every entry it MUST verify the artifact manifest file set, plan digest, source identity,
supported dense Qwen 3.6 architecture, exact runtime-check target directory, validation profile,
candidate directory, immutable candidate revision, measured weight bytes, and required quality,
MTP, memory, speed, and named-host comparisons. All evidence files are bound by SHA-256 in the
output.

The matrix is v1-release-ready only when the observed dense model IDs equal the declared official
scope exactly, every entry is compatible, and each required model binds both profiles to one
immutable source revision and one artifact/candidate/plan identity. Missing or unexpected models,
missing profiles, cross-profile identity drift, and inconsistent checkpoint counts fail closed.
An incomplete matrix is still written for audit and returns exit code `1`; malformed evidence
returns `2`.

## 18. Output artifact layout

### 18.1 Converted checkpoint

An expected converted directory contains:

```text
<model>/
  config.json
  tokenizer files
  model*.safetensors
  model.safetensors.index.json          when sharded
  model-manifest.json                   AX Engine, required for release
  mtp.safetensors                       when external MTP is included
  mtplx_runtime.json                    optional MTP runtime contract
  ax_mtp_sidecar_manifest.json          optional MTP provenance
  axquant_plan.json
  axquant_runtime.json
  axquant_manifest.json
```

After publication preparation it additionally contains:

```text
  axquant_conversion_manifest.json      immutable pre-publication evidence anchor
  quantization_plan.json
  benchmark_report.json
  benchmark_report.md
  general_benchmark_report.json
  general_benchmark_report.md
  benchmark_evidence_index.json
  general_benchmark_evidence_index.json
  release_validation_index.json
  benchmark_evidence/                  packaged agent-coding evaluation bundles
  general_benchmark_evidence/          packaged general evaluation bundles
  reproduction_recipe.yaml
  README.md
  UPSTREAM_README.md                    when an upstream README existed
```

### 18.2 Development run directory

A complete research run SHOULD preserve:

```text
architecture_report.json
feasibility_report.json
feasibility_report.md
calibration_manifest.json
calibration-cache/
sensitivity_map.json
sensitivity_map.safetensors
hardware_profile.json
candidate_allocations.json
candidate_history.json
quantization_plan.json
axquant_quantizer_execution.json
evaluation/
benchmark_report.json
benchmark_report.md
axquant_manifest.json
reproduction_recipe.yaml
```

Files for planned capabilities are required when those capabilities are used; they are not
expected from the current foundation CLI.

## 19. Reproducibility, caching, and resumption

### 19.1 Reproduction identity

A production result is identified by:

```text
source model ID + immutable revision
config digest
AXQuant version
Python version
MLX version
MLX-LM version
AX Engine version
Safetensors and Pydantic versions
calibration dataset digest
evaluation dataset digest
profile
random seed
quantization plan digest
hardware profile
benchmark protocol version
```

A change in any identity-defining field creates a different run. Results MAY be compared but
MUST NOT share a cache entry unless the cache contract explicitly excludes the changed field.

### 19.2 Stable fingerprints

Semantic fingerprints SHOULD exclude nondeterministic `created_at` values when identifying
equivalent inputs. Final artifact manifests still retain creation timestamps and bind file bytes
by SHA-256.

### 19.3 Executable release recipes

`axquant.reproduction.v3` MUST contain ordered argument arrays for:

1. downloading the immutable source revision;
2. converting it with the published plan and calibration manifest;
3. checking AX Engine and MLX-LM;
4. verifying the regenerated artifact.

Commands are argument arrays rather than shell strings. The recipe MUST bind its input files and
immutable conversion manifest by SHA-256. It MUST record the exact expected Safetensors file set,
individual byte sizes, individual SHA-256 values, total weight bytes, and logical parameters.
When conversion consumes an AXQuant-prepared MTP layout, the recipe MUST also bind both
`ax_mtp_sidecar_manifest.json` and `mtplx_runtime.json` by path, size, and SHA-256.

`verify-reproduction` MUST reject unsafe relative paths and report failure when any bound input,
source identity, plan digest, logical-parameter count, total weight byte count, Safetensors path,
file size, or file checksum differs. Exit code `0` means all checks passed; `1` means verification
completed and found a mismatch; invalid input or I/O failure uses exit code `2`.

The M8 `release-audit` wheel check MUST require a single metadata, wheel, entry-point, and `RECORD`
member; reject duplicate or unsafe archive paths; and verify that `RECORD` covers every member
exactly once with the declared byte size and URL-safe base64 SHA-256 digest. The `RECORD` row for
itself MUST have empty hash and size fields. The wheel metadata version MUST equal the required
release version and the version claimed by the artifact manifest, the plan's software provenance,
and both version fields in the reproduction recipe. A final wheel MUST carry the
`Development Status :: 5 - Production/Stable` classifier, declare Python `>=3.11` and the MIT
license, and include the five required runtime dependencies (`huggingface-hub`, `pydantic`,
`pyyaml`, `safetensors`, and `structlog`). M1/M8 MUST also cross-check the artifact and recipe
against the plan's source, profile, calibration, target class, precision distributions, MTP
policy, runtime scope, and random seed as applicable. M8 MUST load the
publication-ready plan, immutable conversion manifest, reproduction recipe, both-profile
validation index and referenced benchmark evidence, hardware registry and referenced evidence,
refinement measurements, and Pareto report from the artifact itself. Artifact-relative path
rewrites are allowed, but their semantic contents and bound checksums MUST match the external
evidence graph audited for M0–M7. M5 MUST checksum-bind the original compatibility request and
reload every requested checkpoint's artifact manifest, plan, AX Engine check, MLX-LM check, and
profile validation. It MUST also bind the scope policy, catalog URL, verification time, required
dense models, required profiles, and the request/matrix checkpoint counts. The matrix entry
identities, architecture profile, precision coverage, measured BPW, MTP status, plan digest,
runtime targets/status, validation status, candidate weight bytes, and file checksums MUST still
match those inputs. The selected candidate entries MUST additionally bind the identical M1 and
both profile-validation inputs supplied to the final audit.

M0 MUST recompute feasibility from the report contents rather than accepting its top-level
status: exactly one complete 4-bit, 6-bit, and mixed baseline; a complete unquantized BF16 source;
logical-parameter equivalence; matching adapter and optimization scope; pinned revisions; MTP
parameters in every checkpoint; passed MLX-LM static checks everywhere; and passed AX Engine
doctor checks for all quantized baselines.

M1 MUST reject duplicate, empty, traversal, absolute, or backslash artifact-manifest paths. Every
Safetensors file below the release artifact MUST appear exactly once in the manifest, and every
listed file MUST retain its recorded size and SHA-256.

M2 MUST reload every available indexed evaluation and recompute required-baseline availability,
entry identity, profile/dataset/seed consistency, immutable model revisions, checkpoint integrity,
complete software and named-hardware provenance, zero kernel fallbacks, complete trial counts,
benchmark-control equivalence, and identical-checkpoint MTP-off/on pairing. A
`release_ready=true` benchmark index is not sufficient by itself. Agent-coding and general
entries MUST bind the loaded reports, use one identical candidate/reference pair, and carry two
distinct non-empty dataset digests.

M3 MUST load the packaged `calibration_manifest.json`, verify the checksum recorded by the
sensitivity/plan calibration evidence, and require matching source, profile, dataset, domains,
sample count, sequence length, and random seed plus an explicit calibration/evaluation separation
attestation. Sensitivity and plan architecture profiles MUST match, and every quantizable
non-preserved tensor requires finite, tensor-scoped measured candidates at its policy-required
precisions.

M6 MUST prove a parent-to-child interaction gain from the bound complete-candidate measurement
set. Both candidates require validated measurements with their own plan hashes and matching
profile; each history BPW and loss MUST equal the worst named-host measurement, and the child loss
MUST be strictly lower. A history-only improvement claim is not release evidence.
Complete-candidate measurement construction MUST independently require the authoritative profile
thresholds, pass/error consistency, passing aggregate-quality, perplexity, MTP acceptance,
effective-speed, peak-memory, and zero-fallback metrics, plus a governed plan-bound exception for
any passing artifact-size overage.

M7 MUST rebuild the complete Pareto report from the checksum- and semantic-digest-bound
measurement set and compare it to the supplied report excluding only its creation timestamp.
Every point, dominator, frontier flag, and frontier candidate ID must be reproducible.

### 19.4 Resume semantics

Resumable stages MUST use:

1. an immutable input manifest;
2. content-addressed or uniquely named partial shards;
3. checksum verification on reuse;
4. an atomic completion marker;
5. fail-closed handling of missing, extra, or mismatched shards.

Changing inputs MUST create a new cache/run directory rather than overwriting an incomplete run.

## 20. Failure, integrity, and security requirements

### 20.1 Fail-closed boundaries

AXQuant MUST fail rather than infer intent when it encounters:

- unsafe or missing index paths;
- duplicate tensor or module mappings;
- an already quantized conversion source;
- an unsupported architecture for conversion;
- an unpinned source for manual release planning;
- an unsupported precision, method, or group size;
- a protected-precision violation;
- unmatched plan modules;
- different MTP sidecar bytes at the destination;
- a missing required runtime executable;
- an incomplete metric pair under a complete validation policy;
- a failed release validation;
- any artifact checksum mismatch.

### 20.2 Data and credential handling

- Calibration and evaluation samples MUST NOT be embedded in public manifests.
- Dataset identifiers and digests SHOULD be sufficient to reproduce public data.
- Private data paths MAY appear in internal development manifests but SHOULD be sanitized before
  publication.
- Hub tokens and other credentials MUST never be logged, serialized, or committed.
- External command arguments MUST be passed as argument arrays, not shell strings.
- Artifact-relative paths MUST reject traversal and absolute references.

### 20.3 Clean-room provenance

Development records MUST distinguish:

- public research references;
- public MLX/MLX-LM APIs;
- externally attributed baseline artifacts;
- AXQuant-generated analysis and measurements.

No external sensitivity allocation, private package behavior, or copied implementation detail
may be presented as an AXQuant design input.

## 21. Testing and CI

### 21.1 Required test layers

Unit tests MUST cover:

- strict schema validation and cross-field invariants;
- path safety and checkpoint membership;
- packed logical parameter accounting;
- Qwen adapter match/support boundaries;
- tensor role and protection classification;
- tied-weight harmonization;
- storage BPW calculations;
- deterministic planning and infeasible budgets;
- MTP sidecar policy;
- predicate exact, suffix, ambiguous, and unmatched behavior;
- atomic conversion success/failure;
- runtime command construction and result parsing;
- validator threshold edges and missing metrics;
- publication gates and checksum validation.

Integration tests SHOULD use tiny synthetic Safetensors fixtures. Tests MUST NOT require the
27B production weights.

### 21.2 Backend tests

MLX/AX Engine integration tests SHOULD be marked separately so schema-only CI can run on
non-macOS hosts. A release candidate MUST additionally run on the reference Apple Silicon
hardware with the pinned runtime matrix.

### 21.3 Quality gates

Every code change SHOULD pass:

```bash
pytest
ruff check .
mypy src
python -m build
```

Documentation-only changes MUST at least validate Markdown relative links and confirm that schema
and CLI names still match the source.

## 22. Extension interfaces

### 22.1 Architecture adapters

A new adapter MUST implement:

```python
matches(model_reference, config) -> bool
profile(model_reference, config) -> ArchitectureProfile
classify_tensor(name, source_file) -> TensorRole | None
```

It also requires:

- an explicit adapter ID and version;
- signature tests for supported and rejected checkpoints;
- tensor role fixtures;
- protection policy review;
- conversion and runtime compatibility evidence;
- an ADR update when the product support boundary changes.

### 22.2 Quantizer plugins

A future quantizer plugin MUST declare:

- method ID;
- supported bits, groups, dtypes, tensor shapes, and roles;
- required calibration or learned state;
- deterministic/cache behavior;
- serialization format;
- conversion fallback;
- AX Engine and MLX-LM compatibility;
- hardware benchmark evidence.

Failure of optional refinement MAY fall back only when the plan records the fallback and the
result is revalidated. Silent method substitution is forbidden.

### 22.3 Measurement backends

A measurement backend MUST declare:

- backend/version ID;
- required runtime;
- capture points;
- metric normalization;
- dataset and tokenization contract;
- deterministic tolerance;
- cache-key fields;
- emitted evidence kind.

Only a backend approved for `measured` evidence may create release-quality sensitivity reports.

### 22.4 Runtime profiles

A hardware profile registry entry MUST be based on measured kernels and identify:

- chip and device class;
- OS, MLX, MLX-LM, and AX Engine versions;
- supported bit-width/group combinations;
- kernel fallbacks;
- shape coverage;
- measurement protocol and tolerance.

Unmeasured hardware recommendations must retain `kernel_evidence = unmeasured`.

`hardware-registry` resolves each entry to an exact stable measurement ID and verifies the
complete measurement, plan, release-quality sensitivity report,
validation report, MTP-off/on evaluation bundles, raw benchmark results, and quantizer execution
manifest. It reconstructs bit/group/method/role/shape coverage from the plan and sensitivity
report, checks every quantized module against conversion records, retains every exact benchmark
command, and refuses measured kernel status when a raw trial fails or any fallback is nonzero.
Multiple entries may certify the same candidate on distinct named hosts without conflating their
measurements. Publication requires a release-ready registry and a Pareto frontier derived from
the identical measurement set and exact registry measurement IDs.

## 23. Current gaps and implementation order

The repository has a verified vertical slice and release-gated pipeline, but no production
AXQuant v1 checkpoint yet. Remaining work proceeds in this order:

1. complete the resumable release-scale per-tensor measurement for the pinned Qwen3.6-27B BF16
   source;
2. generate automated plans and convert complete affine/DWQ refinement candidates;
3. run disjoint `agent-coding` and required `general` quality suites plus the complete benchmark
   evidence index;
4. resolve the current protected-precision size floor, which exceeds the 110% uniform-4 gate;
5. obtain correct, repeatable AX Engine MTP behavior meeting acceptance and 1.20x speed gates;
6. select a complete candidate from measured interaction/refinement results and produce the
   named-host Pareto frontier;
7. reverify the release-time official dense catalog and build the dual-profile compatibility
   matrix for every declared size;
8. run external compatibility testing, reproduce the published recipe, and prepare the guarded
   v1 artifacts.

The pinned BF16 source, uniform baselines, manual vertical slice, actual AX Engine and MLX-LM
checks, calibration cache, benchmark harnesses, publication gates, reproducibility verifier,
compatibility matrix, and evidence indexes are implemented and audited. None of those
development results overrides a failed size/MTP gate or substitutes for the running measured
planner evidence.

## Appendix A: Reference workflow

### A.1 Feasibility

```bash
axquant feasibility \
  --reference-4bit /models/qwen36-27b-4bit \
  --reference-6bit /models/qwen36-27b-6bit \
  --source-bf16 /models/qwen36-27b-bf16 \
  --run-runtime-checks \
  --require-ready \
  --output run/feasibility_report.json \
  --markdown-output run/feasibility_report.md
```

### A.2 Inspection and calibration

```bash
axquant inspect \
  --model /models/qwen36-27b-bf16 \
  --model-id Qwen/Qwen3.6-27B \
  --revision <immutable-revision> \
  --output run/architecture_report.json
```

```bash
axquant calibrate \
  --model Qwen/Qwen3.6-27B \
  --revision <immutable-revision> \
  --dataset data/calibration-agent-coding.jsonl \
  --profile agent-coding \
  --domains coding,json,tool-calling,cjk \
  --max-seq-length 2048 \
  --seed 0 \
  --attest-calibration-eval-separation \
  --output run/calibration-cache
```

### A.3 Analysis and planning

Until the measured backend is implemented, the following produces development evidence only:

```bash
axquant analyze \
  --model /models/qwen36-27b-bf16 \
  --model-id Qwen/Qwen3.6-27B \
  --revision <immutable-revision> \
  --profile agent-coding \
  --bits 4,6,8,16 \
  --group-size 64 \
  --output run/sensitivity_map.json
```

Production planning will consume measured calibration evidence:

```bash
axquant plan \
  --sensitivity run/sensitivity_map.json \
  --target-bpw 4.5 \
  --bits 4,6,8,16 \
  --group-size 64 \
  --minimum-quality 0.98 \
  --minimum-mtp-retention 0.95 \
  --minimum-mtp-speedup 1.20 \
  --max-size-ratio 1.10 \
  --mtp protected \
  --mtp-bits 8,16 \
  --mtp-min-bits 8 \
  --output run/quantization-plans
```

### A.4 Conversion

```bash
axquant convert \
  --model /models/qwen36-27b-bf16 \
  --revision <immutable-revision> \
  --plan run/quantization-plans/plan-01.json \
  --mtp-sidecar /models/qwen36-mtp \
  --ax-engine-manifest required \
  --output /models/AX-Qwen3.6-27B-MLX-AXQuant-4bit
```

### A.5 Validation and release preparation

```bash
axquant validate \
  --reference-evaluation run/evaluation/uniform-6bit.json \
  --candidate-direct-evaluation run/evaluation/axquant-mtp-off.json \
  --candidate-evaluation run/evaluation/axquant-mtp-on.json \
  --profile agent-coding \
  --output run/benchmark_report.json
```

```bash
axquant publish-prepare \
  --model /models/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --validation-index run/release_validation_index.json \
  --hardware-registry run/hardware_profile_registry.json \
  --pareto-report run/pareto_report.json
```

Publication preview:

```bash
axquant publish \
  --model /models/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --validation-index run/release_validation_index.json \
  --hardware-registry run/hardware_profile_registry.json \
  --pareto-report run/pareto_report.json \
  --release-audit run/release_audit.json \
  --release-audit-request examples/release-audit-request.yaml
```

Actual upload requires explicit `--yes`; the audit and request flags are optional for preview but
mandatory for that upload.

## Appendix B: Schema evolution policy

- Additive optional fields MAY remain within a schema major version only when old readers are
  known to reject or safely ignore them according to the declared compatibility policy.
- Removing a field, changing semantics, widening accepted evidence, or changing a release
  invariant requires a new schema version.
- Readers MUST reject unknown major versions.
- Migrations MUST be explicit, deterministic, tested, and preserve the original artifact.
- A migration MUST NOT upgrade `architecture_prior` into release-quality evidence.
- Published artifacts MUST retain their original schema files even when a newer tool can migrate
  them for analysis.
