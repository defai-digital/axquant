# ADR 0001: Pipeline Architecture and Artifact Contracts

**Status:** Accepted for foundation implementation  
**Date:** 2026-07-28

## Context

AXQuant must evolve from manual mixed-precision conversion to measured per-tensor, MTP-aware,
interaction-aware, hardware-aware optimization. The pipeline must be testable without loading a
27B model, resumable across expensive probes, and compatible with public MLX interfaces and the
AX Engine native artifact contract.

Proxy analysis must never be mistaken for release evidence. External MTP sidecars also require a
layout-aware boundary because a generic tensor quantizer cannot safely reinterpret runtime-specific
packing or normalization.

## Decision

AXQuant uses a staged artifact pipeline with Pydantic-validated, versioned boundaries:

```text
inspect
  -> calibrate
  -> analyze
  -> plan
  -> convert
  -> validate
  -> report
  -> publish-prepare
  -> publish
```

Each stage reads immutable upstream artifacts and writes its result atomically. Stages do not
silently infer release evidence from architecture rules.

## Module boundaries

| Module | Responsibility | Must not do |
| --- | --- | --- |
| `inspector` | Safetensors/config inventory, roles, MTP, ties, eligibility | Load all weights into RAM |
| `calibration` | Dataset identity, validation, cache manifest, seed | Bundle unlicensed default data |
| Qwen 3.6 adapter | Family-specific paths, hybrid attention, MTP, and vision classification | Contain planner policy |
| sensitivity backend | Candidate probes and metric evidence | Allocate final precision |
| `planner` | Constraints, objective, deterministic candidate allocation | Claim complete-model quality |
| quantizer backend | Affine/AWQ/DWQ execution | Change plan without recording it |
| MTP evaluator | Draft accuracy, acceptance, verification, throughput | Compare different workloads |
| hardware benchmark | Memory, latency, kernels, device provenance | Substitute theoretical BPW for timing |
| `converter` | Apply plan, export MLX, preserve MTP, request AX native manifest | Quantize unknown external sidecar layouts |
| runtime adapter | Generate AX manifest, run doctor, describe MLX-LM fallback | Invent runtime success |
| `validator` | Complete-model hard release gates | Accept missing or mismatched evidence |
| `reporting` | JSON/Markdown/model card/reproduction recipe | Invent metrics |
| `publisher` | Guarded Hub upload | Upload a failed or incomplete candidate |

Future adapters implement protocols equivalent to:

```python
class ArchitectureAdapter(Protocol):
    adapter_id: str

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool: ...

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile: ...

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None: ...


class SensitivityBackend(Protocol):
    def probe(
        self,
        model: ModelIdentity,
        calibration: CalibrationManifest,
        tensor: TensorSpec,
        candidate: QuantCandidate,
    ) -> CandidateMeasurement: ...


class QuantizerBackend(Protocol):
    def supports(self, assignment: Allocation, hardware: HardwareProfile) -> bool: ...
    def apply(
        self, source: ModelIdentity, plan: QuantizationPlan, output: Path
    ) -> ExecutionLog: ...


class MtpEvaluator(Protocol):
    def evaluate(self, model: ModelIdentity, workload: Workload) -> MtpMetrics: ...
```

`ArchitectureAdapter` reflects the implemented protocol in `axquant.architectures.types`.
Protection floors are planner policy, not adapter behavior; adapters classify roles and the
planner and manual planner enforce the minimum-precision tables.

Protocol packages are added when the corresponding milestone becomes executable; serialized
contracts are stable first.

## Artifact schemas

### `architecture_report.json`

Schema: `axquant.inventory.v1`.

Required data:

- source ID, immutable revision, architecture, and local snapshot;
- config digest and weight file list;
- tensor name, module path, shape, dtype/current precision, logical parameters;
- role, quantization eligibility, protection recommendation, tie relationship;
- MTP presence and integrated/external classification;
- adapter ID, product support level, and text-path optimization scope;
- quantized-source status and warnings.

Checkpoint discovery follows `model.safetensors.index.json` when present and adds only the root
MTP sidecar. Unindexed nested Safetensors are auxiliary data and are excluded. This keeps logical
parameter counts aligned across uniform and mixed checkpoints while still including indexed
nested shards such as a separately stored vision tower.

### `feasibility_report.json`

Schema: `axquant.feasibility.v1`.

The report audits revision pinning, index completeness, native manifest validity, tokenizer
presence, MTP runtime/provenance consistency, logical parameter equivalence, exact weight bytes,
AX Engine doctor, and MLX-LM static availability. Its states are:

- `ready-for-conversion`: required baselines and the BF16 source are complete;
- `baseline-ready`: baselines are complete but the BF16 source was not supplied;
- `blocked`: a supplied artifact, identity comparison, or requested runtime check failed.

### `calibration_manifest.json`

Schema: `axquant.calibration.v1`.

Required data:

- model/tokenizer revisions;
- profile, dataset identity and SHA-256;
- sample count, domains, sequence length, seed;
- calibration/evaluation separation attestation;
- future tokenized-cache and activation-cache digests.

### `sensitivity_map.json`

Schema: `axquant.sensitivity.v1`.

One entry per supported tensor, with one candidate record per precision/method. All metric fields
are loss-oriented: lower is better. Cosine similarity is stored as cosine distance and token
agreement as disagreement.

Evidence kind is one of:

- `measured`: produced by an AXQuant probe;
- `imported`: externally measured with full provenance;
- `architecture_prior`: unmeasured and non-release.

Measured/imported evidence requires a calibration manifest.

### `quantization_plan.json`

Schema: `axquant.plan.v1`.

Required data:

- profile, target mode, seed, source and analysis digests;
- candidate bits, method and group size;
- per-tensor assignment, metrics, predicted loss, and reason;
- target, nominal, and storage-adjusted BPW;
- precision and MTP distributions;
- hard release constraints;
- hardware capability profile;
- MTP policy;
- AXQuant, Python, MLX, MLX-LM, Safetensors, and Pydantic versions;
- `global_validation_required: true`.

### `manual_recipe.yaml`

Schema: `axquant.manual-recipe.v1`.

The v0.1 recipe defines an ordered list of role/tensor/module selectors and explicit precision,
method, group size, and rationale. Mandatory protection floors override the recipe default, while
an explicit unsafe rule fails. Unmatched rules fail unless the recipe opts in, tied weights are
harmonized, and the resulting BPW must satisfy the declared limit. The output remains
`architecture_prior` evidence and is never release-quality.

### `axquant_manifest.json`

Schema: `axquant.artifact.v1`.

Required data:

- source and plan provenance;
- achieved effective BPW and distributions;
- MTP detected/optimized/default status;
- acceptance retention and measured speedup after validation;
- all output file sizes and SHA-256 hashes;
- software versions.

### `model-manifest.json` and `axquant_runtime.json`

`model-manifest.json` belongs to AX Engine and is generated by
`ax-engine-bench generate-manifest --validate`. AXQuant does not maintain a second implementation
of that schema.

`axquant.runtime.v1` records:

- Compatibility Level A AX Engine as primary runtime;
- Compatibility Level B MLX-LM as standard-inference fallback;
- text-path or full-model optimization scope;
- MTP preservation and measured-state fields;
- preferred group size and explicitly unmeasured kernel fields;
- non-binding memory-policy recommendations.

### `benchmark_report.json`

Schema: `axquant.validation.v1`, derived from versioned evaluation bundles.

Every bundle identifies its baseline class, workload, dataset digest, seed, software, and hardware.
The validator rejects dataset/workload mismatches.

### `reproduction_recipe.yaml`

Contains only reproducibility inputs and commands, never secrets:

- immutable source and tokenizer revisions;
- calibration and evaluation digests;
- software versions;
- seed and profile;
- plan and hardware profile digests;
- output repository.

## Planner algorithm

### Foundation greedy solver

1. Validate precision/method/kernel support.
2. Apply hard protection floors.
3. Start every tensor at its lowest policy-compliant candidate.
4. Fail if this minimum exceeds the BPW budget.
5. Compute the next precision upgrade for each tensor.
6. Rank upgrades by marginal proxy-loss reduction per added storage bit.
7. Apply the best feasible upgrade.
8. Repeat until no beneficial upgrade fits.
9. Emit one deterministic plan.

Storage cost for affine precision is:

```text
bits + 32 / group_size
```

BF16 uses 16 BPW. Exact artifact byte size remains authoritative after conversion.

### Candidate generator

Phase 4 adds top-N generation:

- retain alternative upgrades within a configurable efficiency band;
- deduplicate assignment vectors;
- enforce BPW and unsupported-tensor constraints;
- rank by profile proxy objective;
- hand every survivor to complete-model validation.

### Global refinement

Phase 6:

1. evaluate complete candidates;
2. attribute regressions to candidate tensor groups using cached activations;
3. propose upgrades and equal-cost swaps;
4. run bounded coordinate descent;
5. retain Pareto improvements;
6. stop on convergence, wall-time, candidate, or conversion budget.

No proxy constraint marks a candidate production-ready. Only complete-model gates do.

## MTP rules

- The backbone generates token position one.
- Protected mode permits 8-bit or BF16 integrated MTP components.
- Adaptive 6-bit MTP requires measured evidence and complete-model validation.
- MTP norms and output heads remain protected by default.
- External sidecars are copied byte-for-byte and checksum-verified in the foundation backend.
- AX Engine runs MTP-on/off benchmarks on identical checkpoint weights and workloads.
- Acceptance retention, verification overhead, and effective throughput are release gates.

## Validation semantics

Higher-is-better metrics use absolute drop or normalized retention. Lower-is-better metrics use
relative increase. Aggregate retention never divides an error metric.

Production validation requires:

- BF16/high-precision, uniform-4, and uniform-6 baselines;
- identical calibration-independent evaluation inputs;
- AXQuant MTP off/on pair;
- integrity and revision checks;
- device and software provenance.

An attributed mixed-precision, AWQ, or DWQ baseline may be absent only when marked unavailable.

## Resume and concurrency

Every expensive unit uses a deterministic key:

```text
sha256(
  source_revision
  + calibration_digest
  + tensor_name
  + precision
  + quantizer
  + quantizer_config
  + software_versions
)
```

Completed units are immutable. Workers write temporary artifacts and atomically rename them.
Resume skips verified units. A failed unit records structured error state without masquerading as
unsupported evidence.

## Failure behavior

AXQuant fails before conversion when:

- the source is already quantized without explicit inventory-only permission;
- required modules are absent or ambiguous;
- precision/method/group size lacks backend support;
- protection makes the target infeasible;
- evidence is unmeasured without explicit development override;
- top-N behavior is requested before a candidate backend exists.

It fails before publication when:

- validation is missing or failed;
- manifest/plan hashes disagree;
- output files or checksums are incomplete;
- source revision, dataset digest, seed, or hardware provenance is missing;
- the candidate uses unmeasured evidence.

## Current implementation mapping

The foundation repository implements versioned schemas, index-driven Qwen 3.6 Safetensors
inspection, quantized logical-parameter reconstruction, feasibility audits, reviewed manual
recipes, calibration manifests, explicit architecture priors, one-plan greedy allocation, MTP
and vision protection, MLX-LM affine weight conversion, AX Engine manifest/doctor adapters,
atomic conversion staging, runtime tiers, validation evidence gates, reports, guarded
publication, and tests.

The following remain milestone work and are not presented as complete:

- MLX forward sensitivity probes and activation cache;
- built-in calibration packs;
- top-N planning and global coordinate descent;
- AX Engine MTP benchmark runner;
- AWQ/DWQ executor plugins;
- hardware microbenchmark registry;
- layout-aware external MTP quantization.

## Consequences

The project can develop planner and artifact correctness independently of expensive model runs.
It also has more explicit artifacts than a monolithic converter, but those artifacts are necessary
for reproducibility, release gating, and clean external comparisons.
