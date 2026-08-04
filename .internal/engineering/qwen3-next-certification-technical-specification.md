# Qwen3-Next Non-MTP Certification Technical Specification

**Document status:** Accepted for implementation
**Applies to:** QN0–QN8 in `../product/qwen3-next-certification-prd.md`
**Accepted decisions:** AXQ-032, AXQ-033, AXQ-034
**ADR:** `../architecture/decisions/0009-qwen3-next-non-mtp-certification.md`
**Base documents:** `technical-specification.md`, `expansion-technical-specification.md`, and
`../product/release-best-practices.md` remain authoritative where this specification does not
explicitly amend them.
**Last reviewed:** 2026-08-03

## 1. Scope

This specification defines the additive implementation required to certify an exact non-MTP
Qwen3-Next checkpoint without changing the existing Qwen 3.6 M0–M8 audit. It covers:

1. certification-track dispatch and source-derived eligibility;
2. strict N0–N8 request/output schemas;
3. exact-checkpoint certification registry and publication authorization;
4. non-MTP feasibility, benchmark, validation, quality, refinement, hardware, Pareto, and
   reproduction evidence;
5. AX Engine native-manifest support for independently mixed raw/packed attention projections;
6. coding-suite v2 provenance and executable scoring;
7. migration, testing, archival, and operational requirements.

It does not certify the current development artifacts. It builds the policy/tooling required for
a later formal cycle.

### 1.1 Current development evidence snapshot

The following facts are inputs to design and regression testing only:

| Fact | 4-bit | 6-bit |
| --- | ---: | ---: |
| Source revision | `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` | same |
| Planned BPW | 4.7999999974 | 5.9999999936 |
| Measured BPW | 4.8000226491 | 6.0000206560 |
| Total artifact bytes | 47,817,915,336 | 59,768,657,362 |
| MLX-LM generation smoke | pass | pass |
| AX Engine native manifest | generated/valid | blocked by mixed Q/K/V/O layout |
| AX Engine measured trials | 3/3 | 0/3 |
| Development perplexity | 6.6250 | 6.6331 |

BF16 development perplexity was 6.6981. The suite contained 15 tasks and 410 evaluated tokens,
with zero syntax validity on all arms. These values must not be copied into an N0–N8 release
bundle as if they were formal evidence.

### 1.2 Capability truth table

| Capability | QN1 | QN2 | QN3 | QN4 | QN5+ |
| --- | --- | --- | --- | --- | --- |
| Track discriminator and source eligibility | ✅ | | | | |
| Strict non-MTP request/audit schemas | ✅ | | | | |
| Exact-checkpoint registry schema | ✅ | | | | |
| M0–M8 regression harness | ✅ | | | | |
| AX Engine mixed projection manifest/load | | ✅ | | | |
| Formal-host Metal readiness | | ✅ | | | |
| Coding-suite v2 manifest/scorer | | | ✅ | | |
| Matched direct baseline index | | | | ✅ | |
| Formal candidate evidence | | | | | ✅ |
| Publisher authorization | ✅ | | | | exercised QN6 |

## 2. Architectural layout

### 2.1 Modules

```text
src/axquant/
├── certification/
│   ├── __init__.py
│   ├── dispatch.py                 # request schema dispatch; no heuristic default
│   ├── common.py                   # side-effect-free common verification helpers
│   ├── qwen3_next_direct.py        # N0–N8 builder
│   └── registry.py                 # exact-checkpoint certification registry
├── schema/
│   ├── certification.py            # new strict schemas
│   └── coding_suite.py             # suite manifest and executable-score schemas
├── coding_suite.py                 # suite loading, overlap checks, scoring aggregation
├── release_audit.py                # existing v4 builder retained; dispatch wrapper only later
├── publisher.py                    # strict request/audit union and rerun
└── support_policy.py               # renders exact certified checkpoints separately
```

Common verification helpers may be factored out only after golden tests prove their results for
existing v4 requests. The first implementation must not rewrite M0–M8 and N0–N8 simultaneously.

### 2.2 Cross-repository AX Engine surfaces

```text
AX Engine
├── native manifest generator       # per-tensor logical/packed shape validation
├── Qwen3-Next weight loader         # independently mixed Q/K/V/O projections
├── doctor                           # Metal toolchain and artifact readiness
├── generate                         # deterministic parity and benchmark runtime
└── regression fixtures              # BF16-Q + packed-K/V/O and other permutations
```

AXQuant never vendors or reimplements the AX Engine manifest schema. It invokes the released AX
Engine binary and records its structured result, preserving AXQ-004.

## 3. Certification-track schemas

### 3.1 Track identifier

Add a strict enum in `schema/certification.py`:

```python
class CertificationTrack(StrEnum):
    QWEN36_MTP_V1 = "qwen36-mtp-v1"
    QWEN3_NEXT_DIRECT_V1 = "qwen3-next-direct-v1"
```

The existing v4 request has no new field and continues to map to `QWEN36_MTP_V1` only inside the
dispatcher. Its serialized form and stable hash do not change.

### 3.2 Exact source scope

```python
class ArchitectureFingerprint(StrictModel):
    model_type: Literal["qwen3_next"]
    architecture: Literal["Qwen3NextForCausalLM"]
    text_layer_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    full_attention_interval: int = Field(gt=0)
    expert_count: int = Field(gt=0)
    experts_per_token: int = Field(gt=0)
    expert_intermediate_size: int = Field(gt=0)
    mtp_declared: Literal[False]
    vision_present: bool
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExactCertificationScope(StrictModel):
    track: Literal[CertificationTrack.QWEN3_NEXT_DIRECT_V1]
    source_model: ModelIdentity
    architecture: ArchitectureFingerprint
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware_scope_ids: list[str] = Field(min_length=1)
    policy_id: Literal["axquant.qwen3-next-direct-policy.v1"]
```

`source_model.revision` is mandatory and must be a full immutable revision. A symbolic branch,
tag that can move, missing revision, or local path without an `axquant_source.json` binding fails
N0.

### 3.3 Non-MTP audit request

New schema literal:

```text
axquant.qwen3-next-release-audit-request.v1
```

Normative fields:

```python
class Qwen3NextReleaseAuditRequest(StrictModel):
    schema_version: Literal["axquant.qwen3-next-release-audit-request.v1"]
    certification_scope: ExactCertificationScope
    artifact_directory: str
    source_inventory: str
    source_checkpoint_manifest: str
    feasibility_report: str
    sensitivity_report: str
    sensitivity_lineage: list[str]
    refinement_result: str
    refinement_measurements: str
    release_validation_index: str
    benchmark_evidence_index: str
    coding_suite_manifest: str
    coding_suite_self_test: str
    hardware_registry: str
    pareto_report: str
    compatibility_matrix: str
    compatibility_request: str
    reproduction_recipe: str
    reproduction_verification: str
    ax_engine_manifest_check: str
    ax_engine_doctor_check: str
    ax_engine_runtime_check: str
    mlx_lm_runtime_check: str
    evidence_archive_index: str
    toolkit_wheel: str
    required_toolkit_version: str
    policy_sha256: str
    release_exceptions: list[str] = Field(default_factory=list, max_length=0)
```

There is deliberately no MTP field and no release-exception slot. The empty exception list keeps
the refusal machine-readable while preventing a generic request rewriter from adding one.

All paths resolve relative to the request. Every file is required before gate evaluation begins;
the builder emits no partial “passed” audit when request material is missing or unparsable.

### 3.4 Audit output

```python
class NonMtpGateId(StrEnum):
    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    N5 = "N5"
    N6 = "N6"
    N7 = "N7"
    N8 = "N8"


class Qwen3NextReleaseAuditCheck(StrictModel):
    gate_id: NonMtpGateId
    name: str
    passed: bool
    evidence_sha256: dict[str, str]
    issues: list[str]


class Qwen3NextReleaseAudit(StrictModel):
    schema_version: Literal["axquant.qwen3-next-release-audit.v1"]
    certification_scope: ExactCertificationScope
    candidate_model: ModelIdentity
    request_sha256: str
    policy_sha256: str
    toolkit_version: str | None
    checks: list[Qwen3NextReleaseAuditCheck]
    blockers: list[str]
    release_ready: bool
    created_at: datetime
```

Model validation requires N0–N8 exactly once, in order, and
`release_ready == all(check.passed for check in checks) == not blockers`.

### 3.5 Exact-checkpoint registry

New schema literal:

```text
axquant.certified-checkpoint-registry.v1
```

Each entry records:

- exact `ModelIdentity` and `ArchitectureFingerprint`;
- certification track and policy digest;
- candidate artifact manifest digest and release-audit digest;
- target class and measured BPW;
- permitted public claim scope;
- hardware registry entry ids;
- certification and expiry/revalidation timestamps, if policy defines one;
- superseded audit digest, if any.

Registry writes are append-only by identity. Replacing an entry requires an explicit supersession
record; a new source revision is a new entry. The family adapter remains `convertible` until a
separate family-promotion decision.

## 4. Track dispatch and eligibility

### 4.1 Request dispatch

`certification.dispatch.load_request(path)` reads only the top-level `schema_version`, then loads
exactly one strict request type:

| Schema version | Builder |
| --- | --- |
| `axquant.release-audit-request.v4` | existing `build_release_audit` (M0–M8) |
| `axquant.qwen3-next-release-audit-request.v1` | `build_qwen3_next_release_audit` (N0–N8) |

Unknown versions fail. There is no model-name heuristic and no CLI `--non-mtp` override.

### 4.2 Source-derived eligibility

Before N0, the builder independently loads:

1. `config.json` from the immutable BF16 source;
2. `axquant_source.json` and its source revision;
3. the source inventory;
4. the plan packaged in the candidate;
5. the artifact manifest and runtime metadata.

It then proves:

```text
config.model_type == qwen3_next
architectures contains Qwen3NextForCausalLM
inventory.architecture_profile.adapter_id == qwen3-next-v1
inventory.architecture_profile.mtp_declared == false
no inventory tensor role is MTP
no root mtp.safetensors exists in source or artifact
plan.mtp.mode == disabled
artifact.mtp_present == false
runtime metadata declares no MTP capability
all model identities and revisions match the request scope
```

Any contradiction is an eligibility error, not a gate exception. The builder refuses to create an
audit under the wrong track.

## 5. N0–N8 algorithms

### 5.1 N0 — Immutable technical feasibility

N0 recomputes rather than trusts a feasibility status label:

- resolve every indexed Safetensors shard through the safe path rules;
- verify file existence, size, SHA-256, dtype, shape, and full tensor membership;
- verify source `axquant_source.json`, model id, full revision, config/tokenizer digests;
- reproduce the architecture fingerprint and explicit no-MTP eligibility;
- require BF16, uniform-4, uniform-6, and requested candidate feasibility entries;
- require both MLX-LM and AX Engine binaries/build identities to be declared;
- reject an incomplete source, a mutable revision, missing baseline, unsupported host, or stale
  feasibility timestamp beyond the policy window.

N0 evidence includes the source inventory, source manifest, feasibility report, and policy digest.

### 5.2 N1 — Artifact integrity and dual-runtime vertical slice

N1 verifies:

- every artifact Safetensors record has one safe path, size, and checksum;
- logical parameter count and architecture match the BF16 source;
- `axquant_manifest.json` binds `axquant_plan.json`, calibration, capture, source revision, target
  class, and quantizer execution record;
- every planned quantized module was visited by the converter; protected tensors retain their
  required floors;
- `abs(measured_total_bpw - plan.effective_bpw) <= 0.01`;
- total artifact byte ratio meets the target-class policy;
- MLX-LM generation smoke is available and passed on the exact artifact;
- AX Engine native-manifest generation/validation, doctor, and deterministic generation smoke are
  available and passed on the exact artifact;
- runtime reports bind candidate id, artifact revision digest, path, runtime version, host, and OS;
- native manifest was not skipped and no runtime fallback occurred;
- explicit non-MTP facts remain consistent.

Certification mode must invoke conversion with `--ax-engine-manifest required`. `if-available`
and `skip` remain valid development modes but cannot satisfy N1.

### 5.3 N2 — Correct and repeatable direct-decode benchmark

The benchmark evidence index must contain, for each required profile:

```text
bf16
uniform-4bit
uniform-6bit
candidate
```

The requested candidate may be 4-bit or 6-bit. Both uniform controls remain mandatory because
they prove cross-class trade-offs and prevent selecting only a favorable control.

Matched-control invariants:

- identical prompt dataset digest and ordered prompt ids;
- identical tokenizer digest, seed, temperature, top-k/top-p, maximum output tokens;
- identical warmup and measured-trial counts;
- identical AX Engine binary digest, runtime environment, power mode, host id, OS, and background
  workload policy;
- at least two warmups and five successful measured trials per arm;
- all failed/time-out trials retained; required successful count is evaluated after failures;
- zero kernel fallback;
- deterministic greedy AX Engine output exactly matches MLX-LM on the parity corpus;
- throughput, TTFT, peak memory, and output digests are finite and internally consistent.

N2 recomputes medians and ratios from raw trials. Pre-aggregated numbers are not authoritative.

### 5.4 N3 — Measured mixed-precision planner

N3 retains the current measured-evidence discipline:

- calibration manifest has at least 128 samples, required domains, immutable dataset digest,
  random seed, and calibration/evaluation separation attestation;
- tokenized cache and activation capture are complete and checksum-bound;
- sensitivity has complete inventory coverage and at least 8,192 measured tokens for every
  candidate used by the plan;
- metrics are finite, tensor-scoped, and produced by the declared backend version;
- every AWQ/GPTQ allocation has matching captured activations and execution metadata;
- base/refinement lineage digests form one acyclic, completely used chain;
- planner floors and target BPW reproduce the exact plan deterministically;
- architecture-prior, unsupported, dominated, or failed candidates cannot be selected as measured
  release allocations;
- no certification path accepts `--allow-unmeasured`.

### 5.5 N4 — Coding and general quality

N4 loads the coding-suite v2 manifest, raw per-task results, quality comparisons, and both profile
validations. It verifies:

- the suite and general holdout have distinct dataset digests and neither overlaps calibration;
- task count/category/language quotas and scored-token minimums are met;
- every executable scorer records sandbox/toolchain identity and terminates within limits;
- BF16 and candidate model identities, tokenizers, prompts, and generation settings match;
- model errors are zero;
- aggregate retention, category retention, perplexity ratios, syntax/compile validity, JSON/tool
  validity, and unit-test pass rates meet the frozen policy;
- BF16 itself meets the suite validity floor; a broken reference suite cannot authorize a
  candidate through equal failure;
- learned-method allocations are backed by their execution records and measured comparison.

Quality comparisons are recalculated from raw task outcomes. A single aggregate score cannot
hide a failing required category.

### 5.6 N5 — Exact Qwen3-Next architecture proof

N5 verifies exact checkpoint architecture, not a family name:

- 48 text layers and the recorded full-attention cadence;
- linear-attention tensor set and dimensions;
- full-attention Q/K/V/O and Q/K norm dimensions;
- 512 fused experts, experts-per-token, expert intermediate size, and router shapes;
- packed 3-D expert aliases map to every MLX runtime module with no unmatched half;
- embedding/router/LM-head/norm floors and any vision preservation policy;
- every runtime tensor's logical shape is reconstructable from raw/packed metadata;
- no unclassified or unexpected checkpoint tensor;
- architecture fingerprint matches the exact certified registry scope.

One checkpoint passing N5 does not change `qwen3-next-v1` to family-certified.

### 5.7 N6 — Complete candidate optimization

N6 requires the complete refinement graph:

- refinement result binds the N3 sensitivity report;
- every measured candidate binds plan, artifact, quality, validation, size, and benchmark digests;
- complete objective is recomputed with the released evaluator version and authoritative profile
  thresholds;
- selected candidate is validated and improves monotonically over its declared parent/control;
- no missing candidate measurement is silently omitted from selection;
- target class, BPW, method/group distribution, and candidate id match the exact artifact;
- 4-bit and 6-bit lineages cannot reference each other's audit or measurement id.

### 5.8 N7 — Hardware-aware Pareto and reproduction

N7:

- reloads every hardware registry entry and bound raw evidence;
- requires a formal M2 Ultra 192 GB entry with exact host/OS/runtime identity;
- limits claims to registered hardware scope;
- independently reproduces inspect → plan → convert and verifies semantic plan equality,
  artifact membership, and measured BPW tolerance;
- rebuilds every Pareto point from complete measurements and verifies frontier membership;
- reloads the compatibility matrix request and re-hashes all candidate/runtime/validation files;
- requires zero conversion/runtime fallback and no unresolved compatibility issue;
- verifies all digest-referenced evidence has a durable archive record.

A second hardware host is required only for a broader-than-M2-Ultra public claim. Without it, N7
may pass with an explicitly M2-Ultra-scoped certification.

### 5.9 N8 — Release package and claim authorization

N8 verifies:

- stable, non-alpha AXQuant wheel; Python 3.11+; MIT metadata; dependency and `RECORD` integrity;
- artifact, plan, recipe, policy, audit, model card, and wheel versions agree;
- all N0–N7 evidence is packaged under safe relative paths with verified digests;
- model card states exact source revision, non-MTP track, target class, measured BPW/bytes,
  hardware scope, runtime versions, and quality/performance metrics from the audit only;
- model card does not imply family-wide, MTP, VLM, KV, batching, or serving claims;
- publisher reruns the request from packaged/current evidence and reproduces the same passing audit;
- preview and executed publication cannot overwrite a different existing audit.

The exact-checkpoint registry is written **after** a passing audit so an entry can bind the final
audit digest without a self-referential request/audit/registry cycle. The publisher appends the
entry atomically, validates it against the audit and artifact, then packages the updated registry
before upload. N8 verifies that the registry entry is derivable; a pre-existing registry entry
never authorizes its own audit.

## 6. AX Engine requirements

### 6.1 Current failure to preserve as a regression fixture

The 6-bit development plan contains a full-attention layer where:

```text
Q projection: BF16, logical [8192, 2048]
K projection: 4-bit affine, packed physical [512, 256], logical [512, 2048]
V projection: 8-bit affine, packed physical [512, 512], logical [512, 2048]
O projection: 4-bit affine, packed physical [2048, 256], logical [2048, 2048]
```

The current native-manifest validator chooses a layer-level interpretation influenced by Q and
then validates K as if `[512, 256]` were a raw BF16 logical shape. The fix must be per tensor, not
a plan workaround.

### 6.2 Logical shape reconstruction

For every quantizable tensor independently:

```text
if dtype is BF16/F16/F32 and no quantization metadata:
    physical shape == logical shape

if dtype is U32 and quantization.mode == affine:
    packed_last = logical_last * bits / 32
    logical_last = packed_last * 32 / bits
    divisibility must be exact
    scale/bias shapes must match logical rows and group_size
```

The generator records both physical payload shape and quantization metadata while the validator
checks the reconstructed logical role shape. Q/K/V/O may use different bits or raw BF16 in any
combination admitted by the plan/hardware profile.

### 6.3 Required AX Engine tests

- raw Q + packed K/V/O (the current failure);
- packed Q/K/V/O with mixed 4/6/8 bits;
- raw K with packed Q/V/O;
- invalid packed dimension/divisibility;
- missing or contradictory quantization metadata;
- 3-D fused expert 4/6/8-bit logical reconstruction;
- manifest generate → validate → load → deterministic generate;
- no behavior change for existing Qwen 3.6 fixtures.

### 6.4 Formal-host readiness

AX Engine doctor must report:

```text
supported host: true
xcrun metal: available
xcrun metallib: available
Metal toolchain fully_available: true
artifact status: ready
bringup_allowed: true
```

Installing or selecting the Metal toolchain is an operator/host action and must be logged as
formal-host provenance. The audit never treats a successful generation command as a substitute
for a failed doctor result.

## 7. Matched baseline evidence

### 7.1 Required artifacts

| Baseline | Construction | Evidence label |
| --- | --- | --- |
| BF16 | Immutable source, unmodified | `bf16` |
| Uniform 4-bit | All eligible trunk modules 4-bit under identical protection floors | `uniform-4bit` |
| Uniform 6-bit | All eligible trunk modules 6-bit under identical protection floors | `uniform-6bit` |
| AXQuant candidate | Measured plan for the audited target class | `candidate` |

An attributed external OptiQ artifact may appear as `external` but cannot replace a mandatory
baseline or contribute to N0–N8 pass/fail.

### 7.2 Benchmark index extension

The existing benchmark evidence index gains a direct-decode profile or a new strictly versioned
schema if adding it would weaken MTP invariants. Every entry binds:

- artifact manifest and plan digests;
- model identity/revision and tokenizer digest;
- benchmark config and ordered prompt digest;
- raw trial log and evaluation bundle digests;
- runtime executable digest/version and environment allowlist;
- hardware registry id, OS, power mode, warmup/measured trial counts;
- availability, fallback, failure, and timeout status.

Missing mandatory entries are represented as explicit failure blockers, never omitted.

### 7.3 Isolation protocol

Formal trials run sequentially on an otherwise idle host. No candidate arms run concurrently.
Before each arm:

1. verify external volume mount and free space;
2. verify host power mode and Metal toolchain;
3. record background process policy and memory pressure;
4. clear only runtime caches explicitly allowed by the benchmark policy;
5. run two warmups followed by at least five measured trials;
6. archive raw logs before the next evidence artifact binds them.

## 8. Coding-suite v2

### 8.1 Suite manifest

New schema literal:

```text
axquant.coding-suite-manifest.v2
```

```python
class CodingScorer(StrEnum):
    UNIT_TEST = "unit-test"
    COMPILE = "compile"
    AST = "ast"
    JSON_SCHEMA = "json-schema"
    TOOL_EXACT = "tool-exact"
    TEXT_EXACT = "text-exact"


class CodingTaskManifest(StrictModel):
    task_id: str
    category: str
    language: str
    prompt_sha256: str
    reference_sha256: str | None
    scorer: CodingScorer
    license_id: str
    provenance: str
    target_tokens: int
    timeout_seconds: float
    cpu_time_seconds: int
    memory_limit_bytes: int
    process_limit: int
    output_limit_bytes: int
    file_size_limit_bytes: int
    open_file_limit: int
    long_context: bool


class CodingSuiteManifest(StrictModel):
    schema_version: Literal["axquant.coding-suite-manifest.v2"]
    suite_id: str
    version: str
    dataset_sha256: str
    tasks: list[CodingTaskManifest]
    task_shards: dict[str, str]
    calibration_overlap_attested: bool
    calibration_overlap_report: str
    calibration_overlap_report_sha256: str
    toolchains: dict[str, str]
    sandbox_profile_sha256: str
    normalization_algorithm: Literal["axquant-token-5gram-v1"]
    near_duplicate_threshold: float
    random_seed: int
```

Every executable payload also carries a checksum-bound clean-room reference implementation used
only for suite self-test and perplexity evidence. Before freeze, `verify-coding-suite` must prove
that each reference oracle passes and that an empty-output mutant fails under the exact recorded
toolchain and sandbox policy. The checksum-bound `axquant.coding-suite-self-test.v1` report and
its raw logs are mandatory N4 evidence; candidate generation never receives the reference.

Task prompts/references may live in separate checksum-bound shards. The manifest never embeds
credentials, private repositories, or unlicensed source.

### 8.2 Minimum composition

The 128-task minimum is allocated before the formal run:

| Category | Minimum tasks |
| --- | ---: |
| Python generation/repair | 24 |
| JavaScript/TypeScript generation/repair | 20 |
| Rust generation/repair | 16 |
| Go generation/repair | 16 |
| Multi-file/repository-context edits | 16 |
| JSON/tool-call exactness | 16 |
| Algorithm/reasoning with executable oracle | 12 |
| Long-context code navigation | 8 |

At least half of eligible tasks use executable unit-test/compile scorers. No one language or one
scoring heuristic may determine the aggregate result.

The 25,000-token floor applies to the manifest's checksum-bound sum of per-task generation
budgets. Actual generated-token counts are recorded without padding and must remain within each
task's budget; correctness and retention, not verbosity, determine the quality verdict.

### 8.3 Sandbox

Executable scoring runs with:

- no network;
- read-only task fixture plus per-task temporary output directory;
- explicit CPU time, wall time, memory, process, and output-size limits;
- allowlisted compiler/interpreter versions recorded in the result;
- deterministic locale/timezone/environment;
- no repository credentials or user home mounts;
- stdout/stderr digests and truncated diagnostic excerpts.

The scorer reports infrastructure failure separately from model failure. Infrastructure failures
invalidate the affected suite run; they do not score the model as zero and continue silently.

### 8.4 Overlap detection

The suite builder compares normalized prompt/reference hashes and near-duplicate fingerprints
against calibration and any refinement holdout. Exact overlap fails. Near-duplicate matches above
the frozen similarity threshold require removal or a documented dataset replacement before the
formal run; no audit exception is available.

The general holdout has an independent checksum-bound calibration-overlap report. N4 also reloads
both coding payload shards and the general JSONL manifest and recomputes coding↔general exact and
near-duplicate separation. Thus distinct file digests alone are insufficient to claim disjointness.

### 8.5 Direct quality artifacts

Coding and general evaluation both emit `axquant.direct-quality-evaluation.v1`. Each artifact
binds the exact model artifact manifest, evaluation manifest, dataset, tokenizer, deterministic
generation settings, seed, evaluated perplexity tokens, software versions, and raw per-task
outcomes. Each outcome binds the raw model-output file; executable coding outcomes additionally
bind stdout, stderr, toolchain, sandbox policy, resource-limit verdicts, compile/syntax state, and
unit-test state.

Coding generation uses `axquant.coding-evaluation-state.v1`; general generation uses
`axquant.direct-general-quality-state.v1`. Both states are written atomically after every task and
refuse resume when any model/artifact/dataset/tokenizer/generation/seed binding differs.

`axquant.direct-release-validation-request.v1` names the immutable BF16 source manifest,
candidate artifact manifest, coding/general evaluation manifests and result pairs, calibration
digest, general-overlap report, toolkit version, and wheel-owned policy digest. The
`direct-validation-index` command recomputes the policy verdict and emits
`axquant.direct-release-validation-index.v1`; it cannot accept caller-supplied thresholds. N4
then independently recomputes the same aggregates and raw-file bindings instead of trusting the
index declaration.

## 9. Validation and thresholds

### 9.1 Profiles

The track requires:

- `agent-coding`: coding-suite v2 plus agent/tool metrics;
- `general`: a distinct general/long-context holdout with a distinct digest.

Both profiles must use authoritative thresholds stored in released code/policy. A request cannot
supply looser thresholds.

### 9.2 Threshold policy binding

`axquant.qwen3-next-direct-policy.v1` is a versioned, wheel-owned policy object. Its canonical
serialization produces `policy_sha256`, recorded in request, audit, validation, recipe, registry,
and model card. The builder recomputes the digest; an external policy file cannot override it.

The accepted values are in PRD §9. Acceptance of the PRD freezes them for the formal
cycle. Any change increments the policy id/version and starts a new evidence cycle.

## 10. CLI and publication

Before `release-audit`, the direct quality chain uses:

```bash
axquant prepare-coding-suite ...
axquant verify-coding-suite ...
axquant prepare-general-overlap ...
axquant evaluate-coding-suite ...       # BF16, then exact candidate
axquant evaluate-general-quality ...    # BF16, then exact candidate
axquant direct-validation-index --request REQUEST --output INDEX
```

All model-bearing commands run sequentially on the formal host. The general and coding evaluators
archive raw outputs beneath their evidence root and support checksum-bound restart state.

### 10.1 `release-audit`

The public command remains:

```bash
axquant release-audit --request REQUEST --output AUDIT
```

Dispatch is schema-based. Exit behavior:

- `0`: all gates for the selected track pass;
- `1`: a complete audit was produced with one or more failed gates;
- `2`: request/evidence/schema/I/O failure prevented a complete audit.

The existing v4 exit behavior must be regression-pinned before adopting this uniform convention
if it differs today.

### 10.2 `publish-prepare`

For a non-MTP candidate, packaging includes:

```text
certification/
├── request.json
├── audit.json
├── policy.json
├── exact_checkpoint_scope.json
├── coding_suite_manifest.json
├── coding_suite_self_test.json
├── benchmark_evidence_index.json
├── release_validation_index.json
├── refinement_measurements.json
├── hardware_profile_registry.json
├── compatibility_matrix.json
├── pareto_report.json
└── reproduction_verification.json
```

No MTP placeholder files are created.

### 10.3 `publish`

Executed publication requires:

- matching request/audit schema family;
- `release_ready=true` and N0–N8 exactly once;
- fresh audit rerun with identical semantic result and no new blocker;
- exact registry entry;
- model-card claims rendered from audit evidence;
- user `--yes` as the final external-state authorization.

Preview remains non-mutating. A development artifact can be packaged internally, but official
catalog upload under a certified name is refused without the passing track audit.

## 11. Migration and backward compatibility

- Do not edit `ReleaseAuditRequest`, `ReleaseAuditCheck`, or `ReleaseAudit` v4 fields/literals.
- Do not change M0–M8 names, issue strings relied on by tests, or publisher hash behavior.
- New schemas are additive and forward-only; older versions fail on unknown schema versions.
- Current development candidates and logs remain historical evidence; do not inject new audit
  fields into them.
- Final candidate artifacts should be regenerated under the cert-capable toolkit version so
  artifact/plan/recipe/wheel version binding is natural. Reusing an older sensitivity/capture is
  allowed only if N3 verifies its full digest/backend/policy compatibility; any probe/backend
  change requires remeasurement.
- Exact-checkpoint registry absence preserves current `convertible` behavior.

## 12. Testing strategy

### 12.1 Unit tests

- strict schema round-trips and `extra="forbid"`;
- N0–N8 exactly-once/order/status invariants;
- unknown request version and cross-track request/audit mismatch;
- eligibility from source config/inventory, including contradictory MTP facts;
- no exception/applicability bypass;
- exact source revision, architecture fingerprint, tokenizer, policy, and artifact digest binding;
- per-gate positive/negative/tamper fixtures;
- baseline completeness and matched-control invariants;
- suite quotas, sandbox result validation, overlap detection, category threshold failures;
- exact-checkpoint registry append/supersession rules;
- publisher fresh-rerun and packaged-audit tamper detection.

### 12.2 Existing-track regression tests

- load existing v4 request/audit fixtures unchanged;
- stable semantic hashes remain unchanged;
- every M0–M8 issue/gate test still passes;
- Qwen 3.6 publisher preview/executed-path behavior remains unchanged;
- a no-MTP request cannot enter the v4 builder and an MTP source cannot enter the new builder.

### 12.3 AX Engine integration tests

- manifest generation/validation for all raw/packed Q/K/V/O permutations;
- Qwen3-Next 4-bit and 6-bit tiny fixtures;
- fused-expert logical coverage;
- deterministic generation parity against MLX-LM;
- doctor with complete/incomplete Metal toolchain;
- zero-fallback benchmark evidence parsing.

### 12.4 Real-host acceptance tests

On `macstudio-m2u` or the named replacement formal host:

1. immutable source and volume health preflight;
2. MLX-LM smoke for BF16, uniform controls, and candidate;
3. AX Engine manifest/doctor/smoke for each artifact;
4. coding/general quality suites;
5. matched benchmark matrix;
6. reproduction run;
7. N0–N8 audit and publisher preview;
8. executed upload only after review and explicit user authorization.

### 12.5 Repository quality gates

Every code phase must pass:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

MLX/AX Engine integration tests run separately on Apple Silicon and are reported as named evidence,
not silently skipped release tests.

## 13. Security, privacy, and clean-room requirements

- Use public model/runtime interfaces and independently designed AXQuant schemas; do not import
  mlx-optiq implementation, data, or metadata.
- Record license/provenance for coding-suite tasks and do not commit private repository contents.
- Run generated code without network, credentials, user home access, or unrestricted process
  creation.
- Never log Hub tokens or environment secrets.
- Treat model outputs and raw benchmark logs as evidence with controlled, durable storage.
- Resolve every release path safely; reject traversal, symlink escape, duplicate membership, and
  checksum mismatch.

## 14. Operational recovery

- Every expensive capture/analyze/evaluate/benchmark stage is resumable or writes per-unit
  checkpoint state.
- External-volume disconnects must leave atomic progress and no visible partial artifact.
- A resumed run verifies source, config, cache, capture, policy, and progress digests before reuse.
- A failed formal gate produces an immutable failure report. Remediation starts a new candidate
  or evidence cycle under AXQ-030; it never edits the historical audit.
- All load-bearing evidence is archived outside `.internal/tmp/` before a downstream digest binds
  it, per AXQ-031.

## 15. Definition of implementation complete

The toolkit portion is complete only when:

1. decisions AXQ-032–AXQ-034 are accepted;
2. strict track schemas and N0–N8 builder ship;
3. all M0–M8 regression tests remain green;
4. publisher reruns and authorizes the new track only on an exact pass;
5. AX Engine accepts and runs the mixed-projection 6-bit regression fixture;
6. coding-suite v2 and matched baseline index ship with fail-closed tests;
7. full repository quality gates pass;
8. Apple Silicon integration evidence is recorded.

Candidate certification is a later evidence result: implementation complete does not mean the
4-bit or 6-bit artifact is certified. That occurs only when its own final N0–N8 audit passes.
