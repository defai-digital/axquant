# AXQuant Product Requirements Document

**Document status:** Approved baseline for v0.x implementation  
**Product:** AXQuant  
**Organization:** AutomatosX / DEFAI Digital  
**Repository:** `defai-digital/axquant`  
**Initial reference model:** Qwen3.6-27B  
**Primary runtime:** AX Engine  
**Compatibility runtime:** MLX-LM  
**Initial platform:** Apple Silicon with MLX  
**Last reviewed:** 2026-08-01

## 1. Executive summary

AXQuant is a command-line toolkit that converts a supported standard LLM checkpoint into an
AXQuant-optimized deployment checkpoint. It inspects the source, creates an auditable
mixed-precision plan, converts the weights through public MLX-LM interfaces, and emits the
manifests and validation evidence required by AX Engine.

The initial vertical slice supports the Qwen3.6-27B language path on Apple Silicon. It produces
portable MLX weights plus protected model components, MTP metadata or a byte-preserved sidecar,
runtime metadata, provenance, and validation artifacts.

The product objective is to make this conversion reproducible from the CLI while achieving:

> Near-6-bit quality at a near-4-bit footprint, with MTP-aware acceleration.

AXQuant is not a renamed mixed-precision baseline. Its independent technical identity is:

- per-tensor rather than layer-only planning;
- MTP-aware quality and throughput objectives;
- 4-bit, 6-bit, 8-bit, and BF16 allocation;
- workload-aware calibration beyond output KL alone;
- optional AWQ and DWQ refinement;
- complete-model interaction validation;
- real AX Engine latency and unified-memory cost;
- versioned, reproducible, and auditable artifacts.

The first planned public checkpoint is:

```text
AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit
```

MTP capability is recorded in the manifest and model card. It is not normally encoded in the
repository name because MTP-aware processing is a standard AXQuant capability.

## 2. Product vision

AXQuant becomes the optimization layer for the AutomatosX Qwen/MLX ecosystem:

- local private inference on Apple Silicon;
- AX Engine and AX Serving deployments;
- agentic and coding workloads;
- structured output and tool calling;
- English, Traditional Chinese, Simplified Chinese, and Japanese;
- long-context document and repository workflows;
- later expansion to additional model families, MoE, VLM, and runtime KV policy.

AXQuant optimizes a deployment system, not only a weight file. The deployment system includes:

```text
portable MLX weights
+ protected model components
+ MTP sidecar and verification path
+ runtime-specific manifest
+ workload profile
+ measured quality, memory, and speed evidence
```

## 3. Product boundary

### 3.1 v0.x boundary

The v0.x vertical slice supports:

- Qwen3.6-27B only;
- dense language-path PTQ;
- MLX checkpoint input and output;
- AX Engine as the primary runtime;
- MLX-LM standard-inference fallback;
- MTP detection and byte-preserved external sidecars;
- manual mixed-precision plans before measured sensitivity is available;
- 4/6/8/BF16 affine planning;
- `agent-coding` as the primary optimization profile;
- `general` as a required comparison suite;
- vision tensor preservation at BF16 without VLM claims.

### 3.2 v1 boundary

AXQuant v1 supports:

- Qwen3.6-27B and at least one additional validated dense Qwen 3.6 checkpoint;
- measured per-tensor sensitivity;
- MTP-aware planning and validation;
- at least one production learned-refinement method;
- complete-model candidate validation;
- hardware measurements on supported Apple Silicon;
- public reproducibility artifacts and a guarded publication flow.

### 3.3 Explicit non-goals for v1

- Gemma or unrelated model families;
- arbitrary Qwen generations;
- full VLM quantization or VLM quality claims;
- MoE expert-level planning;
- runtime KV-cache quantization;
- arbitrary per-channel execution precision;
- 2-bit, 3-bit, or 5-bit production formats;
- GGUF, CUDA, Windows, or Linux output;
- distributed conversion;
- training from scratch, full fine-tuning, or output identity with the source.

Gemma is a post-v1 adapter candidate. It is not a parallel launch workstream.

## 4. Target users

### 4.1 Primary users

- AutomatosX model engineers;
- AX Engine and AX Serving engineers;
- MLX model publishers;
- Apple Silicon application developers;
- enterprise teams deploying private models on Mac hardware.

### 4.2 Secondary users

- mixed-precision PTQ researchers;
- open-source contributors;
- local LLM developers comparing MLX formats;
- organizations operating under unified-memory constraints.

## 5. User problems

Uniform 4-bit quantization may materially regress:

- reasoning and instruction retention;
- code generation and repair;
- JSON and tool-call validity;
- multilingual quality;
- long-context retrieval;
- generation stability;
- MTP draft accuracy and acceptance.

Uniform 6-bit usually preserves more quality but materially increases storage and unified memory.
Existing mixed-precision approaches may still:

- operate at a layer granularity that hides sensitive tensors;
- use KL divergence as the only proxy;
- ignore interactions between simultaneously quantized tensors;
- exclude MTP from the objective;
- optimize nominal size without measuring actual kernels;
- publish incomplete or irreproducible reports.

## 6. Product goals and claim policy

### 6.1 Primary goal

For Qwen3.6-27B, find a Pareto-efficient checkpoint balancing:

- model and MTP quality;
- actual weight bytes;
- logical and measured BPW;
- peak unified memory;
- prefill throughput;
- ordinary decode throughput;
- MTP acceptance;
- effective MTP throughput.

### 6.2 Claim guardrails

The initial market claim is:

> Near-6-bit quality at a near-4-bit footprint, with MTP-aware acceleration.

The claim:

> 6-bit quality at 4-bit size

is prohibited until repeated results across multiple Qwen checkpoints, workloads, and Apple
Silicon devices support it.

Architecture priors, manual recipes, static compatibility checks, and artifact feasibility audits
are development evidence. They cannot support quality or performance claims.

### 6.3 Differentiation scenario guard

If uniform MLX 4-bit already retains at least 98% of uniform 6-bit aggregate quality on the
declared suite, the aggregate claim becomes uninformative. In that scenario AXQuant's measured
value must be demonstrated on the metrics where uniform 4-bit is weakest: MTP draft acceptance,
structured-output validity, coding repair, multilingual tasks, and long-context retrieval. The
benchmark design MUST retain per-task visibility and statistical power on those tail metrics so
the comparison cannot collapse into a single aggregate number that hides them.

## 7. Measurement definitions

- **Logical parameters:** represented model parameters, reconstructed from packed MLX
  quantization metadata when required.
- **Planned BPW:** assigned weight bits plus the modeled affine scale/bias cost.
- **Measured total BPW:** actual bytes of all inspected model Safetensors, including shipped root
  MTP and protected-tensor sidecars, divided by total logical parameters.
- **Measured main BPW:** actual inspected non-MTP weight bytes, including protected-tensor
  sidecars, divided by non-MTP logical parameters.
- **Weight-file size:** model and MTP Safetensors only; tokenizer, reports, and documentation are
  excluded.
- **Quality retention:** normalized higher-is-better candidate score divided by the uniform
  6-bit score. Error metrics are not divided as quality scores.
- **MTP acceptance retention:** candidate acceptance divided by the high-precision or uniform
  6-bit MTP baseline under the same workload and runtime configuration.
- **MTP speedup:** effective decode throughput of one checkpoint with MTP enabled divided by the
  same checkpoint with MTP disabled.

The converted artifact measurement is authoritative. A plan estimate never replaces actual file
bytes or runtime measurements.

## 8. Success criteria

### 8.1 Required v1 targets

| Metric | Required target |
| --- | ---: |
| Weight-file size | At most 110% of uniform MLX 4-bit |
| Target measured BPW | 4.3–4.8, or an approved Pareto exception |
| Aggregate quality | At least 98% of normalized uniform MLX 6-bit |
| Critical tasks | No major coding, tool, JSON, multilingual, or long-context regression |
| MTP acceptance retention | At least 95% |
| MTP speedup | At least 1.20× over the same candidate without MTP |
| Peak unified memory | At least 15% below uniform MLX 6-bit |
| AX Engine loading and generation | Required |
| MLX-LM standard inference | Required |
| Reproducible manifest and recipe | Required |

#### BPW target interpretation

The 4.3–4.8 range is the Pareto search band for the candidate's measured total BPW (all
inspected model weights, including shipped root MTP and protected-tensor sidecars, divided by
total logical parameters). It is a search target, not an independent hard gate: the binding
constraints are the size ratio (≤ 110% of uniform 4-bit, i.e. ≤ ~5.35 total BPW) and the
quality/MTP floors.
The lower bound is not a requirement to be smaller; a candidate below 4.3 that still passes
quality gates is acceptable but unlikely, because aggressive quantization is what degrades
quality.

Reference points from the 2026-07-28 feasibility audit:

- uniform 4-bit: 4.8677 total / 4.6949 main BPW (BF16-preserved MTP sidecar, 849 MB);
- attributed mixed baseline: 5.7317 total / 5.7509 main BPW (INT8 MTP sidecar, 239 MB).

A byte-preserved BF16 MTP sidecar adds ~0.24 total BPW. Reaching the upper band (~4.8) with
meaningful 6-bit upgrades therefore requires either a small upgrade set or an INT8 MTP sidecar
from a future layout-aware backend. Release claims MUST state whether total or main BPW is
reported and which MTP treatment applies; the manifest reports both.

### 8.2 Stretch targets

| Metric | Stretch target |
| --- | ---: |
| Aggregate quality | At least 99% of uniform 6-bit |
| Weight-file size | At most 105% of uniform 4-bit |
| MTP speedup | At least 1.40× |
| MTP acceptance retention | At least 98% |
| Long-context regression | Less than 2% |
| Structured-output regression | Less than 1% |

### 8.3 Production failure conditions

A checkpoint cannot be released as a production AXQuant model when:

- aggregate quality is below 95% of the uniform 6-bit baseline;
- MTP decreases end-to-end throughput;
- JSON or tool-call failure materially increases;
- weight size exceeds the declared limit without an approved quality tradeoff;
- AX Engine artifact integrity or runtime readiness fails;
- MLX-LM cannot perform the promised standard inference;
- source, data, software, plan, or hardware provenance is missing;
- benchmark evidence is incomplete, mismatched, or internally inconsistent;
- the plan uses architecture-prior evidence;
- the checkpoint cannot be regenerated from its published recipe.

## 9. Functional requirements

Status values refer to the 2026-07-28 working tree and must be reconfirmed for a tagged release.

| ID | Requirement | Interface | Status |
| --- | --- | --- | --- |
| FR-0 | Audit revisions, shards, logical parameters, MTP provenance, and runtimes | `axquant feasibility` | Implemented |
| FR-1 | Inspect local or cached Hub MLX checkpoints without implicit large downloads | `axquant inspect` | Implemented |
| FR-2 | Emit tensor roles, precision, protection, and tie inventory | `axquant.inventory.v1` | Implemented |
| FR-3 | Build a versioned calibration manifest and cache boundary | `axquant calibrate` | Implemented |
| FR-4 | Probe 4/6/8/BF16 per supported tensor using real forward evidence | `axquant analyze` | Implemented |
| FR-4a | Emit explicitly unmeasured architecture priors | `axquant analyze` without calibration | Implemented |
| FR-5 | Generate a constrained automated plan | `axquant plan` | Implemented |
| FR-5a | Apply reviewed v0.1 rules with hard protection and BPW checks | `axquant plan-manual` | Implemented |
| FR-6 | Apply a saved affine mixed-precision plan | `axquant convert` | Implemented |
| FR-7 | Compare reference and candidate evaluation bundles | `axquant validate` | Implemented |
| FR-8 | Measure identical-checkpoint MTP off/on behavior | AX Engine benchmark harness | Implemented |
| FR-9 | Generate JSON and Markdown reports | `axquant report` | Implemented |
| FR-10 | Generate runtime metadata and AX native manifest | conversion/runtime adapters | Implemented |
| FR-11 | Prepare a guarded Hub-ready directory | `axquant publish-prepare` | Implemented |
| FR-12 | Preview or execute a guarded Hub upload | `axquant publish` | Implemented; `--yes` freshly reruns matching M0–M8 audit |
| FR-13 | Run AX Engine doctor or MLX-LM generation checks | `axquant runtime-check` | Implemented |
| FR-14 | Integrate AWQ refinement | quantizer plugin | Portable activation scaling + convert/predicate packing implemented; release validation still requires measured candidate evidence |
| FR-15 | Integrate DWQ refinement | quantizer plugin | Executable portable clipping + affine packing; complete development candidates measured |
| FR-16 | Evaluate full candidate interactions and precision swaps | `axquant refine-run` | Resumable complete-candidate orchestration and measured development selection implemented |
| FR-17 | Measure actual Apple Silicon memory and latency | hardware benchmark | Implemented |
| FR-18 | Emit and verify a checksum-bound executable regeneration recipe | `axquant verify-reproduction` | Implemented |
| FR-19 | Bind every official dense size at release time to artifact, runtime, and dual-profile validation evidence | `axquant compatibility-matrix` | Implemented; current official dense scope is 27B |
| FR-20 | Index every required comparison baseline without silent omission | `axquant benchmark-index` | Implemented |
| FR-21 | Require disjoint passing agent-coding and general release evidence | `axquant validation-index` | Implemented |
| FR-22 | Certify measured named-host kernel and shape coverage | `axquant hardware-registry` | Implemented; release measurements pending |
| FR-23 | Prove every M0–M8 exit condition from bound evidence | `axquant release-audit` | Implemented; final audit pending |

## 10. Core product capabilities

### 10.1 Model inspection

The inspector must:

- resolve a local directory or explicitly permitted Hub snapshot;
- follow `model.safetensors.index.json` when present;
- reject unsafe, missing, or non-Safetensors shard references;
- include only index-referenced shards plus the root MTP sidecar;
- reconstruct logical parameters for packed U32 tensors;
- separate quantization metadata from model parameters;
- classify backbone, MTP, vision, router, expert, norm, embedding, and head tensors;
- mark only supported 2-D language-path matrices as quantizable in v0.x;
- identify tied weights and protected components;
- stream Safetensors metadata rather than load full weights.

### 10.2 Calibration profiles

The initial optimization profile is `agent-coding`:

- source code, completion, and repair;
- JSON and structured output;
- tool schemas and function calls;
- multi-step instructions and agent traces;
- long prompts;
- English, Chinese, and Japanese samples.

The `general` suite covers prose, factual questions, reasoning, instruction following,
multilingual content, and short/medium context.

Calibration and evaluation data must be disjoint and separately identified by revision and
digest.

### 10.3 Sensitivity analysis

Release-quality analysis must support:

- output KL divergence;
- hidden-state mean squared error;
- cosine distance;
- token disagreement;
- task loss delta;
- MTP acceptance loss;
- long-context loss;
- optional measured memory and latency cost.

Each tensor candidate records precision, quantizer, group size, support status, metrics, and
provenance.

### 10.4 Mixed-precision planning

The initial precision set is:

```text
4-bit
6-bit
8-bit
BF16
```

Target modes:

```text
balanced
quality
low-memory
speed
```

The planner applies hard constraints before ranking any weighted objective.

### 10.5 MTP policy

Default policy:

```yaml
mtp:
  mode: protected
  candidate_bits: [8, 16]
  min_bits: 8
  preserve_external_sidecar: true
  protect_norms: true
  protect_output_head: true
  optimize_for_acceptance: true
```

External MTP sidecars remain byte-preserved by default. The explicit
`ax-engine-qwen36-v1` development backend may transform only the seven contract-named BF16 Qwen
3.6 MTP norms after validating checksum-bound raw provenance and exact 15-tensor coverage; it
must prove all eight projection payloads unchanged and emit new source/output payload checksums.
Selecting that backend does not satisfy or weaken any release gate. Integrated MTP tensors may
use 6-bit only after measured adaptive-policy evidence demonstrates that acceptance and
throughput remain within release gates.

Required MTP metrics:

- draft-position token accuracy;
- average accepted tokens;
- acceptance and rejection rate;
- effective tokens per forward;
- verification overhead;
- repetition and divergence;
- effective end-to-end decode throughput.

### 10.6 Runtime compatibility

| Level | Runtime | Product promise |
| --- | --- | --- |
| A | AX Engine | Native manifest, MTP, runtime metadata, and performance authority |
| B | MLX-LM | Portable standard language-model inference |

MLX-LM does not need to implement every AX feature. Unsupported AX metadata must be ignored or
fail gracefully without making the portable backbone unusable.

### 10.7 Export and publication

The output must contain:

- standard MLX model shards and index;
- tokenizer and configuration;
- `model-manifest.json`;
- MTP bundle and provenance when applicable;
- `axquant_plan.json`;
- `axquant_manifest.json`;
- immutable `axquant_conversion_manifest.json` used to bind pre-publication size evidence;
- `axquant_runtime.json`;
- benchmark reports;
- calibration manifest;
- reproduction recipe;
- model card and required notices.

Publication is blocked unless evidence, hashes, source revision, runtime authority, and validation
all pass.

## 11. Benchmark requirements

### 11.1 Required comparison set

1. BF16 or highest available source;
2. uniform MLX 4-bit;
3. uniform MLX 6-bit;
4. an available attributed mixed-precision baseline;
5. an available MLX AWQ baseline;
6. an available MLX DWQ baseline;
7. the AXQuant candidate with MTP disabled;
8. the identical AXQuant candidate with MTP enabled.

Unavailable optional baselines must be marked unavailable rather than silently omitted.

### 11.2 Quality categories

- perplexity and generation stability;
- instruction following and reasoning;
- code generation, repair, syntax, and patch structure;
- tool selection, arguments, JSON, and error recovery;
- English, Traditional Chinese, Simplified Chinese, and Japanese;
- translation consistency;
- long-context retrieval and instruction retention.

### 11.3 Hardware metrics

- model load time;
- peak unified memory;
- prefill tokens per second;
- ordinary decode tokens per second;
- MTP effective tokens per second;
- batch-one latency distribution;
- kernel fallback count;
- dequantization overhead;
- optional energy use.

Every run records the device, chip, memory, OS, AX Engine, MLX, MLX-LM, AXQuant, quantizer, and
dataset versions.

## 12. Non-functional requirements

### 12.1 Reproducibility

Every measured run records:

- immutable source revision;
- tokenizer revision;
- calibration and evaluation dataset identities and SHA-256 digests;
- random seed;
- quantization plan and quantizer versions;
- AXQuant, Python, MLX, MLX-LM, and AX Engine versions;
- hardware and OS profile;
- exact commands or executable recipe.

### 12.2 Transparency

Every released checkpoint publishes:

- measured total and main BPW;
- precision and quantizer distribution;
- protected tensor list;
- baseline comparison;
- MTP status, acceptance, and speedup;
- measured hardware;
- known limitations;
- source and plan provenance.

### 12.3 Safety

The pipeline fails closed on:

- unsafe or incomplete weight indexes;
- duplicate or missing tensors;
- unsupported modules, bits, methods, or group sizes;
- tied-weight inconsistency;
- malformed plans or recipes;
- missing or checksum-invalid MTP artifacts;
- incomplete runtime manifests;
- unmeasured evidence at release gates;
- non-finite or workload-mismatched measurements;
- partial conversion output.

### 12.4 Recovery and performance

- metadata inspection must not load the full model;
- expensive scans use stable cache keys;
- forward probes support resume and partial ranges;
- output files are written atomically;
- conversion occurs in a sibling staging directory and is renamed only after all artifact checks
  complete.

## 13. Roadmap

| Milestone | Product result | Exit condition |
| ---: | --- | --- |
| M0 | Technical feasibility | 4/6/mixed audits, MTP provenance, parameter equivalence, and runtime interfaces pass |
| M1 | AX Engine vertical slice | BF16 source converted from a reviewed manual plan; AX Engine and MLX-LM contracts pass |
| M2 | MTP benchmark harness | Correct, repeatable MTP off/on acceptance and throughput evidence |
| M3 | Measured MTP-aware planner | Per-tensor 4/6/8/BF16 evidence and AX cost integrated |
| M4 | Quality refinement | AWQ/DWQ, agent-coding calibration, and full candidate comparison |
| M5 | Qwen family proof | Every official dense Qwen 3.6 size at release time passes the dual-profile compatibility matrix |
| M6 | Interaction optimization | Bounded coordinate descent and precision swaps improve a complete candidate |
| M7 | Hardware-aware release candidate | Measured Pareto frontier on named Apple Silicon hosts |
| M8 | AXQuant v1.0 | Public toolkit and validated Qwen reference checkpoints |

Release mapping:

```text
v0.1 feasibility + manual Qwen3.6-27B vertical slice
v0.2 AX Engine MTP benchmark harness
v0.3 measured per-tensor and MTP-aware planner
v0.4 AWQ/DWQ and global candidate validation
v0.5 catalog-complete dense Qwen 3.6 compatibility proof
v0.6 interaction-aware refinement
v0.7 hardware registry and Pareto reports
v0.8 release candidate
v0.9 external testing and compatibility fixes
v1.0 public Qwen 3.6 toolkit and reference checkpoints
```

### 13.1 Execution schedule and dependency status

The schedule is dependency-driven. Calendar dates are intentionally not promised while formal
hardware evidence and the approval-bound size decision remain unresolved.

| Order | Milestone | Implementation status | Remaining release work |
| ---: | --- | --- | --- |
| 1 | M0 | Complete | None |
| 2 | M1 | Complete | The manual artifact is development evidence; the selected measured artifact replaces it for release |
| 3 | M2 | Harness, layout backend, phase profiler, and the named exact-profile measurement contract complete; the gate passes on uniform-6 and the size-compliant candidate is now within single-run variance of it | AX Engine's async-draft overlap (b2d6afdd, byte-identical output) lifts the AXQ-026 candidate's 2026-08-01 formal-protocol M5 measurement from 1.0969x to **1.1912x** (direct 32.56 tok/s, MTP 38.79, exactness pass, acceptance 0.8955) — ~0.75% under the 1.20x floor. Closing M2 needs a notarized runtime release carrying the flag plus the formal suite; residual engine headroom (uncovered verify-build time, cross-cycle pipelining) is scoped |
| 4 | M3 | Complete | The resumable 4/6/8/BF16 probe produced 1,199 measured tensor entries and a bound measured plan; under the original floors the policy minimum was 5.5770 BPW, and under the approved AXQ-026 8-bit LM-head floor (measured lm_head 8-bit output-KL 0.000097) the 2026-08-01 measured plan reaches 5.3000 BPW at size ratio 1.0888, inside the 110% gate |
| 5 | M4 | Conversion, dedicated MTP layout, and dual-profile candidate quality complete | The 5.5770-BPW provenance-bound candidate passes MLX-LM generation, AX Engine doctor, and MTP acceptance/exactness at depth 1; the new 5.3000-BPW AXQ-026 development candidate (2026-08-01) additionally passes the size gate at ratio 1.0888 with doctor and generation smokes, and M4 release certification now waits on its dual-profile quality bundles plus the M2 speed gate |
| 6 | M5 | Scope amendment and compatibility tooling complete | Reverify the official catalog at release time and bind the selected artifact to passing `agent-coding` and `general` evidence |
| 7 | M6 | Measured development refinement complete | The earlier grouped `cand-20260729-004` preserves every parent task score while improving agent-coding/general perplexity by 7.44%/2.02%; the current monotonic release lineage is `cand-20260729-002` → `cand-20260729-003` and still requires passing MTP, validation, hardware, and Pareto bindings |
| 8 | M7 | Registry/Pareto tooling complete | Record release-ready measurements on the named supported Apple Silicon host set |
| 9 | M8 | Audit/publication tooling complete | The size-floor direction is resolved by AXQ-026 (governed 8-bit LM-head floor, approved 2026-08-01); plan and validate a candidate on that floor with measured LM-head evidence, pass M0–M7, build the exact wheel, run the final audit, then publish only with explicit authorization |

The critical path is:

```text
M3 measurement
→ M3 plans
→ M4 complete candidates
→ M6 refinement
→ M2/M7 supported-host MTP and hardware certification
→ M8 exact-version reproduction and audit
```

The M5 scope is resolved by AXQ-016: certify every official dense parameter size present at
release time. The protected-weight size decision still requires measured tradeoff evidence and
named approval before M8. Approximate MTP, a favorable-host-only result, an alpha wheel, or an
unmeasured plan cannot shorten this schedule because the release gates reject each one.

## 14. Current verified state

The local feasibility and vertical-slice runs on 2026-07-28 produced:

| Baseline | Logical parameters | Weight bytes | Total BPW | Audit |
| --- | ---: | ---: | ---: | --- |
| Uniform 4-bit | 27,781,427,952 | 16,903,941,980 | 4.8677 | PASS |
| Uniform 6-bit | 27,781,427,952 | 23,627,280,146 | 6.8038 | PASS |
| Attributed mixed baseline | 27,781,427,952 | 19,904,508,887 | 5.7317 | PASS |

All supplied baselines passed:

- immutable revision detection;
- index and tensor integrity;
- logical parameter equivalence;
- Qwen 3.6 adapter classification;
- MTP sidecar and provenance validation;
- AX Engine doctor;
- MLX-LM static compatibility.

The complete revision-pinned source is:

```text
Qwen/Qwen3.6-27B
revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
27,781,427,952 logical parameters
```

M0 is complete and the feasibility report is `ready-for-conversion`.

The corrected manual M1 artifact:

- preserves all 460,730,096 vision parameters in a checksummed BF16 protected sidecar;
- byte-preserves the 849,400,381-byte external MTP sidecar;
- has exact 27,781,427,952-parameter equivalence with the source;
- passes AX Engine doctor and actual MLX-LM load/generation;
- measures 6.0981 total BPW and 21,176,724,385 weight bytes.

The regenerated `axquant.artifact.v2` vertical slice also records 5.9444 measured main BPW,
requires exact total/MTP/vision parameter coverage before the atomic rename, and passes AX Engine
doctor plus actual MLX-LM generation.

The Python distribution also builds as a clean `py3-none-any` wheel and passes an isolated-venv
installation, import-version, and `axquant --help` smoke test.

The current post-hardening wheel is
`.internal/tmp/qwen36-v1-wheel-candidate-20260730-v4/axquant-1.0.0-py3-none-any.whl`
(188,158 bytes, SHA-256
`44d5f98641a68f410f4af088ac48922f842ad9d864dbb7e0c17d1730a346db84`) and passes isolated
installation, version, CLI, and release-workflow smoke checks. Its `RECORD` is valid and its
metadata declares the exact v1.0.0 release version. The source audit currently collects and
passes 301 tests, strict mypy over 46 source files, Ruff lint/format checks, and whitespace
validation.

Publication now emits `axquant.reproduction.v3`: exact argument-array steps download the pinned
source, reconvert the bound plan, run both runtime checks, and verify the regenerated artifact.
The recipe binds its plan, calibration manifest, immutable conversion manifest, optional MTP
sidecar and required provenance/runtime companions, and every expected Safetensors file by
SHA-256. `axquant verify-reproduction` returns a strict machine-readable pass/fail result.

Runtime diagnostics now use `axquant.runtime-check.v2` and bind the candidate model ID, immutable
revision, and resolved artifact directory. Publication rejects an unpinned candidate revision,
and the family compatibility matrix rejects runtime evidence for a different identity or path.

The complete comparison set is now represented by `axquant.benchmark-evidence-index.v1`.
`benchmark-index` requires all eight named baseline entries, checksum-binds every available
evaluation bundle, requires matched hardware, software, power, controls, completed trials,
quantizer provenance, and zero kernel fallbacks, and permits mixed/AWQ/DWQ omissions only with
explicit reasons. Publication requires a release-ready index and packages its available evidence;
a required BF16, uniform-4, uniform-6, or identical-candidate MTP-off/on bundle cannot be omitted.

Publication also requires `axquant.release-validation-index.v1`. `validation-index` accepts
exactly one passing `agent-coding` and one passing `general` validation plus their release-ready
benchmark indexes, requires matched candidate/reference identities and distinct dataset digests,
and checksum-binds all four inputs. Both profiles and their available evaluation bundles are
packaged into the release directory.

Complete refinement execution is now orchestrated by `refine-run`. Its dry-run manifest binds all
inputs and exact commands; `--execute` resumes verified outputs through conversion, quality,
identical-checkpoint MTP A/B, size evidence, validation, complete measurement, deterministic
selection, and Pareto reporting. Actual release candidate executions remain pending the measured
sensitivity result.

The v0.7 hardware-registry contract is implemented. `hardware-registry` checksum-binds
`axquant.refinement-measurements.v5` complete
measurements, plans, release-quality sensitivity, validation, raw and summarized MTP A/B evidence,
and per-module conversion records. It records the named device/chip/memory/OS, MLX, MLX-LM and AX
Engine versions, attested power mode, exact commands, protocol tolerance, bit/group/method/role
and tensor-shape coverage, and fallback counts. A failed trial, nonzero fallback, incomplete
conversion coverage, or provenance mismatch retains `kernel_evidence=unmeasured`. Publication now
requires both a release-ready hardware registry and a Pareto frontier from the identical
measurement set. It verifies and packages that complete measurement set as
`refinement_measurements.json`; the final audit binds the selected M6 interaction improvement to
the same file. Stable measurement IDs allow one candidate to retain distinct named-host records;
selection uses worst-host objective loss and BPW rather than choosing a favorable host. Tooling
for M7 is complete; measured release entries remain pending the release-scale sensitivity and
candidate runs.

`release-audit` emits request-checksum-bound `axquant.release-audit.v4` and is the authoritative
final proof rather than a summary of presumed status. It
rechecks M0–M8 from the feasibility report, measured sensitivity, selected refinement, both
profile validations and every indexed evaluation, raw hardware evidence, Pareto report,
release-time catalog-bound compatibility matrix, both runtime checks, reproduction recipe/result,
prepared release files, and the toolkit wheel. M0 independently recomputes source/baseline
completeness, parameter and
architecture equivalence, immutable revisions, MTP presence, and baseline runtime results rather
than trusting the feasibility status label. M1 rejects duplicate or unsafe manifest records and
requires exact coverage of every artifact Safetensors file. M2 reloads every indexed evaluation
and recomputes required baselines, dataset/seed/control equivalence, immutable identities,
software/hardware provenance, zero fallbacks, complete trials, and identical-checkpoint MTP
pairing. It also requires both workload profiles to bind one candidate/reference pair and distinct
datasets. M3 checksum-binds and reloads the packaged calibration manifest, verifies its complete
provenance and separation attestation, and requires finite tensor-scoped measurements. M6 binds
both the parent and child of a claimed interaction gain to validated,
plan-specific complete-model measurements and recomputes their worst-host loss and BPW. M7 rebuilds
every Pareto point, dominator, and frontier membership from the bound measurement set. It reruns
reproduction verification and inspects wheel name, exact v1.0.0
version, pure-Python metadata, license, required modules, and every `RECORD` member, size, and
SHA-256. The wheel must declare itself production/stable, require Python 3.11 or newer, carry MIT
metadata, and list every runtime dependency. The artifact manifest, plan, and reproduction recipe
must all identify the same AXQuant version as that wheel; the artifact and reproduction recipe
must also match the plan's source, profile, calibration, precision policy, runtime scope, and
random seed as applicable. The audit also reloads the publication-ready plan,
immutable conversion manifest, validation/benchmark indexes, hardware registry/evidence,
refinement measurements, Pareto report, and reproduction recipe from the artifact and requires
them to be semantically identical to the externally audited evidence after normalizing only
artifact-relative path rewrites. It also re-hashes the selected candidate's manifest, both runtime
checks, and profile validation against its compatibility-matrix entry. The v4 audit request
additionally binds the original family-compatibility request; M5 reloads and re-hashes every
required checkpoint's manifest, plan, two runtime checks, and both profile validations so an
official dense size cannot be omitted or represented by detached or stale matrix claims. Any
failed or missing milestone is written as a blocker and makes the command return `1`.

The release validator now requires checksum-bound uniform-4 and candidate size evidence. The
manual slice measures 125.3% of the uniform-4 weight bytes, so it fails the 110% release size gate
independently of its quality result.

On the deterministic 52-task suite v2 with the pinned Qwen chat template, thinking disabled,
64 answer tokens, and identical seed, the manual artifact retained 100.99% of the uniform-6-bit
aggregate score. It matched JSON, tool, multilingual, and long-context category scores, exceeded
the coding and syntax scores in this small suite, and had a 0.68% perplexity increase. This is
valid complete-model quality evidence for the manual vertical slice, but it is not release
evidence for an automated plan and does not override its size, MTP, or evidence-kind failures.

The first complete MTP A/B suite is a failed release result: one of five greedy outputs diverged
and MTP decode throughput was below direct decode throughput on every measured trial. No MTP
quality or acceleration claim is established. Follow-up checksum-bound depth sweeps used AX
Engine's `AX_MLX_MTP_MAX_DEPTH` control: depth 1 improved median throughput to about 1.01x direct
and depth 2 measured about 0.98x, but the same third greedy prompt diverged at both depths. Neither
meets the 1.20x speed or exactness gates.

The measured-candidate investigation then separated sidecar layout from runtime behavior. The
byte-preserved source sidecar and the official uniform-4/uniform-6 BF16 sidecar have the same 15
tensor names. All eight non-norm payloads are byte-identical; the official sidecar adds one, with
BF16 rounding, to each of seven one-dimensional RMSNorm tensors. AX Engine 6.11.1's raw-sidecar
sanitizer applies that correction only when `mean_abs < 0.15`. The raw means are approximately
0.0827, 0.2110, 0.7438, 0.7610, 1.2741, 0.4400, and 0.1792, so only one of seven tensors is
corrected. This explains why active raw-sidecar diagnostics accepted 0 / 40 proposed draft
tokens.

A policy-minimum trunk without the floor plan's two optional 6-bit projection upgrades reproduced
the raw 0 / 8 acceptance result, excluding mixed 4/6-bit packing as the cause. A
development-only symlink view that paired the same trunk with the fully shifted sidecar restored
acceptance. Under the same positive `0.000001` confidence floor as the uniform-6 control, two
depth-1 repetitions each measured 31 / 38 accepted drafts (0.815789), 5 / 5 exact outputs, and
zero kernel fallbacks. This is 1.087719 acceptance retention versus uniform-6, but the repeated
median speeds were only 0.969322x and 0.891567x direct. Depths 2 and 3 measured 0.900772x and
0.932227x; depth 2 also diverged once. The view is not a converted AXQuant artifact and is not
releasable. The uniform-4 and uniform-6 controls measured 0.790698 and 0.750000 acceptance but
only 1.024759x and 1.088955x median speed, respectively.

The dedicated, explicit `ax-engine-qwen36-v1` backend now reproduces that layout without borrowing
the control bundle or its metadata. It validates the raw sidecar's file and per-tensor checksums,
requires the exact 15-tensor BF16 contract, transforms only the seven named norms, verifies all
eight projections unchanged, and writes `axquant.mtp-sidecar-provenance.v3`. A real atomic 27B
conversion produced 5.577043 total / 5.415231 main BPW with 21 manifest-bound files. Its 15 tensor
payloads exactly match the shifted control; AX Engine doctor and MLX-LM generation pass. The
actual candidate's five-trial depth-1 result is 31 / 38 accepted drafts, 5 / 5 exact outputs, zero
kernel fallbacks, and 0.960388x speed. This closes the implementation/layout portion of M4 but
confirms the M2 throughput failure.

Route-phase profiling also identifies a biased AX Engine bypass. A first-cycle draft miss
initializes the `alpha=0.05` acceptance EWMA at zero; seven later accepts raise it only to about
0.302, below the 0.50 threshold, so long trials permanently bypass MTP after eight samples despite
75–87.5% aggregate acceptance. Raising the diagnostic minimum-sample window to 1000 removes all
143 direct-fallback steps and keeps MTP active, but speed is still only 0.967637x. The no-bypass
profile costs 53,215 us/output-token versus 51,903 direct, while the 1.20x gate requires at most
43,253. The remaining ~9,962 us/token gap cannot be closed by planner precision or by eliminating
draft or rollback cost alone.

Source-level investigation of the pinned public AX Engine 6.11.1 tag rules out random tail
sampling in the failing temperature-zero cycle and localizes the divergence to a rejected
strict-MTP verifier tail. The leading testable hypothesis is route-dependent Qwen linear-attention
state or numerics: normal one-token decode uses two default-on `seq == 1` Metal specializations,
while verification evaluates `[last_token] ++ pending_draft` on a cloned cache as a multi-token
sequence. Disabling the post-input Metal path alone and then both documented one-token
linear-attention Metal paths did not restore exactness: each profile diverged on two of five
trials and measured only 0.948156x and 0.970926x direct speed. The exact evidence and source trace
are recorded in `.internal/tmp/ax-engine-v6.11.1-mtp-greedy-exactness-investigation.md` and
`.internal/tmp/qwen36-v1-mtp-sidecar-layout-investigation.md`. This diagnosis does not change the
failed milestone or authorize optimistic/approximate MTP.

The 6.11.1 findings above are superseded at runtime level: upstream published v6.12.0 on
2026-07-29 (the Homebrew formula still serves 6.11.1), and the 2026-07-31 formal M5 suite ran on
a notarized 6.12.1 build ("Harden Qwen linear-attention MTP exact depth-one path") whose
depth-one profile is exact on every measured arm and takes uniform-6 past the 1.20x gate on both
workloads; the remaining M2 shortfall is candidate-specific (see the milestone table and
`.internal/tmp/qwen36-v1-release-continuation-20260730.md`). A separate public Qwen 3.6 MTP
report in llama.cpp also documents deterministic committed-token divergence at higher draft depth
([issue #23302](https://github.com/ggml-org/llama.cpp/issues/23302)). That report is corroborating
runtime context only, not AXQuant implementation evidence; AXQuant continues to fail closed on
its own AX Engine A/B results.

Both former PRD feasibility conflicts now have accepted resolutions:

1. **Resolved in direction by AXQ-026 on 2026-08-01.** The hard BF16 vision/MTP/norm/head and
   8-bit embedding floors produce a planner policy minimum of 5.5770 BPW, about 114.6% of the
   audited uniform-4-bit size — incompatible with the 110% size gate. Of the four options in
   `.internal/tmp/qwen36-v1-size-decision-analysis.md`, the workspace owner approved the governed
   8-bit LM-head floor: a modeled 1,191,936,000-byte saving that fits the gate with
   419,276,754 bytes of margin. The floor is lowered per plan only
   (`plan --lm-head-floor 8bit`, recorded in `constraints.lm_head_min_bits`), the probe now
   measures the LM head at 8-bit so the choice is measurement-backed, and the release audit
   requires the measured 8-bit LM-head candidate before such a plan can certify. Release
   certification still requires full dual-profile quality, MTP, size, and hardware evidence on
   the complete candidate; the exception machinery remains available only if the measured
   LM-head evidence fails.
2. **Resolved by AXQ-016 on 2026-07-30.** The official
   [Qwen 3.6 catalog](https://huggingface.co/collections/Qwen/qwen36) contains one dense parameter
   size (27B, plus a same-size FP8 representation) and one MoE size (35B-A3B). M5 now requires
   every official dense parameter size present at release time rather than a fixed minimum count.
   `axquant.compatibility-request.v2` enumerates that scope and requires both `agent-coding` and
   `general` evidence for one immutable artifact per model;
   `axquant.compatibility-matrix.v2` and `axquant.release-audit.v4` reject missing, unexpected,
   inconsistent, or tampered scope evidence. At the 2026-07-30 verification point, the complete
   required dense scope is the single revision-pinned 27B source.

## 15. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| MTP provides little or negative speedup | Protect MTP, benchmark draft depths, and require positive identical-checkpoint throughput |
| Near-6 quality does not fit the size budget | Use workload focus, learned refinement, and measured Pareto profiles |
| 6-bit is smaller but slower | Measure real kernels and reject theoretical-only improvements |
| Calibration overfits benchmarks | Separate data, publish composition, and use holdout tasks |
| Tensor interactions invalidate isolated scores | Evaluate complete candidates and perform bounded swaps |
| MLX API changes | Pin validated versions and maintain adapter regression tests |
| MLX-LM MTP remains immature | Keep AX Engine authoritative for MTP and preserve ordinary MLX fallback |
| AXQuant appears to rename another method | Maintain clean-room history and visibly independent MTP, tensor, validation, and hardware contracts |
| Licensing concerns | Release AXQuant under MIT; use public APIs/research and preserve all third-party attribution |
| Large conversions leave corrupt output | Use same-filesystem staging and atomic final rename |

## 16. Acceptance

### 16.1 v0.1 acceptance

- feasibility report is `ready-for-conversion`;
- BF16 source is revision-pinned and inspectable;
- a reviewed manual recipe produces a valid plan;
- unsupported and protected components remain BF16 or policy compliant;
- the external MTP bundle is checksum-preserved;
- conversion output is atomic and complete;
- AX native manifest validation passes;
- AX Engine doctor returns `ready`;
- MLX-LM standard inference smoke test passes;
- all provenance artifacts are written;
- no MTP speed or production-quality claim is made.

### 16.2 v1 acceptance

- stable public CLI and versioned configuration;
- every official dense Qwen 3.6 parameter size present at release time, each validated under
  both `agent-coding` and `general`;
- measured per-tensor sensitivity;
- automated 4/6/8/BF16 planning;
- at least one learned-refinement method;
- complete-model global validation;
- MTP acceptance and throughput gates;
- required size, memory, quality, and speed targets;
- reproducible public benchmark commands and model artifacts.

## 17. Governance

- This PRD controls product scope and release claims.
- Architecture changes require an ADR update.
- Interface, schema, algorithm, or runtime changes require a Technical Specification update.
- Public documentation may summarize this document but must not strengthen its claims.
- A release exception must identify the failed target, measured tradeoff, owner, and expiry.
