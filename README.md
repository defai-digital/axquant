# AXQuant

AXQuant is a command-line toolkit that converts a supported standard LLM checkpoint into an
AXQuant-optimized checkpoint for Apple Silicon.

It inspects the model, creates an auditable mixed-precision plan, converts the weights through
public MLX-LM interfaces, and writes the manifests and validation metadata needed by AX Engine.
AXQuant can assign 4-bit, 6-bit, 8-bit, or BF16 precision per tensor while protecting sensitive
components such as normalization layers, output heads, routers, vision tensors, and
multi-token-prediction (MTP) weights.

> AXQuant improves deployment efficiency; it does not train the source model or add new learned
> capabilities. Its goal is to reduce storage and unified-memory cost while preserving important
> model quality and runtime behavior.

## How it works

```text
Revision-pinned BF16 MLX checkpoint
                 │
                 ▼
        inspect → plan → convert → runtime-check / validate
           │                │
           │                └── AXQuant-optimized MLX checkpoint
           │                    + AX Engine runtime metadata
           │                    + plan, manifest, and provenance
           └── model inventory and protection boundaries
```

The intended user journey is:

1. provide a supported, unquantized LLM checkpoint;
2. inspect its architecture and tensors;
3. generate a manual or evidence-based mixed-precision plan;
4. convert it from the command line;
5. verify runtime compatibility, model quality, memory use, and speed;
6. publish only after the required validation gates pass.

## Input and output

### Input

AXQuant currently converts a revision-pinned, unquantized Qwen3.6-27B Safetensors
checkpoint through MLX-LM. The checkpoint must use the expected configuration and indexed
Safetensors layout. Other model families can be inspected, but they are not yet accepted for
conversion.

### Output

A successful conversion produces a new portable MLX model directory containing:

- mixed-precision model weights and standard MLX configuration files;
- the exact quantization plan used for the conversion;
- an AXQuant artifact manifest with checksums and provenance;
- AX Engine and MLX-LM runtime metadata;
- an AX Engine native manifest when the runtime tool is available;
- a byte-preserved external MTP sidecar by default, or an explicitly prepared development
  sidecar with transform-level provenance;
- a raw, checksummed BF16 sidecar for protected vision tensors when MLX-LM excludes them.

The artifact manifest records authoritative main-model and total logical parameters, physical
Safetensors bytes, and measured BPW. The language-model output remains usable as a standard MLX
checkpoint. AX Engine consumes the additional AXQuant metadata for runtime-specific behavior;
MLX-LM may ignore that metadata and use ordinary decode.

## Why AXQuant

Uniform quantization gives every eligible tensor the same precision. AXQuant instead plans
precision at tensor level so the model can spend more bits where they matter and fewer bits where
they do not.

Its design centers on:

- **mixed precision:** 4-bit, 6-bit, 8-bit, and BF16 assignments;
- **quality protection:** hard precision floors for sensitive model components;
- **MTP awareness:** explicit MTP detection, protection, validation, and runtime metadata;
- **workload awareness:** separate objectives for general and agent/coding workloads;
- **real deployment cost:** actual artifact bytes, unified memory, latency, and throughput;
- **reproducibility:** revision-pinned inputs, deterministic artifacts, checksums, and manifests;
- **fail-closed conversion:** incomplete plans or unmatched modules stop conversion;
- **independent implementation:** public APIs and research without reused quantizer internals.

## Current status

The toolkit version is `1.0.0` (packaging classifier: **Beta**). That means the CLI, schemas,
and conversion pipeline are feature-complete and quality-gated for the scoped Qwen 3.6 path—not
that a formal Hub release candidate has passed MTP speed, size, dual-profile validation, and
`release-audit`. Host/candidate certification remains evidence-gated; there is **no** certified
public AXQuant model release claimed here.

AXQuant is deliberately scoped to the validated Qwen 3.6 path; it is not a universal quantizer.

| Area | Current support |
| --- | --- |
| Platform | Apple Silicon with MLX |
| Conversion input | Revision-pinned, unquantized MLX checkpoint |
| Conversion target | Qwen3.6-27B language path |
| Precision choices | 4-bit, 6-bit, 8-bit, and BF16; measured affine, DWQ-clipped affine, and portable AWQ |
| Planning | Manual recipes and a planner that consumes measured sensitivity artifacts |
| MTP | Detection, byte-preserved sidecars, and an opt-in Qwen 3.6 AX Engine layout backend |
| Primary runtime | AX Engine |
| Compatibility runtime | MLX-LM standard inference |
| Output integrity | Atomic conversion, exact parameter coverage, measured BPW, checksums, manifests, and runtime metadata |

Implemented now:

- indexed Safetensors inspection and logical parameter reconstruction;
- deterministic, provenance-bound tokenized calibration caches;
- resumable per-tensor MLX probes with 4/6/8/BF16 affine candidates and targeted DWQ refinement;
- portable AWQ activation-scale search with convert-time refinement and affine packing;
- Qwen 3.6 tensor classification, MTP detection, and vision protection;
- auditable manual recipes with mandatory precision floors;
- mixed-precision planning from compatible sensitivity reports;
- MLX-LM conversion with plan-to-module coverage checks;
- atomic output staging that prevents partial final checkpoints;
- AX Engine manifest generation and runtime readiness checks;
- identical-checkpoint AX Engine MTP off/on benchmarking with greedy-output equality;
- deterministic quality/benchmark suites and complete-model MLX quality evaluation;
- validation gates for externally measured quality and performance evidence;
- guarded Hugging Face publication.

Still incomplete (external evidence / runtime / deferred scope — not missing toolkit commands):

- a **new** release candidate cycle that clears dual-profile MTP speed (≥1.20×), quality, and size
  (or a governed size exception only after non-size gates pass); formal cand-002/003 failed;
- complete-candidate interaction optimization driven by measured holdout results on that candidate;
- validated conversion evidence for any future official dense Qwen 3.6 sizes;
- validated conversion adapters for additional LLM families;
- dedicated quantization of external MTP sidecars;
- KV-cache, VLM, and MoE expert-level planning.

Release discipline (gate order, dual-profile metric completeness, no rewrite of formal roots) is
recorded in `.internal/product/release-best-practices.md`.

Architecture-prior analysis, smoke probes, and manual plans are explicitly marked as
non-release development evidence. They cannot support production-quality or performance claims.

## Installation

Requirements:

- Python 3.11 or newer;
- Apple Silicon for MLX-backed conversion;
- an unquantized MLX source checkpoint;
- `ax-engine-bench` when AX Engine manifest generation is required.

Create an environment and install AXQuant with the MLX backend:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mlx]"
axquant --help
```

For development, install the test and lint tools as well:

```bash
python -m pip install -e ".[dev,mlx]"
```

## Quick start: development conversion

The currently available end-to-end path uses the reviewed manual recipe. This proves the
conversion workflow, but its output remains unmeasured development evidence.

### 1. Inspect the source checkpoint

```bash
axquant inspect \
  --model /models/Qwen3.6-27B-bf16 \
  --model-id Qwen/Qwen3.6-27B \
  --revision REVISION_SHA \
  --output inventory.json
```

Inspection verifies the checkpoint layout, identifies the supported architecture, classifies
each tensor, detects MTP, and records which components must remain protected.

### 2. Create a mixed-precision plan

```bash
axquant plan-manual \
  --inventory inventory.json \
  --recipe examples/qwen36-27b-manual-v0.1.yaml \
  --output manual-plan.json \
  --markdown-output manual-plan.md
```

The included recipe applies 4-bit defaults, keeps attention weights at 6-bit, and preserves
protected components at their required precision. The generated plan records every assignment
and its reason.

### 3. Convert the model

```bash
axquant convert \
  --model /models/Qwen3.6-27B-bf16 \
  --revision REVISION_SHA \
  --plan manual-plan.json \
  --allow-unmeasured \
  --ax-engine-manifest if-available \
  --output AX-Qwen3.6-27B-MLX-AXQuant-4bit
```

If the plan preserves MTP as an external bundle, conversion requires:

```bash
--mtp-sidecar /models/Qwen3.6-27B-bf16/mtp.safetensors
```

The default `--mtp-layout byte-preserved` path never changes tensor payloads. The explicit
development path:

```bash
--mtp-layout ax-engine-qwen36-v1
```

accepts only a checksum-bound raw Qwen 3.6 bundle with the exact 15-tensor BF16 contract. It adds
one, with BF16 rounding, to the seven named MTP RMSNorm tensors, proves the eight projection
payloads unchanged, and writes a new provenance manifest plus a depth-1 AX Engine runtime
contract. This opt-in layout is not a release waiver: identical-checkpoint MTP exactness,
acceptance, and throughput must still pass the ordinary validation gates.

The `--allow-unmeasured` and `--ax-engine-manifest if-available` options are development-only.
Omit them from a release workflow: release conversion requires measured evidence and a valid AX
Engine manifest. A measured plan must also pass
`--calibration-manifest calibration_manifest.json`; conversion verifies its checksum and
provenance against the plan and packages it with the artifact.

### 4. Check the converted model

```bash
axquant runtime-check \
  --model AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --model-id AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --revision candidate-revision \
  --runtime ax-engine \
  --output runtime-check.json
```

Use `--runtime mlx-lm` to perform the MLX-LM generation smoke. `--static-only` is a development
diagnostic.

## CLI workflow

Run `axquant COMMAND --help` for the full options of any command.

| Command | Purpose | Current maturity |
| --- | --- | --- |
| `feasibility` | Audit source and comparison checkpoints before conversion | Implemented |
| `inspect` | Inventory tensors, architecture, quantization, and MTP | Implemented |
| `calibrate` | Validate calibration input and record provenance | Implemented |
| `tokenize-calibration` | Build and verify a deterministic tokenized cache | Implemented |
| `analyze` | Measure resumable affine/DWQ/BF16 sensitivity, including targeted refinement | Implemented |
| `plan` | Allocate 4/6/8/BF16 from a sensitivity report | Implemented; release use requires measured evidence |
| `plan-manual` | Apply an explicit reviewed precision recipe | Implemented for development |
| `convert` | Create the mixed-precision MLX checkpoint and metadata | Implemented for supported Qwen 3.6 scope |
| `runtime-check` | Run AX Engine readiness or actual MLX-LM generation | Implemented |
| `prepare-suite` | Materialize deterministic disjoint benchmark inputs | Implemented |
| `evaluate-quality` | Run MLX perplexity and scored generation tasks | Implemented |
| `compare-quality` | Compare matched quality runs with per-task visibility | Implemented |
| `benchmark` | Collect AX Engine runtime evidence | Implemented |
| `benchmark-ab` | Compare one checkpoint with MTP disabled/enabled | Implemented |
| `benchmark-index` | Bind every required baseline or record why it is unavailable | Implemented |
| `validation-index` | Require disjoint passing agent-coding and general evidence | Implemented |
| `refine` | Generate proxy-ranked bounded precision swaps | Development only |
| `refine-measure` | Build checksum-bound complete-candidate evidence | Implemented |
| `refine-select` | Select only from checksum-bound, validated complete candidates | Implemented |
| `refine-export` | Export standalone executable plans from a refinement result | Implemented |
| `refine-run` | Resume complete conversion, quality, MTP, validation, and selection runs | Implemented |
| `pareto` | Report non-dominated validated candidates on named hardware | Implemented |
| `hardware-registry` | Certify checksum-bound kernel, version, power, and shape coverage | Implemented |
| `release-audit` | Prove every M0–M8 gate from bound release evidence | Implemented |
| `compatibility-matrix` | Bind family-wide artifact, runtime, and validation evidence | Implemented |
| `validate` | Apply release thresholds to external benchmark evidence | Implemented |
| `size-evidence` | Bind authoritative candidate/uniform-4 artifact sizes | Implemented |
| `release-exception` | Record an approved, expiring, evidence-bound size exception | Implemented |
| `report` | Render plan and validation reports | Implemented |
| `publish-prepare` | Assemble a release only after validation | Implemented |
| `publish` | Preview or execute a guarded Hugging Face upload | Implemented |
| `verify-reproduction` | Verify regenerated weight bytes and bound provenance | Implemented |
| `name` | Generate the recommended AXQuant model name | Implemented |

## Measured planning and validation

DWQ release evidence uses the same deterministic 0.1/99.9-percentile clipping implementation
during sensitivity probing and conversion. A targeted run adds measured DWQ candidates to an
existing complete affine report without rewriting any base candidate:

```bash
axquant analyze \
  --model Qwen/Qwen3.6-27B \
  --revision pinned-source-revision \
  --calibration calibration-cache \
  --base-sensitivity measured-affine-sensitivity.json \
  --methods dwq \
  --target-tensor model.language_model.layers.4.mlp.up_proj.weight \
  --state dwq-probe-progress.json \
  --output measured-affine-dwq-sensitivity.json
```

The merged report records the base report's semantic digest, inventory digest, probe backend,
target count, and method set. Release audit requests list every ancestor under
`sensitivity_lineage`; M3 replays the chain and rejects removed or modified base candidates,
undeclared additions, protocol drift, cycles, missing parents, and unused reports.

Once a measured sensitivity report is available, create a plan without the development override:

```bash
axquant plan \
  --analysis measured-analysis.json \
  --target-bpw 4.8 \
  --bits 4,6,8,16 \
  --mtp protected \
  --output quantization-plans
```

Validate externally collected benchmark bundles:

```bash
axquant size-evidence \
  --artifact-manifest candidate/axquant_manifest.json \
  --model-id AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --revision candidate-revision \
  --output candidate-size-evidence.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --profile agent-coding \
  --output validation.json
```

If a measured Pareto candidate misses both the BPW target and the uniform-4 size-ratio gate, a
release authority can record a time-bounded exception. The command computes the observed values
from the two size artifacts; it does not accept caller-authored observed values:

```bash
axquant release-exception \
  --exception-id AXQ-SIZE-001 \
  --plan selected-plan.json \
  --candidate-size candidate-size-evidence.json \
  --size-reference uniform4-size-evidence.json \
  --tradeoff-evidence measured-tradeoff.json \
  --measured-tradeoff "Measured quality, speed, and memory tradeoff approved for release." \
  --owner "AutomatosX release owner" \
  --approved-by "Named release authority" \
  --approval-reference "release-decision-001" \
  --approved-at 2026-07-30T12:00:00Z \
  --expires-at 2027-01-31T00:00:00Z \
  --output release-exception.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --plan selected-plan.json \
  --release-exception release-exception.json \
  --exception-evidence tradeoff=measured-tradeoff.json \
  --profile agent-coding \
  --output validation.json
```

The exception can downgrade only `artifact.weight_size_ratio`; it must also disclose the failed
measured-BPW target. Quality, speed, memory, fallback, integrity, and provenance failures remain
errors. Release audit requests that use an exception must list its file under
`release_exceptions` and provide the exact `plan`, `candidate_size`, `size_reference`, and
`tradeoff` paths under `release_exception_evidence`. M4 reloads and hashes every file, checks both
validation profiles, verifies approval and expiry, and compares the packaged
`release_exception.json` with the approved record.

For a refinement candidate, derive its selection record from the converted manifest, matched
quality comparison, and release validation rather than authoring measurement values:

```bash
axquant refine-measure \
  --refinement refinement.json \
  --candidate-id cand-0000-000 \
  --measurement-id cand-0000-000-m3-max \
  --artifact-manifest candidate/axquant_manifest.json \
  --quality-comparison candidate/quality-comparison.json \
  --validation candidate/validation.json \
  --output measurements.json
```

Use `--existing measurements.json` with a new output path to accumulate another candidate or a
second named-host result for the same candidate. Measurement IDs must be unique. `refine-select`
uses the worst measured objective and BPW across every host record for a candidate, so adding
hardware evidence cannot make selection less conservative. The complete objective combines task
retention and perplexity with MTP acceptance, peak memory, and effective speed. Refinement
parentage is a precision-only monotonic chain: a child may upgrade formats but cannot downgrade
or exchange an unrelated tensor.

Prepare the exact complete-candidate run without executing expensive model work:

```bash
axquant refine-run \
  --request examples/refinement-execution-request.yaml \
  --output-dir run/complete-candidates
```

Review `execution-manifest.json`, then add `--execute`. The runner resumes checksum-verified
completed outputs, skips the remainder of a candidate after an execution failure, treats
validation exit `1` as measured failed-gate evidence, merges complete measurements, and runs
`refine-select` plus `pareto`.

Every release benchmark must name its power mode and quantizer/version. `refine-run` reads
`benchmark_power_mode` from its request, derives the AXQuant identity from each plan, and includes
both raw A/B logs in the resumable output contract. Standalone baseline runs use
`--power-mode`, `--quantizer`, and `--quantizer-version`. `benchmark-ab` derives adjacent-token
repetition directly from emitted token IDs, records depth-one proposal accuracy, and derives
greedy divergence from the matched A/B outputs. For a uniform-6 reference A/B, use
`--direct-baseline-kind uniform-6bit --mtp-baseline-kind uniform-6bit`; the default kinds remain
the AXQuant MTP-off/on release pair. Use `--record-failed-speedup` for an evidence sweep that must
retain both evaluation bundles when only the speed floor fails: the command writes the complete
evidence, returns status `1`, and leaves exactness and matched-control invariants fail-closed.
Build the M7 hardware registry only from the resulting raw
logs, evaluation bundles, validation, plan, converted artifact manifest, sensitivity report,
quality comparison, and quantizer execution manifest:

```bash
axquant hardware-registry \
  --request examples/hardware-registry-request.yaml \
  --output hardware-profile-registry.json
```

The command returns `1` while validation is failing, any runtime or conversion fallback is
present, provenance is inconsistent, the complete objective cannot be rebuilt from the artifact,
quality, and validation files, or the claimed bit/group/role/shape coverage is not measured. The
registry records both the semantic and file digest of its complete-candidate measurement set.
Publication verifies that file, packages it as `refinement_measurements.json`, packages every
objective input, and rewrites the registry to packaged relative paths. Each registry entry
identifies the exact measurement ID, allowing one candidate and plan to be certified on multiple
named hosts.

Prepare the release directory locally, then run the aggregate proof before tagging v1.0:

```bash
axquant publish-prepare \
  --model AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json
```

```bash
axquant release-audit \
  --request examples/release-audit-request.yaml \
  --output release-audit.json
```

This revalidates indexed evaluation, complete-refinement, and hardware file checksums; binds the
selected interaction improvement to the packaged measurement set; reruns reproduction
verification; inspects the wheel metadata, contents, and every `RECORD` member hash/size; and
requires the packaged plan, validation/benchmark evidence, hardware registry/evidence,
refinement measurements, Pareto report, and recipe to match the external evidence graph. The
M0 check recomputes checkpoint completeness, parameter/architecture equivalence, revisions, MTP,
and baseline runtime results rather than trusting the feasibility status label. M1 requires every
artifact Safetensors file to have one safe, size- and checksum-valid manifest record. M2 reloads
the indexed evaluations and rechecks complete trials, matched controls and hardware, provenance,
fallbacks, identical-checkpoint MTP pairing, one cross-profile candidate/reference pair, and
disjoint datasets. M3 reloads the checksum-bound calibration manifest, verifies separation and
provenance, requires finite tensor-scoped measurements, and verifies every targeted-sensitivity
ancestor. M6 reloads the bound artifact, quality comparison, and validation for every
measurement, recomputes the versioned complete objective, and requires a measured, validated,
monotonic parent/child gain. Complete-measurement construction also rejects non-authoritative
profile thresholds, an inconsistent validation pass label, core release metrics below their
active thresholds, nonzero kernel fallbacks, and a passing size overage without its governed
plan-bound exception. M7 rebuilds every Pareto point and frontier member from the bound
measurement set. The
compatibility matrix must bind that same candidate manifest, runtime checks, and validation. The
audit also reloads every checkpoint from the original compatibility request and re-hashes its
manifest, plan, runtime checks, and validation. The wheel must declare Python 3.11+, MIT, and all
runtime dependencies; the artifact, plan, recipe, and wheel must identify the same AXQuant
version. It reports M0 through M8 separately and returns `0` only when all nine milestones pass;
an alpha or pre-1.0 wheel, including one still carrying an Alpha distribution classifier, cannot
pass M8. An executed publication packages that exact authorizing result as `release_audit.json`
and refuses to overwrite a different existing audit.

Preview publication first. Add `--yes` only when the release should be uploaded; an executed
upload also requires the matching audit and its original request so the full M0–M8 proof can be
rerun from current evidence:

```bash
axquant publish \
  --model AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json \
  --release-audit release-audit.json \
  --release-audit-request examples/release-audit-request.yaml
```

Before publication, build the complete comparison index. BF16, uniform 4-bit, uniform 6-bit,
and the identical AXQuant MTP-off/on pair are mandatory. Mixed-precision, AWQ, and DWQ entries
may be unavailable, but they cannot be omitted and must state why:

```bash
axquant benchmark-index \
  --request examples/benchmark-evidence-request.yaml \
  --output benchmark-evidence-index.json
```

Build one benchmark index and validation report for each required profile, using distinct
evaluation datasets. Then bind them into the publication gate:

```bash
axquant validation-index \
  --request examples/release-validation-request.yaml \
  --output release-validation-index.json
```

Publication rejects a missing profile, a reused dataset, differing candidate/reference
identities, a failed validation, or a non-ready benchmark index.

Every prepared release includes `reproduction_recipe.yaml` with argument-array commands for
downloading the pinned source, converting it, checking both runtimes, and verifying every
regenerated Safetensors file. Prepared MTP layouts additionally checksum-bind the provenance and
runtime companion files required to reuse the transformed sidecar without applying the transform
again. After running those commands, verification can also be invoked directly:

```bash
axquant verify-reproduction \
  --recipe reproduction_recipe.yaml \
  --artifact regenerated-model \
  --output reproduction-verification.json
```

Build the M5 family matrix from checksum-bound artifact, AX Engine, MLX-LM, and validation
evidence:

```bash
axquant compatibility-matrix \
  --request examples/qwen36-compatibility-request.yaml \
  --output compatibility-matrix.json
```

The request declares the complete official dense catalog as verified at a timezone-qualified
timestamp. The command returns `1` and still writes the matrix when any declared official dense
Qwen 3.6 model is absent, uses inconsistent candidate evidence, or lacks a compatible
`agent-coding` or `general` validation profile. At the current catalog revision, the only official
dense parameter size is 27B; FP8 is a representation of that size rather than a second size.

## Evidence and safety boundaries

- Architecture priors are never described as measured sensitivity.
- `--allow-unmeasured` is restricted to development conversion.
- Conversion fails if the plan does not cover every module it claims to quantize.
- External MTP sidecars remain byte-for-byte unchanged unless the explicit, provenance-checked
  Qwen 3.6 AX Engine layout backend is selected.
- Output is staged and atomically renamed only after conversion succeeds.
- Release claims require complete-model quality and hardware evidence.
- Credentials and Hugging Face tokens are never written to logs or manifests.

## Model naming

Recommended model names use:

```text
OWNER/AX-BASE-MODEL-MLX-AXQuant-TARGET
```

For example:

```text
AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit
```

The target suffix describes the checkpoint class, not a claim that every tensor uses that bit
width. The manifest contains the actual precision distribution and effective bits per weight.

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

Tests use small synthetic Safetensors fixtures and do not require real model weights.

## Documentation

- [Product requirements](.internal/product/requirements.md)
- [Architecture decision register](.internal/architecture/decision-register.md)
- [Technical specification](.internal/engineering/technical-specification.md)
- [Independent implementation policy](.internal/policies/clean-room.md)
- [Third-party notices and research references](THIRD_PARTY_NOTICES.md)

## License

AXQuant is released under the [MIT License](LICENSE). Dependencies, model checkpoints,
calibration datasets, and external tools retain their own licenses.
