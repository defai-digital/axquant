# AXQuant Architecture Decision Register

**Document status:** Accepted decision set  
**Applies to:** AXQuant v0.x and v1  
**Last reviewed:** 2026-07-30

This document consolidates the product and architecture decisions that constrain AXQuant
implementation. Each decision remains in force until explicitly superseded here.

## Decision index

| ID | Decision | Status |
| --- | --- | --- |
| AXQ-001 | Implement AXQuant independently under a clean-room boundary | Accepted |
| AXQ-002 | Start with Qwen 3.6 and one Qwen3.6-27B vertical slice | Accepted |
| AXQ-003 | Make AX Engine primary and MLX-LM the compatibility runtime | Accepted |
| AXQ-004 | Keep standard MLX weights as the portable base | Accepted |
| AXQ-005 | Define checkpoint membership by the weight index plus declared root sidecars | Accepted |
| AXQ-006 | Optimize the language path and preserve vision at BF16 in v0.x | Accepted |
| AXQ-007 | Use per-tensor 4/6/8/BF16 planning with hard protection floors | Accepted |
| AXQ-008 | Separate development evidence from release evidence | Accepted |
| AXQ-009 | Permit reviewed manual plans only for the v0.1 vertical slice | Accepted |
| AXQ-010 | Treat MTP as a first-class component and protect external sidecars | Accepted |
| AXQ-011 | Prefer hard constraints before a weighted objective | Accepted |
| AXQ-012 | Validate complete candidates before production selection | Accepted |
| AXQ-013 | Measure actual artifact BPW and actual Apple runtime cost | Accepted |
| AXQ-014 | Use versioned artifacts and fail-closed boundaries | Accepted |
| AXQ-015 | Convert atomically and publish only through release gates | Accepted |
| AXQ-016 | Certify every official dense Qwen 3.6 size present at release time | Accepted |

## AXQ-001: Independent clean-room implementation

**Status:** Accepted

### Context

AXQuant may use public PTQ research and public MLX interfaces, but it must have an independent
technical identity and implementation history. Reusing external package internals would create
legal, product, and reproducibility risks.

### Decision

AXQuant:

- uses public papers, equations, and documented algorithms;
- uses public MLX, MLX-LM, Safetensors, Hugging Face, and AX Engine interfaces;
- defines its own schemas, calibration profiles, objectives, planners, and tests;
- may audit a public checkpoint through its standard load contract;
- may compare against an attributed public checkpoint as an external baseline.

AXQuant does not copy, translate, decompile, vendor, or import:

- mlx-optiq implementation code;
- its tests or calibration data;
- its sensitivity outputs or allocation tables;
- its planner metadata or unindexed auxiliary artifacts;
- claims that AXQuant is an official successor.

### Consequences

- Independent design and commit history are mandatory.
- External baseline results remain attributed.
- Only AXQuant-produced measurements may be described as AXQuant measurements.

### Rejected alternatives

- Repackage another quantizer under the AXQuant name.
- Reverse engineer a non-public implementation.
- Import an external sensitivity map and treat it as AXQuant evidence.

## AXQ-002: Qwen 3.6 first

**Status:** Accepted

### Context

AXQuant needs one coherent model family to prove mixed precision, MTP, agent/coding quality,
portable MLX export, and AX Engine acceleration. Supporting Qwen and Gemma simultaneously would
multiply adapter and benchmark uncertainty.

### Decision

- Qwen3.6-27B is the only conversion target through v0.4.
- v0.5 and v1 use the catalog-complete dense-family proof defined by AXQ-016.
- As of the 2026-07-30 catalog verification, the only official dense parameter size is 27B.
- Generic models may be inspected but remain inventory-only.
- Gemma, MoE, VLM optimization, and unrelated models are post-v1 work.

### Consequences

- Qwen-specific behavior is explicit rather than hidden in generic code.
- The first benchmark matrix is smaller and falsifiable.
- AXQuant avoids competing first with Gemma's official QAT path.

### Rejected alternatives

- Launch Qwen and Gemma adapters together.
- Begin with Gemma and postpone MTP proof.
- Claim arbitrary Transformers architecture support.

## AXQ-003: Runtime tiers

**Status:** Accepted

### Context

AX Engine owns the MTP execution path and Apple deployment optimizations. MLX-LM provides
important ecosystem portability but does not need to expose all AX capabilities.

### Decision

| Level | Runtime | Authority |
| --- | --- | --- |
| A | AX Engine | Native manifest, MTP, runtime checks, and performance claims |
| B | MLX-LM | Standard language-model inference from portable MLX weights |

AX Engine is the primary runtime. MLX-LM is the compatibility runtime.

### Consequences

- MTP speed and acceptance claims use AX Engine evidence.
- MLX-LM feature and speed parity is not promised.
- Unsupported AX metadata may be ignored by MLX-LM.
- The portable backbone must not require a closed AXQuant runtime.

### Rejected alternatives

- Make MLX-LM's current MTP support a v0.1 completion dependency.
- Require identical features and throughput from both runtimes.
- Produce an AX-only weight format in v0.1.

## AXQ-004: Standard MLX weights plus additive AX metadata

**Status:** Accepted

### Decision

The deployment directory has separate authorities:

```text
config.json, tokenizer, Safetensors, index
  standard MLX portability

model-manifest.json
  AX Engine native model contract

mtp.safetensors, mtplx_runtime.json, ax_mtp_sidecar_manifest.json
  AX Engine MTP bundle and provenance

axquant_plan.json, axquant_manifest.json, axquant_runtime.json
  AXQuant allocation, evidence, and runtime recommendations
```

AXQuant invokes AX Engine tooling to generate `model-manifest.json`. It does not duplicate the
AX Engine schema or inject AXQuant-only fields into that manifest.

### Consequences

- Runtime innovation does not create an immediately closed model format.
- Artifact ownership remains clear.
- Runtime-specific metadata can evolve without rewriting portable weights.

## AXQ-005: Index-defined checkpoint boundary with declared root sidecars

**Status:** Accepted

### Context

Model repositories may contain nested auxiliary Safetensors that duplicate names or hold research
artifacts. Recursively loading every Safetensors file produces incorrect parameter counts and
violates the clean-room boundary.

### Decision

When `model.safetensors.index.json` exists, checkpoint membership is:

1. every safe relative Safetensors path in `weight_map`;
2. the root `mtp.safetensors`, when present;
3. the root `vision.safetensors`, when emitted with the AXQuant protected-tensor manifest.

Unindexed nested Safetensors are excluded. An indexed nested shard, such as a separately stored
vision shard, is included. Without an index, only root Safetensors files are included.

Unsafe, missing, non-string, or non-Safetensors references fail the inspection.

### Consequences

- Uniform and mixed checkpoints produce comparable logical counts.
- Auxiliary duplicates do not become model parameters.
- Index integrity becomes a feasibility and publication concern.

## AXQ-006: Language-path scope and vision preservation

**Status:** Accepted

### Context

The initial Qwen3.6-27B configuration includes a vision tower even though the first product is a
language-model optimization.

### Decision

- v0.x optimizes the Qwen 3.6 language path only.
- Vision tensors remain BF16.
- Vision loading remains portable when present.
- AXQuant makes no VLM quality, VLM memory, or multimodal acceleration claim.
- Only supported 2-D language-path matrices are quantization candidates.

### Consequences

- The first release avoids an unbenchmarked VLM claim.
- Three-dimensional convolution weights, norms, and unsupported modules stay protected.

## AXQ-007: Per-tensor 4/6/8/BF16 planning

**Status:** Accepted

### Decision

The initial planner operates at tensor/module granularity and supports:

```text
4-bit affine
6-bit affine
8-bit affine
BF16 protected
```

Minimum protection policy:

| Component | Default minimum |
| --- | ---: |
| Robust 2-D backbone linear | 4-bit |
| Embedding | 8-bit |
| Router | 8-bit |
| Norm | BF16 |
| LM head | BF16 |
| Vision | BF16 |
| Protected integrated MTP | 8-bit |
| External MTP sidecar | Byte-preserved |

Hardware capability may raise these floors. It may never silently lower them.

### Consequences

- The plan is more expressive than layer-only 4/8 assignment.
- Nonstandard bit widths are used only when a supported runtime kernel exists.
- Per-channel arbitrary precision remains deferred.

## AXQ-008: Evidence taxonomy

**Status:** Accepted

### Decision

Evidence kinds are:

- `measured`: generated by an AXQuant forward/benchmark backend;
- `imported`: externally measured with full provenance and explicit attribution;
- `architecture_prior`: heuristic or manual development evidence.

Only `measured` and properly validated `imported` evidence are release-quality. Manual recipes and
architecture priors cannot pass publication gates.

### Consequences

- `--allow-unmeasured` is an explicit development override.
- Reports must label evidence kind.
- Missing calibration provenance invalidates measured or imported sensitivity.

### Rejected alternatives

- Treat architecture heuristics as sensitivity measurements.
- Convert a successful smoke test into a quality claim.
- Publish a manual plan because it loads successfully.

## AXQ-009: Manual plans for v0.1

**Status:** Accepted

### Context

The first vertical slice must exercise conversion before the forward sensitivity backend exists.
Directly editing a plan would bypass safety and provenance.

### Decision

`axquant.manual-recipe.v1` is the only supported manual input. It provides:

- ordered role/tensor/module selectors;
- explicit bits, method, group size, and reason;
- default precision;
- MTP and hardware policy;
- BPW limit and release-threshold metadata;
- unmatched-rule behavior and seed.

The first matching rule wins. Unsafe explicit rules fail. Unmatched rules fail unless explicitly
allowed. Tied weights are harmonized at their highest selected precision.

The resulting `axquant.plan.v1` is labeled `architecture_prior`.

### Consequences

- The v0.1 conversion is reviewable and reproducible.
- Manual planning does not establish the v1 technical claim.
- `convert --allow-unmeasured` is required.

## AXQ-010: MTP as a first-class component

**Status:** Accepted

### Context

Small logit changes can preserve ordinary quality while materially lowering multi-token
acceptance. Copying an MTP sidecar without evaluating it is insufficient for a production claim.

### Decision

- Detect and classify MTP tensors separately.
- Apply an independent precision policy.
- Preserve external sidecars byte-for-byte in v0.1.
- Verify sidecar size and SHA-256 against provenance.
- Compare identical checkpoint weights with MTP disabled and enabled.
- Measure draft-position accuracy, acceptance, rejection, verification overhead, repetition,
  divergence, and effective throughput.
- Use MTP acceptance and speed as hard release gates when the benchmark harness is authoritative.

The backbone produces the first token; MTP metrics cover later draft positions.

### Consequences

- v0.1 proves preservation and loading, not acceleration.
- v0.2 makes MTP performance evidence authoritative.
- Layout-aware external sidecar quantization requires a dedicated future backend.

## AXQ-011: Hard constraints before weighted optimization

**Status:** Accepted

### Decision

The planner rejects candidates that violate:

- supported bits, methods, group sizes, or kernels;
- protected tensor floors;
- effective BPW or model-size limits;
- required MTP retention;
- required quality;
- artifact integrity.

A weighted objective ranks only candidates that satisfy applicable hard constraints.

The proxy objective is:

```text
Wkl * output_kl
+ Wh  * hidden_state_error
+ Wc  * cosine_distance
+ Wt  * token_disagreement
+ Wq  * task_loss_delta
+ Wm  * mtp_acceptance_loss
+ Wr  * peak_memory_cost
+ Wp  * prefill_latency_cost
+ Wd  * decode_latency_cost
```

### Consequences

- A strong weighted score cannot compensate for a release-gate violation.
- Proxy-only quality and speed constraints remain unproven until complete-model validation.

## AXQ-012: Complete-candidate validation

**Status:** Accepted

### Context

Independent tensor probes cannot fully predict cumulative hidden-state distortion or tensor
interactions.

### Decision

Final candidates are selected from complete-model results:

```text
independent sensitivity
  -> initial constrained plans
  -> complete checkpoint conversion
  -> quality and MTP evaluation
  -> precision upgrades or equal-cost swaps
  -> bounded re-evaluation
  -> Pareto selection
```

### Consequences

- The foundation greedy plan is a candidate generator, not a production selector.
- Global refinement records every plan change and termination reason.

## AXQ-013: Actual bytes and actual hardware are authoritative

**Status:** Accepted

### Decision

- Planned BPW models assigned bits and affine metadata.
- Measured BPW uses the actual inspected weight files, including declared root MTP and protected
  sidecars.
- Reports expose both total and main-model BPW.
- Runtime selection uses measured AX Engine load, memory, prefill, decode, MTP, and kernel data.
- A smaller checkpoint that is slower may lose the Pareto comparison.

### Consequences

- The target class `4bit` is not a claim that every tensor is four-bit.
- Nonstandard formats are not adopted solely for theoretical compression.

## AXQ-014: Versioned, fail-closed artifact boundaries

**Status:** Accepted

### Decision

Every file/CLI boundary uses strict Pydantic validation and a versioned schema. Unknown fields are
rejected. Required artifacts include immutable identity, digests, versions, seed, and evidence
kind as applicable.

Malformed or incomplete inputs fail before expensive work. Missing evidence cannot be interpreted
as a passing metric.

### Consequences

- Schema changes require an explicit version decision.
- Resume keys and reports remain auditable.
- Release automation can consume machine-readable status.

## AXQ-015: Atomic conversion and guarded publication

**Status:** Accepted

### Decision

Conversion:

1. validates the plan and model coverage;
2. writes into a same-filesystem sibling staging directory;
3. applies quantization and copies verified MTP artifacts;
4. generates runtime manifests;
5. re-inspects staging and requires exact total, MTP, and protected-vision parameter equivalence;
6. writes an `axquant.artifact.v2` manifest with authoritative byte-derived BPW;
7. renames the staging artifact to the final path only after success;
8. removes the staging directory on failure.

Publication requires:

- release-quality evidence;
- an immutable source revision;
- a passing validation report;
- plan/manifest hash agreement;
- complete file sizes and SHA-256 hashes;
- AX Engine primary-runtime metadata;
- required runtime manifests.

### Consequences

- A failed conversion does not leave a misleading final checkpoint.
- Publication cannot silently upgrade development evidence into a release.

## AXQ-016: Release-time official dense Qwen 3.6 scope

**Status:** Accepted

### Context

The original roadmap required two or three dense Qwen 3.6 checkpoints. The official Qwen 3.6
collection verified on 2026-07-30 contains one dense parameter size, 27B, plus a same-size FP8
representation and the MoE `35B-A3B` model. A fixed two-checkpoint minimum therefore cannot be
satisfied without inventing a non-official size or treating a representation as a distinct model
size.

### Decision

- M5 and v1 certify every distinct official dense Qwen 3.6 parameter size present in the official
  collection at release time.
- The release request records the official catalog URL, a timezone-qualified verification time,
  and the complete enumerated dense-size scope.
- Quantized or FP8 representations of the same parameter size do not add a size. MoE checkpoints
  do not satisfy the dense scope.
- Every required dense model uses one immutable source revision and one artifact/plan identity
  across both `agent-coding` and `general` release validations.
- Each required model must pass AX Engine and MLX-LM runtime checks plus both profile validations.
- Missing catalog entries, undeclared dense entries, inconsistent revisions or artifact
  identities, missing profiles, and scope tampering fail closed.
- A later catalog change invalidates the prior scope verification and requires an updated request,
  evidence matrix, and release audit.

The catalog verified on 2026-07-30 contains one required dense size: `Qwen/Qwen3.6-27B`.

### Migration and supersession

This decision supersedes the fixed checkpoint-count clauses in AXQ-002 and ADR 0002. It is
implemented by `axquant.compatibility-request.v2`, `axquant.compatibility-matrix.v2`,
`axquant.release-audit-request.v4`, and `axquant.release-audit.v4`. Older count-based matrices are
development history and cannot authorize a release.

### Consequences

- The scope is truthful to the upstream family rather than padded with derived or unrelated
  checkpoints.
- M5 currently requires one 27B artifact with both release profiles.
- If Qwen publishes another official dense Qwen 3.6 parameter size before release, that size
  automatically becomes required.
- MoE and arbitrary Qwen generations remain outside the v1 conversion boundary.

## Deferred decisions

The following require later ADR updates:

- AWQ and DWQ plugin interfaces and fallback ordering;
- top-N candidate algorithm and global refinement budget;
- hardware profile registry and tolerance model;
- integrated versus external MTP quantization beyond byte preservation;
- MoE expert allocation;
- VLM calibration and vision precision;
- runtime KV-cache recommendation schema;
- additional bit formats and group/column granularity.

## Supersession policy

A decision change must:

1. identify the superseded AXQ ID;
2. document new evidence or constraints;
3. state migration impact on schemas and artifacts;
4. update the Technical Specification and tests;
5. preserve compatibility or explicitly version the broken contract.

The supporting ADRs under [`decisions/`](decisions/) are explanatory projections. This document
is the internal decision authority.
