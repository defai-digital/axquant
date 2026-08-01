# AXQuant Architecture Decision Register

**Document status:** Accepted decision set  
**Applies to:** AXQuant v0.x, v1, and the expansion program (v1.1–v2.0)  
**Last reviewed:** 2026-08-01

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
| AXQ-017 | Tier family support as certified / convertible / inspect-only | Accepted |
| AXQ-018 | Drive dense-family adapters from declarative specifications | Accepted |
| AXQ-019 | Ship one-command quick conversion with unweakened evidence labeling | Accepted |
| AXQ-020 | Publish checksummed recipe bundles that bind measured planning to user conversions | Accepted |
| AXQ-021 | Plan KV-cache precision per layer, prior-based first and measured later | Accepted |
| AXQ-022 | Compete with mlx-optiq by measured merit under clean-room guardrails | Accepted |
| AXQ-023 | Resolve remote recipe bundles only from revision-pinned Hub references | Accepted |
| AXQ-024 | Measure per-layer KV-cache sensitivity and bind measured KV plans to their report | Accepted |
| AXQ-025 | Gate measured KV plans at release by packaged report and deterministic reallocation | Accepted |
| AXQ-026 | Resolve the size gate through a governed 8-bit LM-head floor | Accepted |
| AXQ-027 | Document independent algorithm derivation sources for clean-room provenance | Accepted |

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

## AXQ-017: Tiered family support

**Status:** Accepted

### Context

AXQ-002 deliberately narrowed v0.x/v1 to Qwen 3.6, and the release gates make every supported
family expensive. Competing catalogs win adoption through breadth. Coupling "the adapter can
convert this family" to "this family passed M0–M8" makes breadth impossible; decoupling them
without labels would make breadth dishonest.

### Decision

Family support is recorded as one of three tiers, carried in the architecture profile, the
inventory, and every downstream manifest:

- **inspect-only** — inventory and tensor classification; `convert` and `quantize` refuse;
- **convertible** — conversion permitted; every artifact is development evidence until the family
  is certified; official-catalog publication refuses;
- **certified** — at least one size of the family passed the full M0–M8 audit; release claims
  follow the existing gates.

Tier promotion is evidence-bound and monotonic per release: `convertible` requires adapter unit
tests plus one real-checkpoint conversion smoke with full coverage and integrity checks;
`certified` requires the existing release audit. Public wording must always include the tier.

### Consequences

- Breadth can grow at adapter speed while certification grows at evidence speed.
- The support matrix becomes a first-class, truthful artifact instead of a README table.
- `publish` gains a tier gate in addition to its existing evidence gates.

### Rejected alternatives

- Certifying every family before exposing it (kills breadth; the v1 status quo).
- A single "supported" flag (dishonest breadth; indistinguishable claims).

## AXQ-018: Declarative dense-family adapter specifications

**Status:** Accepted

### Context

`Qwen36Adapter` hand-codes matching, profiling, and tensor-role classification. Most dense
transformer families differ only in config `model_type` values, reference naming, layer-count
location, and a small set of tensor-name conventions. Hand-writing each adapter repeats
protection-critical classification logic and multiplies review cost per family.

### Decision

- A declarative `DenseFamilySpec` (family id, product family, accepted `model_type` values,
  reference patterns, config extraction rules, default support tier, extra classification
  patterns, notes) drives a shared `DenseFamilyAdapter` implementing the existing
  `ArchitectureAdapter` protocol.
- Shared classification remains fail-closed: a tensor no rule classifies stays unclassified and
  blocks conversion, exactly as today.
- Families with bespoke needs (MTP layouts, unusual sidecars, non-dense paths) implement the
  protocol directly; Qwen 3.6 keeps its dedicated adapter.
- The registry resolves adapters in declared order; two adapters matching the same checkpoint is
  an error, not a silent pick.
- New family specs start at `inspect-only` unless their promotion evidence exists.

### Consequences

- A new dense family is data plus tests, typically without new classification code.
- Protection semantics are reviewed once, in the shared classifier.
- The registry stays deterministic and regression-testable as it grows.

### Rejected alternatives

- One hand-written adapter per family (per-family review of protection logic; drift risk).
- Config-file-only adapters loaded at runtime (schema and supply-chain surface without need).

## AXQ-019: One-command quick conversion with unweakened evidence labeling

**Status:** Accepted

### Context

The release pipeline UX is today the only UX, and it is producer-grade: many commands, many
artifacts. Self-converters compare AXQuant to mlx-optiq's single command and leave. The evidence
taxonomy (AXQ-008) must not be weakened to fix this.

### Decision

- `axquant quantize` performs inspect → plan → convert (→ optional runtime smoke) in one
  command with working defaults.
- Quick mode may only run at tier `convertible` or `certified`.
- Quick-mode planning without a measured recipe bundle uses architecture priors or a reviewed
  family default recipe; the resulting plan and artifact carry development evidence kinds, and
  the CLI summary states this in its final output.
- Quick mode cannot publish, cannot emit release claims, and cannot satisfy any M-gate; it is a
  front end over the existing stage implementations, not a parallel pipeline.
- The development-only spelling `--allow-unmeasured` remains for the staged commands; quick mode
  subsumes its role for end users with explicit labeling instead of an unlock flag.

### Consequences

- End-user effort reaches parity with mlx-optiq without any new claim surface.
- The staged pipeline remains the only path to release evidence.
- Documentation gains a clean split: "convert your model" (quick) vs "release a model" (gated).

### Rejected alternatives

- Lowering release gates to make the full pipeline shorter (destroys the differentiator).
- A separate "lite" tool (forks behavior and the evidence taxonomy).

## AXQ-020: Published measured recipe bundles

**Status:** Accepted

### Context

Measured planning is AXQuant's quality edge but is expensive to produce. Users converting their
own copy of a base model would each re-pay the measurement cost, so in practice they would fall
back to priors — losing the edge exactly where adoption happens.

### Decision

- A **recipe bundle** is a versioned, checksummed artifact binding: source model identity
  (model id, revision), a plan or manual recipe, its evidence kind, lineage digests of the
  producing sensitivity/calibration artifacts, and the producing AXQuant version.
- Bundles exported from measured release evidence retain the measured evidence kind; bundles
  from priors are labeled as prior-derived. A bundle never upgrades the evidence kind of its
  inputs.
- `axquant quantize --recipe BUNDLE` verifies bundle checksums and model identity (id and
  revision) before use and records bundle lineage in the artifact manifest; identity mismatch
  fails closed.
- Bundles are published alongside certified models; resolution is local-path first, with remote
  (Hugging Face) resolution as a later, integrity-checked phase.

### Consequences

- One measurement pays for every user conversion of that base model — a structural advantage no
  breadth-only competitor has.
- The evidence chain extends to user machines without trusting user claims.
- Publication tooling gains one more artifact type with existing checksum discipline.

### Rejected alternatives

- Shipping measured plans inside the wheel (stale immediately; wrong granularity).
- Unverified recipe URLs (supply-chain surface; unattributable evidence).

## AXQ-021: Per-layer KV-cache precision planning

**Status:** Accepted

### Context

On unified memory, long-context serving cost is dominated by the KV cache, which weight-only
planning ignores. v1 listed runtime KV-cache quantization as a non-goal; the competing catalog
already allocates KV precision per layer. AX Engine is AXQuant's primary runtime and can consume
richer KV metadata than stock MLX-LM exposes.

### Decision

- The quantization plan gains an optional per-layer KV-cache precision section (bits and group
  size per layer, with policy floors and a recorded allocation basis).
- Phase one allocates KV precision from architecture priors and is labeled prior-based;
  measured KV sensitivity probing is a scheduled follow-up, and no measured-quality KV claim is
  permitted until it exists.
- Converted artifacts emit KV-cache metadata in the AX Engine runtime metadata; MLX-LM
  compatibility guidance (uniform fallback settings derived from the plan) is advisory metadata
  only.
- A plan without the KV section behaves exactly as today; the section is additive and optional.

### Consequences

- Deployment-cost planning covers the real memory bill, not just weight bytes.
- AX Engine gains a runtime feature surface competitors on stock runtimes cannot match.
- The evidence taxonomy extends naturally: prior-based KV now, measured KV later.

### Rejected alternatives

- Runtime-only uniform KV flags (ignores per-layer sensitivity; no planning value).
- Waiting for measured KV probes before any schema work (blocks the metadata contract AX Engine
  needs first).

## AXQ-022: Compete with mlx-optiq by measured merit

**Status:** Accepted

### Context

The expansion program's explicit goal is to win the audience mlx-optiq serves today. That goal
creates pressure to imitate, to over-claim, or to describe AXQuant as a successor — each of which
AXQ-001 already forbids in part; this decision makes the competitive posture explicit.

### Decision

- AXQuant competes on: certified evidence chains, MTP-aware conversion, KV-cache planning,
  reproduction recipes, and measured head-to-head results — never on imitation or naming.
- Every certified model publication includes the mandatory comparison set (BF16, uniform-4,
  uniform-6, AXQuant candidate) and, where an attributed OptiQ checkpoint of the same base
  exists, that checkpoint as an external baseline under AXQ-001's standard-load-contract rules.
- AXQuant never claims to be a successor, replacement, or affiliate of mlx-optiq; public wording
  is "an independent alternative with published evidence".
- Head-to-head publications state the exact revisions, hosts, and suites; a comparison AXQuant
  loses is published with the same prominence as one it wins.

### Consequences

- The replacement ambition becomes a measurable program (catalog coverage plus published
  comparisons) instead of a marketing posture.
- Publishing unfavorable results preserves the credibility that is AXQuant's actual moat.

### Rejected alternatives

- Silent benchmarking (unfalsifiable claims; reputational risk exceeds any upside).
- Successor positioning (forbidden by AXQ-001 and legally fraught).

## AXQ-023: Revision-pinned remote recipe-bundle resolution

**Status:** Accepted

### Context

AXQ-020 shipped local recipe-bundle resolution and deferred the remote trust model. Bundles are
most valuable when users can reference the published bundle next to a Hugging Face model, but a
mutable remote reference (a branch name, `main`, or an unpinned repo) would let the resolved
plan change after review, defeating the checksum discipline the bundle exists to provide.

### Decision

- `quantize --recipe` additionally accepts `hf://OWNER/REPO@REVISION[/PATH]`.
- The revision pin is mandatory: an `hf://` reference without `@REVISION` fails closed with an
  actionable message. The revision should be an immutable commit SHA; AXQuant records exactly
  what was requested.
- `PATH` defaults to the standard bundle location (`axquant_recipe_bundle.json` at the repo
  root); when given, it names the bundle record file inside the repo.
- Resolution downloads the bundle record and its payload through the public `huggingface_hub`
  download API at the pinned revision, then applies the **unchanged** local verification chain:
  strict schema parse, payload checksum, source-model identity and revision match, and
  evidence-kind consistency. Remote origin grants no trust; it only supplies bytes.
- No credentials are ever written to logs or manifests (existing policy). Download failures,
  missing files, and checksum mismatches surface as artifact errors naming the reference.

### Consequences

- Published bundles become directly consumable
  (`--recipe hf://AutomatosX/AX-...@<commit-sha>`), completing the AXQ-020 supply chain.
- A tampered or force-pushed repo cannot silently change a conversion: the pin plus payload
  checksum makes the resolved plan reproducible or loudly broken.
- Mirroring or vendoring bundles stays possible because local resolution is unchanged.

### Rejected alternatives

- Accepting unpinned references with a warning (mutable supply chain; warnings do not survive
  automation).
- A bespoke bundle registry service (new infrastructure and trust root without added integrity;
  the Hub plus checksums already provide content addressing).

## AXQ-024: Measured per-layer KV-cache sensitivity

**Status:** Accepted

### Context

AXQ-021 shipped prior-based KV-cache allocation and reserved the `measured` basis until a
measurement path existed. Without measurement, KV planning cannot claim quality, and conversion
rejects measured KV plans outright. The weight path already has the required machinery: verified
tokenized calibration caches, a backend protocol with an MLX implementation, deterministic
metrics, and evidence-kind discipline.

### Decision

- A KV probe backend runs forward passes with per-layer KV-cache quantization (selected layers
  at candidate bits, all others BF16) over the same verified tokenized calibration caches the
  weight probe uses, and compares logits against the all-BF16 baseline (output KL and token
  disagreement at fixed metric positions).
- Results are recorded as `axquant.kv-sensitivity.v1`: model identity, inventory digest, probe
  backend, group size, complete per-layer candidate coverage (`0..text_layer_count-1`, no gaps),
  and calibration provenance. Measured reports without calibration provenance are invalid.
- `allocate_kv_cache_measured` selects, per layer, the lowest candidate bit-width whose measured
  output KL stays within an explicit budget (falling back to BF16), floors included, and emits a
  `KvCachePlan` with `allocation_basis="measured"` bound to the report by its semantic digest
  (`sensitivity_sha256`).
- Conversion accepts a measured KV plan only when that digest is present; a measured basis
  without its bound report digest fails closed. Prior-based plans are unchanged.
- Like the weight path, probe output is development evidence by default. Release-grade KV
  quality claims and their validation gates remain a deferred decision; nothing in this decision
  permits a KV quality claim in release wording.

### Consequences

- KV planning gains a real measurement path with the same calibration and evidence discipline
  as weight planning, closing the last functional gap against per-layer KV allocation in
  competing tooling.
- The digest binding makes a measured KV plan reproducible or loudly broken, exactly like
  measured weight plans.
- Release gating for KV claims can be designed later without schema breakage.

### Rejected alternatives

- Reusing the weight `SensitivityReport` for KV entries (conflates two different measurement
  protocols under one evidence object).
- Accepting `measured` basis without a bound report (unverifiable claims).

## AXQ-025: Release gating for measured KV-cache plans

**Status:** Accepted

### Context

AXQ-024 deferred release gating for KV quality claims. Without a gate, a release artifact could
carry a measured KV plan whose digest points at a report nobody packaged, or whose per-layer
choices silently diverge from what the report supports. The repository's existing discipline for
exactly this problem is: package the evidence with the artifact, verify it by digest, and prove
the derived result can be recomputed from the packaged evidence.

### Decision

- `allocate_kv_cache_measured` records its selection budget on the plan
  (`KvCachePlan.max_output_kl`), so a measured KV plan is a pure function of
  (report, budget, floors).
- Conversion of a plan whose KV basis is `measured` requires the producing
  `KvSensitivityReport`: `convert` (and `quantize`) accept `--kv-sensitivity`; the converter
  verifies the report's semantic digest equals `kv_cache.sensitivity_sha256` and packages it as
  `kv_sensitivity.json` in the artifact. A measured KV plan without the report fails closed.
- Publication preparation re-verifies the chain: the packaged `kv_sensitivity.json` must parse,
  its digest must equal the plan binding, and re-running the allocator with the recorded budget
  and floors must reproduce the packaged plan's per-layer allocation exactly.
- Prior-based KV plans package nothing and are unaffected.
- Quality claims about KV-quantized serving still require ordinary evaluation evidence through
  the existing dual-profile validation; this gate proves plan provenance and reproducibility, it
  does not by itself authorize a KV quality claim in release wording.

### Consequences

- A measured KV plan in a published artifact is reproducible from packaged evidence or loudly
  broken — identical discipline to `quantization_plan.json` and the calibration manifest.
- The deferred-decision list for KV planning is now empty; future work is evidence collection,
  not contract design.

### Rejected alternatives

- Gating only on digest presence (proves binding, not that the allocation follows the report).
- Blocking release whenever a KV section exists (would punish the prior-based path that AXQ-021
  explicitly allows).

## AXQ-026: Resolve the size gate through a governed 8-bit LM-head floor

**Status:** Accepted (named approval by the workspace owner, 2026-08-01)

### Context

The measured policy floor for Qwen3.6-27B is 5.5770 BPW — 114.57% of the audited uniform-4
reference, 772,659,246 bytes over the 110% size gate. The size-decision analysis
(`.internal/tmp/qwen36-v1-size-decision-analysis.md`) quantified four resolutions; lowering the
LM-head weight floor from BF16 to 8-bit/group-64 saves a modeled 1,191,936,000 bytes and fits the
gate with 419,276,754 bytes of margin — the only single-change option with meaningful headroom.
8-bit output heads are established practice in production quantization stacks, so the expected
quality risk is the lowest of the four options.

### Decision

- The LM-head protection floor may be lowered from BF16 to 8-bit **per plan**, never globally:
  `PlanRequest.lm_head_min_bits` / `ManualPlanRecipe.lm_head_min_bits` (default 16) must be set
  to 8 explicitly (`plan --lm-head-floor 8bit`), and the plan records the deviation in
  `constraints.lm_head_min_bits` with an AXQ-026 allocation reason.
- The measurement probe measures the LM head down to 8-bit so the lowered floor is backed by a
  per-tensor measurement instead of being unmeasurable by construction.
- The release audit widens the LM head's required measured-candidate set to the plan's recorded
  floor: an 8-bit LM-head plan needs measured 8-bit and BF16 LM-head sensitivity or it fails.
- Release certification is unchanged: dual-profile quality, MTP, size, and hardware gates must
  pass on the complete candidate. This decision authorizes the search direction, not a claim.

### Consequences

- A ~5.23-BPW-class candidate satisfying the 110% size gate (modeled ratio ~107.5%) becomes
  plannable from measured evidence; the next probe run must add the LM-head 8-bit measurement
  before such a plan audits.
- Default behavior is byte-identical to AXQ-007 floors; old plans deserialize with floor 16.

### Rejected alternatives

- A permanent ≥114.57% size exception (weakest competitive posture; exception machinery remains
  available if measured LM-head quality evidence fails).
- 8-bit vision + external-MTP sidecar backend (57 MB margin only, touches the fragile MTP
  exactness path, and requires a new validated backend).
- Language-only artifact scope (the ratio worsens to ~115.41% under a consistent denominator).

## Deferred decisions

The following require later ADR updates:

- AWQ and DWQ plugin interfaces and fallback ordering;
- top-N candidate algorithm and global refinement budget;
- hardware profile registry and tolerance model;
- integrated versus external MTP quantization beyond byte preservation;
- MoE expert allocation;
- VLM calibration and vision precision;
- LoRA rank guidance derived from sensitivity artifacts;
- additional bit formats and group/column granularity.

## AXQ-027: Document independent algorithm derivation sources for clean-room provenance

**Status:** Accepted (2026-08-01)

Every core algorithm in the toolkit traces to published academic literature rather than any
competitor's implementation. ADR 0006 records the specific sources:

- Greedy marginal-efficiency allocation: fractional knapsack (Cormen et al.), rate-distortion
  optimization (Shannon 1948, Cover & Thomas).
- KL divergence sensitivity: Kullback & Leibler 1951; standard in PTQ (Dettmers 2022, Frantar 2022).
- Isolated-module probing: layer-wise reconstruction (Li et al. 2016), per-layer sensitivity
  (Dong et al. 2019).
- Fake-quant KV probing: quantize-dequantize round-trip (Jacob et al. 2018).
- Cosine distance and token disagreement: standard distributional metrics.
- Atomic staging conversion: POSIX rename(2) semantics.

Future algorithms must record their derivation source in ADR 0006 or a supplementary ADR.

## Supersession policy

A decision change must:

1. identify the superseded AXQ ID;
2. document new evidence or constraints;
3. state migration impact on schemas and artifacts;
4. update the Technical Specification and tests;
5. preserve compatibility or explicitly version the broken contract.

The supporting ADRs under [`decisions/`](decisions/) are explanatory projections. This document
is the internal decision authority.
