# ADR 0009: Separate Non-MTP Certification for Qwen3-Next

**Status:** Accepted
**Date:** 2026-08-03
**Accepted decisions:** AXQ-032, AXQ-033, AXQ-034
**PRD:** `.internal/product/qwen3-next-certification-prd.md`
**Tech spec:** `.internal/engineering/qwen3-next-certification-technical-specification.md`
**Implementation plan:** `.internal/engineering/qwen3-next-certification-implementation-plan.md`

## Context

AXQuant's current release-audit contract is not a generic checklist whose MTP rows can be marked
“not applicable.” M0–M8 was designed as the Qwen 3.6 vertical slice: AX Engine is primary, MTP is
a first-class component, MTP-off/MTP-on pairs prove correctness and speed, the planner is
MTP-aware, and the dense Qwen family proof is part of the release. The contract deliberately
fails if an artifact does not contain declared MTP weights.

Qwen3-Coder-Next is structurally different. The immutable official source used in the current
development cycle is an 80B-total/3B-active hybrid MoE with alternating linear/full attention,
fused experts, and no declared MTP. AXQuant measured every inventory tensor and produced real
4.8- and 6.0-BPW candidates. Both candidates load in MLX-LM. The 4-bit candidate also runs through
AX Engine's current generation benchmark. None of those facts changes the source capability:
there is no MTP sidecar, integrated MTP head, or MTP behavior to certify.

Three tempting shortcuts are therefore invalid:

1. add an empty or fabricated MTP artifact so M1 sees `mtp_present=true`;
2. make MTP fields optional in the existing release-audit schema and treat missing evidence as
   “not applicable”;
3. use a release exception to waive the MTP gates.

All three would weaken a validated certification track, make an applicability decision from user
input rather than source facts, and allow future MTP-capable models to bypass MTP evidence. A new
track is required if non-MTP models are to become certified honestly.

The current development cycle also exposed two scope issues that the new track must address:

- `qwen3-next-v1` is a family adapter, but one exact checkpoint cannot prove every future
  Qwen3-Next architecture/revision.
- 4-bit and 6-bit have different runtime readiness. The 4-bit native manifest validates; the
  6-bit plan mixes a BF16 Q projection with packed K/V/O projections and exposes an AX Engine
  shape-validation limitation. Treating both as one release unit would either delay the viable
  candidate or hide the failing one.

## Decision summary

| ID | Decision |
| --- | --- |
| **AXQ-032** | Add separately versioned, capability-derived certification tracks. Preserve Qwen 3.6 M0–M8 unchanged; certify Qwen3-Next non-MTP direct decode through a new N0–N8 contract. |
| **AXQ-033** | Certify Qwen3-Next at exact-checkpoint and exact-artifact scope first. One passing checkpoint does not promote the whole adapter family. |
| **AXQ-034** | Require dual-runtime, matched-baseline, executable-coding evidence for non-MTP certification, and audit 4-bit and 6-bit independently with 4-bit first. |

## Decision 1: Isolate certification tracks (AXQ-032)

### Chosen design

Introduce an explicit certification-track discriminator with at least these policies:

```text
qwen36-mtp-v1          existing axquant.release-audit-request.v4 → M0–M8
qwen3-next-direct-v1   new strict request/output schemas          → N0–N8
```

Existing v4 request and audit schemas, gate ids, algorithms, and publisher behavior remain
unchanged. The new request is not a collection of optional MTP fields added to v4; it is a strict
schema containing the evidence required for direct-decode certification.

Track eligibility is computed from immutable source configuration plus inspected inventory:

- `qwen36-mtp-v1` continues to require declared and packaged MTP;
- `qwen3-next-direct-v1` requires `model_type=qwen3_next`, the exact approved architecture
  fingerprint, and `mtp_declared=false` with no MTP tensors or root sidecar;
- a user flag cannot override either result;
- a release exception cannot waive an eligibility mismatch.

The `release-audit` CLI dispatches by request `schema_version`. The publisher loads the matching
request/audit pair, reruns the matching builder, and authorizes upload only when the exact track's
complete gate set passes.

### Why N0–N8 is equally strict

N0–N8 retains immutable source proof, artifact completeness, measured planning, quality,
refinement, hardware registry, Pareto reconstruction, reproducibility, stable wheel inspection,
and publisher re-audit. It replaces only MTP-specific interactions with evidence applicable to a
direct-decode model:

- matched BF16, uniform-4, uniform-6, and candidate controls;
- AX Engine/MLX-LM deterministic greedy parity;
- zero-fallback direct-decode runtime trials;
- executable coding correctness and stronger dataset coverage;
- explicit proof that MTP is absent by source design rather than missing accidentally.

The replacement is a different proof, not fewer checks.

## Decision 2: Exact-checkpoint certification before family promotion (AXQ-033)

Certification initially binds:

```text
source model id
+ immutable source revision
+ architecture fingerprint
+ tokenizer digest
+ plan digest
+ artifact manifest digest
+ artifact file digests
+ certification policy id/version
+ hardware registry scope
```

The result is entered into an exact-checkpoint certification registry. The
`qwen3-next-v1` adapter remains `convertible`; `support-matrix` renders certified checkpoint
entries separately from family tier.

Family promotion requires a later decision and representative official checkpoint coverage. At
minimum it must prove that the adapter's supported architecture variants, layer layouts, expert
packing, and runtime paths are not inferred from one 80B checkpoint. A new upstream source
revision does not inherit the exact-checkpoint audit.

This decision is a scoped refinement of AXQ-017's earlier “first family audit promotes the
adapter” expectation. For heterogeneous MoE/hybrid families, promotion follows representative
coverage, not the first exact artifact. Dense Qwen 3.6 policy is unchanged.

## Decision 3: Dual-runtime direct-decode proof and independent candidates (AXQ-034)

The non-MTP certification track preserves AXQ-003:

- AX Engine remains the primary runtime and must pass native-manifest validation, doctor,
  deterministic generation, and benchmark gates with zero fallbacks;
- MLX-LM remains the compatibility runtime and must pass portable generation smoke;
- MLX-only success is sufficient for a development artifact, not a certified artifact.

Every candidate is audited independently. The 4-bit candidate proceeds first because current
development evidence shows:

- atomic conversion and measured 4.80002 BPW;
- a valid AX Engine native manifest;
- successful MLX-LM smoke;
- three successful AX Engine development benchmark trials.

The 6-bit candidate remains a separate development candidate until AX Engine independently
interprets every mixed Q/K/V/O tensor's raw or packed layout and produces successful trials. Its
failure cannot be hidden by 4-bit evidence, and it cannot block an otherwise passing 4-bit audit.

Both candidates require a new formal evidence cycle after the certification policy, suite,
runtime, and toolkit version are frozen. The 2026-08-03 outputs are discovery/feasibility evidence
and regression fixtures; they are not silently repackaged as the final release lineage.

## Gate correspondence

| Existing MTP track | New direct track | Equivalent rigor |
| --- | --- | --- |
| M0 Technical feasibility | N0 Immutable technical feasibility | Complete source, immutable identity, capability truth, required baselines |
| M1 AX Engine vertical slice/MTP artifact | N1 Artifact integrity and dual-runtime vertical slice | Complete artifact, exact plan binding, MLX + AX pass, explicit non-MTP consistency |
| M2 Matched MTP-off/on benchmark | N2 Matched direct-decode benchmark | BF16/uniform/candidate controls, deterministic parity, repeatable trials, zero fallback |
| M3 Measured MTP-aware planner | N3 Measured mixed-precision planner | Complete measured sensitivity/capture/calibration and deterministic allocation |
| M4 Quality/refinement | N4 Coding/general quality | Strong executable suite, general holdout, learned-method proof |
| M5 Dense Qwen family proof | N5 Exact Qwen3-Next architecture proof | Hybrid attention + fused MoE coverage at exact-checkpoint scope |
| M6 Interaction optimization | N6 Complete candidate optimization | Rebuilt complete objective and measured parent/control improvement |
| M7 Hardware Pareto | N7 Hardware Pareto/reproduction | Hardware registry, frontier reconstruction, independent reproduction |
| M8 v1 release | N8 Track-specific release package | Stable wheel, request/audit package, exact claims, publisher re-audit |

## Rejected alternatives

### Make MTP optional in M0–M8

Rejected because optional fields conflate “model has no MTP” with “operator forgot MTP evidence.”
It would change the meaning of already accepted v4 audits and create a silent bypass for future
MTP-capable models.

### Fabricate an inert MTP sidecar

Rejected because it falsifies model membership, parameter counts, runtime capability, and
performance claims. It also violates AXQ-005 and AXQ-010.

### Use a release exception for MTP

Rejected because release exceptions are not applicability selectors. A required component cannot
be waived into “absent by architecture”; capability must be proven from source facts.

### Certify through MLX-LM only

Rejected because AXQ-003 names AX Engine as the primary runtime. This would be a product/runtime
tier change broader than Qwen3-Next and would reduce the release bar.

### Promote the whole family after one passing artifact

Rejected because Qwen3-Next is a hybrid MoE family whose future checkpoints may vary in expert
packing, attention cadence, dimensions, and protected components. Exact-checkpoint certification
is truthful and sufficient for the initial public artifact.

### Wait for 6-bit before certifying 4-bit

Rejected because candidate artifacts are independent release units. Coupling them increases risk
without adding evidence to either candidate.

### Publish now as certified because perplexity improved

Rejected because the small development suite has 15 tasks, 410 evaluated tokens, and zero syntax
validity for BF16 and both candidates. It is useful smoke evidence, not a release-quality coding
proof.

## Consequences

### Positive

- Qwen3-Coder-Next gets a truthful path to certification without weakening Qwen 3.6.
- Capability applicability becomes explicit and auditable.
- Exact-checkpoint claims prevent accidental family-wide overstatement.
- 4-bit can ship when ready without hiding or waiting for 6-bit.
- The stronger coding suite improves evidence for future coding models, not only this checkpoint.

### Costs

- New strict schemas, audit code, publisher dispatch, tests, and documentation are required.
- AX Engine needs a cross-repository mixed-projection fix and formal-host toolchain readiness.
- A final 80B evidence cycle must be rerun after policy/toolkit freeze.
- Exact-checkpoint certification creates more registry entries than coarse family promotion.

### Risks

- Shared audit helpers could accidentally change M0–M8 behavior. Golden v4 audit fixtures and
  byte/semantic regression tests are mandatory before refactoring.
- A new track could become a template for weaker future tracks. Every new track requires an ADR,
  strict source-derived eligibility, an explicit gate-correspondence table, and publisher tests.
- Coding-suite data could leak into calibration. Content hashes and normalized near-duplicate
  checks across coding, general holdout, and calibration are part of N3/N4, not operator
  convention.
- Executable scoring could mistake harness/toolchain failures for model failures. Per-task
  sandbox, resource, toolchain, stdout/stderr, and infrastructure-error evidence is mandatory;
  infrastructure failure invalidates the run rather than lowering the candidate score.

## Migration and compatibility

- Existing `axquant.release-audit-request.v4` and `axquant.release-audit.v4` remain readable and
  authoritative; no schema literal or field changes.
- Existing Qwen 3.6 model cards, publication bundles, and publisher reruns remain on M0–M8.
- New Qwen3-Next request/audit schemas are additive and cannot be consumed by older AXQuant
  versions; this forward-only behavior matches current strict-artifact policy.
- The exact-checkpoint certification registry is additive. An absent registry means no exact
  checkpoint has been certified.
- Current Qwen3-Coder-Next artifacts remain development evidence and retain their historical
  manifests; they are not mutated or relabeled.

## Supersession and interaction

- **Does not supersede** AXQ-003, AXQ-008, AXQ-010, AXQ-014, AXQ-015, or the Qwen 3.6 M0–M8
  contract.
- **Refines** AXQ-017 for heterogeneous families: exact-checkpoint certification may precede
  family-tier promotion.
- **Applies** AXQ-030 and AXQ-031 to the final formal cycle: new cycle, durable evidence root.
- **Extends** AXQ-022's measured-merit strategy to a coding-focused non-MTP model.

## Approval condition

This ADR accepts the direction and decisions AXQ-032–AXQ-034. Numeric thresholds are frozen in
the accepted PRD. Implementation may not begin by weakening existing M0–M8;
the first code phase must establish regression tests proving the old track is unchanged.
