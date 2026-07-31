# ADR 0005: Release-Time Official Dense Qwen 3.6 Scope

**Status:** Accepted  
**Date:** 2026-07-30  
**Supersedes:** The fixed checkpoint-count clauses in ADR 0002 and AXQ-002

## Context

The original M5 and v1 roadmap required at least two dense Qwen 3.6 checkpoints. The official
Qwen 3.6 collection verified on 2026-07-30 contains:

- `Qwen/Qwen3.6-27B`, a dense model;
- `Qwen/Qwen3.6-27B-FP8`, a representation of the same 27B parameter size;
- `Qwen/Qwen3.6-35B-A3B` and its FP8 representation, whose config declares the MoE architecture
  `qwen3_5_moe`.

There is therefore only one official dense parameter size to certify. Requiring a second dense
size would force AXQuant to substitute an unrelated generation, count an FP8 representation as a
new model size, or create an unofficial derivative. None would prove the stated Qwen 3.6 family
claim.

## Decision

AXQuant v1 certifies every distinct official dense Qwen 3.6 parameter size present in the official
collection at release time.

The compatibility request must:

1. identify the official collection URL;
2. record a timezone-qualified catalog verification time;
3. enumerate the complete official dense model/parameter-size scope;
4. require both `agent-coding` and `general` profiles;
5. bind each profile to one immutable source revision and one candidate artifact/plan identity.

The compatibility matrix fails closed when a required model or profile is missing, an undeclared
dense model is supplied, revisions or artifact identities differ across profiles, runtime or
validation evidence fails, or the catalog scope changes between request and final audit.

Quantized and FP8 variants of one parameter size do not expand the scope. MoE checkpoints do not
satisfy a dense requirement. A later official dense release automatically expands the v1 scope
and requires new compatibility evidence before publication.

At the 2026-07-30 verification point, the complete required dense scope is
`Qwen/Qwen3.6-27B`.

## Schema impact

The accepted contract is encoded by:

- `axquant.compatibility-request.v2`;
- `axquant.compatibility-matrix.v2`;
- `axquant.release-audit-request.v4`;
- `axquant.release-audit.v4`.

Older fixed-count requests and matrices cannot authorize v1 publication.

## Consequences

- M5 can be completed truthfully with one official dense size while the catalog contains only
  27B.
- Both release profiles remain mandatory; reducing the number of sizes does not reduce evidence
  depth.
- The release audit binds the original catalog declaration and rejects scope tampering.
- Future official dense Qwen 3.6 sizes become mandatory without another product-scope exception.
- MoE, VLM optimization, and unrelated Qwen generations remain deferred.

## References

- Official Qwen 3.6 collection:
  <https://huggingface.co/collections/Qwen/qwen36>
- Dense 27B config:
  <https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json>
- MoE 35B-A3B config:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json>
