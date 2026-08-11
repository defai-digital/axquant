# ADR-0001 — Certification freeze discipline governs all roadmap work

- **Status:** accepted
- **Date:** 2026-08-06

## Context

The Qwen 3.6 flagship campaign (`qwen36-mtp-v2`) freezes one candidate, one
released toolkit wheel, matched baselines, disjoint dataset roles, and the
`df-macbookpro-m5` host scope. Meanwhile this roadmap adds algorithms (act-order GPTQ,
latency-aware planning) that would change planner and quantizer behavior. If
those changes reach a frozen campaign's toolchain, the campaign's evidence
becomes unreproducible or, worse, silently tuned.

## Decision

1. A frozen or formal campaign always runs the exact wheel digest recorded at
   freeze. Roadmap code merges to `main` at any time, but behind opt-in flags,
   and never alters default planner/quantizer output for inputs that existing
   evidence covers.
2. Every new algorithm output is a **new candidate label** in sensitivity and
   plan schemas (e.g. a distinct method label for act-order GPTQ), never a
   changed meaning for an existing label. Old reports stay interpretable
   forever.
3. Evidence schemas are versioned additively **only by introducing a new
   `schema_version`**. Under an existing version string, serialized field names,
   types, requiredness, defaults that affect validation, bounds, enum/literal
   membership, discriminators, and unknown-field policy (`extra=forbid`) are
   **immutable**. Because readers forbid unknown fields, even an optional new
   field requires a new version. Repository gates:
   `python scripts/render_schema_contracts.py --check` and
   `schemas/manifest.json` digests (see [schema governance](../../schema-governance.md)).
4. Public certificate records
   (`axquant.public-checkpoint-certification.v1`,
   `axquant.public-mtp-acceleration-certification.v1`) are freeze-class
   `public-certification` and load through StrictModels before documentation
   matrices are generated.
5. Phase 0 (certification closure) takes priority over all other phases. No
   other phase may modify modules on the campaign's critical path
   (`campaign.py`, `claims.py`, `lifecycle.py`, release-audit paths) except
   for bug fixes with independent review.

## Consequences

- Quality/speed improvements only become claimable through a *new* campaign or
  evidence run; nothing retroactively upgrades published packs.
- Some duplication (parallel method labels) is accepted as the cost of
  immutable evidence.
- Failed experiments cannot poison the certification chain: they exist as
  never-selected candidate labels.
- Same-version schema drift fails CI; authors must bump `schema_version` and
  keep prior snapshots loadable.
