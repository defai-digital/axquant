# Tech spec — AXQuant 1.8.0

- **Status:** draft for implementation
- **Date:** 2026-08-14
- **Scope:** R18-01 … R18-32 in [`prd.md`](prd.md)
- **ADRs:** 0007 (layers), 0008 (naming), 0009 (budget)

Conventions: every new serialized contract is a new `schema_version`
(ADR-0001). Flags that change planner defaults stay opt-in until the
1.8.0 tag. Existing completion-program workstreams (act-order, 2/3-bit)
are out of this spec.

```text
B1 spec + schemas  →  B2 verify-cert
D1 interchange freeze
C1 memory accounting  →  C2 axquant optimize
A1 naming + cards
```

B and D may overlap. C depends on D’s interchange types only loosely
(byte accounting already exists). A1 can land after B2 because cards
must emit Spec v1.0 fields.

---

## WS-B1 — Certification Spec v1.0

**Requirements:** R18-01, R18-02, R18-04.

**Modules:** `src/axquant/schema/public_certification.py`,
`schema/registry.py`, `schema_contracts.py`, `schemas/`,
`docs/certification-spec-v1.0.md` (new public spec).

**Design.** Write the public spec first: Tier 1 vs scoped Tier 2,
evidence kinds, host scope, measured-BPW rules, context scope, what
Tier 1 must never imply. Add a **new** schema version for certificates
that require measured main BPW, product class, repo leaf, edition, and
context scope. Keep legacy `axquant.public-checkpoint-certification.v1`
readers. Do not mutate frozen field sets.

**Tests.** Schema snapshot + digest update via
`scripts/render_schema_contracts.py`. A fixture legacy cert still loads.
A Spec v1.0 cert missing measured BPW is rejected.

**Evidence impact.** Additive. Historical certificates stay valid as
legacy; they do not auto-upgrade.

---

## WS-B2 — `axquant verify-cert`

**Requirements:** R18-03.

**Modules:** new `src/axquant/certification/verify.py`;
`certification/dispatch.py`; `cli/_parser.py`; `public_cert_index.py`;
`tests/test_verify_cert.py`.

**Design.** Input: a local artifact directory, or a certificate JSON plus
the bound files it names. Checks:

1. Certificate schema (legacy vs Spec v1.0; report which).
2. Manifest / plan / certificate class-SKU agreement (ADR-0008).
3. File SHA-256 bindings.
4. Recomputed `measured_main_bpw` vs certificate (tolerance: exact
   match on the stored full-precision value).
5. Optional Hub commit / tag binding when those fields are present.
6. Tier 1 must not assert a Tier 2 speed claim.

Exit codes: `0` consistent, `1` failed checks, `2` usage/IO.

Emit JSON + human summary. Fail closed. No network required for the
local-directory path.

**Tests.** Golden consistent bundle; tamper matrix (flipped digest,
class/repo mismatch, BPW rewrite, Tier 2 claimed on a Tier 1-only
record).

---

## WS-D1 — `axq-affine-u32-v1` interchange

**Requirements:** R18-10, R18-11.

**Modules:** new `docs/specs/axq-pack-interchange-v1.md`;
`converter.py`, `predicate.py`, `inspector.py`, `runtime.py`;
optional `src/axquant/schema/interchange.py`; `tests/test_interchange.py`.

**Design.** Document the contract the converter already enforces:
quantized modules are affine-packed U32 with scales/biases; protected
tensors stay BF16/F16; MTP sidecar rules; runtime metadata that MLX-LM
may ignore. Add a conformance helper that walks a pack and rejects
non-affine quantized weights (same spirit as
`converter.py` converted-weight verification).

Dual-runtime test: load the same fixture hashes in the MLX-LM path and,
when the tool is present, AX Engine. Skip AX Engine in CI hosts that
lack it; do not fake a pass.

**Evidence impact.** Documentation + tests. No change to packed bytes.

---

## WS-C1 — Joint memory accounting

**Requirements:** R18-21, R18-22.

**Modules:** new `src/axquant/memory_budget.py`,
`src/axquant/schema/deployment.py` (new schema version);
reuse `planner.py` (`allocate_kv_cache`,
`allocate_kv_cache_measured`), `schema/planning.py`.

**Design.** Pure function:

```text
feasible, breakdown = evaluate_budget(
    weight_bytes, kv_bytes, reserve_bytes, limit_bytes
)
```

`kv_bytes` comes from the existing KV plan at a bound context and batch.
`reserve_bytes` is explicit (CLI default documented; never silent).
Breakdown fields: weights, KV, reserve, limit, remainder, evidence_kind,
context, batch.

Do not fold KV options into the weight upgrade heap.

**Tests.** Infeasible vs feasible fixtures; reserve cannot go negative;
prior-evidence breakdown cannot carry `measured` labels.

---

## WS-C2 — `axquant optimize`

**Requirements:** R18-20, R18-23, R18-24.

**Modules:** new `src/axquant/optimizer.py`; `cli/_parser.py`,
`cli/__init__.py`; `tests/test_optimizer.py`.

**Design.** Orchestrator, not a new solver:

1. Inspect (or accept an inventory).
2. Call existing `plan` with class / BPW / profile / mode / methods.
3. Attach existing KV path when requested.
4. Run WS-C1 accounting.
5. Write a deployment-plan JSON + markdown explanation (floors,
   infeasible reason, selected class, measured vs estimated).

`--mode` must select `objective_for` weights or a documented mode overlay.
Today `plan_quantization` mostly copies `target_mode` through
(`planner.py`); fix that here or in a focused planner patch.

Reuse `--latency-table` behavior from `plan` (ADR-0003).

**Tests.** CLI help; infeasible memory fails before convert; mode overlay
changes recorded objective weights; allow-unmeasured path cannot emit a
certified label.

---

## WS-A1 — Naming guardrails and cards

**Requirements:** R18-30, R18-31, R18-32.

**Modules:** `naming.py`, `claims.py`, `model_card.py`,
`public_cert_index.py`, `README.md` naming paragraphs,
`docs/migration-v1.8.md`, tests in `tests/test_naming.py`,
`tests/test_claims.py`, `tests/test_model_card.py`.

**Design.**

- Hub identity remains `model_name()` class SKUs.
- Card H1 appends rounded measured main BPW.
- Certificate title includes SKU + edition + rounded BPW.
- `build_public_claim` / `render_certified_model_card` compare against
  the **class SKU** repository, not `certified_mixed_precision_name()`.
  Keep the MP helper as a derived `display_claim_label` if flagship
  still wants it.
- Publish-time guard: refuse a new 4-bit sibling that does not beat
  the same-MTP 6-bit complete weight bytes by ≥5%.
- Do not rename any certified repo in this workstream.

**Tests.** Claims accept `AutomatosX/AX-…-MLX-AXQ-4bit-MTP`. Claims
reject an MP-named repository as the public repo. Floor-collapse guard
fires. Card H1 contains `BPW`. Existing sibling-link tests stay green.

---

## Schema impact

| New or changed | Version policy |
| --- | --- |
| Certification Spec document | File at `docs/certification-spec-v1.0.md` |
| Public certificate schema | **New** `schema_version`; legacy v1 untouched |
| Deployment / memory-breakdown schema | **New** `axquant.deployment-plan.v1` (name TBD in implementation) |
| Interchange spec | Document + tests; pack bytes unchanged |
| `axquant.plan.v1` / manifest v2 | Unchanged unless an additive new version is required |

Run `python scripts/render_schema_contracts.py --check` on every PR that
touches schemas.

---

## Implementation sequence and cut line

| Phase | Exit |
| --- | --- |
| B1 + D1 | Specs exist; schemas frozen; interchange tests pass |
| B2 | `verify-cert` green on golden + tamper matrix |
| C1 + C2 | `optimize` fail-closed on budget; mode overlay recorded |
| A1 | Claims/cards match ADR-0008; migration note published |
| Tag | `docs/releases/1.8.0.md`; version bump; no CUDA in notes |

**Cut from 1.8.0:** act-order measurement campaigns, vision-tower
quantization, per-expert precision, full 2.x joint search, CUDA
adapters, fleet recertification, unversioned alias repos.

## PR plan

1. **docs: 1.8.0 planning suite** — this directory (no code).
2. **B1: Certification Spec + new certificate schema** — docs + schema
   snapshots.
3. **D1: interchange spec + conformance tests** — no byte format change.
4. **B2: `verify-cert`** — depends on B1.
5. **C1: memory-budget module + deployment schema** — independent of B2.
6. **C2: `axquant optimize`** — depends on C1.
7. **A1: naming/claims/card alignment** — can overlap C2; must land
   before the 1.8.0 tag.
8. **release: 1.8.0 notes + version bump** — after B2, D1, C2, A1.
