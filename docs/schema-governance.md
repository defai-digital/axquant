# Schema governance

AXQuant treats versioned artifact models as **contracts**. An existing
`schema_version` string is immutable: you may not change serialized field names,
requiredness, defaults that affect validation, enum/literal membership, bounds,
discriminator behavior, or unknown-field policy (`extra=forbid`) under the same
version.

## Three version axes (do not conflate)

| Axis | Example | Meaning |
| --- | --- | --- |
| **Toolkit release** | `1.6.1`, tag `v1.6.1` | Pip package / GitHub Release |
| **Checkpoint edition** | Hub tag `v3`, product class `6bit` | Published model pack identity |
| **Artifact schema** | `axquant.plan.v1` | JSON shape of a plan/manifest/certificate |

A toolkit release may ship many artifact schemas. A Hub pack edition binds one
exact artifact and may reference certificates that freeze a schema version.

## Freeze classes

| Class | Commitment |
| --- | --- |
| `public-certification` | Public cert JSON under `docs/certifications/`; drives README / index / release matrices |
| `release` | Release audit, lifecycle, public claims — long-lived evidence |
| `evidence` | Inventory, plan, sensitivity, benchmark, validation artifacts |
| `operational` | Resumable progress / request helpers — still freeze same-version drift |

## How freeze is enforced

1. **Models** — each top-level `schema_version` has one owning `StrictModel`.
2. **Snapshots** — `schemas/<schema_version>.schema.json` is the canonical JSON Schema lock.
3. **Manifest** — `schemas/manifest.json` lists versions, model owners, and content digests.
4. **Catalog** — `docs/schema-catalog.md` is generated for humans (not authoritative alone).
5. **CI** — `python scripts/render_schema_contracts.py --check` fails on model/snapshot drift
   and on base-ref mutation of an existing version’s digest.

Public certification records additionally load through
`axquant.schema.public_certification` before documentation matrices are rendered.

## Changing a contract

1. Introduce a **new** `schema_version` (e.g. `axquant.plan.v2`) with a new model or
   successor fields.
2. Keep the previous version’s model and snapshot loadable.
3. Add a short migration note in `docs/` (or the relevant `docs/migration-*.md`).
4. Run:

   ```bash
   python scripts/render_schema_contracts.py --write
   python scripts/render_certification_docs.py --write   # if public cert envelope changed
   pytest tests/test_schema_contracts.py tests/test_public_certification_schema.py tests/test_documentation.py
   ```

5. Do **not** edit an existing `schemas/*.schema.json` to match a changed model under
   the same version — CI will reject the digest change vs `origin/main`.

## Related policy

- Campaign freeze discipline: [ADR-0001](roadmap/adr/0001-certification-freeze-discipline.md)
- Public matrices SSOT: `axquant.public_cert_index` + `scripts/render_certification_docs.py`
