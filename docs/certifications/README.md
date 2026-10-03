# AXQuant checkpoint certifications

This directory records public, revision-bound AXQuant certificates. A checkpoint-tier certificate
covers the exact artifact named by its record; it does not promote sibling models, other
revisions, or unscoped runtime claims.

The index table below is **generated** from the `*-tier1.json` / `*-tier2.json` records
(`public_index` fields plus verdicts) for packs with `public_index.listed = true`.
Do not edit the table by hand — update the certificate JSON and run
`python scripts/render_certification_docs.py --write`. CI enforces exact agreement
via `tests/test_documentation.py`.

**Complete inventory** (listed + unlisted records): [full-list.md](full-list.md).

> All per-pack certificate records were withdrawn on 2026-10-03 pending
> re-certification under the current host and privacy rules. The tables below
> are empty until records are re-issued; nothing on this page is a live
> quality or speed claim.

**Host policy:** factory conversion, **all Tier 1**, and **all Tier 2**
certificates **must** be measured on one factory host class: **Mac Studio,
M2 Ultra, 192 GB unified memory**. Records name the hardware spec, never a
machine identity. Historical records keep the host they were measured on;
do not rewrite those files to the new host. The flagship M0–M8 campaign
schema stays frozen to its historical formal host until that contract is
versioned separately.

**Multimodal (1.8.0):** Tier 1 text quality never implies vision/audio quality. Each
certificate may carry a capability-gated `modalities` block: unsupported modalities are
`not-applicable` (disabled); supported ones are `present-not-certified`, `smoke-certified`,
or `quality-certified` only with bound evidence. Spec:
[certification-spec-v1.0 §8](../contracts/certification-spec-v1.0.md).

<!-- BEGIN:AXQUANT_CERTIFICATION_MATRIX -->
| Checkpoint | Edition | Tier 1 (quality) | Tier 2 (MTP -- Scoped) |
| --- | --- | --- | --- |

**Tier 2 (MTP -- Scoped)** is a scoped MTP *acceleration* certification: token-weighted decode speedup >= 1.20x and prompt-median >= 1.10x on the certificate's named authorizing workloads, measured on the host and AX Engine build recorded in that certificate.

No certified Tier 2 row is present, so no engine binding applies. Certified records are historical and are not re-certified for later AX Engine releases.

A Tier 2 certificate is a scoped acceleration claim only. It is **not** the AX Engine MTP ship gate (MTP-S, in-path exactness), **not** AX Engine default promotion (MTP-D), and not a claim for hosts, engines, or workloads outside its recorded binding. See [MTP gate mapping](adr033-mapping.md) for what a Tier 2 record is and is not evidence for.
<!-- END:AXQUANT_CERTIFICATION_MATRIX -->

See [flagship certification](../guides/flagship-certification.md) for the two-tier policy and claim
boundaries (default route vs formal acceleration route; decode-heavy vs short-answer).
