# Archive URI contract (evidence archive index v2)

`axquant.evidence-archive-index.v2` constrains `EvidenceArchiveRecord.durable_uri`.
The field says where evidence is durably archived and it is published inside a
release artifact, so it may no longer name a host path.

## Accepted form

A scheme-qualified, host-independent reference:

```text
nas://models-archive/axquant/000/evidence.json
s3://bucket/axquant/000/evidence.json
https://huggingface.co/org/repo/resolve/<revision>/evidence.json
```

The scheme is required, the location after `://` must be present and must not
begin with `/`, and the value must not contain a backslash, `..`, `~/`, or one of
the host path prefixes the publication privacy scan rejects (the operator home,
volume, private-var, temp, and home directories; see
`src/axquant/schema/certification.py::_durable_uri` for the exact list).

`file://` is refused even though it carries a scheme: a `file:` reference names a
local filesystem, which is what the scan rejects. The in-tree relative location stays where it was, in
`EvidenceArchiveRecord.path`, which is already validated as a safe relative path.

## Compatibility

Evidence written under `axquant.evidence-archive-index.v1` still loads. The v1
envelope is restated in `axquant/schema/frozen_v1.py` rather than subclassed,
because a subclass would inherit the new validator and v1 is exactly the envelope
that carried a host path. Readers dispatch on `schema_version` through
`axquant.schema.loading.load_evidence_archive_index`; `campaign` uses it.

No published digest moved: `schemas/axquant.evidence-archive-index.v1.schema.json`
and its manifest entry are byte-identical, and the v2 snapshot is new.

## Operator action

- Point campaign tooling at scheme-qualified archive references, and re-export
  any `evidence_archive_index.json` that still records a host path or a
  `file:` reference. Rewriting a recorded value does not need a schema change.
- `durable_uri` is not the only published identity that can carry a path:
  `candidate_model.local_path` and `ExactCertificationScope.source_model.local_path`
  in the certification request remain accepted, because the release audit binds
  the artifact by manifest digest and no longer reads them. They are still
  reported by `publish` as `publication_path_shaped_identity`, and the final
  privacy scan will reject them once it runs on every track — so omit them in new
  tooling. Removing the fields themselves needs a later `schema_version`.
