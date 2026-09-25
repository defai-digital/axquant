# Source-plan binding sidecar (AXQ-048)

`axquant plan` now also writes `axquant_source_binding.json` next to the plan, and
convert carries it into the artifact tree. It is a new artifact contract
(`axquant.source-plan-binding.v1`); no existing contract changed.

## Why

Conversion has to prove that the checkpoint it opens is the one the plan was
built from. That proof used to be a local-path equality on
`plan.source_model.local_path`, which meant the path had to travel with the plan
and the artifact — and a published artifact may not record where it was built.

The sidecar replaces the path with a structural fingerprint of the source:

| Field | Meaning |
| --- | --- |
| `plan_sha256` | binds the sidecar to exactly one plan (`semantic_plan_sha256`, path-insensitive) |
| `source_model` | the source identity without `local_path` |
| `config_sha256` | digest of the source `config.json` |
| `index_sha256` | digest of `model.safetensors.index.json`, or `null` for a single-file export |
| `members` | sorted `(path, size_bytes)` of every Safetensors file |

Only `config.json` and the index are hashed; weight files are stat'ed. The cost
is O(files), never O(bytes), so it is safe on multi-hundred-GB checkpoints.

The fingerprint identifies the checkpoint's layout and metadata, **not** its
weight values: two checkpoints with identical member sizes and an identical index
share it. Use it as a conversion-time guard and as an audit record, not as a
content hash of the tensor payloads.

## Bundles

A recipe bundle exported by `axquant recipe-export` carries the producer's
source binding (`axquant_source_binding.json`) when the plan had one beside it,
with its digest recorded in the bundle's `lineage` map as
`source_binding_sha256`. A consumer on another machine verifies its own copy of
the pinned revision against that fingerprint instead of trusting a local
re-derivation. Bundles exported before this version carry none and keep resolving;
a missing or mismatched binding is rejected when a bundle declares one.

The fingerprint identifies the checkpoint's layout and metadata, **not** its
weight values: two checkpoints with identical member sizes and an identical index
share it. It is a producer-origin structural binding, not a content hash of the
tensor payloads.

## Operator action

- `axquant convert --plan <plan>` reads the binding from beside the plan. A plan
  that records no local path and has no binding is rejected — re-run
  `axquant plan` (which writes the binding) or convert the source by hub id.
- A checkpoint whose config or index does not match the binding is rejected
  before conversion starts.
- Nothing changes for plans written before this version: they still record the
  path they were planned from, and conversion keeps honoring it.

## Publication identity

`publish` now rejects an artifact whose `axquant_plan.json` or
`axquant_manifest.json` records a filesystem path as `source_model.model_id`. A
locally sourced run records its `--model` argument as `model_id` when no
`--model-id` is given, which would upload the operator's directory to the Hub.
Re-run the pipeline that produced the evidence with an explicit `--model-id` so
the artifact identifies the checkpoint by name instead of by location.
