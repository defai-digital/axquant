# N-gram table relayout for MTPLX compatibility (2026-09-24)

Status: implemented on `main` (commits `9486731`, `027982b`). This is an
evidence write-up, not a certificate: no pack has been relaid out, no MTPLX
load receipt exists yet, and no public MTP acceleration claim is made.

## Problem

MTPLX 2.5.2 refuses to load AXQ Qwen 3.8 Flash-Next MXFP4 packs. AXQ
packages the hashed n-gram PLE table as 128 ShardedEmbedding shard groups
(384 keys, `...ngram_embedding.shards.<i>.weight`) inside
`model.safetensors.index.json`, while MTPLX expects a standalone
`ngram-table.safetensors` and treats the in-index keys as extraneous
parameters. Stock mlx-vlm loading depends on the same sharded keys, so a
single directory cannot serve both runtimes; the sanctioned fix is an
MTPLX-targeted variant pack, never an in-place edit of a published revision.

## Tooling shipped

- `axquant relayout-ngram-table --directory <pack> --output <new-dir> [--dry-run]`
  builds a variant pack that moves exactly one variable - tensor location.
  Names, dtypes, shapes, and payload bytes are preserved via header-level
  byte slices (no load/save round-trip, no requantize); the source pack is
  never modified; the variant is staged and atomically renamed.
- Full accounting before the rename: per-tensor sha256 equality for moved
  and unmoved tensors, zero lost, zero duplicated, strict re-parse of every
  output Safetensors file, `weight_map` and `metadata.total_size` recomputed.
- `mtplx_runtime.json` in the variant declares `ngram_layout:
  standalone-table` (additive contract; missing value means
  `sharded-index`; explicit unknown values fail closed).
- A provenance manifest `axquant_ngram_relayout_manifest.json` records the
  variant-of binding, per-tensor and per-file hashes, moved/lost/duplicated
  accounting, and the exactness baseline as `unverified` with
  `public_release_blocker: true`, `compatibility-smoke-only`.
- The Hub fleet audit flags both failure directions: sharded n-gram keys
  without the standalone table, and a table beside residual index keys.

## Design review

The layout contract was reviewed headlessly by four independent models
(DeepSeek, GLM, MiniMax, Muse) with a self-contained brief. All four
converged on the same three decisions: keep the sharded keys byte-identical
inside the standalone table (no merge; one variable at a time against a
black-box runtime), header byte-slice rebuild only, and a variant pack in a
new directory as the only path. Their additions - index size recomputation,
unmoved-tensor hash coverage, strict output re-parsing, the additive
`ngram_layout` contract, provenance binding, and bidirectional audit checks -
were all adopted.

## Validation

- Targeted tests (`tests/test_ngram_layout.py`, `tests/test_hub_mtp_audit.py`):
  18/18, including mixed-shard rebuild, pure-n-gram shard removal,
  `total_size` recomputation, dry-run, rerun/output/contract fail-closed
  gates, and the audit directions.
- Full non-integration suite: exit 0. `ruff check` + `ruff format --check`:
  clean. `mypy src`: 144 files, no issues. Installed CLI help: verified.
- Review of the committed diff caught and fixed a private-path leak: the
  manifest now records the variant directory name only, never an absolute
  host path.

## Outstanding

1. Real-pack acceptance: run the relayout against a Flash-Next MXFP4 pack on
   `df-macstudio-m2` + Ext16TR0, load the variant with unmodified MTPLX
   2.5.2, and keep the load receipt. That receipt is the only evidence that
   can lift `public_release_blocker`.
2. The MTPLX Forge exactness baseline is a separate, unresolved workstream;
   layout compatibility is not exactness and no MTP acceleration claim is
   implied by this tooling.
3. Deferred by design: converter emitting an explicit
   `ngram_layout: "sharded-index"` for future packs, and any decision to
   publish a relaid-out variant as a new immutable Hub revision.
