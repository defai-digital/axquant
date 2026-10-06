# Runtime export contracts in 2.1

New MLX conversions include `axquant_compatibility.json` with
`schema_version=axquant.runtime-compatibility.v1`. The conversion artifact
manifest binds this report. It contains independent oMLX/MTPLX profile
verdicts, config/index/runtime/header bindings, required transformations,
launch settings, and modality restrictions. Static-compatible remains
unverified for loading, generation, MTP, and certification.

`export-runtime` creates a new directory and writes
`axquant_runtime_export.json` with `schema_version=axquant.runtime-export.v1`.
It binds source metadata and the source manifest digest when present, final
variant file digests, an explicit legacy norm declaration when supplied,
and the selected runtime requirements. It does not transplant original
certificates or conversion manifests. Unknown profiles fail closed.

The frozen snapshots are new contracts; existing envelope digests are unchanged.
Historical artifacts without the new report retain their versioned readers.
The factory driver requires current bound records for reuse and publication;
old unchecked outputs cannot silently pass this new workflow.

The n-gram layout receipt is now `axquant.ngram-relayout.v2`. Earlier v1 receipts
moved shard keys into a standalone file. A v2 canonical table concatenates
numeric shard rows into `ngram.*` and records actual bits/group metadata.
A v1 filename or receipt does not establish v2 loadability. Complete MTPLX
exports additionally adapt norm representation and native MTP expert paths.

For legacy packs without a trunk norm declaration, an operator may provide
`--source-trunk-norm-layout raw_hf_delta|mlx_multiplier` only after establishing
the source convention. The declaration is bound to the variant and cannot
override a conflicting source declaration. Never guess it from a model name.

Use [the runtime export guide](../guides/runtime-exports.md) to inspect packs,
create distinct variants, honor launch requirements, and run actual factory
smoke checks. Keep private runtime receipts out of published artifacts.
