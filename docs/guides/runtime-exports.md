# Runtime exports and future architecture checks

MLX-LM, MTPLX, oMLX and AX Engine are first-class AXQ compatibility targets,
conditional on model and physical-format support. This guide documents the
implemented export profiles; it does not establish universal runtime support.
See [the ecosystem guide](axq-ecosystem.md) for the broader product direction.


Every new MLX `convert` / `quantize` output includes
`axquant_compatibility.json`, checksum-bound by `axquant_manifest.json`.
It checks current config, tensor headers, index membership, quantization modes,
and MTP discovery separately for oMLX and MTPLX. This is a static export report;
all runtime verification and certification fields remain false.

| Status | Meaning |
| --- | --- |
| `unsupported` | The selected export profile is absent or its physical format is outside the audited scope |
| `requires-export` | A defined profile requires a different index, table layout, or norm representation |
| `static-compatible` | Structural requirements pass; loading and generation remain unverified |

Unknown future architectures fail closed when explicitly exported to a peer.
They can still use ordinary AXQuant conversion, with an explicit unsupported
export profile in their report. An architecture adapter does not automatically
confer oMLX or MTPLX compatibility.

## Commands

Inspect an existing pack without changing it:

```bash
axquant check-runtime-compatibility \
  --directory /path/to/flash-next-pack --output compatibility.json
```

Build a distinct runtime variant:

```bash
axquant export-runtime --directory /path/to/flash-next-pack \
  --output /path/to/flash-next-omlx --target omlx --dry-run
axquant export-runtime --directory /path/to/flash-next-pack \
  --output /path/to/flash-next-omlx --target omlx
axquant export-runtime --directory /path/to/flash-next-pack \
  --output /path/to/flash-next-mtplx --target mtplx
```

Outputs must be new directories outside the source. Exports use staging and
atomic rename. They never replace source weights, silently requantize a trunk,
or transplant source certificates. `axquant_runtime_export.json` binds variant
files and source provenance; the source conversion manifest is not copied as
variant evidence. Neither command publishes a model.

The Flash-Next factory driver rechecks a manifest-bound compatibility record
before reuse and publication. Shared publication preparation/upload gates
also revalidate declared reports. Historical artifacts without this new
record retain their versioned readers. Its explicit local export stage is:

```bash
python scripts/run_qwen38_flash_next_axq.py export-runtime \
  --pack mxfp4 --target omlx
python scripts/run_qwen38_flash_next_axq.py export-runtime \
  --pack axq6 --target mtplx
```

## Flash-Next profiles

`qwen4_exp` / `qwen4_exp_text` use the native `qwen4-exp-mtp` contract in new
conversions. Their head must not use the dense `qwen3-next-mtp` importer.
Ordinary MLX output keeps `mtp.safetensors` separate from the language index.

The oMLX variant keeps the indexed ShardedEmbedding table and original
precision. It includes native MTP tensors in the runtime index and removes
them from any tensor-name exclusion list. This enables native Qwen4 MTP
discovery. Sidecar and trunk payloads remain byte-identical.

The MTPLX `qwen4-exp-v2` profile admits affine and MXFP4 trunks with floating
native MTP heads. MXFP8 remains blocked until separately verified. The MXFP4
variant retains its original precision and was exercised on the factory host
with native MTP disabled/enabled; it is not converted to affine.
The export performs:

1. Numeric shard-row concatenation into `ngram.weight/scales/biases`, with
   actual `ngram_bits` and `ngram_group_size` metadata. An 8-bit embedding
   stays 8-bit regardless of trunk precision.
2. Component, shard-count, shape, dtype, and quantization checks. Concatenation
   uses bounded-memory byte copies and verifies source payload hashes.
3. `ngram_sidecar=true` in text config; removal of table keys from the language
   index and quantization map.
4. Raw-HF RMSNorm delta-to-multiplier rebasing for trunk and applicable MTP
   norms. Pre-FC MTP deltas remain raw because the selected loader rebases
   them during attachment. Precision does not change.
5. Native floating MTP expert splitting/renaming to `switch_mlp` paths.
   Unknown orientations and undeclared norm conventions fail closed.

`relayout-ngram-table` performs only steps 1-3 and writes a v2 layout receipt.
It does not make a complete MTPLX runtime variant. Earlier v1 receipts moved
shard names into a standalone file without canonical concatenation; that
filename alone is insufficient compatibility evidence.

The profiles were inspected on 2026-10-06 against the public
[MTPLX Qwen4 loader](https://github.com/youssofal/MTPLX/blob/main/mtplx/models/qwen4_exp.py),
[oMLX model discovery](https://github.com/jundot/omlx/blob/main/omlx/utils/model_loading.py),
and public MLX-VLM Qwen4 formats. Local MTPLX inspection used version 2.11.3.
This is an implementation observation, not a promised minimum version.

On 2026-10-06, the exact Flash-Next MXFP4 variants passed text generation with
MTP disabled/enabled in MTPLX 2.11.3 (MLX 0.32.2, MLX-LM 0.31.3) and oMLX
0.7.0.dev4 (MLX 0.32.2, MLX-LM 0.32.0, MLX-VLM 0.7.1). Both answered the
arithmetic smoke with `42` and read a synthetic digit image as `7`. oMLX
native head execution was witnessed; MTPLX health confirmed native MTP
engagement. These are scoped load/generation checks, not quality or speed
certificates or a claim about all runtime versions.

On 2026-11-02, the same checks passed for local oMLX and MTPLX variants
exported from the affine six-bit pack
(`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP`) on the factory host, using
the same runtime builds. The six-bit pack predates the trunk norm declaration.
Its 181 trunk RMSNorm tensors were checked byte-identical to the BF16 source,
so the MTPLX export used `--source-trunk-norm-layout raw_hf_delta`. oMLX passed
text and image input with MTP disabled and enabled, and native head execution
was witnessed. MTPLX passed AR text, refused AR image input as documented, and
passed text and image input in MTP mode. Without
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`, the MTPLX MTP-mode prefill on the M2 Ultra
failed with a Metal command-buffer timeout. The smoke runner now sets that
driver scheduling flag when it is unset and records it. With the same flag,
MLX-VLM 0.7.9 (MLX 0.32.4) generated the expected text from the unmodified
six-bit pack on the GPU (about 155 GB, close to the recommended working-set
limit on the 192 GB host); vision and MTP were not checked. These variants are
local and unpublished. These are scoped load/generation checks, not quality,
speed or MTP-acceptance evidence.

## Launch requirements and measured smoke checks

The report and variant manifest carry `launch_environment`, `model_settings`,
and `modality_constraints`. Consumers must honor these profile requirements.
The oMLX profile uses `qwen4_ple_ssd_offload=true`; the per-model setting
overrides its PLE environment variable. Resident PLE plus MTP exceeded the
tested memory admission limit. SSD offload preserves tensor precision.

The MTPLX profile streams the n-gram table and disables the fixed-M4/affine
optimization flags listed in `launch_environment`. The selected MTPLX API
requires MTP mode for image input: AR text works, while AR image input is an
explicit HTTP 400 capability restriction. MTP image input passed.

For legacy packs lacking a trunk norm declaration, the operator may supply
`--source-trunk-norm-layout raw_hf_delta` only after establishing the source
convention. It is recorded in the variant manifest and cannot override an
existing conflicting declaration. The source contract is not rewritten.

Run the bounded, serial factory smoke against a finished variant:

```bash
python scripts/smoke_runtime_export.py --directory /path/to/variant \
  --runtime mtplx --executable /path/to/mtplx \
  --output /path/to/private-evidence/mtplx-smoke.json --vision
python scripts/smoke_runtime_export.py --directory /path/to/variant \
  --runtime omlx --executable /path/to/omlx \
  --output /path/to/private-evidence/omlx-smoke.json --vision
```

The runner selects only the requested model, isolates server state, disables
shared SSD session-cache writes, checks MTP engagement, and retains failures.
It does not publish or grant a certificate. Keep receipts private: they
include host, commands, and local artifact bindings.

## Main-pack layout and the publication smoke gate

A converted MLX pack keeps one layout for every target runtime:

- The weight index lists the trunk and, for Qwen 3.5-family and Gemma 4
  packs, the protected `vision.safetensors` sidecar. MLX-VLM loads only
  indexed shards; an unindexed tower is silently disabled and image input
  fails.
- MTP tensors live only in the unindexed `mtp.safetensors` sidecar. Conversion
  fails closed if the index names an `mtp.*` tensor or an MTP shard: MLX-VLM
  shifts every trunk RMSNorm when it sees MTP keys.
- When `mtplx_runtime.json` declares `mtp_norm_layout: raw_hf_delta`,
  `config.json` also declares `mtp_norm_encoding: delta`. oMLX's one-step
  sidecar import and MTPLX read that key; AX Engine reads the runtime file.
  Packs converted before this declaration landed (2026-10-31) lack the key.

`scripts/repair_pack_layout.py` brings an older pack to this layout without
touching tensor bytes: it indexes a byte-preserved Qwen vision sidecar, applies
the current MTPLX load contract to `config.json`, refreshes the manifest file
records, compatibility record and AX Engine `model-manifest.json`, and backs up
the original small files first. It refuses vision sidecars with other tensor
layouts. Run it without `--apply` to see the planned changes.

Publishing a generative MLX pack (a `...ForCausalLM` or
`...ForConditionalGeneration` architecture) requires
`axquant_publication_smoke.json`, a receipt bound to the digest of every
runtime payload file. Required checks:

| Check | When |
| --- | --- |
| MLX-LM text; AX Engine agreement with MLX-LM on at least 9 of a fixed 10-prompt set | Always |
| MLX-VLM text and image | The pack ships vision weights |
| AX Engine agreement with MLX-VLM's language model, replacing both MLX-LM checks | A vision pack whose model type MLX-LM cannot load (`qwen4_exp`) |
| oMLX with MTP off and on, after `import_mtplx_sidecar` on a private copy | Qwen 3.5-family pack with `mtp.safetensors` |

```bash
python scripts/publication_smoke.py --directory /path/to/pack \
  --omlx-python /path/to/omlx-venv/bin/python
```

`publish` and the catalog publisher refuse a missing, stale, or failing
receipt. For AX Engine agreement, its greedy tokens (16-token budget) are
teacher-forced through MLX-LM (or MLX-VLM, where it replaces MLX-LM); a
prompt agrees when every token is within 0.5 nat of the reference runtime's
top log-probability. Exact token equality and a single
exact answer are not used: engines break exact ties differently, and a small
pack can sit on a near tie. Healthy packs measured 0 or 0.125 nat (one BF16
logit step); tokens from a different model measured up to 14 nat. The gate
recomputes the count from the recorded deficits and checks the prompt-set
digest rather than trusting the recorded status. Embedding and CUDA packs are
outside this gate. A pass shows load and short greedy generation only; it is
not quality, speed, or MTP acceptance evidence. The receipt records the chip,
not the hostname.

## Adding profiles and claiming compatibility

Define an explicit profile covering physical quantization modes, index
discovery, protected modalities, MTP keys, expert orientation, and norm
conventions. Add tiny format regression fixtures, preserve source identity,
and create a distinct variant when a runtime requires different storage.
Unknown families and modes must remain blocked.

Before claiming working runtime compatibility, run load and generation checks
with MTP disabled and enabled on the exact variant. Record file digests,
runtime/dependency versions, host, commands, and results. Check advertised
modalities too. Static reports do not establish these measurements, exactness,
speed, or Tier 1/Tier 2 certification. Existing publication gates still apply.
