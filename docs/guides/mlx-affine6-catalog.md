# Publishing MLX affine6 catalog variants

`scripts/publish_mlx_affine6_catalog.py` runs source-pinned factory jobs and keeps
per-model conversion logs, task smoke receipts and immutable Hub publication
receipts. It uses the native `6bit` preset: affine6/group32 with existing 8-bit
and floating protection floors. This is not MXFP6. Complete storage can exceed
six bits per logical parameter; use the artifact's measured BPW.

Run conversions and smokes on the designated factory host. Use its external
model storage and the authenticated Xet high-performance cache environment from
the family runbook. Keep campaign JSON, logs and credentials outside tracked
files. The script does not commit or push GitHub changes.

Create an English JSON list of jobs. For example:

```json
[
  {
    "key": "embedding-example",
    "source_id": "owner/source-model",
    "source_revision": "0123456789abcdef0123456789abcdef01234567",
    "source_path": "/path/to/verified-original-source",
    "output": "/path/to/new-affine6-pack",
    "repo_id": "AutomatosX/AX-Example-MLX-AXQ-6bit",
    "task": "embedding",
    "license": "apache-2.0",
    "mtp": false
  }
]
```

Verify the original files against the source revision before creating a job.
Do not substitute a published MXFP4/MXFP8 pack for its original. An explicitly
quantized upstream requires `source_quantized: true`; the card discloses the
additional quantization. A configuration flag alone does not establish MTP:
`mtp: true` requires actual protected tensors and matching sidecar provenance.

Prepare and test candidates first:

```bash
python scripts/publish_mlx_affine6_catalog.py \
  --jobs /path/to/jobs.json --state /path/to/campaign-state
```

Publish only prepared passing candidates, without repeating GPU work:

```bash
python scripts/publish_mlx_affine6_catalog.py \
  --jobs /path/to/jobs.json --state /path/to/campaign-state \
  --key embedding-example --publish-ready
```

`--publish` instead performs conversion, smoke and publication in one invocation.
The factory lock serializes conversion and smoke phases. Each conversion and
smoke has a deadline; `convert_timeout` and `smoke_timeout` are per-job seconds.
Task failures remain blocked while other selected jobs continue. A failed smoke
is retained and is not automatically retried. After diagnosis, preserve its
receipt under another filename before running a corrected test.

## Task and runtime evidence

| Task | Required functional smoke |
| --- | --- |
| `embedding` | Public embedding loader, finite normalized vectors, retrieval ordering, and cosine comparison with the source |
| `text` | Two short deterministic generation checks |
| `vision` | Read a synthetic text image through the processor and model |
| `ocr` | Extract a synthetic digit string through the OCR path |
| `translation` | Translate a short source-language sentence |
| `decision` | Included Clef loader, choice probabilities, score output and bounded Noul output |

Embedding comparisons use FP32 arithmetic and a normalization tolerance suitable
for floating outputs. These small checks do not establish MTEB, coding quality,
long-context quality, multimodal quality, MTP execution, acceptance or speed.
The smoke receipt binds the actual runtime payload; modified config or weights
invalidate reuse. Cards distinguish observed behavior from static oMLX/MTPLX
compatibility checks. Unknown profiles remain unsupported.

Decision conversions require an indexed source view that excludes the auxiliary
joint head and any unmerged adapters. `source_origin` points to the original
head assets. The script preserves that BF16 head and includes its parameters
and bytes in complete weight accounting. Merge required source adapters using
the applicable family procedure first; never discard them silently.

## Existing artifacts and external assistants

`artifact_path` can adopt a historical artifact into a separate publication view.
All original manifest records must verify, the source identity must match, and
the pack must contain actual affine6 allocations with no legacy low-bit mix.
Historical native affine6/group64 packing is retained and its actual group sizes
remain in the config and plan. Weights are hardlinked and metadata copied. The
original is not modified.
A new manifest-bound runtime compatibility report is generated for the view.

Gemma external assistants use the existing exact-pair composite contract. Supply
`assistant_path` and `assistant_source_id` in immutable `owner/model@revision`
form, with `mtp: true`. The assistant must use the supported floating profile;
a differently named architecture is not accepted by relabeling its config.
Target-only accounting stays in the base manifest; the composite manifest binds
the separate assistant. The card's complete storage/BPW includes both. Packaging
an assistant does not prove execution, exactness, acceptance or acceleration.

For an existing supported large MoE pack, `runtime: "ax-engine-stream"` runs the
text smoke with SSD expert streaming requested. The script generates the native
manifest only in the publication view. This is a family-specific execution path,
not universal SSD support or a claim that a resident MLX loader can fit the pack.

## Publication and collection behavior

The uploader verifies local manifest members, compatibility records and the
payload-bound task smoke. It creates a new private repository, uploads an exact
allowlist, compares every remote size and LFS SHA-256 or Git blob identity, then
makes the repository public and verifies unauthenticated access at its immutable
revision. Private cache files and local runtime checks are excluded. Existing
repositories require the exact campaign staging receipt; unrelated private or
public repositories are never overwritten.

After publication, the tool adds the model to the 6-bit, MLX and relevant family
collections, plus MTP when a valid payload is packaged. Existing collection
members are preserved. It saves the published card under `docs/hub-cards/`.
Changing a job's identity requires a separate campaign state; do not edit a
receipt to bypass source, smoke or repository guards.

A new floating-source conversion that fails specifically with a Metal command
buffer timeout receives one bounded CPU conversion retry on the same factory
host. Its precision plan and source revision stay fixed, and both attempt logs
are retained. Other failures and quantized upstreams do not receive this retry.
Generation still needs its independent passing task smoke before publication.
Use `--smoke-timeout SECONDS` to change an operational worker deadline without
changing a source or artifact identity.

Hy-MT2 uses AXQuant's bundled MIT-licensed `hy_v3_mlx.py` adapter. Register it
before calling `mlx_lm.load`; unmodified upstream MLX-LM and MLX-VLM do not provide
that architecture. The translation receipt and card name that dependency.

OCR MoEGate router candidates are restricted to BF16 **before** budget
allocation. Their serialized plan distributions and storage estimates therefore
remain consistent. A governed recipe that planned those non-Linear routers at
lower precision must be rebuilt; the converter does not patch a bound recipe.
OCR2 vision-encoder aliases are separate from language-decoder aliases, so a
vision module cannot inherit a language module's lower precision.

Thinking-model smokes compare only an explicit completed source final channel:
Qwen's closing reasoning delimiter or Muse's ATEM `to=user` message. Raw output
is retained in the receipt. An unfinished reasoning channel, or an expected
answer that appears only in reasoning, does not pass.

Native AX Engine receipts also record the explicit runtime environment and cold
smoke elapsed time. A measured `AX_MLX_PIPELINE_GRANULARITY=layer` configuration
submits each layer's lazy graph before loading more expert stacks. Reproduce the
exact recorded setting rather than treating a passing run as proof of the
default configuration. The native MLX preview takes pre-tokenized input tokens;
use the original tokenizer and chat template. SSD paging time and a two-request
functional smoke do not establish useful serving latency or throughput.

Withdrawn model identities are rejected before conversion, inference or Hub
publication. In particular, Ornith 397B is inspection-only following the
2026-10-24 user withdrawal. A historical convertible plan, adopted artifact or
`--allow-unmeasured` does not restore support. Its old campaign job and failure
receipts remain historical evidence; it is not an active publication target.
