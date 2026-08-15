# AXQuant Certification Specification v1.0

- **Status:** normative
- **Version:** 1.0
- **Applies to:** AXQuant checkpoint and scoped MTP-acceleration certificates
- **Verification command:** `axquant verify-cert`

This specification defines what an AXQuant public certificate means and how a third party can
check a local certificate bundle without network access. It standardizes evidence and claims; it
does not introduce a quantizer or broaden AXQuant beyond Apple Silicon.

## 1. Certification tiers

### Tier 1 — checkpoint certification

Tier 1 covers only the bound checkpoint:

- quality against the named, matched reference and recorded thresholds;
- complete weight size and measured bits per weight;
- conversion integrity, file identity, and reproducible provenance; and
- the exact source, artifact revision, profile, datasets, and context scope recorded by the
  certificate.

Tier 1 is **never a speed claim**. A Tier 1 certificate may record that MTP assets are present, but
it may not infer acceleration, throughput, latency, or acceptance performance from their presence.

### Tier 2 — scoped MTP acceleration

Tier 2 certifies MTP acceleration only for the named host, AX Engine build, execution profile,
thresholds, and workload scope in its certificate. A Tier 1 record that reports MTP acceleration as
`certified`, `certified-scoped`, or `certified-see-tier2-record` must point to a consistent certified
Tier 2 record. The repository suffix `-MTP` means that MTP assets are packaged; it does not mean
Tier 2 passed.

Results from one host, engine build, prompt profile, or execution policy do not authorize a claim
for another scope.

## 2. Product identity and measured precision

The Hub repository is a stable product-class SKU:

```text
AX-<Base>-MLX-AXQ-<2bit|3bit|4bit|6bit|8bit>[-MTP]
```

`4bit` and `6bit` are product classes representing requested budget lanes. They do not state that
every tensor has that width and are not GGUF Q4/Q6 labels. Protected tensors and mixed assignments
usually make physical precision differ from the class.

The authoritative weight claim is the full-precision `measured_main_bpw` value bound in the
artifact:

```text
measured_main_bpw = 8 * main_weight_file_size_bytes / main_logical_parameters
```

Verification compares that recomputation exactly with the stored value. Cards and certificate
titles may round it decimal-half-up to two places for display; repository names never contain the
measured value. An artifact edition is bound to an immutable `vN` Hub tag and commit inside the
stable class-SKU repository.

The certificate artifact product class, manifest `target_class`, plan `target_class`, and class in
the repository leaf must agree. Existing certified repositories and legacy certificates are not
renamed or silently rewritten.

## 3. Evidence kinds

Evidence labels retain their literal meaning:

- `measured` is calibration-bound measurement eligible for release gates.
- `imported` is externally produced evidence whose provenance and applicability are explicitly
  bound and reviewed.
- `measured_development` is development evidence and is not a release certification.
- `architecture_prior` is an estimate based on architecture policy. It must never be presented as
  measured or certified.

Passing complete-model quality tests does not rewrite the source evidence label. Any missing,
unknown, inconsistent, or unverifiable binding fails closed.

## 4. Protection floors and artifact integrity

Certificate verification does not weaken planner or conversion policy. Norms remain at least
16-bit; embeddings and routers remain at least 8-bit; the LM head remains 16-bit unless the
governed measured 8-bit path is used; MTP follows its recorded floor; and protected vision/audio
tensors remain BF16/F16. External MTP sidecars remain byte-preserved unless a dedicated validated
transform owns their layout.

The certificate binds an immutable Hub commit and, when present, an edition tag. The local bundle
binds `axquant_manifest.json`, `axquant_plan.json`, and every file hash recorded by the manifest or
certificate. A missing file, unsafe path, changed size, changed digest, plan mismatch, or BPW
rewrite invalidates the verification verdict.

## 5. Context scope

A certificate applies only through its recorded context length and batch/workload conditions.
Certification at a shorter context never implies certification at a longer context. Configuration
metadata such as `max_position_embeddings` is capacity metadata, not evidence that the maximum
context was certified. A longer-context claim requires its own bound evaluation evidence and
certificate scope.

## 6. Offline third-party verification

Outsiders verify a bundle with:

```bash
axquant verify-cert \
  --certificate path/to/checkpoint-tier1.json \
  --artifact path/to/artifact \
  --output certification-verification.json
```

When `--artifact` is omitted, the command uses the certificate's directory if it contains both
`axquant_manifest.json` and `axquant_plan.json`; otherwise it performs certificate-local checks
only. It never contacts the Hub.

The output uses `axquant.certification-verification.v1` and records every check and issue. Exit
codes are:

- `0`: all applicable checks passed;
- `1`: one or more consistency or certification checks failed; and
- `2`: usage, unreadable input, unsafe filesystem input, or other I/O error.

Unknown certificate schemas are rejected. Legacy
`axquant.public-checkpoint-certification.v1` records remain readable as legacy records, but they do
not acquire Certification Specification v1.0 claims merely by being loaded by a newer toolkit.

## 7. Claim boundary

Certification is bound to the exact repository, commit, edition/tag when present, source revision,
artifact hashes, evidence, thresholds, runtime scope, and context stated in the records. It does
not authorize CUDA, GGUF, NVFP4, AutoRound, another physical pack format, another host, a longer
context, vision-tower quality, or an unrecorded MTP speed claim unless a capability-gated
multimodal claim in §8 explicitly authorizes it.

## 8. Multimodal modalities (capability-gated, AXQuant 1.8.0)

Tier 1 text dual-suite quality (**agent-coding** and **general**) never implies vision-tower or
audio quality. Multimodal claims are **capability-gated**:

| Pack capability | Required certification action | Allowed public status |
| --- | --- | --- |
| Vision **not** supported (no declaration and no vision weights/sidecar) | Disable vision checks | `vision.status = not-applicable` |
| Vision **supported** | Smoke and/or quality suite, **or** explicit non-claim | `smoke-certified`, `quality-certified`, or `present-not-certified` |
| Audio **not** supported | Disable audio checks | `audio.status = not-applicable` |
| Audio **supported** | Smoke and/or quality suite, **or** explicit non-claim | same statuses as vision |

### 8.1 Status definitions

| Status | Meaning |
| --- | --- |
| `not-applicable` | Modality disabled for this pack; `supported=false`. No smoke or quality evidence required. |
| `present-not-certified` | Modality weights may be present or BF16-protected; **no** smoke or quality claim. Cards must not say “VLM certified.” |
| `smoke-certified` | Bound runtime smoke only (e.g. MLX-VLM image generation or MLX-Audio transcription). Not a retention-threshold quality claim. |
| `quality-certified` | Bound multimodal quality suite met recorded thresholds against a named reference. |

### 8.2 Evidence binding

Smoke and quality statuses require `evidence_kind` on the modality claim (for example
`runtime-smoke-mlx-vlm` or `multimodal-quality-vision`). Presence of a `vision.safetensors`
sidecar or `vision_config` alone is **not** evidence of smoke or quality certification.

### 8.3 Fail-closed rules

1. Unsupported modality → must be `not-applicable` (do not run multimodal suites).
2. Supported modality without smoke/quality evidence → at most `present-not-certified`.
3. Public “vision quality certified” language requires `quality-certified`.
4. Historical certificates may omit the `modalities` block; toolkits treat omission as
   **legacy-unstated**, not as an implied quality pass.
5. Gemma-style **text-path** packs that strip multimodal configs certify vision/audio as
   `not-applicable` for that SKU even if the upstream BF16 source was multimodal.

### 8.4 Machine-readable field

Optional on checkpoint Tier 1 records (`axquant.public-checkpoint-certification.v1`):

```json
"modalities": {
  "policy": "capability-gated-v1",
  "vision": {
    "status": "present-not-certified",
    "supported": true,
    "reason": "vision weights present/BF16-protected; VLM quality not certified"
  },
  "audio": {
    "status": "not-applicable",
    "supported": false,
    "reason": "audio not supported on this pack"
  }
}
```

Helpers: `axquant.modality_certification` (`build_modalities_block`, `derive_modality_claim`).
