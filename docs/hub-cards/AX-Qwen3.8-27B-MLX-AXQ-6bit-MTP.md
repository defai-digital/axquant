---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- qwen3.8
- 6bit
- 6-bit
- mtp
- vision
---

# AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the multi-token-prediction (MTP) head and vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Checkpoint Tier 1 certified** on `df-macbookpro-m3` (2026-08-14) at Hub commit
> `a5a0b700ea7c` — measured size against a matched uniform baseline, quality retention,
> and conversion integrity. Current `main` preserves that revision's exact Safetensors
> payloads while allowing metadata-only compatibility fixes. Tier 1 is a checkpoint claim,
> **not** a general speed claim: MTP acceleration is certified for the certificate's
> authorizing profiles only; outside that scope there is no speedup claim.
> See the [checkpoint Tier 1 certificate](https://github.com/defai-digital/axquant/blob/main/docs/certifications/qwen38-27b-axq6-mtp-tier1.md) and [Tier 2 MTP acceleration certificate](https://github.com/defai-digital/axquant/blob/main/docs/certifications/qwen38-27b-axq6-mtp-tier2.md) for the bound evidence and thresholds.


## Model details

| Property | Value |
| --- | --- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Product family | `qwen3.8` |
| Source architecture | `Qwen3_5ForConditionalGeneration` (dense); text path optimized |
| Main-model parameters | 27.36B logical parameters |
| Quantizer | AXQuant `1.6.2` |
| Hub budget class | `6bit` |
| AXQuant base precision class | `6bit` |
| Planned storage-adjusted BPW | 6.0000 |
| Measured main-model BPW | 5.8448 |
| Measured total BPW, including MTP | **6.0001** |
| Safetensors weight size | 20.84 GB |
| Approximate complete download | 20.86 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-LM |
| AX Engine native execution | Native manifest included; execution still requires a runtime check |
| MTP present | `True` |
| Vision present | `True` |
| Audio present | `False` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

AXQ names describe a **storage-budget product class**, not one uniform precision applied to every
tensor. Protected tensors remain at higher precision, so the exact measured BPW is authoritative.
In particular, a `6bit`-named mixed plan may retain `4bit` as its base precision while selecting
6-bit, 8-bit, or BF16 for other tensors to meet an approximately 6-BPW total budget. Protection
floors can also raise a `4bit`-named pack close to (or above) a `6bit` budget on small or heavily
protected models. When that collapse happens, AutomatosX does **not** publish a separate
misleading `4bit` sibling for that base.


| Sibling | Intended trade-off |
| --- | --- |
| [4bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP) | Lower-storage AXQ budget; check its exact BPW |
| [6bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP) | Higher average precision near the 6-BPW budget |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --local-dir ./AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP
```

Allow at least 20.86 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \
  --model AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP \
  --prompt "Explain mixed-precision quantization in three sentences." \
  --max-tokens 128 \
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`0.32.0` and MLX-LM `0.31.3` from conversion.

## Serve with AX Engine and MTP

After installing AX Engine, download the complete repository (see
[AXQuant](https://github.com/defai-digital/axquant) for conversion, certificates, and
model-card tooling) and serve the local directory:

```bash
ax-engine serve ./AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP --port 31418
```

AX Engine is the authority for the AXQ runtime contract and native MTP sidecar.
Runtime speedup claims are limited to the authorizing profiles in the linked Tier 2 certificate.
The artifact records AX Engine version `6.16.1`. Native
`model-manifest.json` status: included as `model-manifest.json`.

## Use with oMLX Lightning MTP

Download the complete repository into a writable local directory and add that directory to oMLX.
With an oMLX build that includes the sidecar importer, select **Import MTP side-car** in
**Model Settings**, then enable **Lightning MTP**. The bundled `mtplx_runtime.json` declares the
`qwen3-next-mtp` execution contract required by oMLX's strict importer; this identifier describes
the MTP head layout, not the base model's `qwen3_5` model type.

The one-time import adds the sidecar tensors to the local checkpoint index. oMLX compatibility does
not extend AXQuant's Tier 2 speed claim beyond the certificate's authorizing profiles; benchmark the
result in the intended oMLX workload. At the time of the 2026-08-20 smoke test, direct sidecar import
was available on oMLX `main` (`0.6.3rc2`); stable oMLX `0.5.7` successfully served the already
imported text checkpoint with Lightning MTP. Its VLM loader rejected 333 vision parameters and
fell back to the LLM path, so this is a text-only compatibility result.

## Use with MTPLX

MTPLX `2.5.2` recognizes the bundled depth-1 sidecar and can serve it with the stable profile:

```bash
mtplx quickstart \
  --model ./AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP \
  --profile stable \
  --depth 1 \
  --reasoning off
```

The contract is intentionally marked `unverified` with a public-release blocker because no MTPLX
Forge exactness baseline is bound to this development artifact. It permits compatibility testing
without `--unsafe-force-unverified`; it does not claim MTPLX exactness certification.

## Runtime compatibility speed smoke (not certification)

Measured on an Apple M5 Max with 128 GB unified memory and AC power. Each number is the median of
three measured batch-1 requests after one warmup, with temperature `0`, thinking off, and MTP depth
1. Prefill prompts use a unique salt; decode uses the same fixed prompt and runs to 256 tokens.

| Runtime | Prefill input | Prefill | Decode input / output | Decode |
| --- | ---: | ---: | ---: | ---: |
| oMLX `0.5.7` | 2,158 uncached tokens | **838.9 tok/s** | 155 / 256 tokens | **43.46 tok/s** |
| AX Engine `7.1.5` | 2,122 uncached of 2,154 tokens | 802.5 tok/s | 148 / 256 tokens | 42.56 tok/s |
| MTPLX `2.5.2` | 2,157 uncached tokens | 738.5 tok/s | 155 / 256 tokens | 42.89 tok/s |

Prefill throughput is uncached prompt tokens divided by local loopback time to first content, so it
includes HTTP, scheduling, and first-token overhead. Decode throughput is `(completion tokens - 1)`
divided by the first-to-last-content interval. The three decode results are within about 2.1%; do
not generalize this short smoke to concurrency, long context, vision, quality, or another host.

> **Metadata compatibility update (2026-08-20):** `mtplx_runtime.json` now includes the complete
> Qwen MTP runtime identity, tensor count, MTPLX compatibility version, recommended profile, and
> explicit unverified-exactness status. Safetensors payloads are unchanged.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `4bit` | 24.34B | 87.60% |
| `6bit` | 15.16M | 0.05% |
| `8bit` | 1.27B | 4.58% |
| `bf16` | 2.16B | 7.77% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `32, 64`.
- MTP sidecar: 15 tensors, 424.70M parameters, 0.85 GB, BF16.
- Vision sidecar: 333 tensors, 460.73M parameters, 0.92 GB, BF16.
- Vision weights: protected BF16 sidecar.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 497/497 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | included as `model-manifest.json` |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | certified for the certificate's authorizing profiles only; outside that scope there is no speedup claim |
| AX Engine kernel evidence | `unmeasured` |
| Vision-language quality | Not evaluated or claimed; vision tensors are preserved at BF16 |
| Speech-recognition quality | Not applicable |
| Long-context quality | 262,144-token capacity is config metadata, not a validated claim |
| Release certification | **Checkpoint Tier 1 certified** on `df-macbookpro-m3` (2026-08-14), Hub commit `a5a0b700ea7c`; the formal AXQuant M0-M8 release campaign is a separate process and is not implied |
| Studio recert (`df-macstudio-m2`, 2026-08-15) | **Not certified** as a replacement T1. Candidate means on prepare-suite v2: agent-coding 0.933 (52), general 0.875 (16). BF16 is not on Ext12T so 0.98 retention was not computed. Historical record is unchanged. See [studio evaluation](https://github.com/defai-digital/axquant/blob/main/docs/certifications/qwen38-27b-axq6-mtp-studio-tier1.md). |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
- MTP requires a sidecar-aware runtime. oMLX/MTPLX discovery compatibility does not extend the
  linked AX Engine certificate to those runtimes.
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.
- [`axquant_mtp_sidecar_manifest.json`](axquant_mtp_sidecar_manifest.json): MTP tensor provenance.
- [`axquant_vision_sidecar_manifest.json`](axquant_vision_sidecar_manifest.json): protected vision tensor provenance.
- [`model-manifest.json`](model-manifest.json): AX Engine native tensor manifest.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
the [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) for license terms, model
limitations, and responsible-use guidance.
