---
license: apache-2.0
library_name: mlx
base_model: ornith-ai/Ornith-1.5-9B
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
- qwen3.5
- MXFP4
- MXFP4
- mtp
- vision
---

# AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP — 6.55 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the multi-token-prediction (MTP) head and vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.


## Model details

| Property | Value |
| --- | --- |
| Base model | [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B/tree/489cb97981b8654bcfcf30ce1f94ed1b62e07b53) |
| Source revision | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` |
| Product family | `qwen3.5` |
| Source architecture | `Qwen3_5ForConditionalGeneration` (dense); text path optimized |
| Main-model parameters | 9.41B logical parameters |
| Quantizer | AXQuant `1.9.0` |
| Hub budget class | `MXFP4` |
| AXQuant base precision class | `8bit` |
| Planned storage-adjusted BPW | 7.3259 |
| Measured main-model BPW | 6.5503 |
| Measured total BPW, including MTP | **6.7885** |
| Safetensors weight size | 8.19 GB |
| Approximate complete download | 8.21 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-LM |
| AX Engine native execution | Not established; no validated native manifest is included |
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
| [4bit sibling](https://huggingface.co/AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-4bit-MTP) | Lower-storage AXQ budget; check its exact BPW |
| [6bit sibling](https://huggingface.co/AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP) | Higher average precision near the 6-BPW budget |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP --local-dir ./AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP
```

Allow at least 8.21 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \
  --model AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP \
  --prompt "Explain mixed-precision quantization in three sentences." \
  --max-tokens 128 \
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`0.32.1` and MLX-LM `0.31.3` from conversion.

## AX Engine status

This package does **not** include a validated native `model-manifest.json`, so AX Engine execution
is not established by this release. The AX Engine fields in `axquant_runtime.json` describe the
intended compatibility contract, not observed runtime evidence. Use the architecture-specific MLX
runtime path above. The artifact records AX Engine version
`not recorded`, but version discovery alone is not a runtime check.

## Use the packaged Qwen MTP head with oMLX or MTPLX

This repository keeps the Qwen MTP head in `mtp.safetensors`; stock MLX-LM does not load that
sidecar by itself. Download the complete repository to a writable local directory before using a
sidecar-aware runtime.

In oMLX `0.6.3rc2` or newer, add the local directory, open **Model Settings**, choose
**Import MTP side-car**, and then enable **Lightning MTP**. The import changes only the local copy
so the MTP tensors become visible through the checkpoint index.

MTPLX can consume the packaged sidecar directly:

```bash
mtplx quickstart \
  --model ./AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP \
  --profile stable \
  --depth 1 \
  --reasoning off
```

`mtplx_runtime.json` declares the canonical `qwen3-next-mtp` execution contract. This enables
strict runtime discovery; it does not extend AXQuant quality, exactness, or speed certification to
oMLX or MTPLX.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `4bit` | 6.92B | 71.67% |
| `8bit` | 1.02B | 10.54% |
| `bf16` | 1.72B | 17.79% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `32, 64`.
- MTP sidecar: 15 tensors, 243.29M parameters, 0.49 GB, BF16.
- Vision sidecar: 333 tensors, 456.01M parameters, 0.91 GB, BF16.
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
| Quantizer execution | 249/249 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | not included |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | not measured; no MTP speedup claim |
| AX Engine kernel evidence | `unmeasured` |
| Vision-language quality | Not evaluated or claimed; vision tensors are preserved at BF16 |
| Speech-recognition quality | Not applicable |
| Long-context quality | 262,144-token capacity is config metadata, not a validated claim |
| Release certification | **Not certified**; formal AXQuant M0-M8 gates are not closed |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
- MTP requires a sidecar-aware runtime. oMLX/MTPLX discovery compatibility does not establish exactness or speed certification for those runtimes.
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.
- AX Engine execution is not established because this package has no validated native manifest.

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

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
the [ornith-ai/Ornith-1.5-9B model card](https://huggingface.co/ornith-ai/Ornith-1.5-9B/tree/489cb97981b8654bcfcf30ce1f94ed1b62e07b53) for license terms, model
limitations, and responsible-use guidance.
