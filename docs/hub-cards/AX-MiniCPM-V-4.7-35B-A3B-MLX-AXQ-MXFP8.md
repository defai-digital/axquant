---
license: apache-2.0
library_name: mlx
base_model: openbmb/MiniCPM-V-4.7-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- minicpm-v-4.7
- MXFP8
- vision
---

# AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP8 — 8.71 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.


## Model details

| Property | Value |
| --- | --- |
| Base model | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/tree/896c2d454672476a7b45dcc5e3f0c700a0218577) |
| Source revision | `896c2d454672476a7b45dcc5e3f0c700a0218577` |
| Product family | `minicpm-v-4.7` |
| Source architecture | `MiniCPMV4_7ForConditionalGeneration` (mixture of experts (MoE)); text path optimized |
| Main-model parameters | 35.21B logical parameters |
| Quantizer | AXQuant `2.3.0` |
| Hub budget class | `MXFP8` |
| AXQuant base precision class | `9p4bpw` |
| Planned storage-adjusted BPW | 9.4198 |
| Measured main-model BPW | 8.7148 |
| Measured total BPW | **8.7148** |
| Safetensors weight size | 38.36 GB |
| Approximate complete download | 38.49 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-VLM |
| AX Engine native execution | Unsupported by the tested AX Engine 7.6.4 manifest generator |
| MTP present | `False` |
| Vision present | `True` |
| Audio present | `False` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

This family publishes MXFP4, MXFP8, and one CUDA NVFP4 W4A16 pack. The two MLX packs use
the same role policy at different block formats. Protected tensors stay at higher precision,
so measured BPW is the figure to compare. The NVFP4 format reference uses a CPU backend and is not a CUDA inference checkpoint validation.


| Sibling | Intended trade-off |
| --- | --- |
| [MXFP4 sibling](https://huggingface.co/AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP4) | Same AXQ role policy at the other MLX block format. Vision, merger, and hybrid linear-attention tensors stay at source precision. No MTP tensors were detected. |
| [CUDA NVFP4 W4A16](https://huggingface.co/AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-CUDA-AXQ-NVFP4-W4A16) | CPU reference backend encodes eligible 2D weights as NVFP4; fused 3D experts, vision, and protected tensors stay at source precision. CUDA runtime and quality are unverified. |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP8 --local-dir ./AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP8
```

Allow at least 38.49 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-VLM

```bash
python -m pip install -U mlx-vlm
python -m mlx_vlm.generate \
  --model AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP8 \
  --image ./image.png \
  --prompt "Describe this image." \
  --max-tokens 128 \
  --temperature 0.0
```

The protected vision tower and AXQ language decoder are loaded together by MLX-VLM. The artifact
records MLX `0.32.4`; runtime QA is reported separately from model-quality claims.

## AX Engine status

The tested AX Engine `7.6.4` manifest generator rejects the source model type
`minicpmv4_7`. This package does not include `model-manifest.json`; AX Engine inference is not
supported or validated for this artifact. The AX Engine fields in `axquant_runtime.json` record
the attempted compatibility check, not a successful runtime check.


## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `8bit` | 33.10B | 94.00% |
| `bf16` | 2.11B | 6.00% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `32`.
- MTP sidecar: not included.
- Vision sidecar: not included.
- Vision weights: protected BF16 in main shards.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 231/231 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | not generated; tested AX Engine rejects minicpmv4_7 |
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
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.
- The tested AX Engine manifest generator rejects `minicpmv4_7`; no AX Engine inference support is claimed.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
the [openbmb/MiniCPM-V-4.7-35B-A3B model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/tree/896c2d454672476a7b45dcc5e3f0c700a0218577) for license terms, model
limitations, and responsible-use guidance.
