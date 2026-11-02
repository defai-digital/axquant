---
license: apache-2.0
base_model: openbmb/MiniCPM-V-4.7-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- axquant
- nvfp4
- minicpm-v-4.7
- image-text-to-text
---

# AX-MiniCPM-V-4.7-35B-A3B-CUDA-AXQ-NVFP4-W4A16

> **Development format reference; not a certified AXQuant release.** This repository records an
> unmeasured CPU reference export. It does not establish CUDA inference, model quality, speed, or
> compatibility with any inference engine.

## Source and export

- Source: [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/tree/896c2d454672476a7b45dcc5e3f0c700a0218577)
- Source revision: `896c2d454672476a7b45dcc5e3f0c700a0218577`
- Format: NVFP4 RTN W4A16, block size 16; the plan declares BF16 activations.
- Export backend: `numpy-reference` on CPU.
- Quantized tensors: 160 eligible 2D attention and expert matrices (40 attention and 120 expert
  tensors).
- Vision, merger, hybrid linear-attention, norm, router, embedding, LM-head, and fused 3D expert
  weights retain source precision under the export policy.
- Source index: 1,152 tensors across 16 Safetensors shards. No MTP tensors were indexed.

The `CUDA` repository name identifies the requested format family. The CPU reference backend is
not a CUDA-runtime export validation and should not be treated as a ready-to-serve NVIDIA model.

## Runtime and quality status

- CUDA runtime load and generation: not tested; `runtime_verified=false`.
- Quality evaluation: not performed; `quality_certified=false`.
- AX Engine: unsupported by the tested 7.6.4 manifest generator, which rejects `minicpmv4_7`.
- vLLM and other inference engines: not tested; no inference-engine support is claimed.
- Video, MTP, and vision-language quality: not claimed.

The public MLX MXFP4 and MXFP8 siblings were loaded with MLX-VLM 0.7.9 and passed a synthetic
image-generation smoke. That evidence applies only to those MLX repositories, not this NVFP4
format reference.

## Related packages

- [MLX MXFP4](https://huggingface.co/AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP4)
- [MLX MXFP8](https://huggingface.co/AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-MXFP8)

## License

The upstream source is Apache-2.0. See the [source model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) for the license and model use terms.
