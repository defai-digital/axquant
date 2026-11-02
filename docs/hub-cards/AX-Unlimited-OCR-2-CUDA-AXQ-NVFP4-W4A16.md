---
license: mit
base_model: baidu/Unlimited-OCR-2
base_model_relation: quantized
language:
- multilingual
tags:
- axquant
- nvfp4
- compressed-tensors
- ocr
- development
- runtime-unverified
---

# AX Unlimited-OCR-2 CUDA AXQ NVFP4 W4A16

**Development storage export. No compatible inference runtime has been verified.**

This is an AXQuant RTN NVFP4 W4A16 export of
[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c),
from original BF16 weights at revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`.
The source weight SHA-256 digests were checked against the pinned Hub LFS objects.
This pack is provided for format development and research.

## Storage

- 2,196 language attention, MLP and expert matrices use NVFP4 E2M1 weights,
  E4M3FN block scales (group size 16) and FP32 global scales.
- 623 other tensors retain their exact source values and dtypes, including
  all 108 SR-SWA DASH-router/indexer tensors, vision encoders, projector,
  normalization, embedding and output-head state.
- Activations are declared BF16 (W4A16); activation quantization is disabled.
- The explicit CPU reference encoder produced the checkpoint:
  `backend=numpy-reference`, `status=development`, `runtime_verified=false`,
  `quality_certified=false`.
- `axquant_cuda_plan.json` binds the source files and per-tensor allocations.
  `axquant_cuda_manifest.json` binds the plan and emitted payloads.

## Runtime limitations

Unlimited-OCR-2 adds SR-SWA sparse routing, page-aware rotary positions and
multi-page state. Its architecture differs from Unlimited-OCR v1 despite
sharing `model_type=unlimited-ocr`.

No CUDA, vLLM, SGLang, Transformers or AX Engine load/generation test has
been run for this exact pack. The upstream BF16 inference commands cannot
be assumed to consume these packed tensor names. A runtime needs both the
second-generation graph and the NVFP4 storage integration. There is currently
no verified inference command for this repository.

Storage layout, source hashes, preserved tensor equality and complete output
index coverage were checked. These checks do not establish OCR accuracy,
long-document functionality, throughput, memory bounds or certification.
No MTP or acceleration claim is made. MLX MXFP4/MXFP8 versions are not included.

## License and upstream notices

The upstream model card declares MIT. The included `LICENSE` preserves Baidu's
MIT notice from its official Unlimited-OCR source repository (Git blob
`890d455ae73d1d930eee703ce1e5478783ec9154`). Original model code files and their
notices are preserved; they are not adapted to consume NVFP4 tensors.
The pinned source card is included as `UPSTREAM_README.md`.
