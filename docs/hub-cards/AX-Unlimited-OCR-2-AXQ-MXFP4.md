---
license: mit
base_model: baidu/Unlimited-OCR-2
base_model_relation: quantized
language:
- multilingual
tags:
- axquant
- mxfp4
- reference-storage
- runtime-unsupported
- ocr
---

# AX Unlimited-OCR-2 AXQ MXFP4 reference storage

**Development reference storage. Full model inference is not supported.**

Original source: [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c)
at revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`.
Original BF16 weights were checked against pinned Hub LFS SHA-256 objects.
This pack was exported using the AXQuant development workspace for research
and format development.

## Storage

2,196 language attention/MLP/expert matrices use MLX-native MXFP4 encoding
with group size 32. Packed arrays keep the original `.weight` name as U32;
`.scales` stores one U8 E8M0 exponent per group. No affine biases are present.
Individual experts retain their original names and layout.

All other 623 tensors retain exact source values and dtypes, including all
108 SR-SWA DASH-router/indexer tensors, vision encoders, projector, norms,
embedding and output head. The complete source config is preserved with added
packing metadata and an explicit `axquant_storage_only` marker.

Weight files occupy **2.866 GB**, or
**6.855623 measured storage BPW** including protected weights,
scales and Safetensors headers. Nominal format bits are not whole-pack BPW.

`axquant_plan.json`, `axquant_source_binding.json` and `SOURCE_SHA256SUMS.txt`
record the plan, structural source identity and source member hashes.
`axquant_manifest.json` binds the output files and static compatibility report.
The manual plan is explicitly unmeasured; quantization is not calibration or
a quality result.

## Validation and limits

- Checked all 5,015 indexed output tensors and their storage shapes/dtypes.
- Checked exact equality for every preserved tensor and all source config fields.
- Independently decoded 24 matrix samples and compared them with public MLX
  dequantization and quantized matrix multiplication. This verifies matrix
  storage/operations only.
- oMLX and MTPLX static profiles report `unsupported`. Primary runtime
  metadata has `standard_inference=false`.

Unlimited-OCR-2 requires an SR-SWA graph, page-aware rotary positions and
multi-page KV lifecycle that are not implemented in the current MLX runtime
path. No MLX-LM, MLX-VLM, AX Engine, oMLX or MTPLX full-model load/generation
was run. No OCR quality, long-document behavior, memory bound, speed or MTP
claim is made. No quality certification is attached.

There is **no supported model inference command** for this repository.
Included upstream Python files are unchanged archival source assets; their
BF16 loader cannot be assumed to consume this packed layout. Future runtime
integration needs its own layout validation and exact-pack tests.

## License

The upstream model card declares MIT. `LICENSE` preserves Baidu's official
MIT notice (source repository Git blob
`890d455ae73d1d930eee703ce1e5478783ec9154`). Original code notices are retained.
The pinned source model card is included as `UPSTREAM_README.md`.
