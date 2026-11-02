# MXFP4 and MXFP8

AXQuant supports these formats through different execution surfaces. A format
name does not establish runtime compatibility, measured sensitivity, or quality
certification.

MXFP6 reference export was retired (AXQ-051): no MLX or AX Engine runtime
supports MXFP6 inference, so a reference-only artifact cannot become a usable
pack. The `export-mxfp6` command, the `axquant[mxfp6]` extra, and the
`axquant.mxfp6-pack.v1` envelope no longer exist. Two development MXFP6
reference packs published 2026-10-03 were deleted from the Hub the same day
with the MXFP6 collection; no new ones will be produced.

As checked on 2026-10-21, MXFP6 itself remains an OCP format. NVIDIA provides
[FP6 E2M3/E3M2 types](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP6.html)
and [SM120 MXFP6 GEMM kernels](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu).
[vLLM's Quark integration](https://docs.vllm.ai/en/latest/features/quantization/quark/)
also documents MXFP6 model loading. These are separate execution surfaces;
they do not add MXFP6 support to MLX or AX Engine, or validate every AXQ model family.
There is no verified AXQ MXFP6 publication path at present. A new path requires an explicit
runtime contract and exact-checkpoint load, generation, and modality checks.

| Format | Elements | Shared scale | AXQuant surface | Runtime status |
| --- | --- | --- | --- | --- |
| MXFP4 | E2M1, 4 bits | E8M0 per 32 values | Existing measured planner method and MLX conversion | Existing family-specific runtime gates |
| MXFP8 | E4M3, 8 bits | E8M0 per 32 values | MLX physical repacking with `convert --q-mode mxfp8` | MLX native; AX Engine compatibility must be independently checked |

The OCP standard also defines E5M2 MXFP8. AXQuant's native MXFP8 surface uses
MLX's E4M3 encoding. The existing affine6/affine8 formats are different formats.
For aligned tensors, the element payload plus scales occupy 4.25 and 8.25
bits per value respectively; headers and protected tensors add overhead.

## MXFP4 and MXFP8 conversion

Unlimited-OCR-2 has separate source-layout reference exports:
`AX-Unlimited-OCR-2-AXQ-MXFP4` and `AX-Unlimited-OCR-2-AXQ-MXFP8`.
The development converter packs original matrix names using public MLX APIs;
SR-SWA routing, vision and other protected state retain source values/dtypes.
Full-model inference is unsupported. These packs belong to AXQ Reference
Storage and their peer compatibility reports say `unsupported`.
This source-layout path requires an explicit MX mode, `--allow-unmeasured`,
`--ax-engine-manifest skip` and `--expert-stream off`. It is part of the
development workspace; it does not claim the released PyPI wheel contains it.

Use the existing staged workflow with a source-bound plan. MXFP4 is selected
by the measured planner or explicitly with `--q-mode mxfp4`. MXFP8 uses an
existing plan's unrefined affine8 allocations at group size 32:

```bash
axquant convert --model /path/to/source --plan work/plan.json \
  --q-mode mxfp8 --allow-unmeasured --ax-engine-manifest skip \
  --output /path/to/output-mxfp8
```

`--ax-engine-manifest skip` explicitly requests an MLX development artifact;
it does not declare AX Engine compatibility. Keep the default `required` when
AX Engine is the intended runtime: an unsupported loader must reject the pack.

MXFP8 remaps only 8-bit allocations. BF16 protections and other selected
precisions retain their packing, including explicitly selected MXFP4 tensors.
Per-tensor configuration records the actual physical mode. Fused expert
MXFP8 packing and AWQ/DWQ/GPTQ combinations are rejected pending validation.
Every remapped tensor must use group32; shape and backend coverage checks still
apply. If a plan uses group64, produce a new plan with group32 rather than edit
a checksum-bound artifact.

`--allow-unmeasured` is required even when the input plan has measured affine
sensitivity: those measurements do not measure MXFP8. MXFP8 is not currently a
first-class measured planner method and does not add a certified product SKU.

References: [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
