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

| Format | Elements | Shared scale | AXQuant surface | Runtime status |
| --- | --- | --- | --- | --- |
| MXFP4 | E2M1, 4 bits | E8M0 per 32 values | Existing measured planner method and MLX conversion | Existing family-specific runtime gates |
| MXFP8 | E4M3, 8 bits | E8M0 per 32 values | MLX physical repacking with `convert --q-mode mxfp8` | MLX native; AX Engine compatibility must be independently checked |

The OCP standard also defines E5M2 MXFP8. AXQuant's native MXFP8 surface uses
MLX's E4M3 encoding. The existing affine6/affine8 formats are different formats.
For aligned tensors, the element payload plus scales occupy 4.25 and 8.25
bits per value respectively; headers and protected tensors add overhead.

## MXFP4 and MXFP8 conversion

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
