# MXFP4, MXFP6, and MXFP8

AXQuant supports these formats through different execution surfaces. A format
name does not establish runtime compatibility, measured sensitivity, or quality
certification.

The new MXFP6 export and MXFP8 repacking commands require a build containing
this change. Installing an extra does not backport them to an older wheel.

| Format | Elements | Shared scale | AXQuant surface | Runtime status |
| --- | --- | --- | --- | --- |
| MXFP4 | E2M1, 4 bits | E8M0 per 32 values | Existing measured planner method and MLX conversion | Existing family-specific runtime gates |
| MXFP6 | E2M3 or E3M2, 6 bits | E8M0 per 32 values | Experimental source-bound reference export and readback | No native MLX or AX Engine inference support |
| MXFP8 | E4M3, 8 bits | E8M0 per 32 values | MLX physical repacking with `convert --q-mode mxfp8` | MLX native; AX Engine compatibility must be independently checked |

The OCP standard also defines E5M2 MXFP8. AXQuant's native MXFP8 surface uses
MLX's E4M3 encoding. The existing affine6/affine8 formats are different formats.
For aligned tensors, the element payload plus scales occupy 4.25, 6.25, and
8.25 bits per value respectively; headers and protected tensors add overhead.

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

## Experimental MXFP6 export

Install the NumPy reference extra in your venv:

```bash
python -m pip install 'axquant[mxfp6]'
```

Use a local BF16, FP16, or FP32 source supported by a convertible architecture
adapter. Inspect with an immutable revision and an explicit model ID, then
create a manual plan from this recipe saved as `mxfp6-recipe.yaml`:

```yaml
schema_version: axquant.manual-recipe.v2
default_bits: 6
default_method: affine
group_size: 32
target_bpw: 16.0
```

The 16-bit ceiling leaves room for small fixtures or protected-heavy sources;
the exporter reports actual per-tensor storage rather than promising that BPW.

```bash
axquant inspect --model /path/to/source --model-id org/model \
  --revision IMMUTABLE_COMMIT --output work/inventory.json
axquant plan-manual --inventory work/inventory.json \
  --recipe mxfp6-recipe.yaml --output work/plan.json
axquant export-mxfp6 --model /path/to/source --plan work/plan.json \
  --element-format e2m3 --allow-unmeasured --output /path/to/output-mxfp6
```

Use `--element-format e3m2` for the other FP6 encoding. The binding written
beside the plan is mandatory. The exporter never downloads weights.

Eligible affine6 attention, MLP, and expert matrices become MXFP6. Allocations
at 8 bits or higher are preserved at source precision, so embeddings, routers,
norms, LM heads, vision/audio and MTP keep their original tensor bytes. A shard
containing only preserved tensors is copied byte-for-byte. Lower-bit assignments,
unclassified or protected six-bit tensors, refinement, nonfinite weights,
unaligned shapes, source drift and incomplete coverage abort. Output is staged
and installed only after validation; existing destinations are never replaced.

The artifact contains Safetensors payloads and `axquant_mxfp6.json` with the
distinct `axquant.mxfp6-pack.v1` envelope. It intentionally has no model
`config.json`, tokenizer, native runtime manifest or certificate. It is not a
loadable MLX checkpoint. No existing planning or certificate schema is changed.

Quantized payloads use six-bit codes packed least-significant-bit first: every
four consecutive codes occupy three bytes. Each 32-value block has a separate
U8 E8M0 scale. Scaling uses `floor(log2(amax))` minus the FP6 maximum normal
exponent, bounded to [-127, 127]; zero blocks use exponent zero. FP6 rounding is
ties-to-even with subnormals, signed zero and saturating overflow. NaN/Inf input
and E8M0 NaN scales are rejected. Reference float32 decode saturates values
outside the finite float32 range.

For checksum-verified reference readback:

```python
from axquant.mxfp6_export import load_mxfp6_tensor

weights = load_mxfp6_tensor("/path/to/output-mxfp6", "model.layers.0.mlp.down_proj.weight")
```

Readback verifies the requested shard's checksum, tensor coverage, dtypes,
shapes and per-tensor storage sizes before decoding. It materializes one requested tensor; export uses
bounded row chunks and does not resident-load the whole checkpoint. Native
MXFP6 inference requires a separate runtime implementation.

References: [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
