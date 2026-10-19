# AXDocLayout exports and inference bundles

`quantize-doclayout` is a source-bound exporter for admitted RT-DETRv2 v1.0
and v1.1 checkpoints. It writes custom NVFP4, MXFP4 or MXFP8 storage, with
explicit unmeasured opt-in; those packs reconstruct FP32 for reference CPU
inference. This command does not create the separate native MLX bundles
described below or promote RT-DETR into the language-model architecture
registry.

Run conversion on the designated factory with the pinned local Hub snapshot:

```bash
python -m pip install 'axquant[doclayout]==2.2.0'
# MXFP4/MXFP8 additionally require Apple Silicon MLX.
python -m pip install 'mlx==0.32.3'

axquant quantize-doclayout \
  --model /path/to/AXDocLayout-M-v1.0 \
  --revision 69ddc0da17a7fc35dfde29b06adbb83a75284ae1 \
  --q-mode mxfp4 \
  --allow-unmeasured \
  --output /path/to/AXDocLayout-M-v1.0-AXQ-MXFP4
```

The source directory must contain the original weights, metadata, licenses,
and the complete `inference/` source from that revision. Download with the
normal authenticated factory Hub cache workflow. Weight SHA-256 and an exact
inference-source/metadata digest bind the supported source. Floating revisions,
changed metadata or inference code, changed weights, and existing destinations
fail closed. Conversion stages to a separate directory and verifies the
decoded state before promoting it to the destination. Failed staging remains
for diagnosis; it is never an output pack.

Convolution and attention/MLP matrices are flattened per output channel,
zero-padded to a block boundary and packed. NVFP4 uses E2M1 values with E4M3
block scales and an inverse FP32 global scale (block 16). MXFP4 and MXFP8 use
MLX physical layouts with E8M0 scales (block 32). Norms, biases, proposal anchors,
class/bounding-box heads, query-position heads and embeddings retain original
tensor values and dtypes. Invalid proposal anchors intentionally retain their
source positive infinities; other nonfinite decoded values are rejected.

Outputs carry the frozen `axq-doclayout.development/1` manifest, per-tensor
layout and padding, file digests, a runtime compatibility report, source
licenses and a reference loader. This is a custom storage schema, separate
from the standard AXQuant language-model manifest. Neither existing model
certificates nor the source's FP32 trusted receipt transfer to these packs.

The reference loader reconstructs FP32 weights and runs the bundled PyTorch
architecture on CPU. NVFP4 decoding uses the AXQuant CPU reference codec.
MXFP4/MXFP8 decoding additionally requires Apple Silicon MLX. NVFP4 export
and decoding do not import MLX. Install NumPy and safetensors for packing;
PyTorch and Pillow are needed for page inference. The bundled schema uses
Pydantic and the AXQuant base model provided by the package.

After downloading a pinned published pack, add its directory to `sys.path`:

```python
import axq_doclayout

model = axq_doclayout.load(pack_directory)
elements = axq_doclayout.predict(model, rgb_uint8_array)
```

The returned list contains original-raster elements using source preprocessing,
taxonomy and score threshold 0.30. It is not the source's complete detection
envelope. Source `AXDocLayout.from_pretrained`, ONNX Runtime, Transformers,
MLX-LM/VLM, oMLX, MTPLX, AX Engine and TensorRT do not support this custom pack.
FP32 reconstruction uses full FP32 model memory. No native low-precision
execution, speed improvement, runtime memory saving or quality certification
is claimed.

`scripts/verify_doclayout_axq.py smoke` retains full-weight baseline/candidate
outputs for deterministic blank, text and table pages. These are functional
checks, not a labeled quality evaluation. Initial 4-bit exports produce a
detection on the blank page where FP32 returns none, and all formats change
some detection counts. Treat these reference-storage packs as development
artifacts. The newer native MLX bundles below have separate validation and
limitations.

Current source models and model-specific MLX bundles are grouped in the
[AX DocLayout collection](https://huggingface.co/collections/AutomatosX/ax-doclayout-6acf4438c7708d288bf141d4).
See the [2.2 schema migration](../migrations/doclayout-development-storage.md)
before integrating the custom storage format.

## Published native MLX bundles

The v1.2 and v1.3 MLX models use a separate model-specific inference package.
They execute MXFP8 linear weights with MLX Metal, store convolution weights in
FP16, and preserve normalization and sensitive projections in FP32. Install
the included `ax-doclayout-mlx` source from the immutable revision given in
each model card. MLX 0.32.4 on Apple Silicon is required.

The v1.3 bundle is
[`AutomatosX/AXDocLayout-M-v1.3-MLX-AXQ-MXFP8`](https://huggingface.co/AutomatosX/AXDocLayout-M-v1.3-MLX-AXQ-MXFP8),
derived from the pinned
[`v1.3 CUDA/TF32 source`](https://huggingface.co/AutomatosX/AXDocLayout-M-v1.3-CUDA-TF32).
It passed a local MLX load and synthetic blank/text/table smoke on the factory
Mac Studio. The same text smoke produced a different detection count from the
FP32 CPU source; this is not a labeled quality evaluation and does not establish
accuracy parity, latency improvement, or certification. AX Engine support is
not provided for this model-specific bundle.
