# AXDocLayout development storage in 2.2

AXQuant 2.2 adds two new immutable artifact envelopes:

| Schema | Purpose |
| --- | --- |
| `axq-doclayout.development/1` | Exact source identity, physical format, tensor layouts and file hashes |
| `axq-doclayout.runtime-compatibility/1` | Explicit unsupported native runtimes and the FP32 reference execution path |

These are custom RT-DETRv2 storage artifacts. Existing language-model plans,
manifests, certificates, exports, and their schema digests remain unchanged.
No migration or requantization is needed for existing language-model packs.

Use `quantize-doclayout` only with the admitted source revision and explicit
`--allow-unmeasured`. Install `axquant[doclayout]==2.2.0`; MX packing/decoding
additionally needs Apple Silicon MLX. Keep the complete source snapshot and
choose a fresh output directory. Source metadata, inference code and weight
digests must match. Floating revisions and unrelated weights are rejected.

Load the resulting custom pack through its bundled `axq_doclayout.py`, which
checks file bindings and tensor accounting before restoring the source state.
The bundled schema is included in the file bindings. Standard Hugging Face
cache links to the same model cache's blob directory are accepted; other
external symlink targets are rejected. Intentional positive-infinity proposal
anchors are preserved, while NaN, negative-infinity anchors and nonfinite
quantized tensors fail closed.

All packs remain `status=development`, `quality_certified=false` and
`native_runtime_verified=false`. Reference execution reconstructs full FP32
weights for PyTorch CPU. Standard ONNX/Transformers/MLX-LM/VLM and AX Engine
loaders do not accept this format. Do not transfer the source's trusted FP32
receipt, quality claims, or runtime claims to these files.

The initial synthetic controls expose changed detection counts and blank-page
false positives in both 4-bit packs. No labeled quality evaluation was run.
See the [operator guide](../guides/doclayout.md) for scope and usage.
