# AXQ formats and runtime targets

AXQuant is an independent MIT-licensed AutomatosX / DEFAI Digital product.
AXQ is the model-pack ecosystem: physical weights plus versioned, checksum-bound
metadata and evidence. It is not a single new tensor datatype. Producers and
readers can adopt the published contracts without depending on AX Engine.

## Install and convert

Use `python -m pip install 'axquant[mlx]'` in a virtual environment on Apple
Silicon, then `axquant quantize /path/to/model-bf16`. No repository clone is
needed for released commands. CUDA users install `axquant[cuda]` and use
[quantize-cuda](cuda-nvfp4.md) with a compatible CUDA-enabled PyTorch build.

The `quantize --format` presets below are implemented on this branch and await
the next PyPI release; published 2.2.1 does not expose this option.

| MLX preset | Base representation | Default target BPW | Qualification limits |
| --- | --- | --- | --- |
| `2bit` | Affine 2-bit for eligible MLP/expert tensors | 2.8 | Attention uses at least 6-bit; existing experimental runtime gates remain |
| `3bit` | Affine 3-bit for eligible MLP/expert tensors | 3.8 | Attention uses at least 6-bit; existing experimental runtime gates remain |
| `6bit` | Affine 6-bit | 6.8 | Higher protection floors remain |
| `mxfp4` | MXFP4 blocks | 4.8 | Higher precision protection tensors may use affine or BF16 |
| `mxfp8` | MLX MXFP8 physical repacking | 8.8 | Unrefined affine8/group32 planning priors; not measured MXFP8 sensitivity |

Presets use group size 32. Protected tensors retain their applicable higher
floors, and the planner can raise the target to the policy minimum. Reported
BPW includes metadata and protected tensors. `--target-bpw` can override the
budget; an explicit preset fails if no base-bit allocation survives planning.
`--format` cannot combine with a recipe, measured ladder or legacy affine4.
Without it, the existing mixed prior planner remains unchanged. All presets
produce development evidence. Conversion does not prove quality or speed.

## Runtime coverage

| Target | Implemented path | Evidence still required |
| --- | --- | --- |
| MLX-LM | Public conversion backend and text generation smoke | Exact model, format and version; vision/audio use their dedicated backends |
| MTPLX | Explicit export profiles and MTP sidecar conventions | Supported architecture profile plus load, generation and MTP off/on checks |
| oMLX | Explicit export profiles and compatibility metadata | Supported architecture profile plus load, generation and requested features |
| AX Engine | Manifest integration, runtime checks and existing certification tooling | Exact artifact/runtime binding; optimizations need separate performance evidence |
| vLLM | Primary CUDA NVFP4 target, compressed-tensors exports | GPU/kernel/version compatibility and model-specific execution evidence |

See [runtime exports](runtime-exports.md) for actual profile coverage. Unknown
peer profiles fail closed. A static compatibility receipt does not prove
execution. A runtime supporting the source model does not automatically mean
it supports every AXQ format, modality or MTP layout. Existing certificates
retain their original runtime scope; no cross-runtime certification is implied.

## Build an integration

Start with the [frozen pack interchange v1 contract](../contracts/axq-pack-interchange-v1.md)
and the [certification contract](../contracts/certification-spec-v1.0.md).
Interchange v1 covers its specified affine U32 packing, not every microscaling
or CUDA layout. Read the relevant manifest schema and physical quantization
configuration; reject unknown versions and unsupported layouts explicitly.
Keep immutable artifact digests and runtime evidence together. Layout or
precision changes require a new variant and new evidence.

Forks, independent readers, exporters, conformance fixtures and runtime tests
are welcome under the [MIT License](../../LICENSE). Open a focused issue or PR
with the model architecture, physical format, runtime version and reproducible
result. Start with [CONTRIBUTING](../../CONTRIBUTING.md); you do not need model
weights or accelerator hardware to improve docs, schemas or synthetic tests.
