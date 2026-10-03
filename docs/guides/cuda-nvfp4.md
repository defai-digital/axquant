# Native CUDA NVFP4 conversion

This experimental backend adds AXQuant-owned round-to-nearest (RTN) NVFP4
weight quantization. It does not use AWQ. It is separate from the existing
MLX conversion path and requires a build containing the CUDA commands.
Installing the extra does not backport these commands to an older wheel.

The initial format is **NVFP4 W4A16**: selected weights use E2M1 FP4,
activations execute in FP16/BF16 without activation quantization, and protected
weights retain their source precision. This backend does not calibrate or
quantize activations to FP4 (W4A4).

## Install and convert

Use a venv and a CUDA-enabled PyTorch build matching your NVIDIA environment.
The normal AXQuant install remains unchanged; CUDA dependencies are optional.
See [PyTorch installation](https://pytorch.org/get-started/locally/) for the
appropriate PyTorch wheel. A CPU-only PyTorch build cannot execute CUDA.

```bash
python -m pip install 'axquant[cuda]'
axquant quantize-cuda /path/to/source-bf16 \
  --device cuda:0 --q-mode nvfp4 --allow-unmeasured \
  --output /path/to/output-nvfp4
```

Use an original, unquantized Safetensors checkpoint with a registered
convertible architecture. Already quantized AWQ, MLX or NVFP4 weights are
rejected rather than silently requantized. Hub download is not part of these
commands; supply a local checkpoint. No model remote code is executed.

For a reviewable allocation before conversion:

```bash
axquant plan-cuda /path/to/source-bf16 \
  --model-id organization/model --allow-unmeasured \
  --keep '*layers.0.*' --output work/cuda-plan.json
axquant convert-cuda /path/to/source-bf16 \
  --plan work/cuda-plan.json --device cuda:0 \
  --rows-per-chunk 256 --allow-unmeasured \
  --output /path/to/output-nvfp4
```

`--revision` optionally records a 40-character immutable source revision.
The local weights, config, index and copied tokenizer/processor/code assets
are bound by exact SHA-256 content digests regardless of revision. Plans
contain portable identity and relative filenames, not source host paths.

Both stages require `--allow-unmeasured`: RTN is not measured activation
sensitivity. CUDA conversion fails when CUDA is unavailable; it never falls
back to CPU. GPU scratch memory is bounded by `--rows-per-chunk`; host memory
still holds a source shard and its output tensors.

For format development and CPU regression testing only, explicitly select
`--device cpu`. The manifest then records `numpy-reference`, not CUDA
execution. This does not establish GPU inference compatibility or speed.

## Allocation and output

Eligible attention, MLP and individual expert `.weight` matrices use NVFP4.
They must be BF16, FP16 or FP32, two-dimensional, nonempty and aligned to
16 columns. Router, embedding, norm, LM head, MTP, vision, audio, tied and
unclassified tensors remain at source precision. Additional `--keep` globs
also preserve source precision. Fused three-dimensional expert tables are
preserved; they are not implicitly split or renamed.

Parallel Q/K/V, gate/up and expert w1/w3 projections share the inverse tensor
scale expected by fused runtime modules. Keeping one member preserves its
entire fused unit. Shared groups may span source shards; conversion computes
their common maximum before packing.

Each selected matrix stores these unswizzled checkpoint tensors:

| Suffix | Dtype | Shape |
| --- | --- | --- |
| `weight_packed` | uint8 | rows, columns / 2 |
| `weight_scale` | float8_e4m3fn | rows, columns / 16 |
| `weight_global_scale` | float32 | scalar |

The earlier element occupies the low nibble. Positive E2M1 magnitudes are
`0, 0.5, 1, 1.5, 2, 3, 4, 6`; both element and block-scale rounding use
nearest-even. The reconstruction convention is:

```text
weight = E2M1_value * E4M3_block_scale / weight_global_scale
```

The inverse tensor scale is `2688 / max(abs(weight))`, using the maximum of
the complete fused group when present. For an all-zero matrix or complete
group the inverse scale is 1. Zero blocks use block scales of 1. Positive block scales below the FP8
minimum are clamped to its smallest subnormal to avoid division by zero.
Non-finite weights and unrepresentable FP32 scales fail closed.

The nominal selected-weight payload is 4.5 bits per value plus one FP32
scalar per matrix. Protected weights, Safetensors headers and metadata add
overhead. This estimate does not include activations, KV cache or runtime
scratch memory.

`config.json` uses the public `compressed-tensors` NVFP4A16 configuration
with block size 16 and no activation quantization. Conversion copies allowed
model assets, regenerates the main index and writes:

- `axquant_cuda_plan.json`: `axquant.cuda-plan.v1`.
- `axquant_cuda_manifest.json`: `axquant.cuda-pack.v1`, including output digests.

External MTP and other untouched Safetensors files are copied byte for byte.
Protected tensors inside transformed shards retain dtype and tensor bytes.
Local credentials and runtime evidence are not copied. Source changes,
incomplete plan coverage or conversion failure abort before publication.
Output is staged beside the destination and published by directory rename;
an existing output is rejected.

## Runtime and evidence boundaries

The export targets the public
[compressed-tensors interface](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/).
It is not a renamed MLX MXFP4 pack. NVFP4 and MXFP4 use different scales and
block sizes. The production converter does not depend on compressed-tensors,
LLM Compressor or NVIDIA Model Optimizer.

NVIDIA Blackwell-class hardware has native FP4 acceleration; actual NVFP4A16
kernel availability depends on the exact GPU, runtime version and model
layout. A GPU that can perform conversion is not automatically a supported
NVFP4 inference target. Fused projections, MoE layouts and custom OCR model
plugins require independent runtime checks.

Every generated manifest records `status=development`,
`runtime_verified=false` and `quality_certified=false`. Numerical decoding
and public config/decompression tests do not substitute for an immutable
runtime load, generation, latency or OCR accuracy qualification. Existing
MLX publication and certification commands do not certify this new format.
AX Engine remains the Apple Silicon runtime; AX Serving owns CUDA fleet
deployment. No CUDA inference engine or Tensor Core kernel is added here.

Run the CUDA hardware regressions on a NVIDIA test host with this build:

```bash
python -m pytest tests/test_nvfp4.py -m integration
```

CPU tensor-operation parity and the optional public compressed-tensors
decompression check run with the ordinary CUDA-focused suite:

```bash
python -m pytest tests/test_nvfp4.py tests/test_cuda.py -m 'not integration'
```

The synthetic vLLM smoke script additionally creates a deterministic tiny
Qwen3 BF16 control, performs real CUDA conversion and loads both checkpoints
in isolated vLLM processes. It checks NVFP4 kernel selection and rejects
runtime fused-scale rescaling. It does not measure task quality:

```bash
python scripts/smoke_cuda_nvfp4.py --create-source work/Qwen3-NVFP4-Smoke
python scripts/smoke_cuda_nvfp4.py --source work/Qwen3-NVFP4-Smoke \
  --output work/nvfp4-smoke
```

## Initial GPU validation

The 2026-10-03 development run used the same deterministic, untrained tiny
Qwen3 BF16 checkpoint on both GPUs. Each GPU passed 81 focused regressions,
including real CUDA packing parity, then loaded and generated from both the
BF16 control and AXQuant NVFP4 output in isolated vLLM processes.

| GPU | CUDA | vLLM | NVFP4 W4A16 kernel |
| --- | --- | --- | --- |
| GeForce RTX 5090 | 13.0 | 0.25.1 | MarlinNvFp4LinearKernel |
| Thor | 13.3 | 0.24.0 NVIDIA 26.07 build | MarlinNvFp4LinearKernel |

The generated weight file hashes and token IDs matched across both GPUs.
The final runs had no fused-projection global-scale rescaling warning.
This is synthetic conversion and runtime evidence; trained-model/OCR quality,
performance and W4A4 Tensor Core execution remain unqualified.
