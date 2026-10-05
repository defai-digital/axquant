# Native CUDA NVFP4 conversion

This experimental backend adds AXQuant-owned round-to-nearest (RTN) NVFP4
weight quantization. It does not use AWQ. It is separate from the existing
MLX conversion path and requires a build containing the CUDA commands.
Installing the extra does not backport these commands to an older wheel.

CUDA NVFP4 is the supported experimental conversion format. The short-lived
CUDA FP8 converter and publication were withdrawn. Frozen FP8 artifact
definitions remain solely for historical metadata; no standalone FP8
conversion command or FP8 checkpoint is provided. The FP8 encoder is retained
only as an internal building block of the mixed six-bit lane below.

The default format is **NVFP4 W4A16**: selected weights use E2M1 FP4,
activations execute in FP16/BF16 without activation quantization, and protected
weights retain their source precision. vLLM 0.25.1 selects Marlin for this
weight-only mode, including on FP4-capable GPUs. That saves weight storage
and bandwidth without establishing native FP4 matrix execution.

The opt-in **NVFP4 W4A4** path requires source-bound activation calibration.
Selected Linear and MoE inputs use FP4 with dynamic block scales and calibrated
global scales. Protected vision, router, embedding, norm and head tensors
retain source precision. Separate versioned W4A4 plans and manifests preserve
the frozen W4A16 contracts.

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

Keeping one individual expert projection preserves the complete runtime
expert table. Runtime MoE modules cannot combine missing packed projections
with individual BF16 experts under one quantization scheme.

Runtime vision wrappers may rename internal modules, such as `transformer`
to `encoder`. Protected vision/audio namespaces also receive conservative
regex ignore rules so those aliases retain source precision. An ignore
pattern overlapping a selected NVFP4 tensor aborts conversion.

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

## Mixed six-bit budget class (`--q-mode mix6`)

A CUDA "six-bit" lane is a budget class, not a datatype. NVIDIA defines FP8
E4M3/E5M2, MXFP8 and NVFP4 but no six-bit float, and the only six-bit float
standard was retired because no runtime loads it. `--q-mode mix6` therefore
realizes the project's `6bit` class by mixing 4-bit NVFP4 matrices with 8-bit
FP8 E4M3 matrices in one checkpoint, exactly as
[`target_class_for_bpw`](../../src/axquant/naming.py) already labels a mixed
6.0-BPW plan. The standalone FP8 product remains withdrawn and is not
published.

```bash
axquant plan-cuda /path/to/source-bf16 --q-mode mix6 --target-bpw 6.0 \
  --allow-unmeasured --output work/mix-plan.json
axquant convert-cuda /path/to/source-bf16 --plan work/mix-plan.json \
  --device cuda:0 --rows-per-chunk 256 --allow-unmeasured \
  --output /path/to/output-mix6
```

Planning starts from the NVFP4 protection policy and promotes whole allocation
units to FP8, in ascending unit parameter count, until the language-trunk bits
per weight reaches `--target-bpw`. An allocation unit is a complete expert
table, a fused projection, or the whole attention block, so a runtime unit
never mixes packed and source-precision inputs. The attention block moves as
one unit because vLLM builds `qkv_proj`/`o_proj` without a module prefix and
can only match them by class. Promotion order is a documented unmeasured
heuristic, not measured sensitivity. `--target-bpw` defaults to 6.0 and applies
only to `mix6`. The plan records the requested `target_bpw`, the realized
`trunk_bpw` and the estimated weight bytes; the trunk is the attention, MLP and
expert matrices that carry quantized weights.

Both tensor families are written into one staged checkpoint. NVFP4 matrices
keep the layout and shared fused-group scales described above. FP8 matrices
store the original tensor name as E4M3 plus a per-row
`<prefix>.weight_scale` FP32 tensor of shape `(rows, 1)`. `config.json` uses
one `compressed-tensors` config with two groups: `nvfp4`
(`nvfp4-pack-quantized`, block 16 with an E4M3 block scale) and `fp8`
(`float-quantized`, channel weights and dynamic token activations). The group
that carries the attention method uses the single `Linear` class target, and the
other group uses anchored `re:` name targets, because vLLM matches a target by
exact module name (or regex) and the runtime names carry a backend wrapper
prefix. Ignored modules use anchored `re:` targets too. There is no top-level
`format`, because the checkpoint is mixed. Protected vision, router, embedding,
norm, head and MTP tensors stay at source precision exactly as in the NVFP4
path, and the same MTP and vision/audio ignore rules apply.

Output is development evidence. Every mix manifest records
`status=development`, `runtime_verified=false` and `quality_certified=false`,
and the export uses `axquant.cuda-mix-plan.v1` and
`axquant.cuda-mix-pack.v1`. Native vLLM load and generation on the NVIDIA
development hosts is required before any preview is published, and no
certification, accuracy or speed claim is authorized. The two-group `targets`
semantics are validated on that real vLLM GPU host; a CPU `--device cpu` run
only exercises the exporter. Published mixed previews are named
`AX-<Base>-CUDA-AXQ-NVFP4-FP8-6bit` by `naming.cuda_pack_name`: CUDA packs stay
format-qualified, and the mixed lane must name both datatypes and the project
`6bit` budget class, so ad hoc tokens such as `MIX6` are rejected.

## Calibrated W4A4

First create an ordinary source-bound weight plan. Capture BF16 inputs using
the same portable model identity, revision, protection policy and original
checkpoint. The development OCR capture script uses native vLLM model support,
named worker RPCs and primitive arguments; it does not enable callable pickle
serialization or execute model remote code.

```bash
axquant plan-cuda /path/to/source-bf16 --model-id organization/model \
  --allow-unmeasured --output work/weight-plan.json
python scripts/capture_cuda_ocr.py --model /path/to/source-bf16 \
  --plan work/weight-plan.json --image /path/to/calibration-page.png \
  --output work/activations.json
axquant plan-cuda /path/to/source-bf16 --model-id organization/model \
  --activation-bits 4 --activation-calibration work/activations.json \
  --allow-unmeasured --output work/w4a4-plan.json
axquant convert-cuda /path/to/source-bf16 --plan work/w4a4-plan.json \
  --device cuda:0 --allow-unmeasured --output /path/to/output-w4a4
```

The capture script supports official DeepSeek-OCR-2, Unlimited-OCR and dense
Qwen3-VL Instruct layouts. Qwen3-VL uses its tokenizer chat template and
explicit runtime-to-source language aliases; its visual tower stays BF16.
The smoke reserves 512 MiB KV cache for Qwen3-VL and 128 MiB for the OCR models.
 It observes BF16 Linear inputs and replays every source expert on
observed BF16 hidden states, including experts not routed on that page.
Down-projection statistics come from actual source gate/up matrix operations
and SiLU. The calibration records this replay method, image digests, exact
weight-plan hash, shapes, sample counts and maxima. Missing coverage, wrong
shapes, non-finite values or source drift abort. The default 1.25 activation
headroom is a conservative range allowance, not a measured quality result.

W4A4 shares both weight and input global scales across fused Q/K/V, gate/up,
and each complete MoE expert table. It adds one FP32 `input_global_scale`
per selected matrix and writes `axquant.cuda-w4a4-plan.v1` and
`axquant.cuda-w4a4-pack.v1`. Weight packing remains native AXQuant RTN.
A small development calibration page cannot establish broad OCR accuracy.

Use the image-text smoke with chunked prefill enabled and native FP4
requirements. It inspects actual worker kernels, requires native CUTLASS
Linear, and rejects Marlin, Humming or emulation for quantized MoE. The MoE
selector stays automatic so preserved BF16 expert tables can use their
supported backend. Actual Linear and expert-table counts must match every
fused runtime unit implied by the source allocation plan. It rejects repeated nonblank output lines even when
the three expected test-page lines are present:

```bash
python scripts/smoke_cuda_ocr.py --model /path/to/output-w4a4 \
  --image /path/to/ocr-smoke-page.png --output work/runtime.json \
  --require-native-fp4 --memory-fraction 0.30
```

Both scripts require a build containing these changes and a compatible CUDA
vLLM environment. They are development/operator scripts, not features of
older published AXQuant wheels. Actual GPU/runtime/model compatibility must
be established by load and generation checks.

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
This synthetic run alone does not establish trained-model OCR quality,
performance or W4A4 Tensor Core execution.

Subsequent trained-checkpoint development checks exercised DeepSeek-OCR-2 and
Unlimited-OCR W4A4 packs on both RTX 5090 and Thor, using official vLLM 0.25.1,
PyTorch 2.11.0+cu130 and CUDA 13.0. Every case loaded with
`CutlassNvFp4LinearKernel` and `VLLM_CUTLASS` MoE, enabled chunked prefill,
and recognized all three expected English test-page lines. Exact protected
tensor equality and identical checkpoint hashes across test hosts were checked.
Calibration and smoke used the same generated page; this is not a held-out
OCR evaluation, layout/markup qualification or a speed certification.


Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct are dense, non-MTP sources.
NVFP4 conversion does not add trained MTP heads. The complete visual stack,
embeddings, normalization tensors and LM head retain original precision.
JSON chat templates are checksum-bound source assets and are copied into
the export alongside tokenizer and processor files. Other unpromoted
Qwen3-VL variants remain inventory-only.

Keeping fused text projections emits both source and fused-runtime ignore
entries. Keeping one expert projection protects the complete expert table.
A fully quantized Unlimited-OCR Thor development smoke exposed repeated
prefix lines despite recognizing the expected text. The stricter smoke
rejects that candidate; subsequent conversion uses explicit front-MLP
protection and must pass the same checks before publication.

For Unlimited-OCR on vLLM 0.25.1, MHA attention Linears omit module prefixes,
so per-layer BF16 attention protection cannot be addressed by the quantization
config. The development smoke rejects that layout before loading. Use
MLP-only protection when keeping front layers with this pinned runtime.

## Published native W4A4 development checkpoints

The following exact-checkpoint previews include calibrated W4A4 plans,
protected-tensor equality checks, pinned sources, public checksums and
RTX 5090 plus Thor native FP4 runtime evidence:

- [Qwen3-VL-4B-Instruct](https://huggingface.co/AutomatosX/AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/AutomatosX/AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4)
- [DeepSeek-OCR-2](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4)
- [Unlimited-OCR](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4)

Qwen3-VL has no native MTP. These previews are not quality or speed
certifications. Unlimited-OCR still produced extra prefix text on the
single development page; expected-line and repetition checks do not qualify
clean extraction or layout markup. Consult each model card and its exact
runtime output before treating the pack as an application baseline.

For Qwen3-Embedding and Nemotron-3-Embed exports, see the
[CUDA NVFP4 embedding guide](cuda-nvfp4-embeddings.md). It documents preserved
pooling semantics, BF16 protection and the scope of vector-drift checks.
