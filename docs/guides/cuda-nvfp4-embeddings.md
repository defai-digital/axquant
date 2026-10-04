# CUDA NVFP4 embedding development previews

AXQuant exports the original Qwen3-Embedding and Nemotron-3-Embed BF16
retrieval checkpoints as CUDA compressed-tensors NVFP4 W4A4 packs, without
AWQ. This experimental CUDA path is separate from the MLX product and is
not available in the certified `axquant[mlx]==1.8.1` wheel.

## Preserve embedding behavior

The exporter checksum-binds and retains `modules.json`,
`config_sentence_transformers.json`, `sentence_bert_config.json`, and the
admitted pooling/normalization configuration paths. Embeddings and norms
retain their source precision. Source and output manifests bind these
files, including the nested `1_Pooling/config.json`.

Qwen3 uses its saved query instruction, last-token pooling, one final `<|endoftext|>` token,
and L2 normalization. Nemotron-3-Embed uses saved `query: ` / `passage: `
prefixes, bidirectional attention, mean pooling, and L2 normalization. The
pinned vLLM runner explicitly maps `Ministral3Model` to its native
`Ministral3ForCausalLM` implementation with embedding conversion while
retaining `is_causal=false`; it verifies every actual attention layer as
encoder-only. A causal model declaration does not override that check.

For the factory development build, Qwen3's tested mixed precision policy is
available through `--embedding-protection`:

```bash
axquant plan-cuda /path/to/original-bf16 \
  --model-id Qwen/Qwen3-Embedding-4B --revision SOURCE_COMMIT \
  --embedding-protection --allow-unmeasured --output weight-plan.json
```

This preserves all attention projections plus the first and last two MLP
blocks in source precision. Remaining MLP matrices use native NVFP4 W4A4.
The exact tensor globs are stored in the plan, and runtime ignores cover
both flat source keys and vLLM's wrapper/fused paths. The option rejects
non-Qwen3 or unsupported pooling layouts. It does not imply measured
weight sensitivity: RTN remains explicitly unmeasured.

## Calibration and validation

`capture_cuda_embedding.py` observes BF16 inputs for every selected matrix
using named public vLLM worker RPCs. Source checksums, input column counts,
sample counts and the calibration corpus digest are bound to the complete
capture. Changing the precision policy requires a new capture for that
plan. Retrieval smoke text is excluded from activation collection.

`smoke_cuda_embedding.py` verifies actual CUTLASS FP4 kernels against all
selected fused runtime units, exact attention layer counts/types, finite
full-dimension normalized vectors, and four paired retrieval ranks. Its
optional `--reference` requires a BF16 result from the original raw source
config and the identical retrieval corpus. Development limits are minimum
vector cosine 0.90 and mean cosine 0.95 on eight vectors.

The published evidence uses a small calibration-disjoint corpus also used
for candidate selection. This is development evidence, not independent
retrieval-quality, long-context, concurrency or speed certification. It
contains no MTP or generative capability claim. Rebuild retrieval indexes
with the exact quantized checkpoint; vectors from different checkpoints
are not interchangeable.

The tested recipe uses vLLM 0.25.1, eager execution, full-sequence prefill,
512-token maximum context, one sequence and no remote model code. Full
prefill is required for the tested bidirectional mean-pooling path. Memory
fractions in each model card apply to the recorded development recipe;
production document lengths and concurrency need separate sizing.

## Published exact-checkpoint previews

Each preview includes the original immutable source revision, license/notices,
checksums, calibrated plan, protected-tensor checks and both GPU runtime records.
Every selected runtime module uses `CutlassNvFp4LinearKernel`. All four paired
retrieval queries rank their matching document first on both GPUs.

| Checkpoint | Native FP4 Linear modules | RTX 5090 mean cosine | Thor mean cosine |
| --- | --- | --- | --- |
| [Qwen3-Embedding-0.6B](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-CUDA-AXQ-NVFP4-W4A4/tree/f3b332dcf8f95b9cb509258407e89a7b313eb5d8) | 48 | 0.969200 | 0.968076 |
| [Qwen3-Embedding-4B](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-CUDA-AXQ-NVFP4-W4A4/tree/760de1ca6c8c521296a38ddaab10dc26ef0d5f6a) | 64 | 0.982176 | 0.982622 |
| [Qwen3-Embedding-8B](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-CUDA-AXQ-NVFP4-W4A4/tree/f93bb79d3b6f308750e97e129176a56622af519f) | 64 | 0.986226 | 0.986426 |
| [Nemotron-3-Embed-1B](https://huggingface.co/AutomatosX/AX-Nemotron-3-Embed-1B-CUDA-AXQ-NVFP4-W4A4/tree/68817d7b214f14f3574c5ff3630d325987941256) | 64 | 0.988085 | 0.988092 |
| [Nemotron-3-Embed-8B](https://huggingface.co/AutomatosX/AX-Nemotron-3-Embed-8B-CUDA-AXQ-NVFP4-W4A4/tree/952869891397bf0eaa79ed27fb0f088830e8c147) | 136 | 0.983442 | 0.983408 |

Cosine values compare eight full-dimension vectors with the original BF16
checkpoint on each GPU. The corpus and candidate-selection limitations above
apply. Download the entire repository and use its included `examples/embedding_smoke.py`
with the GPU-specific BF16 reference. Each card records the pinned vLLM image,
precision policy, memory recipe and exact development evidence.

The Hub collection tool keeps CUDA packs in the CUDA and NVFP4 catalogs and
retrieval packs in Embeddings plus their family collection. The MLX catalog
contains only MLX packs. The live-model drift gate checks the union of the
runtime catalogs before a full collection apply.
