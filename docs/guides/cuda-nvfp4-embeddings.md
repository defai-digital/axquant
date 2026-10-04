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
