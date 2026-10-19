---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- axquant
- mlx
- mxfp8
- mtp
---

# AX-Qwen3.5-9B-MLX-AXQ-MXFP8-MTP

AXQuant MXFP8 development conversion of the official Qwen3.5-9B BF16
checkpoint. This is a development artifact, with no Tier 1, Tier 2, quality,
MTP acceptance, or speed certificate.

## Source and storage

- Source: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a).
- Immutable source revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- AXQuant version: `2.3.0`.
- Main-model storage: **9.4643 BPW**, measured from serialized files.
- Total storage including MTP: **9.6290 BPW**.
- Safetensors download: **11.619 GB**.
- Eligible language attention and MLP projections use native MLX MXFP8, group32.
- Vision, MTP, norms, and the output head retain BF16 precision.
- Embeddings use MXFP8 group32 packing.
- Manual planning is unmeasured architecture-prior evidence. MXFP8 is explicit
  unmeasured physical repacking, not a measured sensitivity method.

BPW includes protection floors and metadata overhead; the format name is not
a claim that every tensor has the same precision. Source weights were not
trained or augmented, and no competitor quantization data was used.

## Runtime evidence

The exact payload passed bounded MLX-LM text generation on the factory Mac Studio M2 Ultra. Synthetic image generation was attempted but the standard MLX-VLM loader did not load the protected, unindexed vision sidecar. See `runtime_smoke.json` for the dependency versions, answers and payload digests.

Test environment: MLX `0.32.1`, MLX-LM `0.31.3`,
MLX-VLM `0.6.17`. Smoke results establish only the tested
load/generation cases. Vision generation is not supported by this export through the default MLX-VLM loader.
The BF16 vision payload is retained separately for a future explicit runtime export.
Broad text, image, video, tool-use, long-context, and
task-quality evaluations have not been performed.

`mtp.safetensors` contains the protected native MTP payload. Its presence
does not establish MTP execution, acceptance, exactness, or acceleration.
MTP on/off execution has not been verified for this artifact. The ordinary
MLX-LM example below uses autoregressive text generation.

`axquant_compatibility.json` is manifest-bound static evidence. Its unsupported
oMLX/MTPLX profiles remain unsupported; this publication does not promote
runtime compatibility or claim native AX Engine qualification.

## Text generation

Install MLX-LM on Apple Silicon, then load the complete repository:

```python
from mlx_lm import load, generate

model, tokenizer = load("AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-MXFP8-MTP")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 6 times 7?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=64))
```

Pin the immutable Hub revision for reproducible deployments. Actual memory
use depends on model loading, context, vision input, and runtime settings.

## Provenance and license

`axquant_manifest.json` binds output files, source identity, the sanitized
plan, quantizer execution, protected sidecars, and static compatibility.
The original Apache-2.0 license is included in `LICENSE`; `NOTICE` identifies
the quantization changes. Existing historical certificates do not bind this
new export and are not reused.

Selected for the [AX BI collection](https://huggingface.co/collections/AutomatosX/ax-bi-6ad54be40f8f1d5415a8d15a)
and [AX Work collection](https://huggingface.co/collections/AutomatosX/ax-work-6ad54c25c37f597f21344945).
