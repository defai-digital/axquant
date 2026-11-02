---
license: apache-2.0
library_name: mlx
base_model: google/gemma-4-31B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development
- mtp
---

# AX-gemma-4-31b-MLX-AXQ-6bit-MTP

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475) |
| Immutable source revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Complete weight storage | 29,318,550,922 bytes |
| Complete measured BPW | 7.389072 |
| Manifest budget class | `7p3bpw` |
| MTP | Exact-pair floating external assistant included; execution and speed unverified. |

The precision distribution and exact per-module settings are recorded in
`axquant_manifest.json`, `axquant_plan.json`, and `config.json`. Complete BPW
includes protected tensors and storage overhead. File size does not establish
peak runtime memory requirements. Converted from the pinned original source with existing precision protection floors.
The external assistant comes from `mlx-community/gemma-4-31B-it-assistant-bf16@28e92270316e89288579ec59c17939541d9ca433`. Complete storage/BPW above includes the assistant. The base artifact manifest accounts for target weights; `ax_composite_pack_manifest.json` additionally binds the separate assistant and exact-pair contract.

## Observed runtime evidence

The exact runtime payload passed the following scoped check on Metal. The receipt
binds its runtime input files. This is a functional smoke, not a quality benchmark,
MTEB evaluation, coding evaluation, long-context test or MTP acceleration result.

```json
{
  "task": "vision",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "AXQ 7319",
      "final_answer": "AXQ 7319",
      "output": "AXQ 7319",
      "passed": true,
      "prompt": "Read all the text in the image. Output only the text."
    }
  ]
}
```

| Runtime | Evidence for this artifact |
| --- | --- |
| MLX-VLM 0.7.9 | Passed the task-specific smoke above |
| MLX-LM | Not separately tested |
| AX Engine | Not tested; architecture metadata is not execution evidence |
| omlx | unsupported (static only) |
| mtplx | unsupported (static only) |

Unexercised modalities and MTP behavior remain unverified. Peer runtime-specific
exports require distinct variants and their own testing. See
[the manifest-bound static report](axquant_compatibility.json).

## Use and provenance

```bash
python -m pip install -U huggingface_hub mlx-vlm
hf download AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP --local-dir ./AX-gemma-4-31b-MLX-AXQ-6bit-MTP
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
