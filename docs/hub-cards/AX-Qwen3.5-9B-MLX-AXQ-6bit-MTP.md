---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development
- mtp
---

# AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) |
| Immutable source revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Complete weight storage | 10,642,707,736 bytes |
| Complete measured BPW | 8.820133 |
| Manifest budget class | `8p8bpw` |
| MTP | Protected MTP weights included; execution, acceptance and speed unverified. |

The precision distribution and exact per-module settings are recorded in
`axquant_manifest.json`, `axquant_plan.json`, and `config.json`. Complete BPW
includes protected tensors and storage overhead. File size does not establish
peak runtime memory requirements. Converted from the pinned original source with existing precision protection floors.


## Observed runtime evidence

The exact runtime payload passed the following scoped check on Metal. The receipt
binds its runtime input files. This is a functional smoke, not a quality benchmark,
MTEB evaluation, coding evaluation, long-context test or MTP acceleration result.

```json
{
  "task": "text",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "42",
      "output": "42",
      "passed": true,
      "prompt": "What is 19 plus 23? Reply with only the number."
    },
    {
      "expected": "READY",
      "output": "READY",
      "passed": true,
      "prompt": "Reply with exactly the word READY."
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
hf download AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP --local-dir ./AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template. The tested default text loading path omits the unindexed vision sidecar. Image inference is unsupported by that path for this revision and requires a separately verified vision loader or export.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
