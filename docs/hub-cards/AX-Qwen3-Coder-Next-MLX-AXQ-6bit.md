---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3-Coder-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Qwen3-Coder-Next-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb) |
| Immutable source revision | `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| Complete storage | 70,213,282,892 bytes |
| Complete measured BPW | 7.050023 |
| Manifest budget class | `7p0bpw` |
| MTP | No MTP payload is advertised. |

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
  "runtime": "mlx-lm",
  "version": "0.32.2",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "42",
      "final_answer": "42",
      "output": "42",
      "passed": true,
      "prompt": "What is 19 plus 23? Reply with only the number."
    },
    {
      "expected": "READY",
      "final_answer": "READY",
      "output": "READY",
      "passed": true,
      "prompt": "Reply with exactly the word READY."
    }
  ]
}
```

| Runtime | Evidence for this artifact |
| --- | --- |
| MLX-LM 0.32.2 | Passed the task-specific smoke above |
| MLX-VLM | Not separately tested |
| AX Engine | Not tested; architecture metadata is not execution evidence |
| omlx | unsupported (static only) |
| mtplx | unsupported (static only) |

Unexercised modalities and MTP behavior remain unverified. Peer runtime-specific
exports require distinct variants and their own testing. See
[the manifest-bound static report](axquant_compatibility.json).

## Use and provenance

```bash
python -m pip install -U huggingface_hub mlx-vlm
hf download AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit --local-dir ./AX-Qwen3-Coder-Next-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
