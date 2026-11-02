---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3-VL-32B-Thinking
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Qwen/Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking/tree/7edd10ffd1196091948fb245ff63e406ccb2d4d1) |
| Immutable source revision | `7edd10ffd1196091948fb245ff63e406ccb2d4d1` |
| Complete weight storage | 30,938,516,360 bytes |
| Complete measured BPW | 7.419889 |
| Manifest budget class | `7p4bpw` |
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
  "task": "vision",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "AXQ 7319",
      "final_answer": "AXQ 7319",
      "output": "So, let's look at the image. The text there is \"AXQ 7319\". I need to read all the text and output only that. Let's check each character: A, X, Q, space, 7, 3, 1, 9. Yep, that's it. So the output should be \"AXQ 7319\".\n</think>\n\nAXQ 7319",
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
hf download AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit --local-dir ./AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
