---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3-VL-4B-Instruct
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Qwen3-VL-4B-Instruct-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| Immutable source revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Complete storage | 4,448,622,111 bytes |
| Complete measured BPW | 8.019480 |
| Manifest budget class | `8bit` |
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
hf download AutomatosX/AX-Qwen3-VL-4B-Instruct-MLX-AXQ-6bit --local-dir ./AX-Qwen3-VL-4B-Instruct-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
