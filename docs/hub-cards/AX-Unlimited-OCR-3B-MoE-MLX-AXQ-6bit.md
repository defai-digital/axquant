---
license: mit
library_name: mlx
base_model: baidu/Unlimited-OCR
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Unlimited-OCR-3B-MoE-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR/tree/07dea832e22aefee32ad281d4b80551282e1c168) |
| Immutable source revision | `07dea832e22aefee32ad281d4b80551282e1c168` |
| Complete weight storage | 3,603,015,207 bytes |
| Complete measured BPW | 8.640049 |
| Manifest budget class | `8p6bpw` |
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
  "task": "ocr",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "7319",
      "final_answer": "<|det|>title [54, 294, 648, 580]<|/det|>AXQ 7319",
      "output": "<|det|>title [54, 294, 648, 580]<|/det|>AXQ 7319",
      "passed": true,
      "prompt": "<|grounding|>Convert the document to markdown."
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
hf download AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-6bit --local-dir ./AX-Unlimited-OCR-3B-MoE-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the architecture-aware runtime above and preserve the source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
