---
license: apache-2.0
library_name: mlx
base_model: mlx-community/Qwen3.5-0.8B-decision
base_model_relation: quantized

tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development
- mtp
---

# AX-Qwen3.5-0.8B-decision-MLX-AXQ-6bit-MTP

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [mlx-community/Qwen3.5-0.8B-decision](https://huggingface.co/mlx-community/Qwen3.5-0.8B-decision/tree/36e166ca10c4d16471a231d66cfc22d80039fe34) |
| Immutable source revision | `36e166ca10c4d16471a231d66cfc22d80039fe34` |
| Complete storage | 1,019,917,465 bytes |
| Complete measured BPW | 9.058246 |
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
  "task": "decision",
  "runtime": "mlx-lm",
  "version": "0.32.2",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "answers": {
        "department": {
          "choice": "technical",
          "confidence": 0.5312,
          "probabilities": {
            "billing": 0.4688,
            "technical": 0.5312
          },
          "type": "choice"
        },
        "outage": {
          "noul": 0.4378,
          "type": "noul"
        },
        "urgency": {
          "confidence": 0.8176,
          "legend": {
            "0": "Can wait",
            "1": "Today"
          },
          "probabilities": {
            "0": 0.1824,
            "1": 0.8176
          },
          "score": 0.8176,
          "type": "score"
        }
      },
      "passed": true
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
hf download AutomatosX/AX-Qwen3.5-0.8B-decision-MLX-AXQ-6bit-MTP --local-dir ./AX-Qwen3.5-0.8B-decision-MLX-AXQ-6bit-MTP
```

Pin the immutable Hub revision. Use the included `clef_mlx.load` and `model.systemone` API for decision heads.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
