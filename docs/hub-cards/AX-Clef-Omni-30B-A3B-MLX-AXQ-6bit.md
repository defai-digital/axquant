---
license: apache-2.0
library_name: mlx
base_model: Cloudflare/clef-omni
base_model_relation: quantized

tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Clef-Omni-30B-A3B-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni/tree/0db1cd2607d76a7bdb2a382f659e7b313079f84b) |
| Immutable source revision | `0db1cd2607d76a7bdb2a382f659e7b313079f84b` |
| Complete weight storage | 33,142,672,606 bytes |
| Complete measured BPW | 7.496439 |
| Manifest budget class | `7p5bpw` |
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
  "task": "decision",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "answers": {
        "department": {
          "choice": "billing",
          "confidence": 0.7116,
          "probabilities": {
            "billing": 0.7116,
            "technical": 0.2884
          },
          "type": "choice"
        },
        "outage": {
          "noul": 0.9279,
          "type": "noul"
        },
        "urgency": {
          "confidence": 0.8648,
          "legend": {
            "0": "Can wait",
            "1": "Today"
          },
          "probabilities": {
            "0": 0.1352,
            "1": 0.8648
          },
          "score": 0.8648,
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
hf download AutomatosX/AX-Clef-Omni-30B-A3B-MLX-AXQ-6bit --local-dir ./AX-Clef-Omni-30B-A3B-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the included `clef_mlx.load` and `model.systemone` API for decision heads.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
