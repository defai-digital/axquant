---
license: apache-2.0
library_name: mlx
base_model: Cloudflare/clef
base_model_relation: quantized

tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Clef-27B-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [Cloudflare/clef](https://huggingface.co/Cloudflare/clef/tree/0b331204bb13fbd2ca93a64df1956f5b55478ce5) |
| Immutable source revision | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| Complete storage | 26,484,659,432 bytes |
| Complete measured BPW | 7.708893 |
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
  "task": "decision",
  "runtime": "mlx-lm",
  "version": "0.32.2",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "answers": {
        "department": {
          "choice": "technical",
          "confidence": 0.7495,
          "probabilities": {
            "billing": 0.2505,
            "technical": 0.7495
          },
          "type": "choice"
        },
        "outage": {
          "noul": 0.7585,
          "type": "noul"
        },
        "urgency": {
          "confidence": 0.9546,
          "legend": {
            "0": "Can wait",
            "1": "Today"
          },
          "probabilities": {
            "0": 0.0454,
            "1": 0.9546
          },
          "score": 0.9546,
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
hf download AutomatosX/AX-Clef-27B-MLX-AXQ-6bit --local-dir ./AX-Clef-27B-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use the included `clef_mlx.load` and `model.systemone` API for decision heads.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
