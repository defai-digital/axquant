---
license: "other"
license_name: "openmdw-1.1"
license_link: "https://openmdw.ai/license/1-1/"
library_name: mlx
base_model: nvidia/Nemotron-3-Embed-1B-BF16
base_model_relation: quantized
pipeline_tag: feature-extraction
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/tree/c0c9fea93ea424587517f2c59e20db9f1d6bf615) |
| Immutable source revision | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| Complete weight storage | 1,066,578,565 bytes |
| Complete measured BPW | 7.478738 |
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
  "task": "embedding",
  "runtime": "mlx-vlm",
  "version": "0.7.9",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "finite": true,
      "mars_ranks_above_venus": true,
      "min_cosine_vs_source": 0.9979478120803833,
      "passed": true,
      "shape": [
        3,
        2048
      ],
      "unit_norm": true
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
hf download AutomatosX/AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit --local-dir ./AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use `mlx_vlm.embedding_loader.load_embedding_model` for embeddings, preserving source pooling and task prefixes.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
