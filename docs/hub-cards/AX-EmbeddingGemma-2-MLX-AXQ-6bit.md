---
license: apache-2.0
library_name: mlx
base_model: google/embeddinggemma-2
base_model_relation: quantized
pipeline_tag: feature-extraction
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-EmbeddingGemma-2-MLX-AXQ-6bit

Research development artifact. Base affine6/group32 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) |
| Immutable source revision | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Complete storage | 1,223,744,259 bytes |
| Complete measured BPW | 13.151965 |
| Manifest budget class | `13p2bpw` |
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
      "min_cosine_vs_source": 0.9990512132644653,
      "passed": true,
      "shape": [
        3,
        768
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
hf download AutomatosX/AX-EmbeddingGemma-2-MLX-AXQ-6bit --local-dir ./AX-EmbeddingGemma-2-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Use `mlx_vlm.embedding_loader.load_embedding_model`
for embedding packs, preserving the source pooling and task prefixes. Other
packs require the architecture-aware MLX-VLM loader and source processor/template.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
