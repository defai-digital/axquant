---
license: apache-2.0
library_name: mlx
base_model: tencent/Hy-MT2-30B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development

---

# AX-Hy-MT2-30B-A3B-MLX-AXQ-6bit

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B/tree/d3ead4dba61c09aac60a261a96ad1df3e705febb) |
| Immutable source revision | `d3ead4dba61c09aac60a261a96ad1df3e705febb` |
| Complete storage | 26,682,660,512 bytes |
| Complete measured BPW | 7.100058 |
| Manifest budget class | `7p1bpw` |
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
  "task": "translation",
  "runtime": "mlx-lm",
  "version": "0.32.2",
  "mlx_version": "0.32.4",
  "cases": [
    {
      "expected": "Hello world",
      "output": "Hello, world.",
      "passed": true,
      "prompt": "Translate the following segment into English, without additional explanation.\n\nBonjour le monde."
    }
  ]
}
```

| Runtime | Evidence for this artifact |
| --- | --- |
| MLX-LM 0.32.2 + bundled Hy-V3 adapter | Passed the task-specific smoke above |
| MLX-VLM | Unsupported hy_v3 architecture in this build |
| AX Engine | Not tested; architecture metadata is not execution evidence |
| omlx | unsupported (static only) |
| mtplx | unsupported (static only) |

Unexercised modalities and MTP behavior remain unverified. Peer runtime-specific
exports require distinct variants and their own testing. See
[the manifest-bound static report](axquant_compatibility.json).

## Use and provenance

```bash
python -m pip install -U huggingface_hub mlx-vlm
hf download AutomatosX/AX-Hy-MT2-30B-A3B-MLX-AXQ-6bit --local-dir ./AX-Hy-MT2-30B-A3B-MLX-AXQ-6bit
```

Pin the immutable Hub revision. Hy-MT2 requires the included MIT-licensed `hy_v3_mlx.py` adapter. After downloading, import that module and call `register_hy_v3_model()` before `mlx_lm.load`; unmodified upstream MLX-LM/VLM does not provide hy_v3. The adapter license is retained in `AXQuant-Adapter-LICENSE`.
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
