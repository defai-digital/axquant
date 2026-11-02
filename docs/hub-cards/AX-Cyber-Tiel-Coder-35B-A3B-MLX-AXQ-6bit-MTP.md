---
license: mit
library_name: mlx
base_model: peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- axquant
- axq
- 6-bit
- mixed-precision
- mtp
- development
---

# AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP

**Research development artifact, not certified.** Affine six-bit base packing with
higher-precision protection; this is not MXFP6 or a uniformly six-bit checkpoint.

Re-quantized from an already quantized oQ6e checkpoint. The source Sharp chat template and protected MTP payload are retained. This conversion does not recover BF16 information already lost in the source, reproduce its calibration process, or inherit upstream GGUF, SWE-bench, quality or MTP-speed results.

| Property | Value |
| --- | --- |
| Source | [peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP/tree/a443d4e30fd5228942cb7695f916f4b14a88fae9) |
| Immutable source revision | `a443d4e30fd5228942cb7695f916f4b14a88fae9` |
| Manifest budget class | `7p5bpw` |
| Complete measured storage | **7.490084 BPW** |
| Weight storage | 33,660,273,640 bytes / 31.3486 GiB |
| MTP | Protected payload included; execution, acceptance and acceleration unverified |
| Quality | No quality benchmark or certificate for this revision |

## Precision distribution

The following shares are logical parameters, not file-byte shares.

| Precision | Parameter share | Parameters |
| --- | ---: | ---: |
| 6bit | 93.0325% | 33,446,887,424 |
| 8bit | 1.9582% | 704,004,096 |
| bf16 | 5.0093% | 1,800,931,184 |

Group sizes and per-module overrides are recorded in `config.json` and
`axquant_plan.json`. Weight bytes do not establish peak runtime memory requirements.
All advertised BPW values include protection and storage overhead.

## Runtime evidence and limits

MLX-VLM 0.7.9 / MLX 0.32.4 passed two short text smokes on `Device(gpu, 0)`: arithmetic produced `42` and exact-word output produced `READY`. The text smoke used the public MLX-VLM `load` and `generate` APIs on Metal.

Short text checks, when successful, establish only scoped text load/generation. They do not establish
coding quality, long-context behavior, vision/video quality, MTP execution or speed.
Vision and MTP weights are present; neither feature is behavior-verified by this release.
The `-MTP` suffix describes packaged weights, not a measured acceleration claim.
The tested text loading path disables the unindexed vision tower. Image inference is not supported by that tested path for this revision; the retained vision sidecar is not a verified vision runtime export.

| Runtime | Status for this artifact |
| --- | --- |
| MLX-VLM | Scoped text smoke as described above |
| MLX-LM | Not tested; do not infer equivalence from successful loading |
| AX Engine | Not measured; an included native manifest is not execution evidence |
| omlx | unsupported; no load, generation or MTP claim for this artifact |
| mtplx | unsupported; no load, generation or MTP claim for this artifact |

See the manifest-bound [runtime compatibility record](axquant_compatibility.json).
Any runtime-specific index, norm or n-gram layout export is a distinct artifact and
needs its own load/generation validation. No peer variant is silently substituted here.

## Download and evaluation

```bash
python -m pip install -U huggingface_hub mlx-vlm
hf download AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP --local-dir ./AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP
```

Pin the resulting immutable Hub commit for reproducibility. Use a runtime build
that supports the exact source architecture and declared physical quantization.
Do not assume a particular memory capacity, modality or MTP setting is supported.

## Provenance

The manifest binds model files, precision plan, quantizer execution and runtime
compatibility checks. Protected MTP provenance is in
`axquant_mtp_sidecar_manifest.json`. Source certificates, upstream benchmark plates,
and measurements from other AXQ models do not transfer to this checkpoint.

## License and attribution

The pinned upstream repository declares MIT in its model-card metadata. Retain the upstream attribution and consult its [model card](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP/tree/a443d4e30fd5228942cb7695f916f4b14a88fae9) for provenance and terms. Upstream attribution: peculiar-ragdoll and ornith-ai; Cyber Tiel also credits huihui-ai.
