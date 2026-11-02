---
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
library_name: mlx
base_model: ukisai/Swift-1.5-Qwen3.8-27b
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

# AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-6bit-MTP

**Research development artifact, not certified.** Affine six-bit base packing with
higher-precision protection; this is not MXFP6 or a uniformly six-bit checkpoint.

Converted from the pinned original BF16 checkpoint, with protected higher-precision tensors.

| Property | Value |
| --- | --- |
| Source | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/tree/b4c84d42903a8646b25857eb2827288d92ed85a4) |
| Immutable source revision | `b4c84d42903a8646b25857eb2827288d92ed85a4` |
| Manifest budget class | `8bit` |
| Complete measured storage | **7.800057 BPW** |
| Weight storage | 27,087,090,027 bytes / 25.2268 GiB |
| MTP | Protected payload included; execution, acceptance and acceleration unverified |
| Quality | No quality benchmark or certificate for this revision |

## Precision distribution

The following shares are logical parameters, not file-byte shares.

| Precision | Parameter share | Parameters |
| --- | ---: | ---: |
| 6bit | 87.2070% | 24,227,348,480 |
| 8bit | 5.0199% | 1,394,606,080 |
| bf16 | 7.7731% | 2,159,473,392 |

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
hf download AutomatosX/AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-6bit-MTP --local-dir ./AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-6bit-MTP
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

The Swift contribution follows the [Swift Open License v1.0](LICENSE). The source Apache-2.0 license and notices are retained in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [NOTICE](NOTICE). The Swift license does not license commercial use by a legal entity with US$1,000,000 or more gross revenue in its most recent fiscal year. Upstream attribution: UkisAI / Swift.
