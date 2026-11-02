---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
base_model: Qwen/Qwen3.8-Flash-Next
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

# AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP

**Research development artifact, not certified.** Affine six-bit base packing with
higher-precision protection; this is not MXFP6 or a uniformly six-bit checkpoint.

Converted from the pinned original BF16 checkpoint, with protected higher-precision tensors.

| Property | Value |
| --- | --- |
| Source | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) |
| Immutable source revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Manifest budget class | `8bit` |
| Complete measured storage | **7.423369 BPW** |
| Weight storage | 167,025,781,506 bytes / 155.5549 GiB |
| MTP | Protected payload included; execution, acceptance and acceleration unverified |
| Quality | No quality benchmark or certificate for this revision |

## Precision distribution

The following shares are logical parameters, not file-byte shares.

| Precision | Parameter share | Parameters |
| --- | ---: | ---: |
| 6bit | 68.7413% | 123,734,261,760 |
| 8bit | 29.2041% | 52,567,449,600 |
| bf16 | 2.0546% | 3,698,270,099 |

Group sizes and per-module overrides are recorded in `config.json` and
`axquant_plan.json`. Weight bytes do not establish peak runtime memory requirements.
All advertised BPW values include protection and storage overhead.

## Runtime evidence and limits

No successful load/generation smoke was completed for this exact artifact. MLX-VLM 0.7.9 / MLX 0.32.4 failed the high-level CPU wrapper's Metal wired-memory query. A second attempt using the public lazy-load and generate_step APIs with CPU requested reached an internal Gated DeltaNet Metal operation and failed with a GPU command-buffer timeout. This is a checksum-verified research weight release, not a working-inference claim. A suitable architecture-aware runtime and memory/offload configuration still require verification. Do not use these failed attempts as CPU or GPU support evidence.

Short text checks, when successful, establish only scoped text load/generation. They do not establish
coding quality, long-context behavior, vision/video quality, MTP execution or speed.
Vision and MTP weights are present; neither feature is behavior-verified by this release.
The `-MTP` suffix describes packaged weights, not a measured acceleration claim.


| Runtime | Status for this artifact |
| --- | --- |
| MLX-VLM | Smoke failed; inference unverified |
| MLX-LM | Not tested; do not infer equivalence from successful loading |
| AX Engine | Not measured; an included native manifest is not execution evidence |
| omlx | requires-export; no load, generation or MTP claim for this artifact |
| mtplx | requires-export; no load, generation or MTP claim for this artifact |

See the manifest-bound [runtime compatibility record](axquant_compatibility.json).
Any runtime-specific index, norm or n-gram layout export is a distinct artifact and
needs its own load/generation validation. No peer variant is silently substituted here.

## Download and evaluation

```bash
python -m pip install -U huggingface_hub mlx-vlm
hf download AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP --local-dir ./AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP
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

The complete Qwen Community License 1.0 and its notices are included in [LICENSE](LICENSE). This research development release does not claim a separate Qwen commercial license. Consult the original terms for downstream use.
