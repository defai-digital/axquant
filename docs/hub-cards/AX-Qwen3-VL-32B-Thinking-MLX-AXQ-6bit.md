---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3-VL-32B-Thinking
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- qwen3-vl
- 6bit
- 6-bit
- vision
---

# AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit — 6.94 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.


## Model details

| Property | Value |
| --- | --- |
| Base model | [Qwen/Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking/tree/7edd10ffd1196091948fb245ff63e406ccb2d4d1) |
| Source revision | `7edd10ffd1196091948fb245ff63e406ccb2d4d1` |
| Product family | `qwen3-vl` |
| Source architecture | `Qwen3VLForConditionalGeneration` (dense); text path optimized |
| Main-model parameters | 33.36B logical parameters |
| Quantizer | AXQuant `1.8.1` |
| Hub budget class | `6bit` |
| AXQuant base precision class | `8bit` |
| Planned storage-adjusted BPW | 6.9379 |
| Measured main-model BPW | 6.9380 |
| Measured total BPW | **6.9380** |
| Safetensors weight size | 28.93 GB |
| Approximate complete download | 28.94 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-VLM |
| AX Engine native execution | Not established; no validated native manifest is included |
| MTP present | `False` |
| Vision present | `True` |
| Audio present | `False` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

AXQ names describe a **storage-budget product class**, not one uniform precision applied to every
tensor. Protected tensors remain at higher precision, so the exact measured BPW is authoritative.
In particular, a `6bit`-named mixed plan may retain `4bit` as its base precision while selecting
6-bit, 8-bit, or BF16 for other tensors to meet an approximately 6-BPW total budget. Protection
floors can also raise a `4bit`-named pack close to (or above) a `6bit` budget on small or heavily
protected models. When that collapse happens, AutomatosX does **not** publish a separate
misleading `4bit` sibling for that base.


| Sibling | Intended trade-off |
| --- | --- |
| [4bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-4bit) | Lower-storage AXQ budget; check its exact BPW |
| [6bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit) | Higher average precision near the 6-BPW budget |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit --local-dir ./AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit
```

Allow at least 28.94 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-VLM

```bash
python -m pip install -U mlx-vlm
python -m mlx_vlm.generate \
  --model AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit \
  --image ./image.png \
  --prompt "Describe this image." \
  --max-tokens 128 \
  --temperature 0.0
```

The protected vision tower and AXQ language decoder are loaded together by MLX-VLM. The artifact
records MLX `0.32.0`; runtime QA is reported separately from model-quality claims.

## AX Engine status

This package does **not** include a validated native `model-manifest.json`, so AX Engine execution
is not established by this release. The AX Engine fields in `axquant_runtime.json` describe the
intended compatibility contract, not observed runtime evidence. Use the architecture-specific MLX
runtime path above. The artifact records AX Engine version
`not recorded`, but version discovery alone is not a runtime check.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `6bit` | 31.21B | 93.55% |
| `8bit` | 777.91M | 2.33% |
| `bf16` | 1.37B | 4.12% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `64`.
- MTP sidecar: not included.
- Vision sidecar: not included.
- Vision weights: protected BF16 in main shards.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 449/449 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | not included |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | not measured; no MTP speedup claim |
| AX Engine kernel evidence | `unmeasured` |
| Vision-language quality | Present, not certified; text Tier 1 does not imply VLM quality |
| Speech-recognition quality | Not applicable (audio disabled for this pack) |
| Long-context quality | 262,144-token capacity is config metadata, not a validated claim |
| Release certification | **Not certified**; formal AXQuant M0-M8 gates are not closed |

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision tower BF16-protected; VL quality not certified |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.
- AX Engine execution is not established because this package has no validated native manifest.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
the [Qwen/Qwen3-VL-32B-Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking/tree/7edd10ffd1196091948fb245ff63e406ccb2d4d1) for license terms, model
limitations, and responsible-use guidance.
