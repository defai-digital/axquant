---
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- qwen3.8
- 8bit
- 8-bit
- vision
---

# AX-Qwen3.8-27B-MLX-AXQ-8bit — 8.00 BPW measured main

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. The language path is quantized while the vision tower are preserved at BF16 in the checkpoint (or a bound sidecar when present).

> **Checkpoint Tier 1 certified** on `df-macstudio-m2` (2026-08-15) for this exact
> revision — measured size against a matched uniform baseline, quality retention, and
> conversion integrity. Tier 1 is a checkpoint claim, **not** a speed claim: MTP
> acceleration is **not certified**; no MTP speedup claim for this checkpoint.
> See the [checkpoint Tier 1 certificate](https://github.com/defai-digital/axquant/blob/main/docs/certifications/qwen38-27b-axq8-tier1.md) for the bound evidence and thresholds.


## Model details

| Property | Value |
| --- | --- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Product family | `qwen3.8` |
| Source architecture | `Qwen3_5ForConditionalGeneration` (dense); text path optimized |
| Main-model parameters | 27.36B logical parameters |
| Quantizer | AXQuant `1.8.1` |
| Hub budget class | `8bit` |
| AXQuant base precision class | `8bit` |
| Planned storage-adjusted BPW | 8.0000 |
| Measured main-model BPW | 8.0001 |
| Measured total BPW | **8.0001** |
| Safetensors weight size | 27.36 GB |
| Approximate complete download | 27.38 GB |
| Configured maximum context | 262,144 tokens; practical limits depend on unified memory |
| Primary MLX runtime | MLX-LM |
| AX Engine native execution | Native manifest included; execution still requires a runtime check |
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
| [4bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit) | Lower-storage AXQ budget; check its exact BPW |
| [6bit sibling](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit) | Higher average precision near the 6-BPW budget |

See the [AutomatosX collections](https://huggingface.co/AutomatosX/collections)
for the family catalog, or the [complete index](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

## Download

```bash
python -m pip install -U huggingface_hub
hf download AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit --local-dir ./AX-Qwen3.8-27B-MLX-AXQ-8bit
```

Allow at least 27.38 GB of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \
  --model AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit \
  --prompt "Explain mixed-precision quantization in three sentences." \
  --max-tokens 128 \
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`0.32.0` and MLX-LM `0.31.3` from conversion.

## Serve with AX Engine

After installing AX Engine, download the complete repository (see
[AXQuant](https://github.com/defai-digital/axquant) for conversion, certificates, and
model-card tooling) and serve the local directory:

```bash
ax-engine serve ./AX-Qwen3.8-27B-MLX-AXQ-8bit --port 31418
```

AX Engine is the authority for the AXQ runtime contract.
This development package does not claim runtime speedups until identical-checkpoint benchmarks are
published. The artifact records AX Engine version `6.16.1`. Native
`model-manifest.json` status: included as `model-manifest.json`.

## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
| `6bit` | 17.79B | 65.05% |
| `8bit` | 7.83B | 28.61% |
| `bf16` | 1.73B | 6.34% |

- Quantization methods: `affine, bf16`.
- Group sizes used by quantized assignments: `32, 64`.
- MTP sidecar: not included.
- Vision sidecar: 333 tensors, 460.73M parameters, 0.92 GB, BF16.
- Vision weights: protected BF16 sidecar.
- Optimization scope: `text-path`.
- Support tier: `convertible`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `architecture_prior` |
| Calibration | none; the allocation is based on architecture priors |
| Quantizer execution | 497/497 recorded module conversions succeeded; 0 fallbacks |
| AX Engine native manifest | included as `model-manifest.json` |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | **not certified**; no MTP speedup claim for this checkpoint (No MTP weights; checkpoint Tier 1 is non-MTP direct-decode only.) |
| AX Engine kernel evidence | `unmeasured` |
| Vision-language quality | Present, not certified; text Tier 1 does not imply VLM quality |
| Speech-recognition quality | Not applicable (audio disabled for this pack) |
| Long-context quality | 262,144-token capacity is config metadata, not a validated claim |
| Release certification | **Checkpoint Tier 1 certified** on `df-macstudio-m2` (2026-08-15), Hub commit `bf7c63b10c17`; the formal AXQuant M0-M8 release campaign is a separate process and is not implied |

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision sidecar present; mlx-vlm smoke not a quality pass (prefixes=['model.visual']) |
| Audio | `not-applicable` | `false` | audio not supported on this pack |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
- Vision weights are preserved at BF16, but this release does not claim validated VLM quality.
- The configured context window can require substantially more memory as the KV cache grows.

- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX compatibility metadata; runtime checks remain separate evidence.
- [`axquant_vision_sidecar_manifest.json`](axquant_vision_sidecar_manifest.json): protected vision tensor provenance.
- [`model-manifest.json`](model-manifest.json): AX Engine native tensor manifest.

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
the [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) for license terms, model
limitations, and responsible-use guidance.
