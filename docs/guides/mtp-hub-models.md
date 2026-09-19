# AXQ MTP Hub model runtime matrix

Verified against the `AutomatosX` Hugging Face organization on 2026-08-30. The inventory contains
25 populated AXQ MTP checkpoints and one explicitly reserved repository. An `-MTP` name means MTP
assets are packaged; it does not by itself mean that speculative decoding is enabled, compatible
with every runtime, or Tier 2 certified.

Run the architecture-aware, header-only fleet audit with:

```bash
python scripts/audit_mtp_hub_fleet.py --json-output /path/to/mtp-fleet-audit.json
```

The audit validates immutable revisions and the distinct Gemma assistant, resident Qwen,
expert-stream Qwen, and DeepSeek `nextn` packaging contracts without downloading full model
weights. It is a static publication gate, not runtime or certification evidence.

## Resident Qwen sidecars

These nine checkpoints carry the canonical `qwen3-next-mtp` contract in `mtplx_runtime.json`.
Their packaged sidecars can be discovered by AX Engine and strict Qwen sidecar importers. oMLX
`0.6.3rc2` or newer can import the sidecar into a writable local snapshot, and MTPLX `2.5.2` or
newer can consume it directly. Import compatibility is not an exactness or speed certificate;
follow each model card's AX Engine Tier 2 status.

| Hugging Face model | Packaged MTP form |
| --- | --- |
| [AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP) | Resident Qwen sidecar |
| [AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP) | Resident Qwen sidecar |

## Gemma 4 assistant-MTP bundles

These six checkpoints package an exact-paired drafter under `assistant/`, governed by
`ax_gemma4_assistant_mtp.json`. AX Engine 7.1.5 validates the pair, enables assistant-MTP by
default, and caps the default draft depth at two. Set `AX_MLX_GEMMA4_ASSISTANT_MTP=0` to force
direct decode or `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH=1` to cap drafting at one token.

The same bundle can use oMLX's **VLM MTP** path when the target was built with AXQuant's
`mlx-vlm-gemma4-v1` protected-vision layout. That layout removes the source-only `model.` prefix
from `vision_tower.*` and `embed_vision.*` tensors. For the 12B `gemma4_unified` source it also
normalizes `vision_embedder.*` and `embed_audio.*`, then restores `model_type=gemma4_unified` and
the upstream audio contract after text conversion. Every normalized tensor is mapped to
`vision.safetensors` in `model.safetensors.index.json`. Older revisions without that layout marker,
exact index coverage, or the restored unified config must be rebuilt and recomposed; editing only
`config.json` cannot repair their weights or index.

Download the complete repository, register the target and its `assistant/` directory as separate
local oMLX models, and configure the target as follows:

```yaml
vlm_mtp_enabled: true
vlm_mtp_draft_model: /absolute/path/to/checkpoint/assistant
vlm_mtp_draft_block_size: 2
```

Do not use **Lightning MTP**, **Import MTP side-car**, or synthetic MTP head-count fields for this
layout. Those settings describe embedded/native heads, while Gemma 4 uses an external
`gemma4_assistant` model. oMLX 0.6.4 officially pins MLX 0.32.0 and ABI-matched custom kernels.
An MLX 0.32.2 diagnostic stack requires MLX-VLM 0.6.17 or newer and rebuilt oMLX native
extensions; older MLX-VLM releases use an RNG-state mutation that MLX 0.32.2 rejects. Overriding
the oMLX pin is not an oMLX-supported installation. Load and generation smokes are compatibility
evidence only. Follow each immutable revision's Tier 2 status for exactness, acceptance, and speed
claims.

| Hugging Face model | Packaged MTP form |
| --- | --- |
| [AX-gemma-4-12b-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP) | Exact-paired `assistant/` bundle |
| [AX-gemma-4-12b-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP) | Exact-paired `assistant/` bundle |
| [AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP) | Exact-paired `assistant/` bundle |
| [AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP) | Exact-paired `assistant/` bundle |
| [AX-gemma-4-31b-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP) | Exact-paired `assistant/` bundle |
| [AX-gemma-4-31b-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP) | Exact-paired `assistant/` bundle |

## DeepSeek V4 `nextn` sidecars

Four populated checkpoints package a native DeepSeek V4 `mtp.safetensors` sidecar. AX Engine
7.1.5 recognizes this `nextn` layout, but product policy remains direct fallback until a
revision-bound Tier 2 acceptance, exactness, and speed certificate exists. Stock MLX-LM can run
the text backbone but does not activate the sidecar. These files are not Qwen sidecars and must
not be assigned `qwen3-next-mtp` or sent through the oMLX/MTPLX Qwen importer.

| Hugging Face model | Status |
| --- | --- |
| [AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP) | Populated; checkpoint Tier 1 (experimental); MTP Tier 2 not certified |
| [AX-DeepSeek-V4-Flash-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-4bit-MTP) | Populated; MTP Tier 2 not certified |
| [AX-DeepSeek-V4-Flash-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-6bit-MTP) | Populated; MTP Tier 2 not certified |
| [AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP) | Populated; checkpoint and MTP Tier 2 not certified |
| [AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP) | Reserved name; no weights or MTP sidecar uploaded |

## Qwen 3.8 Flash Next preview sidecars

These four preview checkpoints carry `mtp.safetensors` together with `ax_expert_stream.json`.
They use the MLX-VLM `qwen4_exp` text/vision path; AX Engine MTP acceleration is not claimed.
Their sidecars pass the expert-stream packaging audit but are not resident
`qwen3-next-mtp` oMLX/MTPLX import targets.

| Hugging Face model | Status |
| --- | --- |
| [AX-Qwen3.8-Flash-Next-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-2bit-MTP) | Preview; packaged MTP sidecar; AX Engine MTP not claimed |
| [AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP) | Preview; packaged MTP sidecar; AX Engine MTP not claimed |
| [AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP) | Preview; packaged MTP sidecar; AX Engine MTP not claimed |
| [AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP) | Preview; packaged MTP sidecar; AX Engine MTP not claimed |

## Super-class expert-stream packs

These checkpoints cannot resident-load on any shipping Mac. `ax_expert_stream.json` is required.
They are not resident `qwen3-next-mtp` import targets, cannot be loaded with `mlx_lm.load`, have
no 4-bit sibling, and will not be certified in this revision. Sidecars are packaged; MTP / DSpark
acceleration is not claimed.

| Hugging Face model | Notes |
| --- | --- |
| [AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP) | Qwen 3.8 2.4T; native Qwen MTP sidecar |
| [AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP](https://huggingface.co/AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP) | DeepSeek V4 Pro 0813; native `mtp.safetensors` DSpark sidecar, not a Qwen sidecar |
| [AX-MiniMax-M3-MLX-AXQ-2bit](https://huggingface.co/AutomatosX/AX-MiniMax-M3-MLX-AXQ-2bit) | MiniMax M3; **no** packaged MTP (config `num_mtp_modules` only). Listed here because it is the Super-class stream queue, not because it has a sidecar |
| [AX-MiniMax-M3-MLX-AXQ-MXFP4](https://huggingface.co/AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4) | MiniMax M3 MXFP4; **no** packaged MTP; `vision.safetensors` sidecar |
| [AX-Kimi-K3-MLX-AXQ-2bit](https://huggingface.co/AutomatosX/AX-Kimi-K3-MLX-AXQ-2bit) | Kimi K3; **no** packaged MTP; native MXFP4 dequant → affine 2-bit; `vision.safetensors` sidecar |
