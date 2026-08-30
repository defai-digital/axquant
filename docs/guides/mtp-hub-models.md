# AXQ MTP Hub model runtime matrix

Verified against the `AutomatosX` Hugging Face organization on 2026-08-21. The inventory contains
20 populated AXQ MTP checkpoints and one explicitly reserved repository. An `-MTP` name means MTP
assets are packaged; it does not by itself mean that speculative decoding is enabled, compatible
with every runtime, or Tier 2 certified.

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
direct decode or `AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH=1` to cap drafting at one token. Stock
MLX-LM does not consume the assistant bundle, and the Qwen oMLX/MTPLX sidecar workflow does not
apply. The bundle is not an embedded MTP-head layout. An oMLX error that the model has no MTP
heads in its config is therefore expected; adding synthetic head-count fields would make that
runtime request embedded weights that the bundle does not contain. Use AX Engine for the paired
assistant layout. The published checkpoint cards do not claim Tier 2 MTP acceleration.

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
