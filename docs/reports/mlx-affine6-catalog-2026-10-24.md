# MLX affine six-bit catalog publication

Evidence date: 2026-10-24. This audit covers 45 distinct source models,
including historical GitHub model cards, the live Hub catalog, two detector
versions and one diffusion source. The GitHub inventory was pinned at
`729f43d753bad035c6deaa9950d64403a084aa4d`.

30 additional source models have published MLX AXQ six-bit variants,
including the resumed Swift conversion; 11 include native MTP weights or paired assistants. Three earlier Flash Next/Tiel/Cyber
artifacts remain unchanged. See the [6-bit collection](https://huggingface.co/collections/AutomatosX/6-bit-6adc1885130a725e8595678b).

These are development research artifacts, not quality certificates. New packs
use affine6/group32 plus protected 8-bit/floating tensors. Historical verified
Ornith affine6/group64 packs retain their actual packing. This is not MXFP6;
complete storage BPW can exceed six. Exact source pins, measured storage, task
receipts and manifest-bound compatibility reports are included in each pack.

## Published in this campaign

Each link pins the immutable verified publication revision. Weight GiB and complete
BPW include protected payloads and paired assistants, not just the language
trunk; BPW is a storage measurement, not a quality score. A passing task smoke
covers only the listed task and loader. MTP in this table means packaged native
weights or an exact-pair external assistant; execution, exactness, acceptance
and acceleration remain unverified in this campaign.

| Model | Task / measured loader | Weight GiB | Complete BPW | MTP payload |
| --- | --- | --- | --- | --- |
| [AX-Clef-27B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Clef-27B-MLX-AXQ-6bit/tree/0aabab998de972d8b5a382146fa047c0a8b449b4) | decision / mlx-lm | 24.67 | 7.709 | No |
| [AX-Clef-Flash-9B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Clef-Flash-9B-MLX-AXQ-6bit/tree/900bb3d60af72aa5d81bd00096febb0f5637c414) | decision / mlx-lm | 9.68 | 8.724 | No |
| [AX-Clef-Omni-30B-A3B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Clef-Omni-30B-A3B-MLX-AXQ-6bit/tree/af8d29a1640678b04577406f39791ccef2288efd) | decision / mlx-vlm | 30.87 | 7.496 | No |
| [AX-Qwen3-Coder-Next-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit/tree/3fcf7bb0c59142da013e721e370f478d2d51b1a0) | text / mlx-lm | 65.39 | 7.050 | No |
| [AX-Qwen3-Embedding-0.6B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-6bit/tree/14afbb22e1d1f1deb8e4a90535d78182413afd81) | embedding / mlx-vlm | 0.52 | 7.530 | No |
| [AX-Qwen3-Embedding-4B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-6bit/tree/181e6e6a6f2b30c6cea13e85aad7c87276bc2ab1) | embedding / mlx-vlm | 3.37 | 7.199 | No |
| [AX-Qwen3-Embedding-8B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-6bit/tree/08b68f241b630099a7f68eeb165c8d0ca6604366) | embedding / mlx-vlm | 6.32 | 7.169 | No |
| [AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit/tree/0143ec44d5432c8041c178bca572fe1fae33b7d4) | vision / mlx-vlm | 28.81 | 7.420 | No |
| [AX-Qwen3-VL-4B-Instruct-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-VL-4B-Instruct-MLX-AXQ-6bit/tree/0a4cd56aa9b45f3f9cb083b95444636f4b6d9c31) | vision / mlx-vlm | 4.14 | 8.019 | No |
| [AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit/tree/8a6191ac5aedff4f6b5fabde41c5305f622b52d0) | vision / mlx-vlm | 8.55 | 8.380 | No |
| [AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP/tree/fde94cbc5bb5d6671b933831f1666a622d1bd604) | text / mlx-vlm | 9.91 | 8.820 | Yes |
| [AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP/tree/78a56c2b89f0beb28f12561939d45dedc890a2f4) | text / mlx-vlm | 25.23 | 7.800 | Yes |
| [AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP/tree/473f8a168a47e853fe800bb057818f4b432a8d0a) | text / mlx-vlm | 31.35 | 7.490 | Yes |
| [AX-Unlimited-OCR-3B-MoE-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-6bit/tree/83daae13bf093cfca4a5cd894fa432297c7cdffa) | ocr / mlx-vlm | 3.36 | 8.640 | No |
| [AX-DeepSeek-OCR-2-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-6bit/tree/c29b5e4b26d40a263aec3701dec91bd6eddbeaa4) | ocr / mlx-vlm | 3.45 | 8.749 | No |
| [AX-EmbeddingGemma-2-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-EmbeddingGemma-2-MLX-AXQ-6bit/tree/ec05e8332b01359877e88b99bdaf0d422db4181b) | embedding / mlx-vlm | 1.14 | 13.152 | No |
| [AX-gemma-4-12b-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit/tree/292265a2924c078a28a6d2d1a529e98b1d248d44) | vision / mlx-vlm | 10.04 | 7.210 | No |
| [AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP/tree/a772139a221fbf6f2fdc962ebb0747d8961c1a67) | vision / mlx-vlm | 22.59 | 7.400 | Yes |
| [AX-gemma-4-31b-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP/tree/de5906d049f00b464673f5f48dfb2b473df0ff2d) | vision / mlx-vlm | 27.31 | 7.389 | Yes |
| [AX-Muse-Glimmer-30B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit/tree/91a934817338bedbc147fc79b5e23a9aa07f5698) | text / mlx-lm | 28.01 | 8.080 | No |
| [AX-Qwen3.5-0.8B-decision-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Qwen3.5-0.8B-decision-MLX-AXQ-6bit-MTP/tree/44c3035febb7ed3c9cf90969afef75fee38779c4) | decision / mlx-lm | 0.95 | 9.058 | Yes |
| [AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-6bit-MTP/tree/38da39d73e9a50f9285d315692689c7423f8dbe2) | text / mlx-vlm | 104.62 | 7.270 | Yes |
| [AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-6bit-MTP/tree/3ab22b88e289b4508643966411cc71a11485adb1) | text / mlx-vlm | 28.70 | 7.490 | Yes |
| [AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit/tree/3b96de385ea2bdee75f2d7857138678ef26f1bfa) | embedding / mlx-vlm | 0.99 | 7.479 | No |
| [AX-Nemotron-3-Embed-8B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Nemotron-3-Embed-8B-MLX-AXQ-6bit/tree/292318b688f36642e280505661950c0a1817a0e7) | embedding / mlx-vlm | 6.61 | 7.140 | No |
| [AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-MiniCPM-V-4.7-35B-A3B-MLX-AXQ-6bit/tree/601dac023b6e22837f91486ce0e08925aa57b4af) | vision / mlx-vlm | 30.99 | 7.560 | No |
| [AX-Ornith-1.5-35B-A3B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Ornith-1.5-35B-A3B-MLX-AXQ-6bit-MTP/tree/fe662b72298cd3cec8e90f870f05509cb365d573) | text / mlx-vlm | 29.32 | 7.005 | Yes |
| [AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP/tree/28d59d9be4805a6abc97a7f82bcad4f544632726) | text / mlx-vlm | 9.44 | 8.401 | Yes |
| [AX-Hy-MT2-30B-A3B-MLX-AXQ-6bit](https://huggingface.co/AutomatosX/AX-Hy-MT2-30B-A3B-MLX-AXQ-6bit/tree/75459bc989d0d819e136b147b864f183d2634b58) | translation / mlx-lm | 24.85 | 7.100 | No |
| [AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-6bit-MTP](https://huggingface.co/AutomatosX/AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-6bit-MTP/tree/603b1ecf466bf8fa58ce78d453a59b3af2a98f07) | text / mlx-vlm | 25.23 | 7.800 | Yes |

Embedding checks include finite normalized outputs, retrieval ordering and source
cosine comparisons. Vision/OCR checks read a synthetic text image; decision
checks exercise the protected joint head. Text and translation checks are short
functional probes. These do not establish benchmark quality, long-context
behavior or every advertised modality. Clef Omni has decision/text evidence;
its audio and vision behavior were not verified. Text-tested packs with a
preserved vision sidecar do not inherit image evidence.

Hy-MT2 requires the bundled AXQuant MIT-licensed architecture adapter; unmodified
upstream MLX-LM/MLX-VLM does not supply that architecture. Gemma 26B/31B use
source-bound floating external assistants. Gemma 12B remains without MTP because
its new unified assistant profile is not supported by the composition contract.
MiniCPM-V 4.7 contains no actual MTP payload in the pinned source despite its
configuration flag, so its name has no MTP suffix.

Compatibility checks are bound to each artifact. A static compatibility result
is not an executed AX Engine/oMLX/MTPLX or MTP test. Use the model card's runtime
table for measured support and unsupported/untested distinctions.

## Existing publications

- [AX-Qwen3.8-Flash-Next](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP/tree/3c1dd547d70cb7bf01dd0a89c41c400dde32ed39): Previously published research artifact; unchanged. Flash Next retains its disclosed runtime failures and separate user-authorized research exception.
- [AX-Cyber-Tiel-Coder-35B-A3B](https://huggingface.co/AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP/tree/b0b979ad6be74740571eeedb9af35c56158f03e8): Previously published research artifact; unchanged.
- [AX-Tiel-Coder-35B-A3B](https://huggingface.co/AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP/tree/ad22433661ef9292e09b6b0eeb602e06c0aa2474): Previously published research artifact; unchanged.

## Other source dispositions

| Source | State | Reason |
| --- | --- | --- |
| `MiniMaxAI/MiniMax-M3` | blocked-adapter | No exact supported conversion adapter. Inspection or a similar family is not conversion support. |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | blocked-conversion | Current compressed-source dequantization/requantization materializes full output despite the streaming switch. A bounded-memory requantization implementation is required before these above-RAM exports. Multi-terabyte source plus output storage also needs a verified mounted archive; no terabyte download started. |
| `Qwen/Qwen3.8-27B` | excluded | Earlier explicit user exclusion retained; distinct from Flash Next and Swift. |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | blocked-conversion | Current compressed-source dequantization/requantization materializes full output despite the streaming switch. A bounded-memory requantization implementation is required before these above-RAM exports. Multi-terabyte source plus output storage also needs a verified mounted archive; no terabyte download started. |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | blocked-conversion | Current compressed-source dequantization/requantization materializes full output despite the streaming switch. A bounded-memory requantization implementation is required before these above-RAM exports. Multi-terabyte source plus output storage also needs a verified mounted archive; no terabyte download started. |
| `deepseek-ai/DeepSeek-V4.1-Flash` | blocked-adapter | No exact supported conversion adapter. Inspection or a similar family is not conversion support. |
| `moonshotai/Kimi-K3` | blocked-adapter | No exact supported conversion adapter. Inspection or a similar family is not conversion support. |
| `ornith-ai/Ornith-1.5-397B` | withdrawn | Support withdrawn by explicit user direction on 2026-10-24 (AXQ-065). Inspection only; planning, conversion, runtime checks and publication disabled. Failed exact-payload native runtime evidence is retained. |
| `vllm-sr/d3` | blocked-adapter | No exact supported conversion adapter. Inspection or a similar family is not conversion support. |
| `AXDocLayout M v1.2 detector` | unsupported-format | RT-DETRv2 detector has no native MLX affine6 exporter. Preserve existing repository visibility and collection memberships. |
| `AXDocLayout M v1.3 detector` | unsupported-format | RT-DETRv2 detector has no native MLX affine6 exporter. Preserve existing repository visibility and collection memberships. |
| `Qwen/Qwen-Image-2.1-Turbo` | unsupported-format | Existing AXQ diffusion pack is a custom reference container reconstructing floating Diffusers weights. No native MLX affine6 diffusion exporter or direct MLX-LM/AX Engine loader is implemented. |

The large-model blocker concerns the current conversion path. AX Engine has a
family-specific SSD expert streaming path, so a checkpoint exceeding resident
RAM alone does not prove inference is unsupported. The combined source plus
minimum six-bit output for Qwen 2.4T and DeepSeek Pro exceeds 6.4 TB before the
rest of the campaign; available factory space was about 5.16 TB at preflight and
the model archive was not mounted. Source and published checkpoints were kept.

## Publication and implementation

The resumable [campaign driver](../../scripts/publish_mlx_affine6_catalog.py)
checks actual quantization, protected payloads, safe artifact members and an
exact-payload passing task receipt. It stages new repositories privately,
verifies every remote member's size and content identity, then verifies public
access at the immutable revision. Collection updates add published items while
preserving previous members. Existing private detector repositories were not
made public by this campaign.

See the [operator guide](../guides/mlx-affine6-catalog.md) for job format,
resumption, bounded CPU fallback and runtime-specific requirements. The source
changes also fix OCR alias separation and restrict non-Linear MoEGate routers to
BF16 before allocation, keeping serialized plan accounting consistent.

No certificate, package release, GitHub commit or push is part of this campaign.

## Ornith 397B native runtime failure

The 337 GB affine6/group64 candidate was not published. Native AX Engine 7.9.0
selected its MLX backend and completed two requests, but returned `!!` for
`19 + 23` and `!` for the required `READY` response. The emitted token IDs were
`[0, 0]` and `[0]`. These are unrelated answers, not truncated prefixes of the
expected answers. The two-request test took 3307.4 seconds.

The attempt used `AX_STREAM_EXPERTS=on`,
`AX_MLX_PIPELINE_GRANULARITY=layer` and bounded sequential page-cache prefetch
(current shard plus two following shards). The earlier default lazy-graph
attempt was stopped without a generation result. A resident MLX-LM/MLX-VLM
load was not attempted on the 192 GB factory. MTP was not verified.

The cause is not localized. This is an observed failure for this artifact and
native execution path, not proof that the original source, all MLX loaders or
affine6 quantization is generally unsupported or poor quality. Full original
artifact integrity and a passing generation task are separate gates; the
publication gate remained closed. The user subsequently withdrew Ornith 397B support on 2026-10-24. AXQuant
now rejects its planning, conversion, runtime checks and publication, including
stale plans and artifact aliases. No AutomatosX repository or collection entry
exists for this model. A later passing probe alone does not restore support;
a new explicit user direction would be required. Ornith 9B/35B and other
Qwen 397B checkpoints retain their separate support.

## Validation

The final host suite passed **2,150 tests**, with 2 skipped and 21 integration
cases deselected. The isolated non-MLX suite passed **2,052 tests**, with 102
skipped and 10 deselected. Ruff lint, source type checking (171 files), frozen
schema contracts, generated certification docs and diff whitespace checks passed.

The complete local CI command stops at four unchanged pre-existing Hub-card
Python formatting failures: Qwen3.5-9B and Qwen3.6-35B-A3B, each in MXFP4 and
MXFP8. The remaining gates were run separately; a fully green CI run is not
claimed. This campaign's code, new cards and report pass formatting checks.

The scoped additive audit verified 18 affected collections, including the method, runtime and family views; every prior member was preserved.
