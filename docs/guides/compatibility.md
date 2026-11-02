# Environment compatibility matrix

This matrix covers **runtime environments**. For the model-family support matrix, see the
`compatibility-matrix` command and the README support tiers.

## Withdrawn model support

Ornith 397B support was withdrawn on 2026-10-24. It is inspection-only:
AXQuant rejects planning, conversion, runtime checks, model-card preparation
and publication for the known source ID and its local/config/pack aliases.
Saved convertible plans and development opt-ins do not override this withdrawal.
AutomatosX does not offer Ornith 397B Hub models or collection entries.

Ornith 9B and 35B-A3B retain their model-specific support; the existing support
for other Qwen 397B checkpoints is unchanged. Historical checkpoints and failed
runtime evidence are retained. See the [native runtime failure](../reports/mlx-affine6-catalog-2026-10-24.md#ornith-397b-native-runtime-failure).

## Platforms

| Platform | Status | Notes |
| --- | --- | --- |
| macOS on Apple Silicon (arm64) | Full support | All stages, including MLX capture/probe/convert/runtime |
| macOS on Intel | Partial | Inspection, planning, reporting only (no MLX backend) |
| Linux / Windows | Partial | Inspection, planning and reporting; experimental CUDA conversion with a compatible PyTorch/CUDA environment (see the [CUDA guide](cuda-nvfp4.md)); MLX execution requires Apple Silicon |

## Python

| Version | Status |
| --- | --- |
| 3.11 | Supported (`requires-python >= 3.11`) |
| 3.12 | Supported |
| 3.13 | Supported; primary development target |

## MLX execution backend (optional `mlx` extra)

| Component | Requirement | Used by |
| --- | --- | --- |
| `mlx` | `>= 0.32.2` | capture-activations, analyze (measured), analyze-kv, convert, evaluate-quality, runtime paths |
| `mlx-lm` | `>= 0.31` | same as above; DeepSeek V4 (`model_type=deepseek_v4`) requires a build that ships `mlx_lm.models.deepseek_v4` (not all PyPI 0.31.x releases) |
| `mlx-audio` | `>= 0.4.7` | Qwen3-ASR BF16 normalization, conversion, and transcription smoke |
| `mlx-vlm` | `>= 0.6.17` | Qwen3-VL; MiniCPM-V 4.7 needs its model module (tested with 0.7.9) |

The AXQ MiniCPM-V 4.7 adapter is scoped to the official 35B-A3B checkpoint and its validated
configuration signature. It quantizes the language path while preserving the vision tower, merger,
and hybrid linear-attention tensors at source precision. The runtime path is MLX-VLM. AX Engine
7.6.4's manifest generator rejects `minicpmv4_7`; other inference runtimes have not been tested.
Conversion artifacts remain development evidence, without video, quality, or MTP claims.

Without the `mlx` extra these commands fail closed with `BackendUnavailableError`; everything
else (inspect, calibrate/tokenize, plan, plan-manual, plan-experimental-mix, reporting, release audit) runs anywhere.

| Stage | Needs MLX | Needs macOS arm64 |
| --- | --- | --- |
| feasibility / inspect | No | No |
| calibrate / tokenize-calibration | No | No |
| capture-activations | Yes | Yes |
| analyze (architecture priors) | No | No |
| analyze (measured, incl. AWQ/GPTQ) / analyze-kv | Yes | Yes |
| plan / plan-manual / plan-experimental-mix / refine | No | No |
| convert | Yes | Yes |
| runtime / evaluate-quality / benchmark | Yes | Yes |
| report / release-audit / publish-prepare | No | No |

## Tested configuration

The toolkit line is **v2.3.0 (official release)**; **v1.8.1** is a historical validated-toolkit pin, not a model certificate. CI (`.github/workflows/ci.yml`) runs the full
suite on `macos-14` (Apple Silicon) with Python 3.13 and the `mlx` extra. A historical
v1.2.0 development environment was macOS arm64, Python 3.13, `mlx 0.32.0`,
`mlx-lm 0.31.3`, `mlx-audio 0.4.7`, and `mlx-vlm 0.6.10`.
