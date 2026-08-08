# Environment compatibility matrix

This matrix covers **runtime environments**. For the model-family support matrix, see the
`compatibility-matrix` command and the README support tiers.

## Platforms

| Platform | Status | Notes |
| --- | --- | --- |
| macOS on Apple Silicon (arm64) | Full support | All stages, including MLX capture/probe/convert/runtime |
| macOS on Intel | Partial | Inspection, planning, reporting only (no MLX backend) |
| Linux / Windows | Partial | Inspection, planning, reporting only (MLX is Apple-Silicon-only) |

## Python

| Version | Status |
| --- | --- |
| 3.11 | Supported (`requires-python >= 3.11`) |
| 3.12 | Supported |
| 3.13 | Supported; primary development target |

## MLX execution backend (optional `mlx` extra)

| Component | Requirement | Used by |
| --- | --- | --- |
| `mlx` | `>= 0.29` | capture-activations, analyze (measured), analyze-kv, convert, evaluate-quality, runtime paths |
| `mlx-lm` | `>= 0.31` | same as above; DeepSeek V4 (`model_type=deepseek_v4`) requires a build that ships `mlx_lm.models.deepseek_v4` (not all PyPI 0.31.x releases) |
| `mlx-audio` | `>= 0.4.7` | Qwen3-ASR BF16 normalization, conversion, and transcription smoke |
| `mlx-vlm` | `>= 0.6.10` | Qwen3-VL conversion and image-to-text smoke |

Without the `mlx` extra these commands fail closed with `BackendUnavailableError`; everything
else (inspect, calibrate/tokenize, plan, plan-manual, reporting, release audit) runs anywhere.

| Stage | Needs MLX | Needs macOS arm64 |
| --- | --- | --- |
| feasibility / inspect | No | No |
| calibrate / tokenize-calibration | No | No |
| capture-activations | Yes | Yes |
| analyze (architecture priors) | No | No |
| analyze (measured, incl. AWQ/GPTQ) / analyze-kv | Yes | Yes |
| plan / plan-manual / refine | No | No |
| convert | Yes | Yes |
| runtime / evaluate-quality / benchmark | Yes | Yes |
| report / release-audit / publish-prepare | No | No |

## Tested configuration

CI (`.github/workflows/ci.yml`) runs the full suite on `macos-14` (Apple Silicon) with
Python 3.13 and the `mlx` extra. The v1.2.0 development environment was macOS arm64,
Python 3.13, `mlx 0.32.0`, `mlx-lm 0.31.3`, `mlx-audio 0.4.7`, and
`mlx-vlm 0.6.10`.
