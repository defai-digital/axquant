# AX Coder

Development campaign for the product **AX Coder**. The naming decision is
AXQ-061 in the internal decision register. This is an axquant MLX pack
of pinned `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. AX Coder Flash remains the separate
Flash Next policy experiment. A score does not grant a public release.

## Where it runs

Run measured calibration, sensitivity, and planning on the Mac Studio M2 Ultra
192 GB factory host. The commands are `calibrate`, `analyze`, and
`plan`. They do not call `plan-cuda` or `convert-cuda`. An NVIDIA GPU is not
required for this pack. CUDA NVFP4 remains a separate, unmeasured track.

The pinned BF16 directory has to be local before `measure`. The existing
MXFP8 pack cannot replace that source. On 2026-10-06 the Studio had the
uniform `AX-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP` control pack and did not have
`/path/to/models/Qwen3.8-27B-bf16`.

## Cells

| Cell | Weights | Template |
| --- | --- | --- |
| `control` | existing uniform MXFP8-MTP | stock Qwen template |
| `sensitivity` | code-sensitivity plan, converted with `--q-mode mxfp8` | stock Qwen template |
| `template` | existing uniform MXFP8-MTP | `examples/qwen38-27b-coding-preamble.txt` as a system message |
| `combined` | sensitivity MXFP8 plan | the same preamble |

The format follows `examples/qwen38-27b-axq-mxfp8-mtp-v0.1.yaml`. Attention
and MLP are 8-bit affine at group 32, then `convert --q-mode mxfp8` remaps
those allocations. Candidate bits are only 8 and 16. The planner policy
minimum on this probe is 9.2238 BPW, so the executed budget is 9.224 BPW.
A 9.0 target is infeasible in the planner's units. Vision and norm stay
at 16 bits, embeddings and the LM head at 8, and MTP in `protected` mode.
An external MTP sidecar stays 16-bit. MXFP8 conversion requires
`--allow-unmeasured` because affine sensitivity does not measure the MXFP8
repack. The preamble is not written into the pack's `chat_template.jinja`.

The calibration mix is `data/calibration/qwen38-27b-code-sensitivity-v0.1.jsonl`.
It is clean-room CC0 text. Coding samples are the majority, and the file also
contains the json, tool, multilingual, and long-context samples the toolkit
requires. It is not a Tiel imatrix.

## Commands

```sh
PYTHONPATH=src .venv/bin/python scripts/run_qwen38_27b_code_sensitivity.py prepare \
  --work /path/to/factory/work/qwen38-27b-code-sensitivity \
  --bf16 /path/to/models/Qwen3.8-27B-bf16

PYTHONPATH=src .venv/bin/python scripts/run_qwen38_27b_code_sensitivity.py measure \
  --work /path/to/factory/work/qwen38-27b-code-sensitivity \
  --bf16 /path/to/models/Qwen3.8-27B-bf16
```

`measure` prints the axquant command sequence and returns 2 when the BF16
directory is missing. It does not launch conversion. Do not add
`--allow-unmeasured` or `--allow-mtp-unmeasured` for a product claim.

## Score

After the four cells exist, write one JSON object keyed by cell id. Each cell
records `trials`, `successes`, `schema_failures`, `retries`,
`reserved_device_seconds`, `human_seconds`, `per_device_hour`, `human_hour`,
and `decode_tokens_per_second`. Rates and trial counts must match the control.

```sh
PYTHONPATH=src .venv/bin/python scripts/run_qwen38_27b_code_sensitivity.py score \
  --cells /absolute/path/cells.json
```

A baseline solve rate below 0.25 stops the study. A candidate is eligible only
when its solve rate is at least the baseline, schema failures and retries do
not rise, and estimated cost per success is lower. Cost includes failed trials,
reserved device time, and human time. Decode speed is reported and is not the
selection rule. `combined` also needs both single cells to be eligible. The
report always sets `release_tag` to false. The product name is AX Coder, so
`ax_coder_name_authorized` is true. Public release stays false.
