# Qwen3.8-27B MXFP4 flagship development

The current development flagship is **Qwen3.8-27B AXQ MXFP4 with MTP enabled by
default**. The source is pinned to
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
This study uses the existing measured planner; it does not migrate the frozen
`qwen36-mtp-v2` certification policy or certify any existing Hub revision.
The earlier MXFP8/template study remains historical and separate.

## Plan and acceptance

1. Freeze the BF16 source, stock tokenizer/template, current MXFP4-MTP control,
   coding/general development holdouts, and calibration data. Verify control
   checksums and reject exact/near-duplicate calibration/holdout overlap.
2. Replay coding-oriented calibration through the source chat template. Probe
   native MXFP4 at group 32 and higher-precision affine/BF16 candidates.
   Retain architectural protection floors. Measure attention, recurrent-state
   projections, and FFN tensors without assuming which upgrades will help.
3. Allocate within the control plan's effective storage budget. Keep MTP at
   BF16. Check actual converted weight bytes: the same planner budget does not
   establish equal physical size. A difference above 1% stops this study's
   matched-size comparison; repeat with a new work directory and revised budget
   policy rather than relabeling the result.
4. Evaluate BF16, current MXFP4, and calibrated MXFP4 on identical coding and
   general tasks, with the same tokenizer, context/output limits, seeds, cache
   state, and tool harness. A higher-budget 6-bit pack may be an experimental
   reference, not a replacement flagship. Record task-level outcomes and
   uncertainty; do not infer a quality win from KL/loss proxies.
5. Evaluate the default accelerated route on independent repository tasks with
   multi-turn tool use. Record success rate, total tokens, wall time, memory,
   tool/schema failures, observed drafter route, and acceptance. Retain failed
   trials. Compare direct decode and MTP on the same artifact; DFlash2 is a
   separate route. Do not tune against a sealed formal holdout after consuming it.

The script implements preparation, measurement, conversion, physical-size
comparison, required-MTP serving, and the existing native MTP A/B benchmark.
Executable coding scoring remains available through `prepare-coding-suite`,
`verify-coding-suite`, and `evaluate-coding-suite`; the full repository agent
harness in step 5 is external. No quality improvement or certification is
established merely by completing conversion or the native benchmark.

## Calibration and MTP sensitivity

The default clean-room CC0 development mix has 160 records: 96 coding, plus
16 each for JSON, tools, multilingual instructions, and long context. It
includes source inspection, tool calls/results, failing tests, patches, and
follow-up checks. These intentionally small buffer-repair fixtures validate the
workflow; production calibration should provide representative, independently
authored repository traces with `--calibration`. The script never overwrites a
different existing study input.

`analyze --measure-trunk-mtp` measures the unchanged MTP head conditioned on
the quantized trunk's hidden states, as well as existing MTP-head probes. The
metric is teacher-forced draft-token disagreement. A preserved BF16 head can
still produce different drafts after trunk quantization. This proxy is neither
live draft acceptance nor a DFlash2 measurement. Missing MTP-forward capability
fails the flagship probe; failed candidate measurements cannot authorize an
allocation. Probe resume/refinement binds this protocol separately from legacy
head-only measurement.

## Run

Install the checkout with MLX extras on a suitable Apple Silicon host. Source
weights, the control pack, and JSONL holdouts must already be local. Holdout
records need unique `id` or `task_id` and `text` or `prompt` fields. For this
development script, both holdouts are development validation data; sealed
formal release holdouts stay under the separate certification procedure.

```sh
python scripts/run_qwen38_27b_mxfp4_flagship.py prepare \
  --work /path/to/studies/qwen38-mxfp4 \
  --bf16 /path/to/models/Qwen3.8-27B-bf16 \
  --control /path/to/models/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP \
  --coding-holdout /path/to/datasets/coding-dev.jsonl \
  --general-holdout /path/to/datasets/general-dev.jsonl
```

Repeat with `measure` to execute calibration, resumable sensitivity probing,
planning, and conversion. It runs subprocesses and retains stage logs/exit
receipts. It does not merely print commands. It uses a 32,768-token probe
budget and never sets `--allow-unmeasured` or `--allow-mtp-unmeasured`.
The control's plan and manifest must identify the same pinned BF16 source.
Default native-manifest generation remains required during conversion.

After measurement succeeds:

```sh
python scripts/run_qwen38_27b_mxfp4_flagship.py serve \
  --work /path/to/studies/qwen38-mxfp4

python scripts/run_qwen38_27b_mxfp4_flagship.py benchmark \
  --work /path/to/studies/qwen38-mxfp4 \
  --prompts /path/to/datasets/mtp-development-prompts.jsonl
```

Serving explicitly requests `--mlx-mtp-policy required`: unavailable MTP fails
startup rather than silently selecting direct decode. A current engine is
required. DFlash2 may be selected by AX Engine for eligible requests when its
separate local drafter is present; it is not packaged or downloaded here.
The recorded source model ID stays `Qwen/Qwen3.8-27B` so engine artifact
selection can identify the correct target.

The MTP A/B command records `AX_MLX_DFLASH2=0` in both arms to isolate native
MTP. Existing exactness/speed gates remain in force; failures stay failures.
Measure DFlash2 separately with its exact drafter revision, actual route,
request eligibility, memory cost, and task quality recorded. Its availability
does not establish native-MTP correctness or acceleration.

All study outputs remain development evidence. Model quality, real task
efficiency, native runtime admission, and public release each need their own
artifact-bound results. No model download, Hub upload, or certification record
is performed by this script.
