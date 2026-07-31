# ADR 0003: Feasibility Gates and Manual v0.1 Plans

**Status:** Accepted  
**Date:** 2026-07-28

## Context

The first Qwen3.6-27B conversion is too large and expensive to begin from assumptions. Before
acquiring the BF16 source, AXQuant needs to prove that its 4-bit, 6-bit, mixed-precision, MTP, AX
Engine, and MLX-LM comparison paths refer to complete and equivalent artifacts.

The first vertical slice also needs a reviewed mixed-precision plan before the measured sensitivity
backend exists. Editing `quantization_plan.json` directly would bypass validation, protection,
provenance, and BPW constraints.

## Decision

AXQuant adds two explicit development boundaries:

```text
axquant feasibility
  -> axquant inspect BF16
  -> axquant plan-manual
  -> axquant convert --allow-unmeasured
```

Neither a passing feasibility report nor a manual plan is quality evidence. They establish
artifact and execution readiness for v0.1.

## Checkpoint boundary

When `model.safetensors.index.json` exists, the inspector reads exactly:

- every safe relative Safetensors path referenced by `weight_map`;
- the root `mtp.safetensors` sidecar when present.

It rejects missing, unsafe, or non-Safetensors index references. Unindexed nested Safetensors are
excluded as auxiliary artifacts. This rule includes an indexed nested vision shard but excludes
duplicate research or packaging files that are not part of the loadable checkpoint.

For checkpoints without an index, only root Safetensors files are inspected.

Packed U32 tensors are converted back to logical parameter counts using their per-tensor or global
MLX quantization metadata. Scale and bias packing metadata contributes storage bytes but not
logical model parameters.

## Feasibility contract

`axquant.feasibility.v1` records:

- immutable model identity and local snapshot revision;
- adapter and optimization scope;
- logical and MTP parameter counts;
- total, main-model, and MTP weight bytes;
- measured total and main-model BPW;
- precision counts and fractions;
- config, index, tokenizer, AX native manifest, MTP runtime, and MTP provenance integrity;
- AX Engine doctor and MLX-LM static checks;
- cross-checkpoint parameter and architecture equivalence.

Quantized baselines require a valid AX native manifest, root MTP sidecar, MTP runtime contract, and
checksum-valid MTP provenance. A BF16 source requires a supported revision-pinned Qwen 3.6
inventory with MTP tensors but does not require output-runtime manifests.

The state machine is:

```text
invalid supplied artifact or failed requested AX doctor
  -> blocked

complete comparison baselines, no BF16 source supplied
  -> baseline-ready

complete comparison baselines and complete BF16 source
  -> ready-for-conversion
```

`--require-ready` makes `baseline-ready` return a nonzero status for automation.

## Manual recipe contract

`axquant.manual-recipe.v1` contains:

- profile, target BPW, default precision/method/group size;
- ordered tensor, module, and role selectors;
- an ID and rationale for every rule;
- MTP, hardware, runtime, release-threshold, and seed metadata.

The first matching rule wins. Every rule must match unless explicitly allowed. An explicit rule
below a protected minimum fails. Defaults are automatically raised to mandatory floors for
non-quantizable tensors, norms, embeddings, LM head, vision, router, and MTP. External MTP
sidecars remain BF16 and byte-preserved in v0.1. Tied weights are harmonized at their highest
selected precision.

The planner fails when the resulting storage-adjusted BPW exceeds the recipe limit. A successful
manual plan is marked `architecture_prior`, requires `convert --allow-unmeasured`, and cannot pass
publication gates.

## Initial local evidence

The 2026-07-28 feasibility run used complete immutable local snapshots and AX Engine 6.11.1 on a
128 GiB Apple Silicon host:

| Baseline | Logical parameters | Weight bytes | Total BPW | Main BPW | MTP bytes | Audit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform 4-bit | 27,781,427,952 | 16,903,941,980 | 4.8677 | 4.6949 | 849,400,381 | PASS |
| Uniform 6-bit | 27,781,427,952 | 23,627,280,146 | 6.8038 | 6.6610 | 849,400,381 | PASS |
| Attributed mixed baseline | 27,781,427,952 | 19,904,508,887 | 5.7317 | 5.7509 | 238,934,201 | PASS |

All three passed parameter equivalence, Qwen adapter, MTP provenance, AX Engine doctor, and
MLX-LM static checks. The report status is `baseline-ready`; the only current conversion blocker
is the absence of a complete revision-pinned BF16 source checkpoint.

These numbers are artifact measurements, not quality or speed results. The mixed checkpoint
remains an attributed external baseline; AXQuant does not import its sensitivity data, code,
tests, calibration data, or unindexed auxiliary files.

## Consequences

Large downloads and conversion runs now have a machine-readable precondition. The v0.1 manual
slice is reviewable and deterministic without being confused with measured PTQ. Exact artifact
BPW is separated from planner estimates, and package-wide BPW cannot hide protected MTP storage.

The remaining M0/M1 work is to obtain and pin the BF16 source, inspect it, verify or construct the
external MTP bundle under the AX Engine contract, apply the manual recipe, and run conversion plus
runtime load checks.
