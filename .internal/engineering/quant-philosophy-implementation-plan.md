# Quant Philosophy Program — Multi-Phase Implementation Plan

**Status:** Complete (QP0–QP3)  
**PRD:** `../product/quant-philosophy-prd.md`  
**ADR:** `../architecture/decisions/0007-quant-philosophy-nvfp4-and-recovery.md`  
**Tech spec:** `quant-philosophy-technical-specification.md`  
**Decisions:** AXQ-028, AXQ-029  
**Last updated:** 2026-08-01

## Phase overview

| Phase | Name | Status | Goal |
| --- | --- | --- | --- |
| **QP0** | 3-D configuration + strategy metadata | **Done** | Plan/probe/report `bits x group_size x method`; record scale/outlier strategy |
| **QP1** | Role policy + holdout refinement | **Done** | Prefer fine groups / AWQ for sensitive roles; bind measured refine |
| **QP2** | Optional recovery | **Done** | Calibration-only retention restore; recovery manifest; no domain SFT |
| **QP3** | Ultra-low bit experimental | **Done** | 2/3-bit + fine group + experimental labeling |

## QP0 — completed checklist

- [x] PRD / ADR / tech spec / decision register / README index
- [x] `ScaleStrategy`, `OutlierStrategy` enums
- [x] Candidate uniqueness `(bits, method, group_size)`
- [x] `PlanRequest.candidate_group_sizes` / `candidate_methods`
- [x] `Allocation` strategy fields + `QuantizationPlan.candidate_group_sizes`
- [x] `architecture_prior_report` multi-group emission
- [x] Planner Pareto option ladder
- [x] `ProbeConfig.candidate_group_sizes` + probe loop
- [x] CLI `--candidate-group-sizes` on analyze/plan
- [x] Tests + full suite green

### QP0 usage

```bash
# Multi-group architecture-prior analysis (development evidence)
axquant analyze --model /path/to/bf16 --candidate-group-sizes 32,64,128 -o sensitivity.json

# Plan using the multi-group report
axquant plan --analysis sensitivity.json --allow-unmeasured \
  --candidate-group-sizes 32,64,128 --target-bpw 4.8 -o plan.json
```

## QP1 — completed checklist

- [x] `role_policy.py` with measured-only method/group preferences
- [x] Planner collapse + knapsack ranking uses role preferences under measured evidence
- [x] `HardwareProfile` defaults include AWQ
- [x] Refinement proxy labeling (`evidence_label=proxy-development`)
- [x] Holdout digest binding on `select_complete_candidate`
- [x] Tests in `tests/test_role_policy_qp1.py`

## QP2 — completed checklist

- [x] `RecoveryManifest` / `RecoveryRequest` (`axquant.recovery.v1`)
- [x] `axquant recover` CLI (opt-in; convert/quantize untouched)
- [x] Fail-closed incomplete provenance
- [x] Retention-restore-only claim language
- [x] Tests in `tests/test_recovery_qp2.py`

## QP3 — completed checklist

- [x] `experimental_bits.py` labels and target_class suffix
- [x] Planner + manual plan annotation for 2/3-bit
- [x] Example recipe `examples/qwen36-experimental-2bit-v0.1.yaml`
- [x] Tests in `tests/test_experimental_bits_qp3.py`

## Non-goals (standing)

- NVFP4 / E2M1 format on MLX
- Domain SFT / DPO / Lab-style fine-tune product
- Default convert path requiring training
