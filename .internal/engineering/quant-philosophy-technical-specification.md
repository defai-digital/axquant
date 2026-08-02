# AXQuant Quantization Philosophy — Technical Specification

**Document status:** Accepted for implementation  
**Applies to:** Program QP phases QP0–QP3  
**Decisions:** AXQ-028, AXQ-029  
**Base document:** `technical-specification.md` remains authoritative unless amended here  
**Last reviewed:** 2026-08-01

## 1. Scope

| Phase | Implementation status target |
| --- | --- |
| **QP0** | **Implemented** (2026-08-01): schema, planner Pareto ladder, analyzer multi-group priors, probe multi-group grid, strategy metadata, CLI flags, tests |
| **QP1** | **Implemented** (2026-08-01): role preferences + holdout-bound refinement |
| **QP2** | **Implemented** (2026-08-01): optional recovery stage + provenance |
| **QP3** | **Implemented** (2026-08-01): experimental ultra-low bit labeling + recipe |

## 2. Capability truth table

| Capability | QP0 | QP1 | QP2 | QP3 |
| --- | --- | --- | --- | --- |
| Candidate uniqueness `(bits, method, group_size)` | ✅ | | | |
| `PlanRequest.candidate_group_sizes` | ✅ | | | |
| Planner Pareto option ladder (3-D) | ✅ | | | |
| Allocation `scale_strategy` / `outlier_strategy` | ✅ | | | |
| Architecture-prior multi-group emission | ✅ | | | |
| `ProbeConfig.candidate_group_sizes` + multi-gs probe | ✅ | | | |
| CLI multi-group plan/analyze/probe flags | ✅ | | | |
| Role-preferred group / AWQ policy | | ✅ | | |
| Holdout-measured refinement binding | | ✅ | | |
| `axquant recover` + recovery manifest | | | ✅ | |
| Experimental 2/3-bit + fine-group recipes | | | | ✅ |

## 3. Schema amendments (QP0)

### 3.1 Enums

```python
class ScaleStrategy(StrEnum):
    NONE = "none"  # BF16 / preserved
    GROUP_AFFINE = "group-affine"  # standard MLX group scales
    CHANNEL_AWQ = "channel-awq"  # AWQ channel scales then affine pack


class OutlierStrategy(StrEnum):
    NONE = "none"
    PERCENTILE_CLIP_DWQ = "percentile-clip-dwq"
```

### 3.2 `CandidateMeasurement` uniqueness

`TensorSensitivity` (and probe progress keys) treat candidates as unique by:

```text
(bits, method, group_size)   # group_size is None for BF16
```

Legacy reports with a single group size continue to validate.

### 3.3 `PlanRequest`

Additive fields:

```python
candidate_group_sizes: tuple[int, ...] = ()  # empty => (group_size,)
candidate_methods: tuple[QuantMethod, ...] = ()  # empty => no method filter beyond hardware
```

Validators:

- every entry in `candidate_group_sizes` ∈ `hardware.supported_group_sizes`;
- if empty, effective grid is `(group_size,)` and `group_size` must still be supported;
- `candidate_methods` if non-empty must subset hardware supported methods (excluding accidental BF16-only lists for quantized work is caller responsibility; BF16 remains allowed via 16-bit candidates).

`QuantizationPlan` records:

```python
candidate_group_sizes: tuple[int, ...] = ()  # effective grid used
```

`group_size` remains the **default / primary** group size for compatibility and KV advisory paths.

### 3.4 `Allocation`

Additive fields with defaults:

```python
scale_strategy: ScaleStrategy = ScaleStrategy.GROUP_AFFINE
outlier_strategy: OutlierStrategy = OutlierStrategy.NONE
strategy_metadata: dict[str, str | int | float | bool] = {}
```

Derivation rules when building an allocation from a selected measurement:

| method / bits | scale_strategy | outlier_strategy |
| --- | --- | --- |
| BF16 / 16-bit | `none` | `none` |
| affine | `group-affine` | `none` |
| dwq | `group-affine` | `percentile-clip-dwq` |
| awq | `channel-awq` | `none` |
| gptq (if ever selected) | `group-affine` | `none` |

### 3.5 `ProbeConfig`

```python
candidate_group_sizes: tuple[int, ...] = ()  # empty => (group_size,)
```

Probe loops `for group_size in effective_group_sizes` inside the bits × methods loops.
Existing progress resume keys must include group size (same triple as uniqueness).

## 4. Architecture priors (QP0)

`architecture_prior_report(..., group_size=64, candidate_group_sizes=())`:

- effective groups = `candidate_group_sizes or (group_size,)`;
- for each quantizable tensor and each quantized bit, emit one affine candidate **per group**;
- prior metrics scale noise by `sqrt(group_size / 64.0)` so smaller groups are weakly preferred
  (development evidence only; never release quality).

## 5. Planner algorithm (QP0)

### 5.1 Effective grids

```text
groups = request.candidate_group_sizes or (request.group_size,)
methods_filter = request.candidate_methods or <no extra filter>
```

Candidates from the sensitivity entry must pass existing floors, hardware bits/methods/groups,
and (if `methods_filter` non-empty) method ∈ filter **or** method is BF16 for 16-bit.

### 5.2 Option construction

1. Filter supported candidates.
2. Collapse by key `(bits, group_size)` keeping **minimum loss** method (same storage).
3. Sort by `(storage_bpw, loss, bits, group_size or 0, method)`.
4. Drop **dominated** options: option B is dominated if some A has
   `storage_bpw(A) <= storage_bpw(B)` and `loss(A) <= loss(B)` (strict improvement on at least
   one axis, or equal on both with lexicographically smaller key — keep one).
5. Marginal-efficiency upgrade loop unchanged (AXQ-011 / ADR 0006 knapsack).

### 5.3 MoE fused groups

Unchanged policy: after selection, force fused expert members to the **minimum storage**
selected option index among the group (budget-safe).

### 5.4 Strategy attachment

When emitting `Allocation`, set scale/outlier strategies from §3.4 and include:

```python
strategy_metadata = {
  "storage_bpw": <float>,
  "selected_from_candidates": <int count available before collapse>,
}
```

## 6. CLI (QP0)

| Command | Flag | Behavior |
| --- | --- | --- |
| `plan`, `analyze` (prior path), `probe` | `--candidate-group-sizes 32,64,128` | Comma-separated ints; omit for single `group_size` |
| `plan` | `--candidate-methods affine,dwq` | Optional method filter |

Invalid group sizes fail closed with hardware message.

## 7. QP1 (planned contracts)

### 7.1 Role policy table (draft)

| Role | Preferred min group when measured | Preferred methods |
| --- | --- | --- |
| attention | 32 | awq, affine |
| mlp / expert | 64 | affine, dwq |
| embedding / router floors | policy bits | affine |
| lm_head (if 8-bit AXQ-026) | 64 | affine |

Policy only **reorders or floors** within legal candidates; it cannot lower AXQ-007 bits floors.

### 7.2 Holdout refinement

`RefinementConfig` gains optional binding to complete-candidate measurement digests. Proxy-only
runs stay development evidence (existing).

## 8. QP2 (planned contracts)

### 8.1 Recovery artifact

```text
schema_version: axquant.recovery.v2
source_artifact_sha256
plan_sha256
calibration digests
algorithm_id
steps, seed, hyperparameters
parameter_update_scope  # scales | biases | lora-merged
quality_before / quality_after refs
weight_mutation_applied  # v2: explicit disclosure, no longer implied
```

### 8.2 CLI

```text
axquant recover --artifact DIR --calibration ... --output DIR
```

Convert/quantize do not call recover.

## 9. QP3 (planned)

Experimental 2/3-bit candidates remain hardware-gated. Recipes may combine `bits∈{2,3}`,
`group_size=32`, and optional recovery. Release audit still requires full quality gates; no
shortcut claims.

## 10. Tests (QP0 minimum)

1. `storage_bpw` unchanged for classic pairs; multi-group storage ordering.
2. Sensitivity report accepts two candidates at same bits/method different group sizes.
3. Prior report with `candidate_group_sizes=(32,64)` emits both.
4. Planner with multi-group report selects finer group when budget allows and prior loss favors it.
5. Allocation strategies match method mapping.
6. ProbeConfig validates multi-group subset of hardware when wired through plan request.

## 11. Migration

- Old plans without strategy fields load with defaults (`group-affine` / `none`) — acceptable for
  display; re-plan recommended for accuracy of strategy labels.
- Old single-group sensitivity reports plan as today.
- No schema_version bump for `axquant.plan.v1` / `axquant.sensitivity.v1` (additive + uniqueness
  relaxation).

## 12. Clean-room note

Pareto ladder filtering is classical multi-objective dominance (Pareto efficiency). Group-size
noise prior is an AXQuant development heuristic, not measured science. Recovery algorithms (QP2)
must cite independent sources under AXQ-027 before merge.
