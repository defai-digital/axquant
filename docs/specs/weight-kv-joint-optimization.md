# AXQuant Weight + KV Joint Optimization — Technical Specification

**Status:** Draft
**Target:** AXQuant v2.x
**Language:** Python
**Primary Initial Backend:** MLX / AX Engine

# 1. Architecture

Introduce the following logical modules:

```text
src/axquant/
    deployment/
        planner.py
        candidate.py
        pareto.py
        constraints.py
        schema.py

    kv/
        inspector.py
        estimator.py
        sensitivity.py
        planner.py
        policy.py
        evaluator.py

    runtime/
        capabilities.py
        mlx_kv.py
        ax_engine_kv.py

    evidence/
        deployment.py
```

Existing weight planning remains reusable.

---

# 2. Core Data Model

## 2.1 KVPolicy

```python
@dataclass(frozen=True)
class KVPolicy:
    dtype: str
    bits: int
    group_size: int | None = None
    per_layer: dict[str, "KVLayerPolicy"] | None = None
```

Example:

```json
{
  "dtype": "int4",
  "bits": 4,
  "group_size": 64
}
```

---

## 2.2 WeightPolicyRef

Do not duplicate existing weight plans.

```python
@dataclass(frozen=True)
class WeightPolicyRef:
    plan_id: str
    bpw: float
    artifact_id: str | None
```

---

## 2.3 DeploymentCandidate

```python
@dataclass
class DeploymentCandidate:
    weight_policy: WeightPolicyRef
    kv_policy: KVPolicy

    context_length: int
    batch_size: int

    estimated_memory_bytes: int | None
    measured_memory_bytes: int | None

    quality_score: float | None
    quality_delta: float | None

    prefill_ms: float | None
    decode_tok_s: float | None

    evidence: dict
```

---

# 3. KV Memory Model

For transformer layer (l):

[
M_{KV,l}
========

2
\times
B
\times
L
\times
H_{KV,l}
\times
D_l
\times
Bytes(q_l)
]

where:

* factor 2 = K + V;
* (B) = batch size;
* (L) = context length;
* (H_{KV,l}) = number of KV heads;
* (D_l) = head dimension;
* (Bytes(q_l)) = effective bytes per KV element.

Total:

[
M_{KV}
======

\sum_l M_{KV,l}
+
M_{metadata}
]

For grouped quantization, metadata overhead MUST include:

* scales;
* zero points if used;
* alignment;
* runtime padding.

---

# 4. Hardware Memory Model

Deployment memory:

[
M_{deployment}
==============

M_W
+
M_{KV}
+
M_A
+
M_R
]

where:

* (M_W): weight memory;
* (M_{KV}): KV memory;
* (M_A): activation workspace;
* (M_R): runtime overhead.

MVP planner MAY estimate (M_A + M_R) using a calibrated runtime reserve.

Example:

```yaml
runtime_reserve:
  mode: measured
  bytes: 2147483648
```

Certification MUST use measured peak memory.

---

# 5. Search Space

Initial search:

```text
Weight policies:
    existing AXQ candidate plans

KV:
    BF16
    INT8
    INT4
```

Therefore:

[
N_{candidate}
=============

N_W\times N_{KV}
]

This is deliberately small for MVP.

Later:

[
N
=

N_W
\times
\prod_l N_{KV,l}
]

which will require pruning or optimization.

---

# 6. Feasibility Filter

Before expensive quality testing:

```python
for weight in weight_candidates:
    for kv in kv_candidates:
        candidate = combine(weight, kv)

        memory = estimate(candidate)

        if memory > memory_budget:
            continue

        if not runtime.supports(candidate):
            continue

        emit(candidate)
```

This prevents expensive evaluation of impossible configurations.

---

# 7. Evaluation Pipeline

```text
Candidate
   ↓
Static Validation
   ↓
Memory Estimation
   ↓
Runtime Compatibility
   ↓
Short Quality Evaluation
   ↓
Long-Context Evaluation
   ↓
Runtime Benchmark
   ↓
Evidence Record
   ↓
Pareto Filtering
```

---

# 8. Quality Metrics

Minimum recommended evaluation:

## 8.1 Perplexity

Use identical evaluation data for all candidates.

## 8.2 Token/logit divergence

Measure divergence against BF16 baseline.

Possible metric:

[
D_{KL}
(
P_{BF16}
\parallel
P_Q
)
]

## 8.3 Long-context retrieval

Evaluate at:

```text
8K
32K
128K
```

where architecture permits.

## 8.4 Task quality

At least one:

* reasoning;
* coding;
* instruction following;
* domain-specific evaluation.

---

# 9. KV Sensitivity

Phase 1:

Uniform precision.

Phase 2:

Measure per-layer KV sensitivity.

For layer (l):

[
S_l(q)
======

## Q_{baseline}

Q(KV_l\rightarrow q)
]

Then solve:

[
\min_{{q_l}}
\sum_l S_l(q_l)
]

subject to:

[
M_W
+
\sum_l M_{KV,l}(q_l)
\le B
]

---

# 10. Joint Quality Interaction

Do NOT initially assume:

[
\Delta Q(W_q,KV_q)
==================

\Delta Q(W_q)
+
\Delta Q(KV_q)
]

Measure joint effect.

Define:

[
I_{W,KV}
========

## \Delta Q(W_q,KV_q)

## \Delta Q(W_q)

\Delta Q(KV_q)
]

This metric SHOULD be persisted even in MVP.

It may later become a research contribution if joint interaction is substantial.

---

# 11. Pareto Algorithm

Candidate (A) dominates candidate (B) if:

[
Q_A\ge Q_B
]

[
M_A\le M_B
]

[
T_A\le T_B
]

and at least one inequality is strict.

Pseudo-code:

```python
def pareto_frontier(candidates):
    frontier = []

    for a in candidates:
        dominated = False

        for b in candidates:
            if b is a:
                continue

            if dominates(b, a):
                dominated = True
                break

        if not dominated:
            frontier.append(a)

    return frontier
```

MVP candidate count is small enough for (O(n^2)).

---

# 12. Recommendation Policy

After Pareto filtering, choose recommendation according to user objective.

Examples:

```yaml
objective: balanced
```

```yaml
objective: max_quality
```

```yaml
objective: min_memory
```

```yaml
objective: max_throughput
```

For publication/certification, AXQuant MUST retain the complete Pareto frontier.

---

# 13. CLI

## Optimize

```bash
axquant deployment optimize \
    --model MODEL \
    --weight-plans ./plans \
    --kv bf16,int8,int4 \
    --context 32768 \
    --batch-size 1 \
    --memory-budget 28GB \
    --hardware auto \
    --output deployment-plan.json
```

## Evaluate

```bash
axquant deployment evaluate \
    --plan deployment-plan.json \
    --suite long-context
```

## Certify

```bash
axquant deployment certify \
    --plan deployment-plan.json \
    --hardware current \
    --output certificate.json
```

---

# 14. Deployment Manifest

Example:

```json
{
  "schema_version": "2.0",
  "model": {
    "id": "model-name",
    "revision": "immutable-revision"
  },
  "weight": {
    "plan_id": "axq-plan-001",
    "bpw": 5.17
  },
  "kv": {
    "dtype": "int4",
    "bits": 4,
    "group_size": 64
  },
  "workload": {
    "context_length": 32768,
    "batch_size": 1
  },
  "constraints": {
    "memory_budget_bytes": 30064771072
  },
  "results": {
    "peak_memory_bytes": 25769803776,
    "decode_tok_s": 41.3,
    "quality_ratio": 0.991
  },
  "evidence": {
    "memory": "measured",
    "quality": "measured",
    "runtime": "measured"
  }
}
```

---

# 15. Runtime Adapter Interface

```python
class KVRuntimeAdapter(Protocol):
    def capabilities(self) -> KVCapabilities: ...

    def estimate_memory(
        self,
        model_info,
        policy: KVPolicy,
        context_length: int,
        batch_size: int,
    ) -> int: ...

    def benchmark(
        self,
        model,
        deployment_policy,
    ) -> RuntimeResult: ...
```

---

# 16. Fail-Closed Rules

AXQuant MUST reject certification if:

* runtime does not support requested KV format;
* measured context differs from claimed context;
* model revision differs;
* calibration overlaps formal holdout;
* required benchmark evidence is missing;
* runtime fallback silently changes precision;
* peak memory is estimated but certificate claims measured memory.

---

# 17. Testing

## Unit Tests

```text
test_kv_estimator.py
test_kv_policy.py
test_deployment_candidate.py
test_pareto.py
test_constraints.py
test_runtime_capabilities.py
```

## Integration Tests

```text
test_weight_kv_joint_plan.py
test_mlx_kv_runtime.py
test_ax_engine_kv_runtime.py
test_deployment_manifest.py
```

## Certification Tests

```text
test_context_binding.py
test_runtime_precision_binding.py
test_memory_evidence.py
test_joint_quality_evidence.py
```

---

# 18. Benchmark Matrix

Minimum research benchmark:

```text
Models:
  Dense model A
  Dense model B
  MoE model A

Weight:
  BF16
  uniform Q4
  uniform Q6
  AXQ-4
  AXQ-6

KV:
  BF16
  INT8
  INT4

Context:
  8K
  32K
  128K
```

Important comparisons:

```text
Uniform Q4 + BF16 KV
Uniform Q6 + BF16 KV
AXQ4 + KV8
AXQ4 + KV4
AXQ6 + KV8
AXQ6 + KV4
AXQuant Joint Optimized
```

---

# 19. Required Ablations for Paper

Ablation A:

```text
Weight-only optimization
vs
Weight + KV optimization
```

Ablation B:

```text
Memory-estimate-only
vs
Measured hardware cost
```

Ablation C:

```text
Independent quality estimate
vs
Measured joint quality
```

Ablation D, future:

```text
Uniform KV
vs
Per-layer KV
```

---

# 20. Acceptance Criteria

MVP is complete when:

1. Existing AXQuant weight plans can be reused without modification.
2. KV BF16/8/4 candidate policies can be represented.
3. Memory can be estimated for each candidate.
4. Unsupported backend combinations fail before conversion.
5. Joint candidates can be evaluated.
6. A Pareto frontier is generated.
7. Deployment policy is serializable.
8. AX Engine can consume at least one policy format.
9. Measured runtime evidence can be attached.
10. Certification binds policy to context length and hardware.

---

# 21. Recommended Implementation Order

### Milestone 1

```text
KVPolicy
KV model inspector
KV memory estimator
backend capability registry
```

### Milestone 2

```text
DeploymentCandidate
weight/KV combination
constraint filtering
Pareto frontier
```

### Milestone 3

```text
MLX/AX Engine runtime measurement
quality evaluator
deployment manifest
```

### Milestone 4

```text
certification
benchmark matrix
ablation
```

### Milestone 5

```text
per-layer KV sensitivity
interaction-aware Weight × KV optimization
```

---

# 22. Long-Term Research Extension

After MVP evidence exists, extend the objective to:

[
\boxed{
P^*
===

\arg\min_P
[
\Delta Q(P)
+
\lambda_I I(P)
+
\lambda_T T(P,H)
]
}
]

subject to:

[
M(P,H,L,B_s)\le B
]

where the complete policy may eventually contain:

```text
Weight precision
Weight quantizer
KV precision
Activation precision
MoE expert precision
Router protection
Interaction terms
Hardware deployment policy
```

This provides a technically coherent path from today's AXQuant to a general **LLM inference precision optimizer**.
