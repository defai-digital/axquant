# ADR-002 — Joint Weight and KV Precision Optimization

**Status:** Proposed
**Decision Date:** 2026-08-10
**Component:** AXQuant Planner
**Decision Type:** Architecture

## Context

AXQuant currently treats quantized model weights as the primary optimization domain.

For long-context inference this is insufficient because KV-cache memory grows with context length and can become a material or dominant runtime memory component.

Weight quantization and KV quantization compete for the same finite device memory.

Optimizing the two independently can therefore produce suboptimal deployment configurations.

We need an architecture that:

1. preserves existing AXQuant weight planners;
2. introduces KV policies without tightly coupling them to one runtime;
3. supports measured hardware evidence;
4. can later expand to activation and MoE optimization;
5. maintains AXQuant certification and reproducibility guarantees.

---

# Decision

AXQuant will introduce a **Deployment Precision Planner** above the existing weight planner.

Architecture:

```text
                   Deployment Planner
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Weight Planner              KV Planner
             │                         │
             ▼                         ▼
       Weight Policy                KV Policy
             │                         │
             └────────────┬────────────┘
                          ▼
                    Joint Candidate
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         Quality Evaluator      Runtime Evaluator
              │                       │
              └───────────┬───────────┘
                          ▼
                   Pareto Optimizer
                          │
                          ▼
                  Deployment Policy
```

The planner will use a **candidate-based architecture**, rather than hard-coding one specific quantization algorithm.

---

# Decision 1 — Keep weight and KV planners independent

The weight planner remains responsible for:

* tensor precision;
* quantization method;
* weight BPW;
* existing sensitivity analysis.

The KV planner is responsible for:

* KV precision;
* optional layer grouping;
* KV memory;
* KV quality effects.

The Deployment Planner combines the two.

### Rationale

This allows:

* weight-only workflows to remain unchanged;
* KV backends to evolve independently;
* easier testing;
* easier addition of activation optimization later.

---

# Decision 2 — Optimize Pareto frontier first

AXQuant will NOT initially collapse all objectives into a single arbitrary scalar.

Instead it will model:

```text
Quality
Memory
Latency
```

and compute non-dominated candidates.

Users or policy profiles may later select a candidate.

### Rationale

A scalar function such as:

[
Q+\lambda M+\mu T
]

requires arbitrary coefficients.

The Pareto frontier retains more information and is easier to audit.

---

# Decision 3 — Static KV precision before dynamic KV precision

MVP supports static:

```text
BF16
INT8
INT4
```

or backend-supported equivalents.

Dynamic token-level or query-aware KV policies are deferred.

### Rationale

Dynamic policies materially increase:

* runtime complexity;
* kernel requirements;
* certification difficulty;
* evaluation complexity.

Static policies are sufficient to validate the joint-optimization hypothesis.

---

# Decision 4 — Runtime capability registry

KV precision MUST be resolved through a backend capability registry.

Example:

```yaml
backend: ax-engine
device: apple-m-series

kv:
  bf16: true
  int8: true
  int4: true
  int6: false
```

Unsupported policies MUST fail closed.

### Rationale

AXQuant must never produce a theoretically valid plan that the selected runtime cannot execute.

---

# Decision 5 — Separate analytical estimates from measured evidence

Every metric MUST carry provenance:

```text
estimated
measured
certified
```

Example:

```json
{
  "peak_memory_gb": 21.4,
  "evidence_level": "measured"
}
```

### Rationale

Analytical memory estimates are useful during search but cannot substitute for runtime certification.

---

# Decision 6 — Context length is part of the deployment policy

A deployment plan MUST declare its intended context profile.

A policy certified at 8K MUST NOT automatically be treated as certified at 128K.

Example:

```yaml
context:
  min: 1
  certified: 32768
  max_tested: 65536
```

---

# Decision 7 — AXQuant remains runtime-neutral at planner level

Core planner abstractions MUST NOT depend directly on MLX objects.

Backend-specific runtime execution belongs behind adapters.

Example:

```text
core/
runtime/
  mlx/
  ax_engine/
  cuda/
  rocm/
```

### Rationale

This is necessary if AXQuant later supports CUDA/ROCm.

---

# Alternatives Considered

## A. Build KV quantization directly into the existing weight planner

Rejected.

Reason:

* increases coupling;
* makes future activation optimization difficult;
* mixes fundamentally different memory models.

## B. Keep weight and KV optimization completely separate

Rejected.

Reason:

* fails to optimize the real shared memory budget;
* misses the primary research/product opportunity.

## C. Optimize only total memory

Rejected.

Reason:

A low-memory configuration can have unacceptable quality or latency.

## D. Implement dynamic KV quantization first

Rejected for MVP.

Reason:

Research potential is high, but implementation and evaluation risk are too high for the first joint-optimization release.

---

# Consequences

## Positive

* expands AXQuant beyond weight-only quantization;
* high relevance to long-context inference;
* provides a strong research hypothesis;
* improves AX Engine integration;
* architecture naturally extends to activation and MoE optimization.

## Negative

* significantly larger benchmark matrix;
* runtime-specific KV implementations required;
* certification becomes context dependent;
* quality evaluation becomes more expensive.

---

# Future Extension

The same architecture allows:

```text
Deployment Planner
   ├── Weight Planner
   ├── KV Planner
   ├── Activation Planner
   └── MoE Planner
```

The long-term optimization problem becomes:

[
P^*
===

\arg\min_P
\Delta Quality(P)
]

subject to deployment constraints:

[
Memory(P,H,L,B_s)\le B
]

[
Latency(P,H)\le T
]

where (P) represents the complete inference precision policy.
