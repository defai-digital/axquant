# AXQuant Weight + KV Joint Optimization — PRD

**Status:** Draft (full joint optimizer)
**Target:** AXQuant v2.x for the complete design; **1.8.0 extracts only one shared memory budget** — see [`docs/roadmap/v1.8.0/adr/0009-one-deployment-budget.md`](../roadmap/v1.8.0/adr/0009-one-deployment-budget.md)
**Owner:** DEFAI Digital / AXQuant
**Document Type:** Product Requirements Document

## 1. Executive Summary

AXQuant currently optimizes model weight quantization using measured sensitivity, mixed precision, multiple quantization methods, runtime evidence, and certification.

However, for long-context LLM inference, model weights are only one component of memory usage.

Total inference memory is approximately:

[
M_{total}
=========

M_{weights}
+
M_{KV}
+
M_{activation}
+
M_{runtime}
]

As context length grows, KV-cache memory becomes increasingly important.

AXQuant should therefore evolve from a weight-only quantization optimizer into an **LLM precision deployment optimizer** that jointly selects:

* weight precision;
* weight quantization method;
* KV-cache precision;
* optional layer-specific KV policy;

under explicit quality, memory, and latency constraints.

The core product goal is:

> Find the best weight + KV precision configuration for a target model, hardware platform, context length, and memory budget.

---

# 2. Problem Statement

Current deployment choices are typically made independently.

Example:

* Model weights: Q4
* KV cache: FP16

or:

* Model weights: Q6
* KV cache: Q4

These configurations are usually selected manually.

This creates three problems.

## 2.1 Weight and KV precision compete for the same memory budget

A lower-weight precision may permit a larger KV cache.

Conversely, preserving higher-quality weights while compressing KV may produce better end-to-end model quality.

The best configuration therefore cannot always be found by optimizing weights independently.

## 2.2 Optimal configuration changes with context length

For short context:

```text
Q4 Weight + KV8
```

may be efficient.

For long context:

```text
Q6 Weight + KV4
```

may provide a better quality-memory tradeoff.

The correct policy therefore depends on:

* context length;
* batch size;
* model architecture;
* workload;
* hardware.

## 2.3 Current quantization tools expose choices but do not optimize the joint deployment problem

AXQuant should solve:

[
(q_W^*,q_{KV}^*)
================

\arg\min
\Delta Q(q_W,q_{KV})
]

subject to:

[
M_W(q_W)
+
M_{KV}(q_{KV},L,B_s)
\le M_{budget}
]

where:

* (q_W): weight quantization policy;
* (q_{KV}): KV precision policy;
* (L): context length;
* (B_s): batch size;
* (M_{budget}): available memory.

---

# 3. Product Vision

AXQuant becomes:

> **A measurement-driven precision optimizer for deployed LLM inference.**

Longer-term:

```text
AXQuant
   │
   ├── Weight Optimizer
   ├── KV Optimizer
   ├── MoE Optimizer
   └── Activation Optimizer
            │
            ▼
     Deployment Planner
            │
            ▼
 Quality / Memory / Latency
       Pareto Frontier
```

MVP scope is limited to:

```text
Weight + KV
```

---

# 4. Goals

## G1. Joint optimization

AXQuant MUST be able to jointly select weight and KV-cache precision.

## G2. Explicit deployment budgets

Users MUST be able to specify constraints such as:

* maximum model BPW;
* maximum total memory;
* target context length;
* batch size;
* minimum quality threshold;
* optional latency target.

## G3. Measurement-driven decisions

Planner decisions MUST be based on measured or certified evidence whenever available.

## G4. Pareto frontier generation

AXQuant SHOULD generate multiple non-dominated candidate configurations rather than returning only one answer.

Example:

| Plan | Weight | KV  | Memory | Quality | tok/s |
| ---- | ------ | --- | -----: | ------: | ----: |
| A    | AXQ-4  | KV8 |  21 GB |   98.7% |    42 |
| B    | AXQ-6  | KV4 |  22 GB |   99.2% |    45 |
| C    | AXQ-4  | KV4 |  18 GB |   97.9% |    48 |

## G5. Certification

Weight + KV plans SHOULD integrate with AXQuant's existing evidence and certification model.

---

# 5. Non-Goals — Initial Release

The first release will NOT attempt to provide:

* W4A4 activation quantization;
* dynamic per-token KV precision;
* KV eviction;
* semantic KV compression;
* learned cache compression;
* expert prefetching;
* router fine-tuning;
* training-aware quantization;
* arbitrary CUDA/ROCm backend support.

These can be future extensions.

---

# 6. Primary Users

## 6.1 Local LLM users

Need to fit larger models and longer contexts into fixed unified memory.

## 6.2 Model publishers

Need reproducible Q4/Q6 releases with documented context-memory behavior.

## 6.3 Enterprise deployment teams

Need predictable:

* memory requirements;
* quality;
* throughput;
* maximum context.

## 6.4 AX Engine

Can consume AXQuant-generated runtime policy and execute the recommended configuration.

---

# 7. Core User Stories

### US-1

As a user with 32 GB RAM, I want AXQuant to determine the best weight and KV precision for a 27B model at 32K context.

### US-2

As a model publisher, I want to certify that a model stays within an acceptable quality envelope at 8K, 32K, and 128K context.

### US-3

As an AX Engine user, I want the runtime to load the deployment policy produced by AXQuant automatically.

### US-4

As a researcher, I want to compare uniform quantization against jointly optimized weight + KV configurations.

---

# 8. Functional Requirements

## FR-1 Model inspection

AXQuant MUST identify:

* architecture;
* attention layers;
* number of KV heads;
* head dimension;
* layer count;
* supported KV data types;
* backend capabilities.

## FR-2 KV memory estimator

AXQuant MUST estimate KV memory as a function of:

* context length;
* batch size;
* layer count;
* KV heads;
* head dimension;
* precision.

The estimator MUST later be validated against actual runtime memory measurements.

## FR-3 KV candidate generation

Initial precision candidates:

```text
FP16/BF16
INT8
INT6 where backend-supported
INT4
```

Support SHOULD be backend capability driven.

## FR-4 Weight candidate integration

Use existing AXQuant candidates such as:

```text
AXQ-4
AXQ-6
Q8
BF16
GPTQ
AWQ
Affine
```

where supported.

## FR-5 Joint planner

Planner MUST evaluate combinations of:

```text
WeightPlan × KVPlan
```

and return feasible candidates.

## FR-6 Quality evaluation

Quality evaluation MUST include at least:

* short-context quality;
* long-context quality;
* generation consistency;
* perplexity or equivalent model metric;
* selected downstream tasks.

## FR-7 Runtime evaluation

Where supported, AXQuant SHOULD measure:

* peak memory;
* steady-state memory;
* prefill latency;
* decode latency;
* tokens/sec.

## FR-8 Pareto selection

Dominated configurations MUST be removed.

A plan is dominated if another plan is:

* equal or better in quality;
* equal or lower in memory;
* equal or faster in latency;

with at least one strict improvement.

## FR-9 Evidence output

Every measured plan MUST retain:

* model revision;
* AXQuant version;
* backend version;
* hardware identifier;
* context length;
* batch size;
* quantization policy;
* quality dataset;
* calibration dataset;
* measurement timestamp;
* artifact hashes where applicable.

## FR-10 Certification

AXQuant SHOULD support certification profiles such as:

```text
Short Context
8K

Medium Context
32K

Long Context
128K
```

---

# 9. Planner Objective

Initial objective:

[
J(q_W,q_{KV})
=============

\Delta Q(q_W,q_{KV})
+
\lambda_T \hat T(q_W,q_{KV},H)
]

subject to:

[
M_W(q_W)
+
M_{KV}(q_{KV},L,B_s)
\le B
]

Where:

* (\Delta Q): measured or estimated quality degradation;
* (\hat T): measured/predicted runtime cost;
* (H): hardware profile;
* (B): memory constraint.

AXQuant SHOULD primarily expose the Pareto frontier rather than hiding all tradeoffs behind fixed coefficients.

---

# 10. UX / CLI

Example:

```bash
axquant optimize \
  --model ./Qwen-model \
  --target-memory 28GB \
  --context 32768 \
  --batch-size 1 \
  --weights axq4,axq6 \
  --kv bf16,int8,int4 \
  --hardware auto
```

Example output:

```text
AXQuant Joint Precision Optimization

Target memory: 28.0 GB
Context:       32768
Batch:         1

Recommended:
  Weight: AXQ-6
  KV:     INT4

Estimated memory: 24.8 GB
Measured quality: 99.1% baseline
Decode:           41.2 tok/s

Alternatives:
  AXQ-4 + KV8
  AXQ-4 + KV4
```

---

# 11. Success Metrics

## Product

MVP considered successful if:

1. AXQuant can generate valid weight + KV plans for at least two supported model families.
2. Predictions are within 10% of measured peak-memory use.
3. Planner produces at least one configuration that improves the quality-memory Pareto frontier over uniform baselines.
4. AX Engine can consume the generated policy.
5. Results are reproducible under AXQuant certification rules.

## Research

Strong research result if:

[
JointOptimization

>

IndependentWeightOptimization
]

at equal memory budget across several models/context lengths.

Target:

* ≥3 model architectures;
* ≥3 context lengths;
* ≥2 hardware classes;
* complete ablation.

---

# 12. Research Hypothesis

Primary hypothesis:

> Independently selecting weight and KV precision is suboptimal under fixed deployment resource constraints.

Secondary hypothesis:

> Context-aware joint precision allocation can deliver higher quality at equal memory, or lower memory at equal quality, than uniform weight/KV baselines.

---

# 13. Rollout

## Phase 1

* KV memory estimator
* uniform KV4/KV8/BF16 candidates
* joint weight/KV planner
* static context profile

## Phase 2

* measured runtime profiles
* Pareto optimization
* certification

## Phase 3

* per-layer KV precision
* quality sensitivity measurements

## Phase 4

* interaction-aware joint optimization
* MoE-aware policies
* activation precision

---

# 14. Product Principle

AXQuant should not attempt to invent a new KV codec merely to create novelty.

Its core advantage should remain:

> **Measure available precision strategies, select the correct combination for the actual deployment constraint, and provide evidence that the decision is valid.**
