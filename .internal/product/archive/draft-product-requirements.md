# AXQuant Product Requirements

> **Archived:** This draft is retained for historical context. The current, authoritative
> requirements are in [product requirements](../requirements.md).

**Status:** Draft, reviewed 2026-07-28  
**Product:** AXQuant  
**Organization:** AutomatosX / DEFAI Digital  
**Repository:** `defai-digital/axquant`  
**Initial model:** Qwen3.6-27B MLX  
**Initial platform:** Apple Silicon with MLX  
**Category:** Post-training quantization toolkit

## Executive summary

AXQuant is an AX Engine-optimized, MLX-compatible post-training quantization toolkit for
Qwen 3.6. Its initial product goal is:

> Near-6-bit quality at a near-4-bit footprint, with MTP-aware acceleration.

AXQuant produces portable MLX weights plus AX Engine manifests, MTP sidecars, runtime metadata,
transparent plans, and complete-model validation. It combines per-tensor evidence, 4/6/8/BF16
allocation, optional learned refinement, global candidate evaluation, workload profiles, and
measured Apple Silicon costs.

The initial release target is:

```text
AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit
```

MTP is a standard capability recorded in manifests and model cards. It is not normally included
in the repository name.

## Product vision

AXQuant is the Qwen 3.6 optimization layer for the AutomatosX MLX ecosystem:

- local Apple Silicon inference;
- AX Engine and AX Serving;
- private enterprise deployments;
- agent and coding workloads;
- multilingual and long-context applications;
- additional architectures, MoE, VLM, and KV-cache recommendations after the Qwen 3.6 v1.

It optimizes deployment quality, memory, and throughput rather than model size alone.

The v1 product boundary is deliberately narrow:

- primary runtime: AX Engine;
- secondary compatibility runtime: MLX-LM;
- first checkpoint: Qwen3.6-27B;
- v1 family coverage: two or three validated Qwen 3.6 dense checkpoints;
- primary workload: agent-coding, with a general-quality baseline;
- language-path PTQ only; bundled vision weights are preserved at BF16.

Gemma is a post-v1 architecture adapter and is not a parallel launch workstream.

## Problem

Uniform 4-bit quantization may damage reasoning, coding, structured output, multilingual quality,
long-context behavior, or MTP acceptance. Uniform 6-bit generally costs materially more memory.
Existing mixed-precision methods may remain too coarse, optimize proxy metrics without
complete-model validation, exclude MTP, or ignore actual Apple Silicon performance.

AXQuant addresses:

- tensor-level sensitivity within transformer blocks;
- MTP as an explicit quality and speed objective;
- 4/6/8/BF16 choices;
- workload signals beyond KL divergence;
- full-candidate interaction effects;
- real kernel, latency, memory, and throughput measurements;
- reproducible plans and benchmark evidence.

## Primary goal and claims

The target range is `4.3–4.8` storage-adjusted bits per weight. A stronger claim such as “6-bit
quality at 4-bit size” is prohibited until repeated results across models, workloads, and devices
support it.

### Measurement definitions

- Planned storage-adjusted BPW includes assigned weight bits and affine scale/bias metadata.
- Measured artifact BPW uses the actual bytes of every indexed Safetensors shard plus the shipped
  root MTP sidecar; it is authoritative after conversion.
- The BPW denominator is the logical parameter count represented by the checkpoint.
- MTP weight bytes are included when MTP is shipped.
- Reports publish both total package BPW and language/backbone BPW so that protected MTP or vision
  storage cannot be hidden.
- Tokenizer, documentation, and report bytes are excluded from weight BPW and baseline size.
- Model-size ratios compare the same set of model and MTP weight files.
- Aggregate quality retention applies only to normalized higher-is-better scores.
- Perplexity, invalid-output rate, repetition, and divergence use directional thresholds.
- MTP retention compares the same MTP implementation and workload on the candidate and its
  high-precision or uniform-6 baseline.
- MTP speedup compares the identical AXQuant candidate with MTP disabled and enabled.

## Success criteria

### Required targets for Qwen3.6-27B

| Metric | Target |
| --- | ---: |
| Weight-file size | At most 110% of uniform MLX 4-bit |
| Effective BPW | 4.3–4.8 |
| Aggregate quality | At least 98% of normalized uniform MLX 6-bit score |
| Critical tasks | No major coding, tool, JSON, multilingual, or long-context regression |
| MTP acceptance retention | At least 95% of the BF16 or uniform-6 MTP baseline |
| MTP speedup | At least 1.20× over the same candidate with MTP disabled |
| Peak unified memory | At least 15% below uniform MLX 6-bit |
| Standard MLX backbone loading | Required |
| AX Engine model and MTP execution | Required |
| MLX-LM standard inference fallback | Required |
| Reproducible manifest and recipe | Required |

### Stretch targets

- 99% aggregate uniform-6 quality retention;
- weight-file size at most 105% of uniform 4-bit;
- 1.40× MTP speedup;
- 98% MTP acceptance retention;
- less than 2% long-context regression;
- less than 1% structured-output regression.

### Production failure conditions

A candidate cannot be released when:

- normalized quality retention is below 95% of uniform 6-bit;
- MTP reduces end-to-end throughput;
- JSON or tool-call failure materially increases;
- size exceeds the declared limit without an approved quality tradeoff;
- standard supported MLX cannot load the backbone;
- AX Engine integration or artifact integrity fails;
- the run cannot be reproduced from its recipe;
- required benchmark evidence is missing or inconsistent.

## Users

Primary users are AutomatosX model engineers, AX runtime developers, MLX publishers, Apple Silicon
application developers, and enterprise teams deploying private models. Secondary users include
quantization researchers and open-source contributors.

## Core capabilities

### Model inspection

The inspector identifies:

- architecture and transformer blocks;
- attention and MLP tensors;
- embeddings, norms, and LM head;
- MTP components and whether they are integrated or external;
- tied weights;
- current precision and quantization state;
- MoE and vision components when present;
- unsupported and recommended protected components.

The Qwen3.6-27B source declares a vision tower. v0.x preserves these tensors at BF16 but neither
optimizes nor benchmarks the vision path. A language-path AXQuant release must not claim VLM
quality or VLM memory savings.

### Per-tensor sensitivity

Initial tensor targets include `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, embeddings, LM head, and MTP projections, blocks, and output heads.

Evidence includes:

- output KL divergence;
- hidden-state mean squared error;
- cosine distance;
- output-token disagreement;
- task-loss delta;
- MTP acceptance degradation;
- measured latency and memory costs.

Architecture priors may guide experiments but are not measured sensitivity and cannot satisfy a
release gate.

### Mixed-precision planning

The initial precision set is 4-bit, 6-bit, 8-bit, and BF16. Plans are constrained by effective
BPW, weight-file size, quality, MTP acceptance, measured hardware support, and protected tensors.

Target modes:

```text
balanced
quality
low-memory
speed
```

### MTP policy

The default policy is:

```yaml
mtp:
  mode: protected
  candidate_bits: [8, 16]
  protect_norms: true
  protect_output_head: true
  optimize_for_acceptance: true
```

Selected integrated MTP tensors may use 6-bit only with measured adaptive-policy evidence.
External sidecars remain byte-preserved until a layout-aware backend is validated.

The backbone produces the first token. MTP metrics cover subsequent draft positions.

### Refinement

AXQuant orchestrates public MLX-LM quantization methods:

| Sensitivity | Precision | Preferred method |
| --- | ---: | --- |
| Low | 4-bit | affine |
| Medium | 4-bit or 6-bit | AWQ |
| High | 6-bit or 8-bit | DWQ |
| Critical | 8-bit or BF16 | protected |

GPTQ and other methods remain plugin extensions. Failed refinement must fall back safely and be
recorded.

### Global validation

AXQuant generates complete checkpoint candidates, evaluates them, identifies interaction
regressions, applies bounded precision swaps or upgrades, and repeats until convergence or budget
exhaustion. Initial search methods may include greedy allocation, beam search, coordinate descent,
and top-N candidate ranking.

### Calibration

Initial profile:

- `agent-coding`: source code, completion, repair, JSON, tool schemas, function calls, structured
  output, multi-step instructions, long prompts, and agent traces.

The `general` profile remains a required baseline suite rather than a second v0.x optimization
target. Future profiles include coding, translation, long-context, CJK, RAG, OCR, and VLM.
Calibration and evaluation data must be separated and independently identified.

### Hardware benchmarking

Required measurements:

- load time;
- peak unified memory;
- prefill and decode tokens per second;
- MTP effective tokens per second;
- batch-one latency;
- kernel fallback and dequantization overhead;
- optional energy use.

Every result records device, chip, unified memory, OS, AX Engine, MLX, MLX-LM, and AXQuant
versions.

### Export

The output includes a standard MLX checkpoint, tokenizer/config files, quantization configuration,
AXQuant manifest, benchmark reports, model card, calibration manifest, plan, and reproduction
recipe.

Runtime compatibility has two explicit levels:

| Level | Runtime | Contract |
| --- | --- | --- |
| A | AX Engine | Full mixed precision, native MTP, runtime metadata, measured performance |
| B | MLX-LM | Standard language-model inference; MTP is runtime-dependent |

The portable base is standard MLX weights. `model-manifest.json` is authoritative for AX Engine;
`axquant_manifest.json` and `axquant_runtime.json` carry optimization provenance and
recommendations. MLX-LM may ignore AX-specific metadata and fall back to ordinary decode.

## Functional requirements

| ID | Requirement | Command or artifact |
| --- | --- | --- |
| FR-0 | Audit revision, shards, logical parameters, MTP provenance, and runtime readiness | `axquant feasibility` |
| FR-1 | Load local or Hub MLX source, without implicit large download | `axquant inspect` |
| FR-2 | Emit tensor, precision, MTP, protection, and tied-weight inventory | `architecture_report.json` |
| FR-3 | Build versioned calibration cache from built-in or custom data | `axquant calibrate` |
| FR-4 | Probe 4/6/8/BF16 per supported tensor | `axquant analyze` |
| FR-5 | Generate one or more constrained plans | `axquant plan` |
| FR-5a | Apply reviewed v0.1 precision rules with hard protection and BPW checks | `axquant plan-manual` |
| FR-6 | Apply a saved plan through supported MLX backends | `axquant convert` |
| FR-7 | Compare BF16, uniform-4, uniform-6, and candidate evidence | `axquant validate` |
| FR-8 | Compare the same candidate with MTP off and on | MTP benchmark report |
| FR-9 | Emit JSON and Markdown evidence | `axquant report` |
| FR-10 | Prepare a guarded Hub-ready artifact | `axquant publish-prepare` |
| FR-11 | Generate and validate the AX native manifest | `ax-engine-bench generate-manifest` |
| FR-12 | Verify runtime readiness without loading benchmark claims | `axquant runtime-check` |

## Non-functional requirements

### Reproducibility

Every run records source and tokenizer revisions, AXQuant/MLX/MLX-LM/quantizer versions,
calibration digest, evaluation digest, seed, plan, hardware profile, and execution recipe.

### Transparency

Released checkpoints publish BPW, precision and quantizer distributions, protected tensors,
baseline comparisons, MTP status and metrics, hardware, and limitations.

### Modularity

Inspection, architecture adapters, calibration, probes, planning, quantizer backends, MTP
evaluation, hardware benchmarking, export, and reporting are separate interfaces.

### Safety

The pipeline fails closed on tensor-shape mismatch, inconsistent tied weights, tokenizer/config
drift, missing MTP artifacts, unsupported modules, malformed plans, incomplete checkpoints,
non-finite measurements, or unvalidated production claims.

### Performance and recovery

Long scans support cached activations, resume state, partial tensor ranges, deterministic seeds,
bounded sample counts, and atomic artifact writes.

## Optimization contract

For tensor `t` and candidate `b`, the proxy objective combines quality, MTP, and measured hardware
cost:

```text
proxy_loss(t, b) =
    Wkl * output_kl
  + Wh  * hidden_state_error
  + Wc  * cosine_distance
  + Wt  * token_disagreement
  + Wq  * task_loss_delta
  + Wm  * mtp_acceptance_loss
  + Wr  * peak_memory_cost
  + Wp  * prefill_latency_cost
  + Wd  * decode_latency_cost
```

Hard constraints:

```text
effective_bpw <= configured limit
weight_size_ratio <= configured limit
validated_quality_retention >= configured minimum
validated_mtp_retention >= configured minimum
validated_mtp_speedup >= configured minimum
unsupported tensors remain protected
artifact integrity passes
```

The first two constraints can be checked during plan construction. Complete-model validation is
the only authority for the quality, MTP, speed, and integrity constraints.

## Benchmark framework

Required baselines:

1. BF16 or highest available source;
2. uniform MLX 4-bit;
3. uniform MLX 6-bit;
4. an available attributed mixed-precision baseline;
5. available MLX AWQ;
6. available MLX DWQ;
7. AXQuant candidate with MTP disabled;
8. the identical candidate with MTP enabled.

Required categories are general quality, coding, agentic use, Traditional Chinese, Simplified
Chinese, Japanese, English, translation, long context, and MTP. Initial hardware includes one
Max-class machine and one Ultra-class machine when available; exact hardware is mandatory in the
report.

## Scope exclusions for v1

- Gemma and other non-Qwen model families;
- arbitrary Qwen generations;
- full VLM quantization;
- arbitrary per-channel precision execution;
- dynamic runtime KV-cache quantization;
- complete expert-aware MoE planning;
- distributed quantization;
- GGUF, CUDA, Windows, or Linux runtime outputs;
- model training or full fine-tuning;
- every MLX architecture;
- production 2/3/5-bit formats.

Qwen 3.6 vision tensors may be present in the portable checkpoint, but preservation is not VLM
optimization support.

The architecture must permit these later without a rewrite.

## Milestones

| Milestone | Result | Exit condition |
| ---: | --- | --- |
| M0 | Technical feasibility | Versioned 4/6/mixed audits pass; Qwen 3.6, MTP, kernels, and benchmark interfaces are mapped |
| M1 | AX Engine vertical slice | Manual PTQ, portable MLX output, both runtime contracts |
| M2 | MTP benchmark harness | Trustworthy AX Engine MTP-on/off evidence |
| M3 | MTP-aware planner | Per-tensor 4/6/8/BF16 allocation with runtime costs |
| M4 | Quality refinement | AWQ/DWQ, global candidates, agent-coding calibration |
| M5 | Family proof | A second Qwen 3.6 dense checkpoint and MLX-LM hardening |
| M6 | Global optimization | Complete-model interaction correction |
| M7 | Hardware-aware RC | Measured Apple cost and Pareto frontier |
| M8 | AXQuant v1.0 | Public toolkit and two or three validated Qwen 3.6 checkpoints |
| M9 | Workload profiles | Coding, translation, CJK, and long context |
| M10 | Hardware registry | Reusable Apple Silicon profiles |
| M11 | MoE | Router/shared/expert-aware planning |
| M12 | KV optimization | AX Engine runtime recommendations |
| M13 | VLM | Vision/projector/cross-attention support |
| M14 | Adaptive precision | Hardware-aligned groups and new formats |

Release sequence:

```text
v0.1 Qwen3.6-27B feasibility gate, AX Engine vertical slice, MTP preservation, manual mixed precision,
     AX manifest, standard MLX output, MLX-LM fallback; no MTP speed claim
v0.2 AX Engine MTP benchmark harness, correctness, draft-position accuracy, acceptance,
     identical-checkpoint MTP-on/off throughput
v0.3 per-tensor sensitivity and automated MTP-aware planner using AX Engine latency and RAM
v0.4 AWQ/DWQ, full-candidate validation, precision swaps, agent-coding calibration,
     4.3–4.8 BPW Pareto search
v0.5 second Qwen 3.6 dense checkpoint and MLX-LM compatibility hardening
v0.6 interaction-aware validation
v0.7 hardware benchmark registry and Pareto reports
v0.8 Qwen3.6 release candidate
v0.9 external testing and compatibility fixes
v1.0 public Qwen 3.6 toolkit and validated reference checkpoints
```

## v1 acceptance

AXQuant v1 requires:

- stable versioned CLI and configuration;
- Qwen3.6-27B plus at least one additional validated Qwen 3.6 dense checkpoint;
- AX Engine Compatibility Level A and MLX-LM Compatibility Level B;
- per-tensor 4/6/8/BF16 analysis;
- at least one production learned-refinement method;
- global complete-model validation;
- automatic MTP detection, policy, acceptance, and throughput measurement;
- at least 98% normalized uniform-6 aggregate quality;
- no major critical-task regression;
- at most 110% of uniform-4 weight-file size;
- lower peak memory than uniform 6-bit;
- at least 1.20× candidate MTP speedup;
- complete software, data, seed, plan, benchmark, and recipe provenance.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| MTP provides little or negative speedup | Protect MTP, tune horizon, require measured positive throughput |
| Near-6 quality does not fit the budget | Use 4.3–4.8 BPW, refinement, workload focus, balanced/quality variants |
| 6-bit or new formats are slower | Microbenchmark kernels and use latency as a constraint |
| Calibration overfits benchmarks | Separate data, publish composition, use hidden holdouts |
| Tensor interactions invalidate proxies | Evaluate full candidates and run bounded precision swaps |
| MLX API changes | Pin validated versions, adapters, regression tests, manifest versions |
| MLX-LM Qwen MTP remains immature | Keep v0.1 PTQ independent of MTP speed; use AX Engine as MTP authority |
| Product appears to rename another method | Demonstrate per-tensor, MTP, 4/6/8/BF16, refinement, global and hardware evidence |
| Licensing concerns | Independent implementation, public APIs/research, attribution, release license review |

## Required artifacts

```text
architecture_report.json
feasibility_report.json
feasibility_report.md
model-manifest.json
calibration_manifest.json
calibration_cache/
sensitivity_map.json
sensitivity_map.safetensors
hardware_profile.json
candidate_allocations.json
quantization_plan.json
manual_recipe.yaml
quantizer_execution_log.json
benchmark_report.json
benchmark_report.md
axquant_manifest.json
axquant_runtime.json
runtime_check.json
reproduction_recipe.yaml
```

## Workstreams

- MLX and model adapters: inspection, Qwen, tensor mapping, export, compatibility.
- Quantization research: sensitivity, planner, refinement, interaction search.
- MTP: detection, acceptance, throughput, AX Engine integration.
- Benchmarking: calibration, quality, agent/coding, multilingual, hardware, reproducibility.
- Product and release: CLI, schemas, logs, manifests, model cards, Hub, automation.

## Independent implementation

AXQuant uses public MLX and MLX-LM interfaces, public research, and independently licensed
calibration data. It does not copy mlx-optiq code, tests, text, calibration samples, or generated
metadata, and it does not claim to be an official successor.
