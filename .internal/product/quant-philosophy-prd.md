# AXQuant Quantization Philosophy Program — Product Requirements

**Document status:** Accepted for planning and phased implementation  
**Program ID:** QP (Quant Philosophy)  
**Applies to:** AXQuant v1.x follow-on (post-expansion)  
**Related:** AXQ-007, AXQ-008, AXQ-011, AXQ-012, AXQ-022, AXQ-028, AXQ-029  
**Last reviewed:** 2026-08-01

## 1. Executive summary

AXQuant’s differentiation is **auditable mixed-precision PTQ** on Apple Silicon: tensor-level
bit budgets, protection floors, MTP awareness, and release gates. Competitors such as mlx-optiq
win on end-to-end local UX (quantize + fine-tune + serve). NVIDIA NVFP4 shows that **fine-grained
scaling and dynamic-range handling** can keep ~4-bit quality close to higher precision — but as a
**Blackwell hardware format**, not as an MLX portable weight contract.

This program upgrades AXQuant’s quantization philosophy **without**:

- adopting NVFP4 or any non-MLX weight format;
- making domain fine-tune / SFT / DPO a product line;
- weakening evidence taxonomy or clean-room boundaries.

It does so by expanding the planner search space to **bits × group size × method**, making
**scale / outlier strategy first-class and reproducible**, closing the **measured refinement**
loop, and adding an **optional, provenance-bound recovery** stage that restores retention after
PTQ — never marketed as “new model capability.”

## 2. Product vision (one sentence)

> Within the portable MLX weight contract, allocate `(bits × group_size × method)` under a
> memory budget to minimize task loss; use calibration-driven range and scale refinement to
> approach low-BPW high-fidelity; treat training-style recovery as optional, fully provenance-bound
> evidence — never as the default convert path.

## 3. Product boundary

### 3.1 In scope

| Phase | Theme | User-visible outcome |
| --- | --- | --- |
| **QP0** | 3-D configuration space + strategy metadata | Plans can assign different group sizes and methods per tensor; allocations record scale/outlier strategy |
| **QP1** | Measured closed-loop refinement + role policies | Holdout-driven plan swaps; sensitive roles default to finer groups / AWQ when evidence exists |
| **QP2** | Optional quantization recovery | `recover` restores retention without domain SFT; recovery manifest is required for claims |
| **QP3** | Experimental ultra-low bit | 2/3-bit paths gated experimentally; only with fine groups + optional recovery evidence |

### 3.2 Explicit non-goals

- NVFP4 / MXFP4 / E2M1 hardware formats or CUDA export.
- Full QAT productization as the default path.
- Domain SFT, DPO, preference alignment, or coding-agent training inside AXQuant.
- Becoming a local “Lab + serve + fine-tune” stack (mlx-optiq territory).
- Weakening `architecture_prior` vs `measured` gates, or reusing mlx-optiq assets (AXQ-001).
- Claiming recovery “improves the model” beyond pre-quant capability retention.

### 3.3 Competitive posture (AXQ-022)

| Competitor | AXQuant response under this program |
| --- | --- |
| **mlx-optiq** | Stay the **auditable PTQ + optional recovery** path; do not chase full fine-tune UX |
| **NVFP4** | Learn **block scale / dynamic range** ideas; implement as group size + affine/AWQ/DWQ strategies on MLX |
| **Uniform 4-bit MLX-LM** | Keep mixed precision + protection floors; prove retention at comparable or slightly higher BPW |

## 4. Users and jobs

1. **Release engineers** — need reproducible plans with explicit group/method/strategy and gates.
2. **Self-converters** — benefit when recipe bundles encode better 3-D assignments without re-probing.
3. **Runtime owners (AX Engine)** — need plans that stay within portable MLX + documented experimental bits.

## 5. Goals and success metrics

### 5.1 Primary goals

1. **Configuration completeness:** planner and probe grids cover `bits × group_size × method`
   allowed by hardware profiles.
2. **Fidelity at budget:** at fixed target BPW, dual-profile retention ≥ prior bit-only planner
   on the same measured report (or equal within documented noise).
3. **Provenance:** every allocation records bits, method, group size, and scale/outlier strategy;
   recovery (if used) has its own manifest with digests.
4. **Default purity:** `convert` / `quantize` never require recovery or training.

### 5.2 Phase acceptance (product)

| Phase | Acceptance |
| --- | --- |
| QP0 | Schema + planner + prior analyzer + tests green; CLI can request multi-group candidates; plans deserialize backward-compatibly |
| QP1 | Role policy documented and enforced when measured AWQ candidates exist; refinement can consume holdout measurements without inventing metrics |
| QP2 | Optional `recover` produces checkpoint + recovery manifest; quality gate requires dual-profile pass; default pipeline unchanged |
| QP3 | Experimental 2/3-bit only behind existing AX Engine gates; release wording forbids production claims without full audit |

### 5.3 Claim policy

- PTQ improvements: “higher retention / lower loss at the same BPW” with suite + revision.
- Recovery: “restores retention after quantization” — never “fine-tuned for coding/chat.”
- NVFP4 comparisons: external format baseline only; never “AXQuant ships NVFP4.”

## 6. Requirements

### R-QP-001 — Three-dimensional candidates

The sensitivity report and planner SHALL treat a candidate as unique by
`(bits, method, group_size)` (BF16: method only, no group size).

### R-QP-002 — PlanRequest multi-group grid

`PlanRequest` SHALL accept `candidate_group_sizes`. Empty means “use `group_size` only”
(backward compatible). Non-empty grids MUST subset `hardware.supported_group_sizes`.

### R-QP-003 — Pareto option ladder

For each tensor, the planner SHALL build a storage-ordered option ladder, collapsing same
`(bits, group_size)` to the best-loss method, dropping storage-dominated options, then apply
existing marginal-efficiency upgrades under the BPW budget.

### R-QP-004 — Strategy metadata on allocations

Each `Allocation` SHALL record:

- `scale_strategy` (e.g. `group-affine`, `channel-awq`, `none` for BF16);
- `outlier_strategy` (e.g. `none`, `percentile-clip-dwq`);
- optional `strategy_metadata` map for digests / alphas (strict, JSON-serializable scalars).

### R-QP-005 — Architecture priors for multi-group

Architecture-prior reports SHALL emit multi-group candidates when requested, with prior metrics
that **weakly prefer smaller groups** (development evidence only).

### R-QP-006 — Probe multi-group (QP0 complete when wired)

`ProbeConfig` SHALL accept `candidate_group_sizes` and measure each `(bits, method, group_size)`
pair under the same calibration contract. Resume keys include group size.

### R-QP-007 — Role-aware defaults (QP1)

When measured candidates exist, policy MAY prefer smaller groups and AWQ for attention / high
sensitivity roles without lowering protection floors (AXQ-007 / AXQ-026).

### R-QP-008 — Holdout refinement (QP1)

Global refinement SHALL be able to bind complete-candidate holdout measurements; proxy-only
refinement remains development evidence.

### R-QP-009 — Optional recovery (QP2)

A recovery stage MAY update scales/biases or merged low-rank adapters on a converted checkpoint
using calibration-only data. It MUST write `axquant.recovery.v1` with full provenance. Domain
task fine-tune is out of scope.

### R-QP-010 — No default training

Quick convert, staged convert, and certified release paths MUST succeed without recovery.

## 7. Milestones

| Milestone | Phase | Deliverables |
| --- | --- | --- |
| M-QP0 | QP0 | ADR AXQ-028/029, tech spec, schema+planner+analyzer(+probe grid), tests, decision register |
| M-QP1 | QP1 | Role policies, measured refine binding, quality evidence on ≥1 development candidate |
| M-QP2 | QP2 | `recover` CLI, recovery schema, validation integration, docs claim language |
| M-QP3 | QP3 | Ultra-low-bit experimental recipes + gate wording |

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| Probe cost multiplies with group × method grid | Cap grid defaults; allow tensor-target subsets; early termination remains |
| Smaller groups raise BPW and fail size gate | Planner budget already uses `storage_bpw`; size gate unchanged |
| Users confuse recovery with fine-tune | Claim policy + CLI labeling + non-goals in README |
| Schema break | Additive fields + safe defaults; uniqueness key change needs migration note |

## 9. Dependencies

- Portable MLX-LM conversion remains the weight packing authority (AXQ-004).
- AWQ / DWQ plugins already exist; QP0–QP1 route planner selection, not new formats.
- Clean-room: algorithm provenance for multi-group Pareto and recovery must be recorded
  (extend ADR 0006 / AXQ-027 as algorithms land).

## 10. Open questions (resolved for QP0)

| Question | Resolution |
| --- | --- |
| Adopt NVFP4 format on MLX? | **No** (AXQ-028). |
| Is fine-tune a mainline feature? | **No**; only optional recovery in QP2 (AXQ-029). |
| Change plan schema major version? | **No** for QP0 additive fields; uniqueness key change is backward-compatible for old single-gs reports. |
