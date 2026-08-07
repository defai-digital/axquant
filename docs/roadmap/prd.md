# PRD — AXQuant completion and improvement program

- **Status:** draft
- **Date:** 2026-08-06
- **Owner:** AXQuant maintainer
- **Related:** README "Still incomplete" list, `docs/known-issues.md`,
  `docs/flagship-certification.md`, ADRs 0001–0006

## 1. Background

AXQuant converts Safetensors checkpoints into quality-protected mixed-precision
MLX artifacts (4/6/8-bit + BF16, experimental 2/3-bit behind AX Engine gates)
with an auditable evidence chain. The toolkit commands are implemented; what
remains open is (a) the first certified flagship release, (b) a set of scope
items README explicitly lists as incomplete or deferred, and (c) algorithm and
runtime headroom identified in the 2026-08 competitive review.

The strategic conclusion of that review, which this PRD adopts:

- Quality headroom is algorithmic (act-order GPTQ, interaction optimization,
  better bit/group/method frontier) and fits the existing architecture.
- Speed headroom on Apple Silicon is dominated by measured BPW, MTP, and
  KV-cache precision — decode is usually memory-bandwidth-bound — plus
  aligning the planner with what the runtime actually executes fast.
- Hardware-native GPU formats (e.g. Blackwell NVFP4) are a different product
  track and never enter the MLX certification narrative (ADR-0006). The lesson
  we take from them is *hardware-aligned formats*, applied to Apple/AX Engine.

## 2. Goals

1. **G1 — Close the flagship.** Complete Qwen 3.6 certification under the
   `qwen36-mtp-v2` policy (M0–M8, dual profiles, `df-macbookpro-m5`), or record a
   documented no-go. This is the trust anchor for everything else.
2. **G2 — Quality pack.** Measurably improve quality-at-BPW using act-order
   GPTQ, measured-holdout-safe interaction optimization, and a better
   bit/group/method Pareto frontier, without breaking the portable affine
   packing contract.
3. **G3 — Speed evidence pack.** Turn the existing MTP and measured-KV
   machinery into formal, hardware-scoped speed and serving-quality evidence,
   and push measured BPW down at held quality.
4. **G4 — Runtime co-design.** Give the planner a measured kernel-latency cost
   model so it prefers configurations that are actually fast on the target
   runtime, not just low-bit on paper.
5. **G5 — Deferred scope.** Land dedicated MTP-sidecar quantization,
   vision-tower quantization, and 2/3-bit hardening behind their gates;
   track (not fork) upstream support for per-expert MoE precision.

## 3. Non-goals

- **No NVFP4/FP8 GPU port.** Any NVIDIA-format work is a separate research
  track with its own artifacts and narrative (ADR-0006).
- **No new certified claims from this program directly.** Every quality/speed
  improvement ships as development evidence first; certification claims only
  through the ordinary M0–M8 chain.
- **No mid-campaign algorithm changes.** While a campaign is frozen or formal,
  algorithm changes may merge behind flags but must not alter that campaign's
  evidence (ADR-0001).
- **No implementation-level reference to competing toolkits** (AXQ-001).
- **No training-based recovery.** Recovery remains calibration-only; `recover`
  stays provenance-only, weight mutation goes through refinement (ADR-0006).

## 4. Users and stakeholders

- **Release operator / evaluation custodian / independent reviewer** — run the
  campaign machinery; need the freeze discipline to stay intact.
- **Downstream MLX-LM / AX Engine users** — consume packs; need the portable
  packing contract preserved byte-for-byte in meaning (dequant-identical).
- **Maintainer** — needs each workstream to be independently landable and
  evidence-versioned so a failed experiment cannot poison the chain.

## 5. Requirements

IDs are namespaced `RM-` (roadmap). Priorities: P0 blocks the program,
P1 core value, P2 valuable, P3 opportunistic.

### 5.1 Certification closure (G1)

- **RM-01 (P0).** Execute the frozen `qwen36-mtp-v2` campaign end to end:
  overlap checks, frontier, freeze, preflight on `df-macbookpro-m5`, formal run, holdout
  completion, M0–M8 release audit. Acceptance: `release-audit` returns 0 for
  the real candidate, or a `closed_no_go` / `formal_failed` record exists with
  archived evidence.
- **RM-02 (P0).** No algorithm or planner behavior change enters the frozen
  campaign's toolchain. Acceptance: campaign wheel digest unchanged from
  freeze to audit.

### 5.2 Quality pack (G2)

- **RM-10 (P1).** GPTQ act-order, group-preserving (ADR-0002), behind an
  opt-in flag, producing packs that satisfy the existing portable affine
  contract. Acceptance: round-trip dequant test proves layout compatibility;
  measured sensitivity on at least one reference model shows the act-order
  candidate is selected over static-grid GPTQ for ≥1 tensor class at 4-bit, or
  the result is recorded as a negative finding.
- **RM-11 (P1).** Complete-candidate interaction optimization driven by
  measured results on non-holdout roles only (ADR-0004). Acceptance: the
  refinement loop can consume a `RefinementMeasurementSet` produced by real
  evaluation runs, emits an immutable candidate history, and structurally
  cannot read holdout artifacts.
- **RM-12 (P1).** Bit/group/method frontier improvement: planner exploits the
  existing multi-group candidate probing (`candidate_group_sizes`) to produce
  a strictly better measured BPW–quality Pareto front on the reference model,
  or records that the current front is already optimal for the probed grid.
- **RM-13 (P2).** Floor tuning study: quantify the quality/BPW effect of
  `--lm-head-floor` and attention-related floors on both workload profiles;
  outcome is a documented default recommendation, not a silent default change.
- **RM-14 (P2).** DWQ×(AWQ/GPTQ) composition rules in the planner: when both
  a clipping and a compensation method are candidates for a tensor, selection
  is measured, not heuristic. Acceptance: sensitivity report records the
  compared candidates.

### 5.3 Speed evidence pack (G3)

- **RM-20 (P1).** Formal MTP speed evidence: `benchmark.py` A/B (MTP off/on)
  executed under the campaign hardware contract, reporting acceptance rate and
  effective tok/s as an `EvaluationBundle`. Acceptance: bundle is digest-bound
  to `df-macbookpro-m5` and reproducible on a clean host for path neutrality.
- **RM-21 (P1).** Measured-KV serving-quality evidence: dual-profile quality
  evaluation executed with measured per-layer KV plans via
  `runtime-check --runtime mlx-lm-kv`. Acceptance: quality deltas vs BF16-KV
  recorded per profile; gate thresholds proposed from data.
- **RM-22 (P2).** BPW push at held quality: using RM-10/12 outputs, produce at
  least one reference pack with lower measured BPW at unchanged eval verdicts.

### 5.4 Runtime co-design (G4)

- **RM-30 (P1).** Kernel-latency measurement harness producing a
  hardware-scoped latency table per (bits, group size, layout/method) for the
  runtimes we ship to (MLX-LM, AX Engine) (ADR-0003).
- **RM-31 (P1).** Planner cost model consumes the latency table when present
  (fall back to abstract BPW when absent), so candidate selection can prefer
  kernel-fast configurations at equal measured quality. Acceptance: plan
  output records which cost model was used; abstract-BPW behavior is
  bit-identical when no table is supplied.

### 5.5 Deferred scope (G5)

- **RM-40 (P2).** Dedicated quantization of external MTP sidecars as separate
  checksummed artifacts, byte-preserved remains the default; requires AX
  Engine runtime support for the quantized layout (ADR-0005).
- **RM-41 (P2).** Vision-tower quantization for Qwen3-VL behind vision-specific
  evaluation evidence; protection floors for patch-embedding/merger tensors
  (ADR-0005).
- **RM-42 (P3).** 2/3-bit hardening: restrict experimental low-bit assignment
  to robust trunk tensor classes and record stability evidence under the
  existing `AX_ENGINE_{2,3}BIT_EXPERIMENTAL` gates.
- **RM-43 (P3).** Per-expert (unfused) MoE precision: tracked as an upstream
  MLX-LM dependency; AXQuant does not fork MLX-LM (ADR-0005).
- **RM-44 (P3).** Secondary family evidence (Nemotron Super/Ultra promotion
  beyond inspect-only) as capacity allows.

## 6. Success metrics

- Flagship: one certified release or one fully documented no-go (binary).
- Quality: measured BPW–quality Pareto front dominates the pre-program front
  on the reference model for both workload profiles.
- Speed: MTP effective-tok/s uplift and measured-KV memory savings recorded as
  hardware-scoped evidence, not estimates.
- Planner: with a latency table, ≥1 real plan changes selection vs abstract
  BPW and wins on measured decode latency at equal quality verdicts.
- Integrity: zero holdout-leak events; zero unversioned evidence-schema
  changes.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| New algorithms invalidate existing measured evidence silently | Evidence schema versioning + ADR-0001 freeze discipline; act-order and latency-model outputs are new candidate labels, never mutations of old ones |
| Interaction optimization overfits or leaks holdout | ADR-0004: structural role separation; holdout consumption recorded; failed formal runs are archived, never tuned against |
| Act-order breaks portable packing | Group-preserving permutation only (ADR-0002); round-trip dequant CI test |
| Latency table over-fits one host | Table is hardware-scoped evidence bound to a registry entry, like every other performance artifact |
| Vision quantization judged by text-only evals | RM-41 requires vision-specific eval tasks before any tower quantization claim |
| Scope creep into NVIDIA formats | ADR-0006 hard boundary |
