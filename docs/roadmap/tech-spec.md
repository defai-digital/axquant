# Tech spec — completion & improvement workstreams

- **Status:** draft
- **Date:** 2026-08-06
- **Scope:** engineering design for PRD requirements RM-10 … RM-44.
  RM-01/02 (certification closure) are operational, governed by
  `docs/flagship-certification.md` and ADR-0001, and not specced here.

Conventions used below: every new output is a new candidate label or a new
artifact kind with a versioned schema (ADR-0001); all flags default off; CI
gains a test per contract stated.

---

## WS-1 GPTQ act-order (RM-10, ADR-0002)

**Modules:** `gptq.py`, `quantizers.py`, `probe.py`, `schema/sensitivity.py`,
CLI parser.

**Design.** Extend the GPTQ refinement entry point with an ordering mode:

- `static` — today's behavior, label unchanged.
- `act-order` — group-preserving ordering: sort groups by aggregate Hessian
  diagonal mass (descending), and columns within each group by Hessian
  diagonal (descending); build the permuted Hessian, run the existing
  Cholesky/error-propagation loop over the permuted columns, then scatter
  refined columns back to original positions before grid packing. Group
  membership, scales, and packed layout are computed from original positions,
  so the portable affine contract holds by construction.

**Surface.** New sensitivity/plan method label (e.g. `gptq-act`) registered
wherever method labels are enumerated (probe candidates, planner method
tables, `quantizer_defaults.yaml` if GPTQ has a section there). CLI flag on
`analyze`/`convert` opts the label into the candidate grid.

**Tests.**
- Layout contract: pack a tensor via `static` and `act-order` paths and assert
  identical shapes/dtypes/group structure; dequant both and assert both sit
  exactly on their affine grids.
- Quality direction: on a fixture with a skewed Hessian, act-order reduces
  proxy error vs static (deterministic seedless fixture).
- Determinism: same inputs → byte-identical refined weights.

**Evidence impact.** New label only; existing GPTQ evidence untouched.

---

## WS-2 Measured-holdout interaction optimization (RM-11, ADR-0004)

**Modules:** `refinement.py`, `head_to_head.py` / `direct_quality.py`,
`schema` (RefinementMeasurementSet provenance), campaign role definitions.

**Design.** Keep the existing loop skeleton (top-N generation, bounded
coordinate descent, beam, immutable history). Add a measured evaluation
backend implementing the same measurement interface the loop already
consumes, backed by real `evaluate-quality`/`compare-quality` runs on a bound
candidate. Each measurement records: dataset role id, artifact digests, and
profile. The set constructor validates every role id against the campaign's
role table and **fails closed on any holdout role** — the optimizer cannot be
handed holdout data even deliberately.

**Budgeting.** Variant count is bounded by the loop's existing beam/iteration
caps; each variant costs a convert + eval cycle, so the CLI exposes a
max-variant budget and the history records exhausted-budget as a terminal
state distinct from converged.

**Tests.** Role-guard rejection test; history immutability test (existing);
end-to-end smoke on a tiny model with a stub evaluator proving the loop
selects the measured winner, not the proxy winner, when they disagree.

**Evidence impact.** Produces a new refinement-history artifact packaged with
the candidate; no change to existing sensitivity schemas.

---

## WS-3 Bit/group/method frontier & composition (RM-12, RM-14)

**Modules:** `pareto.py`, `planner.py`, `analyzer.py`, `probe.py`.

**Design.** Multi-group probing exists (`candidate_group_sizes`,
`AX_ENGINE_EXECUTABLE_GROUP_SIZES`); the gap is exploitation:

- Frontier: ensure the planner's budget search treats (bits, method,
  group size) as one candidate axis per tensor rather than fixing group size
  globally, and that `ParetoReport` points carry the full triple so fronts
  from different grids are comparable.
- Composition (RM-14): where both DWQ-clipped affine and AWQ/GPTQ are probed
  for a tensor, selection already happens on measured error; add a report
  section that surfaces near-ties (within a configurable epsilon) so floor
  and profile tuning can see where composition choices are fragile.

**Tests.** Planner selects a mixed-group plan on a fixture where the measured
frontier demands it; serialization round-trip for extended Pareto points.

**Evidence impact.** Additive fields in Pareto/plan schemas (version bump per
ADR-0001).

---

## WS-4 Kernel-latency cost model (RM-30/31, ADR-0003)

**Modules:** new `kernel_latency.py`, `hardware_registry.py`, `planner.py`,
`profiles.py`, CLI (`benchmark-kernels` command name TBD), schema addition.

**Design.**

- *Harness:* for each (runtime ∈ {mlx-lm, ax-engine}, bits ∈ {2,3,4,6,8,16},
  group size ∈ executable set, layout/method packing), time decode-shaped
  (batch 1, short M) and prefill-shaped GEMMs at representative hidden sizes
  drawn from the target architecture registry. Warmup + trimmed-mean over
  fixed iteration counts; record dispersion so noisy entries are flaggable.
  Output: `axquant.kernel-latency.v1` — host-scoped (hardware-registry
  binding), runtime-versioned, checksummed.
- *Planner:* `objective_for` gains an optional latency provider. Cost of a
  candidate = interpolated relative decode latency from the table (fallback:
  abstract bits, current behavior). The plan artifact records
  `cost_model: abstract-bpw | kernel-latency@<digest>`.
- *Neutrality guarantee:* with no table supplied, plan output must be
  byte-identical to pre-WS-4 output (golden-file CI test).

**Tests.** Golden-file neutrality; interpolation unit tests; a fixture table
that flips a selection and a test asserting the flip and its recorded
provenance.

**Evidence impact.** New artifact kind; plans gain an additive provenance
field. Latency tables are planning inputs, never quality evidence.

---

## WS-5 Formal MTP speed evidence (RM-20)

**Modules:** `benchmark.py`, `benchmark_evidence.py`, `campaign.py` glue.

**Design.** The A/B harness (MTP off/on, strict invariants, EvaluationBundle)
exists. Work: bind bundle emission to the campaign hardware contract
(`df-macbookpro-m5` scope id, preflight freshness) so an MTP bundle is admissible as
formal evidence; add acceptance-rate and effective-tok/s summary fields if not
already first-class; wire the release-audit MTP milestone to consume the
bundle digest.

**Tests.** Bundle admissibility validator (wrong host / stale preflight →
rejected); A/B invariant regression tests stay green.

---

## WS-6 Measured-KV serving-quality evidence (RM-21)

**Modules:** `kv_probe.py` (AXQ-024), `kv_exec.py`, quality harness, gate
schema.

**Design.** Execute dual-profile `evaluate-quality` with the measured KV plan
applied via the existing `runtime-check --runtime mlx-lm-kv` execution path
(MLX-LM public `QuantizedKVCache`), comparing against BF16-KV on the same
candidate. Emit a KV-quality report binding: kv plan digest (already
digest-bound to sensitivity), per-profile deltas, and context-length matrix
(short/long). Threshold proposal comes after data exists; the gate lands as
report-only first, enforcing later (two-step, per ADR-0001 additive rule).

**Tests.** Report binding validation; a smoke asserting the executed KV
precisions match the plan per layer (extend existing runtime-check
verification).

---

## WS-7 MTP sidecar quantization (RM-40, ADR-0005)

**Modules:** `mtp_sidecar.py`, `converter.py`, manifest schema, AX Engine
check in `runtime.py`/`kv_exec.py`-style gate.

**Design.** Opt-in convert flag produces a quantized sidecar artifact
alongside (never replacing) byte-preserved default: affine-quantized sidecar
tensors using the same portable packing, own manifest with
`mtp_sidecar_bits` describing actual per-tensor bits, own checksum. Gate:
conversion refuses to emit the quantized sidecar unless an AX Engine
capability check (subprocess, lazy, like `benchmark.py`'s pattern) confirms
the runtime executes the layout with MTP enabled. Naming: quantized sidecar
recorded in the manifest; measured BPW of the pack updates accordingly.

**Tests.** Byte-preserved default unchanged (digest test); quantized sidecar
round-trip load; refusal path when the capability check fails.

---

## WS-8 Vision-tower quantization (RM-41, ADR-0005)

**Modules:** `multimodal_backend.py`, `probe.py`, `module_paths.py` (tower
tensor classes + floors), eval suite under `data/eval/`.

**Design (two stages, evidence first).**
1. *Eval:* author clean-room vision eval tasks (same authorship discipline as
   the existing 60-task suite) and integrate into `evaluate-quality` scoring;
   until this exists no tower tensor may drop below BF16.
2. *Quant:* extend per-tensor probing to tower tensors through the MLX-VLM
   backend; add role floors for patch-embedding/merger classes; planner treats
   tower tensors like any measured tensor once vision eval gates consume the
   scores.

**Tests.** Floor enforcement on tower classes; probe coverage test that tower
tensors appear in sensitivity reports when the backend is MLX-VLM.

---

## WS-9 2/3-bit hardening (RM-42)

**Modules:** `experimental_bits.py`, `planner.py`.

**Design.** Constrain experimental low-bit assignment to a robust-trunk
allowlist of tensor classes (middle-layer MLP projections first; never
embeddings, heads, norms, routers — floors already forbid most). Keep the
existing env-var gates; add plan annotation listing every 2/3-bit tensor and
its measured sensitivity so instability is diagnosable. Stability evidence =
long-generation smoke + eval deltas recorded per pack.

---

## WS-10 Floor tuning study (RM-13)

**Modules:** none (uses existing `--lm-head-floor` and probe machinery);
outputs a doc + recommended defaults.

**Design.** Grid: lm-head floor {8bit, 16bit} × attention-class floors ×
both profiles on the reference model; measure eval deltas and BPW. Deliverable
is `docs/` guidance and possibly changed *documented recommendations* — no
silent default changes (ADR-0001).

---

## Cross-cutting

- **Schema discipline:** every additive field lands with a schema version note
  in `docs/migration-*.md` style; `serde` round-trip tests extended per
  change.
- **Flags:** `--gptq-act-order`, `--interaction-opt` (budgeted),
  `--latency-table <path>`, `--quantize-mtp-sidecar`, vision-quant flag —
  all default off through at least one release.
- **Test tiers:** unit (fixtures, no MLX), integration (tiny real model,
  MLX required, existing markers), evidence (formal-host only, never CI).
- **Docs:** README "Still incomplete" list shrinks only when the
  corresponding acceptance criterion in the PRD is met, and `known-issues.md`
  entries close in the same PR as the code.
