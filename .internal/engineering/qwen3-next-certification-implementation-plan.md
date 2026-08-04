# Qwen3-Next Non-MTP Certification — Multi-Phase Implementation Plan

**Status:** Active
**PRD:** `../product/qwen3-next-certification-prd.md`
**ADR:** `../architecture/decisions/0009-qwen3-next-non-mtp-certification.md`
**Technical specification:** `qwen3-next-certification-technical-specification.md`
**Accepted decisions:** AXQ-032, AXQ-033, AXQ-034
**Last updated:** 2026-08-04

## 1. Delivery objective

Produce the first exact-checkpoint certified, non-MTP Qwen3-Next AXQuant release without changing
or weakening the existing Qwen 3.6 M0–M8 contract. Certification targets are independent:

1. certify a new 4-bit-class artifact derived from
   `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`;
2. remediate and then independently audit a 6-bit-class artifact;
3. keep the `qwen3-next-v1` family adapter at `convertible` until a later representative-family
   program is approved.

The current measured 4-/6-bit artifacts are development/feasibility evidence. They provide
regression fixtures and planning input, not a shortcut around the formal phases.

## 2. Phase overview

| Phase | Name | Status | Primary exit | Estimated engineering | Formal-host time |
| --- | --- | --- | --- | ---: | ---: |
| **QN0** | Governance and threshold freeze | Complete | PRD/ADR/spec/plan accepted; thresholds frozen | 1–2 days | none |
| **QN1** | Certification-track infrastructure | Complete | Strict N0–N8 schemas/builder/publisher; M0–M8 unchanged | 5–8 days | 1–2 h integration |
| **QN2** | AX Engine and host readiness | Host-blocked | Mixed Q/K/V/O supported; doctor ready; parity green | 5–10 days | 4–8 h |
| **QN3** | Coding-suite v2 | Complete | ≥128-task licensed/provenance suite and sandbox scorer | 5–8 days | 2–4 h pilot |
| **QN4** | Matched controls and evidence orchestration | In progress | BF16/uniform4/uniform6/candidate matrix reproducible | 3–5 days | 8–16 h |
| **QN5** | New formal 4-bit evidence cycle | Not started | Complete archived evidence graph for one exact artifact | 2–4 days ops | 10–18 h |
| **QN6** | 4-bit N0–N8 and guarded publication | Not started | Passing audit, registry entry, publisher rerun/upload | 1–3 days | 2–5 h |
| **QN7** | Independent 6-bit cycle | Blocked on QN2 | Passing audit or durable named failure verdict | 2–5 days ops | 10–18 h |
| **QN8** | Optional family/hardware expansion | Deferred | Second checkpoint and/or second hardware scope | separate program | TBD |

Estimates are planning ranges, not release promises. AX Engine review, coding-suite licensing, and
formal-host availability dominate calendar time.

## 3. Dependency graph

```text
QN0
 └── QN1 certification contracts
      ├── QN2 AX Engine + host ─────┐
      ├── QN3 coding suite ─────────┼── QN4 matched controls
      └── evidence CLI contracts ───┘          │
                                               ▼
                                      QN5 formal 4-bit cycle
                                               │
                                               ▼
                                      QN6 audit + publish

QN2 ───────────────────────────────────────────┐
QN3/QN4 reusable controls ─────────────────────┼── QN7 formal 6-bit cycle
                                               ▼
                                      independent audit verdict

QN6 + representative second checkpoint/hardware evidence ──► QN8 (optional)
```

QN2 and QN3 may execute in parallel after QN1 freezes their artifact contracts. QN4 may begin
building BF16/uniform controls while QN2 finishes, but no formal performance row is accepted until
the released AX Engine and host doctor are ready.

## 4. Standing rules for every phase

1. Never edit existing M0–M8 schema literals or relax existing tests.
2. Never infer track eligibility from a user-provided `--non-mtp` flag.
3. Never fabricate an MTP sidecar or use an exception to waive capability applicability.
4. Do not run multiple 80B model processes concurrently on the M2 Ultra; benchmark arms run
   sequentially and alone.
5. Every formal source revision is immutable and every downstream artifact binds it by digest.
6. New formal cycle means new evidence root; do not rewrite the 2026-08-03 development lineage.
7. Archive digest-referenced measured evidence durably before downstream use.
8. Candidate 4-bit and 6-bit identities, plans, outputs, and audits remain separate.
9. No public upload occurs until the exact candidate passes all gates and the user explicitly
   authorizes `publish --yes`.
10. A failed gate produces a durable failure artifact; it does not cause threshold editing.

## 5. Phase QN0 — Governance and threshold freeze

### Goal

Accept the product boundary and prevent implementation from accidentally turning “non-MTP” into
“MTP evidence optional.”

### Work items

- [x] Draft Qwen3-Next certification PRD.
- [x] Draft ADR 0009 with decisions AXQ-032–AXQ-034.
- [x] Draft technical specification.
- [x] Draft this multi-phase plan.
- [x] Accept the numeric quality/runtime thresholds in PRD §9 as policy v1.
- [x] Set initial public claim scope to M2 Ultra 192 GB only; require a separate registry entry
      before claiming any other hardware.
- [x] Confirm exact source revision; bind tokenizer/config fingerprints from the immutable source
      into N0 rather than copying mutable operator assertions.
- [x] Require clean-room-authored or evaluation/publication-compatible coding tasks with per-task
      provenance and license metadata.
- [x] Accept decisions AXQ-032–AXQ-034 in the decision register.
- [x] Freeze `axquant.qwen3-next-direct-policy.v1` before formal candidate measurement.

### Exit criteria

- PRD status becomes accepted for planning.
- ADR status becomes accepted and AXQ-032–AXQ-034 become authoritative.
- Tech spec becomes accepted for implementation.
- Thresholds and hardware claim scope are explicit and versioned.
- No unresolved decision changes the schema/gate architecture.

### Stop conditions

- Stakeholders prefer MLX-only certification (requires a broader AXQ-003 ADR, not this program).
- Stakeholders want family-wide certification from one checkpoint (requires revising AXQ-033).
- Coding-suite data cannot satisfy licensing/provenance requirements.

## 6. Phase QN1 — Certification-track infrastructure

### Goal

Implement additive non-MTP certification schemas and N0–N8 dispatch while proving M0–M8 remains
unchanged.

### PR slice QN1-A — Golden regression boundary

- [x] Capture semantic hashes and expected gates/issues for representative passing/failing v4
      release-audit fixtures.
- [x] Add publisher preview/rerun golden tests for existing Qwen 3.6 artifacts.
- [x] Add a test proving a no-MTP artifact still fails the current M1 requirement.
- [x] Run full suite before any audit refactor and record the baseline.

Exit: existing behavior has executable protection against accidental change.

### PR slice QN1-B — Strict schemas and dispatch

- [x] Add `CertificationTrack`, `ArchitectureFingerprint`, `ExactCertificationScope`.
- [x] Add strict Qwen3-Next request/audit/check schemas with N0–N8 exactly-once validation.
- [x] Add schema-version dispatcher; unknown versions fail closed.
- [x] Add source-derived eligibility loader and contradictory-MTP negative tests.
- [x] Add wheel-owned policy object and canonical policy digest.

Exit: valid requests load only under their exact track; no gate algorithms yet claim pass.

### PR slice QN1-C — N0–N8 builder

- [x] Implement N0/N1 common artifact and runtime checks using side-effect-free helpers.
- [x] Implement N2 matched direct benchmark verification.
- [x] Implement N3 measured sensitivity/plan reconstruction.
- [x] Implement N4 coding/general quality verification.
- [x] Implement N5 exact architecture fingerprint/coverage.
- [x] Implement N6 complete refinement measurement reconstruction.
- [x] Implement N7 hardware/Pareto/reproduction verification.
- [x] Implement N8 wheel/package/claim verification.
- [x] Add one negative/tamper test for every material invariant.

Exit: synthetic/tiny fixtures can produce complete passing and intentionally failing N0–N8
audits; no real model claim yet.

### PR slice QN1-D — Registry and publisher

- [x] Add exact-checkpoint certification registry with append/supersession rules.
- [x] Render exact certified checkpoint entries separately in `support-matrix`.
- [x] Extend `publish-prepare` for non-MTP evidence package without placeholder MTP files.
- [x] Extend `publish` to load a matching request/audit union and rerun the exact builder.
- [x] Refuse cross-track request/audit, wrong registry entry, stale policy, or modified artifact.

### Tests and quality gates

```bash
.venv/bin/pytest tests/test_release_audit.py tests/test_publisher.py
.venv/bin/pytest tests/test_certification_dispatch.py
.venv/bin/pytest tests/test_qwen3_next_release_audit.py
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

### Exit criteria

- QN-T1–QN-T3 pass.
- Existing v4 semantic hashes and publisher behavior are unchanged.
- New track cannot pass with missing, contradictory, prior-based, or cross-track evidence.

## 7. Phase QN2 — AX Engine and formal-host readiness

### Goal

Make both target classes valid AX Engine artifacts and make the M2 Ultra a truthful formal host.

### Workstream QN2-A — Reproduce and freeze the 6-bit failure

- [x] Create a minimal native-manifest fixture with BF16 Q and packed 4-/8-bit K/V/O matching the
      observed logical/physical shapes.
- [x] Confirm the current validator fails with the recorded `attention_k [512,256]` diagnostic.
- [x] Add the fixture and failure contract before changing the loader.

### Workstream QN2-B — Per-tensor logical shape validation

- [x] Reconstruct Q/K logical shapes independently from each tensor's dtype/bits/group metadata.
- [x] Preserve quantization scale/bias tensor-shape and packed-divisibility validation.
- [x] Remove the observed layer-level “Q determines K” assumption.
- [x] Cover the failing raw-Q/packed-K path and inverse raw-K/packed-Q path.
- [x] Validate both real development artifact manifests, including fused-expert metadata.

### Workstream QN2-C — Runtime load and parity

- [x] Validate mixed raw/packed projection manifests without materializing an incorrect logical
      shape.
- [ ] Add deterministic greedy parity against MLX-LM on a tiny fixture.
- [ ] Run real 4-bit and 6-bit artifact smoke on M2 Ultra.
- [ ] Require zero kernel fallbacks and record executable commit/version/digest.
- [x] Confirm the AX Engine core regression suite remains green on `macstudio-m2u`: 641 unit
      cases (639 passed, 2 ignored), 2 integration tests passed, and 3 doctests ignored.
- [x] Commit and push the isolated mixed-projection fix as AX Engine commit `6ded0023` on
      `codex/qwen3-next-mixed-projection-v2`.
- [ ] Review, merge, and release the AX Engine fix; formal evidence must pin the released commit.

### Workstream QN2-D — Metal host readiness

- [x] Record current `xcode-select`, `xcrun metal`, and `xcrun metallib` state.
- [ ] Install/select the approved Metal toolchain through the normal host-management process.
- [ ] Record Xcode/Metal versions and checksums in host provenance.
- [ ] Re-run AX Engine doctor until `bringup_allowed=true` and artifact status is ready.
- [ ] Prevent background automatic toolchain switching during formal runs.

Installing system components is an explicit operator action and must not be hidden inside an
AXQuant test or release command.

Current blocker: `macstudio-m2u` selects CommandLineTools and has no compatible full Xcode;
`xcrun metal` and `xcrun metallib` are unavailable. The isolated AX Engine fix passes
`cargo fmt`, four focused mixed-projection regressions, 641 core unit cases, and both core
integration tests. Strict Clippy reports no diagnostic in any changed range; the repository-wide
`--all-targets --all-features -D warnings` baseline remains red with 1,563 pre-existing rule
diagnostics (2 library and 1,561 library-test diagnostics) and is not represented as a passing
gate. No formal Metal runtime, parity, zero-fallback, or performance evidence is accepted until the
approved Xcode is installed and selected and the AX Engine fix is reviewed, merged, and released.

### Exit criteria

- QN-T4 and QN-T5 pass.
- Both development artifacts generate/validate a native manifest and complete deterministic
  smoke; the results remain development evidence.
- AX Engine change is reviewed, committed, and released before formal QN4/QN5 measurements.

### Rollback

If mixed-projection runtime parity fails, retain the failure fixture and keep 6-bit blocked. Do
not rewrite the 6-bit plan to make the runtime test pass. The 4-bit track may continue if its exact
runtime path remains green.

## 8. Phase QN3 — Coding-suite v2

### Goal

Replace the 15-task smoke suite with a release-quality, licensed, deterministic executable
evaluation that can distinguish equal failure from retained coding ability.

### Workstream QN3-A — Data and provenance

- [x] Select/author at least 128 tasks under compatible licenses.
- [x] Meet category/language quotas from the tech spec.
- [x] Record task id, prompt/reference digest, provenance, license, scorer, limits, and toolchain.
- [x] Create immutable coding-suite manifest and sharded payloads.
- [x] Run exact and near-duplicate overlap checks against calibration and general holdout.
- [x] Require every reference oracle to pass and an empty mutant to fail before freezing the
      suite; archive the checksum-bound self-test and raw logs.

### Workstream QN3-B — Sandbox scorer

- [x] Implement network-disabled per-task execution.
- [x] Enforce wall/CPU/memory/process/output limits.
- [x] Implement unit-test, compile, AST, JSON-schema, tool-exact, and text-exact scorers.
- [x] Separate infrastructure failure from model failure.
- [x] Record stdout/stderr digests and toolchain identity.

### Workstream QN3-C — Quality artifact integration

- [x] Extend quality result/comparison schemas additively or version them explicitly.
- [x] Aggregate pass rate, syntax/compile validity, tool validity, perplexity, errors, and scored
      tokens by required category.
- [x] Make `direct-validation-index` enforce authoritative track thresholds.
- [x] Make N4 recompute aggregates from raw per-task outcomes.

### Pilot

- [x] Run BF16 only first; BF16 must clear the absolute suite-validity floor.
- [x] Repair invalid tasks/scorers before freezing the suite, never after seeing candidate scores.
- [x] Freeze suite id/version/digest and policy thresholds.

### 2026-08-04 pilot record

- Attempt 1 was stopped after two tasks because its state did not bind the source artifact
  manifest. It is retained as `bf16-pilot-attempt-001-provenance-incomplete`.
- Attempt 2 completed all 128 BF16 tasks with zero model/infrastructure errors, but correctly
  failed the frozen absolute floor: syntax/compile validity was 0.6635. All 35 syntax failures
  were generation-budget truncations (10 JavaScript, 16 Rust, 9 algorithm tasks); no candidate
  result had been run or inspected.
- The failed suite and pilot are retained as attempt 004 / attempt 002 evidence. The pre-freeze
  repair raised only the affected generation budgets to 768 tokens and made outer code-fence
  extraction tolerant of a missing closing fence while leaving compiler/unit-test verdicts
  authoritative.
- The repaired suite contains 128 tasks and 58,112 target tokens. Its self-test passed 128/128
  reference oracles and rejected 128/128 empty mutants. The immutable manifest digest is
  `d5b3a9db0c4d53c1d6eeb4d1b2aafbab1b355b4b5dd0074403c902f593f3760b`.
- The second BF16-only run completed all 128 tasks with zero model/infrastructure errors and no
  timeouts. It recorded PPL 2.066259 over 59,604 evaluated tokens, aggregate score 0.828125,
  syntax/compile validity 103/104 (0.990385), JSON/tool validity 16/16, and unit-test pass
  72/94. The quality artifact digest is
  `005fa8a8fd4134b6f2f95f04dbe191759ec69f1a2c8673b201068e106d6fe0dc`.
- The distinct BF16 general reference completed 16 tasks with zero model errors, PPL 87.247595,
  aggregate score 0.875, and 16 checksum-bound raw outputs. Coding↔calibration,
  general↔calibration, and coding↔general overlap checks all passed with zero matches. No
  candidate result was observed before this freeze.

### Exit criteria

- QN-T6 passes.
- BF16 pilot has nonzero, meaningful syntax/compile/unit-test results.
- Calibration overlap report is clean and checksum-bound.
- Full test/format/type gates pass.

## 9. Phase QN4 — Matched controls and evidence orchestration

### Goal

Create reproducible baseline artifacts and a direct-decode evidence index before spending a new
formal candidate cycle.

### Workstream QN4-A — Immutable controls

- [x] Verify/reuse the immutable BF16 source snapshot and checksum-bound 42-file source manifest.
- [x] Freeze matched BF16 coding/general reference evidence before any candidate evaluation.
- [ ] Generate uniform-4 and uniform-6 plans under exactly the candidate's protection floors.
- [ ] Convert atomically under the released cert-capable AXQuant/MLX-LM version.
- [ ] Require AX Engine native manifest and both runtime smokes.
- [ ] Record measured BPW, weight bytes, total artifact bytes, parameter counts, and digests.

### Workstream QN4-B — Benchmark evidence index

- [ ] Add/directly use strict BF16, uniform-4, uniform-6, candidate entries.
- [ ] Enforce matched prompt/config/runtime/hardware invariants.
- [ ] Enforce ≥2 warmups and ≥5 successful trials, retaining every failure.
- [ ] Compute ratios from raw trials.
- [ ] Add missing-baseline and mismatched-control negative tests.

### Workstream QN4-C — Evidence requests

Prepare, but do not fabricate pass results for:

- feasibility request/report;
- coding and general release validation requests;
- benchmark evidence request/index;
- refinement execution/measurement requests;
- hardware registry request;
- compatibility matrix request;
- reproduction recipe;
- Pareto report request;
- Qwen3-Next release-audit request.

All request paths are relative and portable. Durable archive roots are explicit.

### Exit criteria

- QN-T7 passes for BF16/uniform controls.
- Controls are complete and reusable by both 4-bit and 6-bit candidate cycles.
- Formal host is idle/ready and all evidence commands have rehearsed on tiny/development fixtures.

## 10. Phase QN5 — New formal 4-bit evidence cycle

### Goal

Produce one complete, durable, internally consistent 4-bit candidate evidence graph after code,
runtime, policy, suite, and thresholds are frozen.

### Preflight

- [ ] Pin AXQuant, MLX, MLX-LM, AX Engine, OS, Xcode/Metal, source, tokenizer, policy, suites.
- [ ] Verify external volume SMART/mount/free space and disable accidental disconnect risk.
- [ ] Verify no competing model/runtime workload.
- [ ] Create durable run root and off-machine/archive plan.
- [ ] Verify current low-bit public claims remain disabled.

### Formal stage chain

1. **feasibility** — BF16 and mandatory controls complete; runtime available.
2. **inspect** — exact source inventory and architecture fingerprint.
3. **calibrate/tokenize** — release calibration, separation attestation, cache verification.
4. **capture** — checksum-bound activations; resumable coverage.
5. **analyze** — complete affine grid; measured tokens and progress state.
6. **refine analyze** — frozen AWQ/GPTQ target selection and lineage.
7. **plan** — target 4.8 BPW, candidate id unique to the formal cycle.
8. **convert** — `--ax-engine-manifest required`; atomic artifact.
9. **quality** — coding-suite v2 and disjoint general holdout for BF16/controls/candidate.
10. **runtime** — MLX-LM and AX Engine structured checks.
11. **benchmark** — matched BF16/uniform/candidate trials, sequentially.
12. **validate** — both profiles under authoritative thresholds.
13. **refine-measure/select** — complete objective and candidate selection.
14. **hardware/compatibility/Pareto/reproduction** — full governance graph.
15. **archive** — verify durable digest copies before audit request references them.

### Evidence reuse rule

- Immutable source shards may be reused after full checksum verification.
- Calibration/capture/sensitivity may be reused only if exact backend, policy, source, tokenizer,
  suite-separation, and released version contracts permit it and N3 recomputes their bindings.
- Any probe/backend, tensor classification, source, or policy change forces remeasurement.
- The 2026-08-03 plan/artifact is not renamed into this cycle.

### Exit criteria

- Every QN5 stage has a complete checksum-bound artifact.
- 4-bit measured BPW and bytes meet frozen thresholds.
- Both validation profiles pass.
- Registry/Pareto/reproduction inputs load and cross-bind without manual digest editing.
- No publication has occurred.

### Failure handling

If a quality/runtime gate fails, write a formal failure status with exact values. Fixing code or
changing a candidate starts a new evidence root at the earliest affected stage; do not overwrite
the failed bundle.

## 11. Phase QN6 — 4-bit audit and guarded publication

### Goal

Obtain and independently verify a complete N0–N8 verdict for the exact 4-bit artifact.

### Work items

- [ ] Assemble strict request from QN5 durable evidence paths.
- [ ] Run `release-audit`; require exit 0 and N0–N8 exactly once.
- [ ] Review every gate, raw blocker list, policy digest, hardware scope, and model-card claim.
- [ ] Add exact checkpoint to certification registry using audit/artifact digests.
- [ ] Run `publish-prepare` into a new release directory.
- [ ] Re-run audit against packaged relative paths; require semantic equality.
- [ ] Run `publish` preview; inspect file list and repository destination.
- [ ] Obtain explicit user authorization for executed publication.
- [ ] Run `publish --yes` and verify remote commit/files/model card.
- [ ] Record publication revision and rerun read-only Hub verification.

### Exit criteria

- QN-T8, QN-T9, QN-T11, and QN-T12 pass.
- Only the exact 4-bit checkpoint and M2 Ultra hardware scope are claimed unless additional
  registry evidence exists.
- Public artifact contains request, audit, recipe, and all permitted evidence.

### Rollback

Before upload, rollback is local and no public state changes. After upload, never silently replace
an artifact under the same revision; publish a corrective commit/model-card notice or withdraw the
claim explicitly while preserving audit history.

## 12. Phase QN7 — Independent 6-bit cycle

### Entry conditions

- QN2 mixed-projection manifest/load/parity is released.
- Formal host doctor is ready.
- QN3/QN4 suites and controls are frozen.
- 4-bit outcome is recorded but not used as 6-bit evidence.

### Work items

- [ ] Reproduce the original 6-bit failure fixture as a now-passing AX Engine regression.
- [ ] Start a new 6-bit candidate/evidence root at target 6.0 BPW.
- [ ] Run the same complete chain as QN5 with `--ax-engine-manifest required`.
- [ ] Produce independent quality/benchmark/validation/refinement/governance evidence.
- [ ] Run a separate N0–N8 audit and publisher preview.
- [ ] Publish only after an exact pass and explicit authorization.

### Exit criteria

- QN-T10 passes: either a certified exact 6-bit artifact or a durable audit naming the exact
  blocker.
- No 4-bit file, measurement id, audit, or registry entry is substituted into the 6-bit graph.

## 13. Phase QN8 — Optional expansion

This phase is intentionally not required for the first exact-checkpoint release.

Possible goals:

- certify a second official Qwen3-Next checkpoint/revision with a distinct architecture
  fingerprint;
- add a second supported high-memory Apple Silicon host and broaden the hardware claim;
- propose family-tier promotion from `convertible` to `certified` with representative evidence;
- certify batching, KV-cache, or serving-concurrency capabilities through separate ADRs/gates.

Family promotion must not be inferred from QN6/QN7 alone.

## 14. Pull-request sequence

| PR | Scope | Must not include |
| --- | --- | --- |
| PR-1 | Governance documents and accepted decisions | Runtime/schema implementation |
| PR-2 | Existing M0–M8 golden regression harness | New gate behavior |
| PR-3 | Certification schemas, policy, dispatch, eligibility | Publisher mutation |
| PR-4 | N0–N8 builder and strict fixtures | AX Engine changes |
| PR-5 | Registry, package, publisher rerun/preview | Real model publication |
| PR-6 | AX Engine mixed-projection manifest/load tests and fix | Threshold changes |
| PR-7 | Coding-suite v2 schema/sandbox/scorers | Candidate-specific tuning |
| PR-8 | Direct baseline evidence index/orchestration | Formal evidence files in source code |
| Evidence cycle | QN4–QN6 generated artifacts under durable evidence storage | Code behavior changes mid-cycle |
| Later evidence cycle | QN7 6-bit | Reuse of 4-bit audit identity |

Each code PR passes the full repository gates. Cross-repository AX Engine work uses its own review,
test, commit, and release process before AXQuant pins the runtime version.

## 15. Formal run operations

### 15.1 Host policy

- one formal model process at a time;
- no batch-size experiments during formal evidence;
- no benchmark concurrency;
- stable power mode and thermal preconditioning;
- external SSD cable/volume protected from accidental disconnect;
- progress/status polled without modifying model/runtime state;
- all commands and versions recorded without credentials.

### 15.2 Checkpointing

- capture and analyze state files are checksum-bound and resumable;
- conversion remains atomic staging → rename;
- quality writes per-task progress or restarts only the affected suite arm;
- benchmark raw trials are written before aggregation;
- controller never publishes and never marks a failed required stage “completed” without a
  blocker status.

### 15.3 Status vocabulary

```text
running
completed
completed_with_blockers
failed
release_ready
published
```

`completed` means all planned commands produced valid outputs; it does not imply `release_ready`.
`completed_with_blockers` is the correct state when optional development diagnostics finish but a
required certification output fails or is absent.

## 16. Definition of done

### Toolkit done

- QN1–QN4 implementation and tests pass.
- Existing M0–M8 behavior is unchanged.
- AX Engine and coding-suite prerequisites are released.
- Evidence commands can be rehearsed end to end on fixtures.

### 4-bit certification done

- QN5 evidence graph is complete and archived.
- N0–N8 passes on the exact artifact.
- Publisher rerun reproduces the pass.
- Exact-checkpoint registry and model card are correct.
- Executed upload is explicitly authorized, verified, and recorded.

### Program minimum done

- 4-bit certification has a truthful pass or durable named failure verdict.
- 6-bit has at least a durable independent N0–N8 verdict; publication is conditional on pass.
- No unsupported low-bit, MTP, family-wide, or hardware-wide claim exists.

## 17. Immediate next actions

1. Install/select the approved full Xcode on `macstudio-m2u` through host management.
2. Review, merge, and release AX Engine commit `6ded0023`, then pin that released revision.
3. Run formal Metal smoke and deterministic MLX-LM parity using the already installed,
   checksum-bound AXQuant 1.2.0 wheel and the released AX Engine revision.
4. Generate matched uniform-4/uniform-6 controls and finish the direct evidence requests/indexes.
5. Start the new formal 4-bit capture→analyze→plan→convert cycle only after QN2–QN4 exits pass.

No formal 80B rerun should start before QN0–QN4 freeze the policy, runtime, suite, controls, and
toolkit version.
