# Qwen3-Next Non-MTP Certification Product Requirements Document

**Document status:** Accepted for implementation
**Applies to:** Exact-checkpoint certification of Qwen3-Coder-Next, phases QN0–QN8
**Accepted decisions:** AXQ-032, AXQ-033, AXQ-034
**ADR:** `../architecture/decisions/0009-qwen3-next-non-mtp-certification.md`
**Technical specification:** `../engineering/qwen3-next-certification-technical-specification.md`
**Implementation plan:** `../engineering/qwen3-next-certification-implementation-plan.md`
**Last reviewed:** 2026-08-03

## 1. Executive summary

AXQuant can now produce real, measured Qwen3-Coder-Next mixed-precision checkpoints from an
immutable BF16 source. The 2026-08-03 development cycle proved the important mechanical and
measurement path on `Qwen/Qwen3-Coder-Next` at revision
`a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`:

- 807/807 inventory tensors received measured sensitivity evidence;
- calibration used 160 samples and 8,192 measured tokens per candidate;
- the 4-bit-class candidate measured 4.80002 BPW and 47.82 GB of total artifact bytes;
- the 6-bit-class candidate measured 6.00002 BPW and 59.77 GB of total artifact bytes;
- both checkpoints passed stock MLX-LM generation smoke checks;
- the 4-bit checkpoint completed three AX Engine benchmark trials at approximately
  16.78 decode tokens/second on an Apple M2 Ultra;
- measured perplexity did not show degradation against the BF16 reference on the small
  development suite.

Those facts make the checkpoints credible **development candidates**. They do not make either
checkpoint an AXQuant certified release. Qwen3-Coder-Next has no MTP component, while the current
M0–M8 release audit is intentionally a Qwen 3.6/MTP certification contract. Treating MTP as
optional inside that contract, fabricating an empty MTP sidecar, or waiving MTP through an
exception would weaken a proven release boundary and misstate the model.

This program introduces an equally strict, separately versioned **Qwen3-Next non-MTP direct-decode
certification track**. It preserves the existing M0–M8 contract byte-for-byte and defines a new
N0–N8 gate set whose rigor comes from immutable source identity, complete mixed-precision
coverage, dual-runtime correctness, matched direct-decode baselines, a substantially stronger
coding suite, independent reproduction, hardware-scoped claims, and the same checksum-bound
governance graph used by existing AXQuant releases.

Certification is initially scoped to one exact source revision and one exact converted artifact,
not the entire Qwen3-Next family. The 4-bit candidate is the first certification target because
its AX Engine manifest and measured benchmark path are closer to complete. The 6-bit candidate is
an independent later target and cannot block or inherit the 4-bit verdict.

## 2. Product problem

### 2.1 What is already solved

The current implementation already proves:

- immutable source resolution and source-checksum binding;
- complete inventory and fail-closed tensor classification for the hybrid Qwen3-Next MoE layout;
- checksum-bound calibration and activation capture;
- measured affine/AWQ/GPTQ sensitivity lineage;
- mixed 4/6/8/BF16 planning with protected norms, embeddings, router, and LM head;
- packed fused-expert conversion through public MLX-LM APIs;
- atomic output, measured BPW, artifact byte accounting, and MLX-LM generation compatibility.

### 2.2 What remains unsolved

The project does not yet have a truthful route from those development facts to a certified
non-MTP release:

1. `qwen3-next-v1` is tier `convertible`, and policy explicitly calls its artifacts development
   evidence.
2. The current release audit requires declared MTP weights and matched MTP-off/MTP-on evidence.
   Qwen3-Coder-Next declares no MTP; this gate is inapplicable, not failed evidence that can be
   waived.
3. AX Engine doctor is not release-ready on the M2 Ultra host because the Metal compiler
   toolchain is incomplete.
4. The 6-bit plan exposes an AX Engine native-manifest bug for independently mixed attention
   projections: Q is BF16 while K/V/O use packed quantized layouts.
5. The development quality suite contains only 15 tasks and 410 evaluated tokens. Its syntax
   validity is zero for BF16 and both candidates, so equal aggregate scores do not prove coding
   correctness.
6. There are no checksum-matched BF16, uniform-4, and uniform-6 direct-decode controls across two
   disjoint release profiles.
7. The validation index, complete-refinement measurement set, hardware registry, compatibility
   matrix, Pareto report, reproduction verification, and release wheel binding required for a
   formal audit have not been assembled for this checkpoint.

## 3. Product vision

A user can download an exact, certified Qwen3-Coder-Next AXQuant checkpoint and verify all of the
following without trusting a model-card claim:

- which immutable BF16 revision produced it;
- which tensors were measured, protected, or quantized and why;
- its planned and measured BPW and exact artifact bytes;
- its quality relative to BF16 and matched uniform quantization baselines;
- its direct-decode runtime behavior on the named Apple Silicon hardware scope;
- that stock MLX-LM loads it and AX Engine runs it without unreported fallback;
- that the release contains no MTP and makes no MTP claim;
- that N0–N8 passed on the exact artifact, evidence files, toolkit wheel, and reproduction recipe
  packaged with the release.

## 4. Certification vocabulary

| Term | Meaning | Allowed public claim |
| --- | --- | --- |
| Development candidate | Measured or prior-based artifact that has not passed a complete release audit | “Measured development artifact”; no certified quality/performance claim |
| Exact-checkpoint certified | One artifact derived from one immutable source revision passed N0–N8 on a named hardware scope | “AXQuant certified for `<model>@<revision>` on `<hardware scope>`” |
| Family certified | Multiple representative official checkpoints passed the family-promotion evidence contract | May describe the family adapter as `certified`; not part of the initial program |
| Runtime compatible | A named runtime passed its exact smoke/doctor contract | Compatibility only; does not imply certification |

An exact-checkpoint certification never silently promotes unrelated Qwen3-Next checkpoints or
future revisions. The adapter remains `convertible` until a separately approved family-promotion
program proves representative coverage.

## 5. Product boundary

### 5.1 In scope

- A separately versioned certification policy for non-MTP Qwen3-Next checkpoints.
- Exact source scope:
  `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`.
- Candidate scope:
  - first: 4-bit class at target 4.8 BPW;
  - second: 6-bit class at target 6.0 BPW, only after its AX Engine layout is supported.
- New N0–N8 audit request/output schemas and publisher authorization.
- Capability-derived track eligibility that proves `mtp_declared=false` from source configuration,
  inventory, plan, and artifact.
- AX Engine fixes for per-projection mixed packed attention and supported-host readiness.
- Matched BF16, uniform-4, uniform-6, and AXQuant candidate performance/quality controls.
- A versioned, provenance-recorded coding-suite v2 with executable correctness scoring.
- Complete validation, refinement, hardware, compatibility, Pareto, reproduction, wheel, and
  publication evidence.
- Exact-host certification on the formal M2 Ultra host; broader Apple Silicon claims require
  additional registry entries.

### 5.2 Explicit non-goals

- Adding, synthesizing, or pretending that Qwen3-Coder-Next contains MTP.
- Modifying, weakening, renaming, or making optional any existing Qwen 3.6 M0–M8 requirement.
- Using a release exception to bypass track eligibility, runtime correctness, quality,
  provenance, or benchmark completeness.
- Promoting all Qwen3-Next checkpoints to `certified` from one exact-checkpoint audit.
- Publishing the current 2026-08-03 artifacts as certified without a new, policy-frozen formal
  evidence cycle.
- Certifying 6-bit by inheriting 4-bit runtime or quality results.
- Claiming VLM, MTP, LoRA, KV-cache, batching, or multi-user serving behavior not measured by this
  program.
- Treating attributed OptiQ results as AXQuant evidence. An external comparison may be published
  later under the existing clean-room policy, but it is not a certification gate.

## 6. Target users

1. **Model downloaders** need a Qwen3-Coder-Next checkpoint that is smaller than BF16, loads on
   stock MLX-LM, and has verifiable coding/runtime evidence.
2. **Apple deployment engineers** need exact memory, throughput, compatibility, and hardware-scope
   facts rather than a generic “Apple Silicon” claim.
3. **Enterprise release engineers** need an audit whose non-MTP applicability is explicit and
   cannot be confused with a waived MTP gate.
4. **AXQuant maintainers** need a reusable pattern for certifying future non-MTP architectures
   without weakening the primary Qwen 3.6 certification track.

## 7. Product requirements

### 7.1 Track isolation and eligibility

- **QN-R1:** The existing `axquant.release-audit-request.v4` and
  `axquant.release-audit.v4` M0–M8 behavior remains unchanged and regression-pinned.
- **QN-R2:** A new request explicitly selects `qwen3-next-direct-v1`; absence of a track never
  defaults a model into the non-MTP path.
- **QN-R3:** Track eligibility is derived from the immutable source config and inspected inventory.
  The non-MTP track refuses a source that declares or contains MTP. The MTP track continues to
  refuse an artifact that lacks it.
- **QN-R4:** No exception mechanism may waive certification-track applicability.

### 7.2 Exact-checkpoint scope

- **QN-R5:** Certification binds source model id, full immutable source revision, architecture
  fingerprint, tokenizer digest, plan digest, artifact-manifest digest, and artifact file digests.
- **QN-R6:** A certification registry records exact certified checkpoints separately from the
  adapter’s family-level `SupportTier`.
- **QN-R7:** A source revision change, tokenizer change, planner-policy change, quantizer change,
  or weight mutation starts a new candidate and audit lineage.

### 7.3 Candidate independence

- **QN-R8:** 4-bit and 6-bit have separate plans, artifacts, evaluations, runtime checks,
  benchmark indexes, audit requests, and audit outputs.
- **QN-R9:** A passing 4-bit audit neither passes nor blocks 6-bit. A failing 6-bit audit cannot
  prevent an independently passing 4-bit exact-checkpoint release.
- **QN-R10:** Current development measurements may guide engineering and feasibility, but final
  thresholds are frozen before the formal run and final release evidence is regenerated or
  explicitly revalidated under the released toolkit and policy version.

### 7.4 Dual-runtime contract

- **QN-R11:** Stock MLX-LM must pass generation smoke from portable MLX weights.
- **QN-R12:** AX Engine must generate and validate a native manifest, pass doctor on the named
  host, and complete deterministic generation plus measured benchmark trials with zero kernel
  fallbacks.
- **QN-R13:** `--ax-engine-manifest skip`, an absent native manifest, an unavailable runtime, or a
  compatibility fallback is development-only and blocks certification.
- **QN-R14:** Every Q/K/V/O tensor is validated according to its own dtype and quantization
  metadata; a BF16 Q projection cannot cause a packed K/V/O projection to be interpreted as raw
  BF16.

### 7.5 Matched baseline matrix

- **QN-R15:** Each candidate is compared against immutable, checksum-bound BF16, uniform-4, and
  uniform-6 controls on identical prompts, seeds, sampling, token limits, power mode, runtime
  build, hardware, and OS.
- **QN-R16:** Agent-coding and general evaluations use distinct, recorded dataset digests; neither
  overlaps calibration. Exact and normalized near-duplicate checks cover coding↔calibration,
  general↔calibration, and coding↔general; different file digests alone are not sufficient.
- **QN-R17:** Every performance arm has at least two warmups and five successful measured trials.
  Failed, timed-out, or fallback trials remain recorded and cannot be silently dropped.
- **QN-R18:** Hardware and software conditions are recorded in the evaluation bundle and checked
  by the benchmark index.

### 7.6 Coding-suite v2

- **QN-R19:** The release suite contains at least 128 deterministic tasks and 25,000 scored target
  tokens across code generation, repair, multi-language compilation, JSON/tool use, reasoning,
  and long-context repository tasks.
- **QN-R20:** Executable tasks run in a network-disabled, time- and memory-limited sandbox with
  language/toolchain versions recorded.
- **QN-R21:** The suite manifest records task ids, licenses/provenance, prompt/reference digests,
  category, language, scoring method, and calibration-overlap attestation.
- **QN-R22:** Syntax/compile validity, deterministic unit-test pass rate, task score, JSON/tool
  validity, perplexity, model errors, and scored tokens are first-class metrics. A suite where
  both BF16 and candidate have zero syntax validity cannot authorize a release. Every raw model
  output and executable stdout/stderr stream is checksum-bound and durably archived; coding and
  general generation persist per-task restart state that refuses mismatched provenance.

### 7.7 Evidence and governance

- **QN-R23:** N0–N8 consumes strict, checksum-bound Pydantic artifacts; missing evidence fails
  before any gate can be marked passed.
- **QN-R24:** The release contains a complete refinement measurement set, validation index,
  hardware registry, compatibility matrix, Pareto report, reproduction recipe and verification,
  runtime checks, benchmark evidence index, coding-suite manifest, and stable toolkit wheel.
- **QN-R25:** Digest-referenced evidence is archived durably before downstream use, per AXQ-031.
- **QN-R26:** `publish --yes` reruns the exact N0–N8 request and refuses any mismatch with the
  packaged audit.

## 8. N0–N8 certification gates

| Gate | Name | Required proof |
| --- | --- | --- |
| **N0** | Immutable technical feasibility | Complete BF16 source/index, exact revision, Qwen3-Next architecture fingerprint, no MTP declared or present, baseline availability |
| **N1** | Artifact integrity and dual-runtime vertical slice | Atomic/checksummed artifact, logical parameter equivalence, plan/manifest/calibration bindings, measured BPW tolerance, MLX-LM pass, AX Engine manifest/doctor/generation pass |
| **N2** | Correct and repeatable direct-decode benchmark | Matched BF16/uniform/candidate controls, ≥2 warmups + ≥5 successful trials, deterministic greedy parity, zero fallbacks, complete hardware metadata |
| **N3** | Measured mixed-precision planner | Complete measured sensitivity, checksum-bound calibration/capture, target coverage, finite metrics, deterministic plan reproduction |
| **N4** | Coding and general quality | Coding-suite v2 + disjoint general suite, executable correctness, syntax/tool validity, perplexity and retention thresholds, zero model errors |
| **N5** | Exact Qwen3-Next architecture proof | Hybrid full/linear attention, fused 512-expert MoE, router/protected tensor floors, packed expert conversion coverage, exact checkpoint scope |
| **N6** | Complete candidate optimization | Measured refinement lineage, complete objective reconstruction, candidate improvement over its parent/control, independent 4-/6-bit identity |
| **N7** | Hardware-aware Pareto and reproduction | Candidate lies on measured quality/size/runtime frontier, formal M2 Ultra registry entry, independent reproduction, compatibility matrix |
| **N8** | Release package and claim authorization | Stable wheel and schema versions, full evidence bundle, exact model card claims, publisher re-audit, no development or unsupported claims |

N0–N8 is not a shortened M0–M8. It replaces MTP-specific proof with direct-decode correctness,
matched controls, executable coding quality, and explicit non-MTP capability consistency while
retaining the same artifact, provenance, hardware, reproducibility, and publication rigor.

## 9. Frozen release thresholds

Thresholds must be accepted and committed before the formal evidence cycle. They may not be tuned
after observing the formal candidate.

| Metric | Frozen minimum |
| --- | ---: |
| Calibration samples | ≥128 |
| Measured tokens per sensitivity candidate | ≥8,192 |
| Coding-suite tasks / scored target tokens | ≥128 / ≥25,000 |
| Successful benchmark trials / warmups per arm | ≥5 / ≥2 |
| Candidate model/runtime errors | 0 |
| Perplexity ratio to BF16 | ≤1.02 agent-coding; ≤1.03 general |
| Aggregate task-score retention | ≥0.99 per profile |
| Eligible syntax/compile validity | ≥0.95 absolute and no more than 0.01 below BF16 |
| JSON/tool exact-validity | ≥0.98 absolute and no more than 0.01 below BF16 |
| Greedy AX Engine vs MLX-LM token agreement | 1.00 on parity corpus |
| Kernel fallbacks | 0 |
| Decode speedup vs BF16 | ≥1.20× on formal host |
| Throughput retention vs same-class uniform control | ≥0.95× |
| TTFT ratio vs same-class uniform control | ≤1.10× |
| Measured vs planned BPW difference | ≤0.01 BPW |
| 4-bit artifact bytes / BF16 weight bytes | ≤0.35 |
| 6-bit artifact bytes / BF16 weight bytes | ≤0.45 |

“Scored target tokens” means the frozen sum of per-task generation budgets in the checksum-bound
suite manifest, not a minimum verbosity requirement on model output. Raw generated-token counts
remain recorded per outcome and may not exceed the task budget; concise correct code is never
padded merely to satisfy the suite-size threshold.

If feasibility evidence shows a threshold is structurally inappropriate, it may be changed only
through an accepted ADR before the formal candidate cycle begins. A failed formal result starts a
new evidence cycle; it does not trigger an in-place threshold edit.

These values were accepted on 2026-08-03 as
`axquant.qwen3-next-direct-policy.v1`. The initial public claim scope is Apple M2 Ultra with
192 GB unified memory only. Coding-suite inputs must be clean-room authored or distributed under
a license permitting evaluation and result publication; every task records provenance and
license metadata, and no private or license-ambiguous task may satisfy a certification quota.

## 10. Success criteria

| ID | Target |
| --- | --- |
| QN-T1 | Existing Qwen 3.6 M0–M8 fixtures and golden audits remain unchanged |
| QN-T2 | A non-MTP request cannot select a model that declares MTP, and no-MTP cannot be supplied as a user-only assertion |
| QN-T3 | N0–N8 request, output, publisher rerun, and tamper tests are implemented with strict schemas |
| QN-T4 | AX Engine validates independently mixed Q/K/V/O packed/raw projections and reproduces the current 6-bit failure as a regression test |
| QN-T5 | Formal M2 Ultra doctor is ready, including `xcrun metal` and `metallib` |
| QN-T6 | Coding-suite v2 satisfies QN-R19–QN-R22 and is disjoint from calibration by digest/content checks |
| QN-T7 | BF16, uniform-4, uniform-6, and candidate benchmark evidence is complete for both required profiles |
| QN-T8 | The new formal 4-bit candidate passes N0–N8 and is registered only for the exact source revision and hardware scope |
| QN-T9 | 4-bit publication contains its audit and request and survives publisher re-audit |
| QN-T10 | 6-bit receives an independent N0–N8 verdict; it is published only if all gates pass |
| QN-T11 | No model card or support matrix implies MTP or family-wide certification |
| QN-T12 | Every load-bearing evidence file has a durable archive location and verified digest |

## 11. Claim policy

### 11.1 Claims allowed after 4-bit N0–N8 passes

- “AXQuant exact-checkpoint certified non-MTP direct-decode artifact.”
- Exact measured BPW, artifact bytes, quality deltas, memory, and throughput from the packaged
  evidence.
- MLX-LM and AX Engine compatibility on the named tested versions and hardware scope.

### 11.2 Claims not allowed

- “Qwen3-Next is certified” without exact checkpoint scope.
- “Apple Silicon certified” without naming the hardware registry scope.
- Any MTP, speculative decoding, VLM, KV-cache, batching, or serving-concurrency claim not in the
  audit.
- “Better than BF16” from the small 2026-08-03 development perplexity result.
- A 6-bit claim based on 4-bit evidence.

## 12. Roadmap

| Phase | Product result | Exit condition |
| ---: | --- | --- |
| QN0 | Governance and threshold freeze | PRD, ADR, tech spec, plan, and decisions accepted |
| QN1 | Certification-track infrastructure | New strict schemas, N0–N8 builder, publisher dispatch, M0–M8 regression suite green |
| QN2 | AX Engine and formal-host readiness | Mixed-projection manifest/runtime support; Metal toolchain; dual-runtime parity green |
| QN3 | Coding-suite v2 | Provenance, sandbox, ≥128 tasks, metrics and overlap checks implemented |
| QN4 | Matched baseline evidence | BF16/uniform-4/uniform-6 controls complete across both profiles |
| QN5 | New 4-bit formal candidate cycle | Capture → analyze → refine → plan → convert → quality → benchmark evidence archived |
| QN6 | 4-bit certification and publication | N0–N8 passes; exact checkpoint entry and guarded upload complete |
| QN7 | 6-bit remediation and certification | Independent 6-bit candidate completes N0–N8 or records a durable failure verdict |
| QN8 | Optional family/hardware expansion | Second representative checkpoint and additional hardware scope proven separately |

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| New track becomes an easier escape hatch | Invalid certification | Separate schema, capability-derived eligibility, no applicability exceptions, parity tests against M0–M8 rigor |
| Coding suite overfits Qwen output style | Inflated quality claim | Freeze before formal run, disjoint holdout, executable scoring, category/language quotas |
| AX Engine fix changes numerics | Quality or parity regression | Golden raw/packed projection fixtures, greedy parity corpus, zero-fallback gate |
| One host is generalized to all Macs | Misleading claim | Hardware-scoped registry and exact model-card wording |
| Current development evidence is repackaged as formal | Stale toolkit/policy binding | New formal cycle after policy/toolkit freeze; AXQ-030 evidence-root discipline |
| 6-bit delays 4-bit | Lost time-to-value | Independent candidate audits and 4-bit-first sequencing |
| 80B evidence runs are interrupted | Lost compute/evidence | Resumable stages, atomic files, durable archive, external-disk health preflight |

## 14. Release decision

Until QN6 exits successfully, the current 4-bit and 6-bit artifacts remain development evidence
and no low-bit quality or production-readiness claim is restored. If QN6 passes while QN7 fails,
only the exact 4-bit artifact may be released as certified. If any mandatory 4-bit gate fails, the
program records the failing evidence and starts a new candidate cycle rather than weakening the
gate.
