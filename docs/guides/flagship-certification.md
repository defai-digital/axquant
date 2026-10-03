# Qwen 3.6 flagship certification

AXQuant’s first flagship policy is `qwen36-mtp-v2`. It applies to one exact source:

```text
Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

It does not certify existing development packs automatically and does not change any M0–M8
quality, size, MTP, protection-floor, or fallback threshold.

## Current certification status

Pack-level verdicts live in the [certification index](../certifications/README.md),
not here. All per-pack records were withdrawn on 2026-10-03 pending
re-certification under the current host and privacy rules; this guide covers
the operator procedure, and the index is the source of truth for what is
certified today.

## Factory host

Factory **conversion**, **all Tier 1**, and **all Tier 2** certifications
**must** run on one factory host class: **Mac Studio, M2 Ultra, 192 GB
unified memory**. Records name the hardware spec, never a machine identity.

Historical certificates remain bound to the host they were measured on. The
flagship M0–M8 campaign schema and `PublicClaimManifest` performance-scope
literal stay frozen to the historical formal host (the literal lives in
`src/axquant/factory.py`) until that contract is versioned separately.

## Two-tier claim policy

AXQuant checkpoint certification is the first tier: it proves the bound artifact's size,
quality, conversion integrity, and standard-runtime compatibility. An MTP sidecar may be part of
that artifact, but its presence is not a performance claim.

AX Engine MTP acceleration is a separate second tier. It may be claimed only when the same
candidate has formal-host-bound evidence for greedy-stream exactness (100%), token-weighted decode speedup
(at least 1.20x), and prompt-median speedup (at least 1.10x). A failing or unavailable MTP gate
does not rewrite first-tier checkpoint evidence; it prevents an acceleration claim and an
`-MTP` certified-performance label.

Operationally, `axquant scoreboard --require-complete` is the checkpoint-tier scorecard. Run it
for both `agent-coding` and `general` with measured plan, matched size, and quality evidence; use
`--evaluation-profile` when the evaluation workload differs from the plan's optimization profile.
Adding `--require-mtp-acceleration` selects the acceleration tier and independently requires all
three MTP gates; the prompt guardrail cannot be hidden inside a token-weighted average. The
historical `qwen36-mtp-v2` M0–M8 release audit remains an acceleration-bearing track for full
flagship publication. Prior failed exactness runs remain archived development evidence; they do
not revoke Tier 1 and are superseded for this v3 artifact by the Tier 2 certificate once metric
gates and release-ready A/B bindings pass on the formal host.

The formal MTP harness sets
`AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1`. AX Engine also requires the loaded checkpoint
to satisfy its exact-arithmetic capability gate and emits explicit candidate telemetry. Without
that opt-in, Qwen linear-attention MTP remains on canonical direct decode; this switch is test
plumbing, not a public certification override.

## MTP flagship trust model

- `CheckpointKey` identifies source config, tokenizer, Safetensors index, and checkpoint members
  without including an absolute mount path.
- `CandidateKey` additionally binds policy, calibration, activation capture (or an explicit
  no-capture sentinel), sensitivity, semantic plan, converted manifest, and checkpoint bytes.
- A campaign freezes exactly one candidate, released toolkit wheel, runtime builds, three matched
  baselines, six disjoint dataset roles, formal-host scope, role assignments, cycle budget, and
  durable storage proof.
- Formal performance, MTP, memory, and hardware-registry evidence is authorizing only when bound
  to the formal host. A different clean host proves reproduction and path neutrality, not performance.
- The formal holdout is recorded as consumed on either pass or model failure.
- Certification and publication require different operators and an independent reviewer.

## State flow

```text
draft → frozen → formal_running → release_ready
                           └────→ formal_failed
draft/frozen → closed_no_go

development → candidate → frozen → certified → superseded
                                      └──────→ revoked
```

Campaign files and lifecycle registries are written as new atomic artifacts. Do not edit a prior
state or lifecycle event in place.

## MTP flagship operator sequence

1. Create all bound files under a durable campaign root. The repository-local disposable
   temporary-report directory is not a durable root. The campaign request, state transitions, raw
   evidence, reviews, no-go/publication records, and outputs must all remain beneath that exact
   non-symlinked root.
2. Run `campaign-overlap` once for each dataset against every other campaign dataset (algorithm
   `axquant-token-5gram-v2`; default `--id-field` order is `id` then `task_id` so calibration
   corpora and strict `QualityTask` suites can share one run). Reports contain record digests
   and similarities, not private record ids or text. Direct-track coding/general overlap
   commands use the same tokenizer — regenerate any pre-v1.5.1 coding-suite manifests before
   freeze.
3. Run `campaign-frontier` from `axquant.flagship-frontier-request.v1` to build a complete
   cheapest-failure-first `axquant.flagship-frontier.v1`; it retains failed candidates, rechecks
   every gate-evidence checksum, and derives formal eligibility rather than accepting it as a
   summary assertion.
4. Run `campaign-freeze`, then run `campaign-preflight` on the formal host. Preflight requires fresh
   doctor, Metal, zero-fallback, storage, power, and thermal results bound to the frozen host
   contract, and independently checks the live macOS/arm64/hostname/OS-build/free-disk state.
5. Run `campaign-start-formal`; the evaluation custodian executes both formal profiles.
6. Write `axquant.formal-holdout-completion.v1` with a checksum-bound raw-evidence index,
   custodian attestation, `verdict`, and
   `gate_issues`, then run `campaign-complete-formal`. The CLI derives the outcome from that bound
   record. A failure is archived; it is not tuned against the same holdout.
7. Build the preliminary `axquant.flagship-release-audit-request.v1`. `release-audit` must report
   `authorization_ready=true`; it remains `release_ready=false` until claim closure.
8. Append legal lifecycle transitions through `frozen → certified`, binding the authorization
   audit and the generated measured-BPW repository identity.
9. Run `claim-render` from `axquant.public-claim-render-request.v1`.
10. The independent reviewer signs `axquant.flagship-publication-review.v1`, binding the exact
   authorization audit, public claim, and generated card under the durable campaign root.
11. Add the authorization audit, lifecycle registry, public claim, generated card, and publication
   review to the final flagship request. Rerun `release-audit`; all M0–M8 checks must pass.
12. Run publisher preview. `publish --yes` reruns the final audit again before Hub access.
   Both paths also scan every public text file and filename for credentials, home/temp paths,
   private network addresses, invalid UTF-8, and packaged formal raw evidence.
13. Download the published revision, inventory every file, verify AX Engine/MLX-LM and
   zero-fallback behavior, then run `campaign-record-publication` with
   `axquant.flagship-publication-verification.v1`. Only that record can move the campaign to
   `published`. Future semantic changes trigger an impact scan and, when necessary, a certified
   reaffirmation, supersession, or revocation.

If the search budget is exhausted without an eligible candidate, `campaign-close-no-go` requires
an `axquant.flagship-no-go.v1` record binding the complete frontier and reviewer attestation. It
cannot consume or discard the formal holdout.

The generated repository grammar is:

```text
AX-<Base>-MLX-AXQ-MP-<measured-main-BPW>bpw[-MTP]
```

BPW is rounded to two decimal places with decimal `ROUND_HALF_UP`; the unrounded manifest values
remain authoritative. Numeric public metrics must be structured `BoundMetricClaim` records with
evidence, profile, metric key, unit, operands where relevant, and comparison direction. Free-form
marketing metrics are rejected.

## Compatibility and migration

- Historical `axquant.release-audit.v4` remains readable and reproducible.
- Qwen3-Next `N0`–`N8` remains unchanged.
- Existing development packs keep their current names and status. The only current exception is
  the exact Qwen 3.6 27B AXQ 6-bit v3 revision listed in the Tier 1 certificate.
- A package containing `public-claim.json` or a flagship lifecycle registry cannot be published
  through an older request.
- Qwen3-Next/Coder artifacts produced before the v1.2.0 fused-expert classification fix remain
  `regeneration_required`; lifecycle impact scans model this class of invalidation explicitly.
## Release operator flows (measured plan to publication)

The staged commands below turn a measured sensitivity report into a validated,
audited release. They live here — not in the README — because they are
certification-operator procedures, not evaluator onboarding.

DWQ release evidence uses the same deterministic 0.1/99.9-percentile clipping implementation
during sensitivity probing and conversion. A targeted run adds measured DWQ candidates to an
existing complete affine report without rewriting any base candidate:

```bash
axquant analyze \
  --model Qwen/Qwen3.6-27B \
  --revision pinned-source-revision \
  --calibration calibration-cache \
  --base-sensitivity measured-affine-sensitivity.json \
  --methods dwq \
  --target-tensor model.language_model.layers.4.mlp.up_proj.weight \
  --state dwq-probe-progress.json \
  --output measured-affine-dwq-sensitivity.json
```

The merged report records the base report's semantic digest, inventory digest, probe backend,
target count, and method set. Release audit requests list every ancestor under
`sensitivity_lineage`; M3 replays the chain and rejects removed or modified base candidates,
undeclared additions, protocol drift, cycles, missing parents, and unused reports.

Once a measured sensitivity report is available, create a plan without the development override:

```bash
axquant plan \
  --analysis measured-analysis.json \
  --target-bpw 4.8 \
  --bits 4,6,8,16 \
  --mtp protected \
  --output quantization-plans
```

`--lm-head-floor 8bit` is the governed size-gate path: it lowers the LM-head weight
floor from BF16 to 8-bit for that plan only, records the deviation in
`constraints.lm_head_min_bits`, and requires a measured 8-bit LM-head sensitivity candidate
before the release audit accepts the plan. The default floor stays BF16.

Validate externally collected benchmark bundles:

```bash
axquant size-evidence \
  --artifact-manifest candidate/axquant_manifest.json \
  --model-id AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --revision candidate-revision \
  --output candidate-size-evidence.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --mtp-ab candidate-mtp-ab.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --profile agent-coding \
  --output validation.json
```

For an MTP speed claim, `--mtp-ab` binds the matched AX Engine direct/MTP comparison used for
token-weighted decode speedup, prompt-median speedup, and greedy-output exactness. AXQuant rejects
the bundle when its model identity, workload, software, hardware, controls, or environment do not
match the candidate evidence.

For a `6bit` certification, freeze the class explicitly and derive the size reference from the
matching complete uniform-6 baseline. The same `max_weight_size_ratio` threshold is then applied
to the uniform-6 denominator; a 4-bit candidate cannot switch denominators opportunistically:

```bash
axquant size-evidence \
  --feasibility-report feasibility.json \
  --reference-kind uniform-6bit \
  --output uniform6-size-evidence.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --mtp-ab candidate-mtp-ab.json \
  --size-reference uniform6-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --target-class 6bit \
  --profile agent-coding \
  --output validation.json
```

If a measured Pareto candidate misses both the BPW target and the uniform-4 size-ratio gate, a
release authority can record a time-bounded exception. The command computes the observed values
from the two size artifacts; it does not accept caller-authored observed values:

```bash
axquant release-exception \
  --exception-id AXQ-SIZE-001 \
  --plan selected-plan.json \
  --candidate-size candidate-size-evidence.json \
  --size-reference uniform4-size-evidence.json \
  --tradeoff-evidence measured-tradeoff.json \
  --measured-tradeoff "Measured quality, speed, and memory tradeoff approved for release." \
  --owner "AutomatosX release owner" \
  --approved-by "Named release authority" \
  --approval-reference "release-decision-001" \
  --approved-at 2026-07-30T12:00:00Z \
  --expires-at 2027-01-31T00:00:00Z \
  --output release-exception.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --plan selected-plan.json \
  --release-exception release-exception.json \
  --exception-evidence tradeoff=measured-tradeoff.json \
  --profile agent-coding \
  --output validation.json
```

The exception can downgrade only `artifact.weight_size_ratio`; it must also disclose the failed
measured-BPW target. Quality, speed, memory, fallback, integrity, and provenance failures remain
errors. Release audit requests that use an exception must list its file under
`release_exceptions` and provide the exact `plan`, `candidate_size`, `size_reference`, and
`tradeoff` paths under `release_exception_evidence`. M4 reloads and hashes every file, checks both
validation profiles, verifies approval and expiry, and compares the packaged
`release_exception.json` with the approved record.

For a refinement candidate, derive its selection record from the converted manifest, matched
quality comparison, and release validation rather than authoring measurement values:

```bash
axquant refine-measure \
  --refinement refinement.json \
  --candidate-id cand-0000-000 \
  --measurement-id cand-0000-000-m3-max \
  --artifact-manifest candidate/axquant_manifest.json \
  --quality-comparison candidate/quality-comparison.json \
  --validation candidate/validation.json \
  --output measurements.json
```

Use `--existing measurements.json` with a new output path to accumulate another candidate or a
second named-host result for the same candidate. Measurement IDs must be unique. `refine-select`
uses the worst measured objective and BPW across every host record for a candidate, so adding
hardware evidence cannot make selection less conservative. The complete objective combines task
retention and perplexity with MTP acceptance, peak memory, and effective speed. Refinement
parentage is a precision-only monotonic chain: a child may upgrade formats but cannot downgrade
or exchange an unrelated tensor.

Prepare the exact complete-candidate run without executing expensive model work:

```bash
axquant refine-run \
  --request examples/refinement-execution-request.yaml \
  --output-dir run/complete-candidates
```

Review `execution-manifest.json`, then add `--execute`. The runner resumes checksum-verified
completed outputs, skips the remainder of a candidate after an execution failure, treats
validation exit `1` as measured failed-gate evidence, merges complete measurements, and runs
`refine-select` plus `pareto`.

Every release benchmark must name its power mode and quantizer/version. `refine-run` reads
`benchmark_power_mode` from its request, derives the AXQuant identity from each plan, and includes
both raw A/B logs in the resumable output contract. Standalone baseline runs use
`--power-mode`, `--quantizer`, and `--quantizer-version`. `benchmark-ab` derives adjacent-token
repetition directly from emitted token IDs, records depth-one proposal accuracy, and derives
greedy divergence from the matched A/B outputs. Its release speed gate defaults to
`token-weighted-decode-tps`: total output tokens divided by total generation wall time, with the
same calculation applied to both arms. The artifact also records the legacy prompt-median TPS
ratio and requires it to remain at or above `1.10x`, preventing a long decode from hiding a
typical-prompt regression. Release MTP evidence therefore requires token-weighted decode speedup
`>=1.20x`, prompt-median speedup `>=1.10x`, and exact greedy outputs. Use
`--speedup-metric prompt-median-tps` only when reproducing the
legacy protocol. For a uniform-6 reference A/B, use
`--direct-baseline-kind uniform-6bit --mtp-baseline-kind uniform-6bit`; the default kinds remain
the AXQuant MTP-off/on release pair. Use `--record-failed-speedup` for an evidence sweep that must
retain both evaluation bundles when only the speed floor fails: the command writes the complete
evidence, returns status `1`, and leaves exactness and matched-control invariants fail-closed.
Build the M7 hardware registry only from the resulting raw
logs, evaluation bundles, validation, plan, converted artifact manifest, sensitivity report,
quality comparison, and quantizer execution manifest:

```bash
axquant hardware-registry \
  --request examples/hardware-registry-request.yaml \
  --output hardware-profile-registry.json
```

The command returns `1` while validation is failing, any runtime or conversion fallback is
present, provenance is inconsistent, the complete objective cannot be rebuilt from the artifact,
quality, and validation files, or the claimed bit/group/role/shape coverage is not measured. The
registry records both the semantic and file digest of its complete-candidate measurement set.
Publication verifies that file, packages it as `refinement_measurements.json`, packages every
objective input, and rewrites the registry to packaged relative paths. Each registry entry
identifies the exact measurement ID, allowing one candidate and plan to be certified on multiple
named hosts.

### Flagship campaign closure

The certified Qwen 3.6 path starts from the exact source
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. It is separate from the
historical v4 development audit:

```bash
axquant campaign-freeze \
  --request flagship-campaign-request.json \
  --output flagship-campaign.json

# Authorizing preflight must run on the formal host
# (MacBook Pro, M5, 128 GB, 18-core class).
axquant campaign-preflight \
  --campaign flagship-campaign.json \
  --output flagship-campaign-preflight.json

axquant release-audit \
  --request flagship-release-audit-request.json \
  --output flagship-authorization-audit.json
```

An authorization-ready audit proves the frozen campaign and current M0–M8 evidence but is
deliberately not publication-ready until the independent lifecycle and claim closure is present.
The campaign request and every transition, raw-evidence, review, no-go, and publication record
must remain inside the declared non-symlinked durable root. Formal preflight also requires fresh
doctor, Metal, zero-fallback, storage, power, and thermal results bound to the exact frozen
host contract (MacBook Pro, M5, 128 GB, 18-core class).
After the legal `frozen → certified` event, `claim-render` creates `public-claim.json` and the
measured-BPW `README.md`. An independent final publication review binds those exact files and the
authorization audit under the durable campaign root; the final flagship request must pass M0–M8
again. Preview and executed publication both rerun that exact final request. A v4 audit cannot
authorize a package containing flagship claims or lifecycle metadata.

Prepare the release directory locally, then run the aggregate proof before publishing a certified
checkpoint:

```bash
axquant publish-prepare \
  --model AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json
```

```bash
axquant release-audit \
  --request examples/release-audit-request.yaml \
  --output release-audit.json
```

This revalidates indexed evaluation, complete-refinement, and hardware file checksums; binds the
selected interaction improvement to the packaged measurement set; reruns reproduction
verification; inspects the wheel metadata, contents, and every `RECORD` member hash/size; and
requires the packaged plan, validation/benchmark evidence, hardware registry/evidence,
refinement measurements, Pareto report, and recipe to match the external evidence graph. The
M0 check recomputes checkpoint completeness, parameter/architecture equivalence, revisions, MTP,
and baseline runtime results rather than trusting the feasibility status label. M1 requires every
artifact Safetensors file to have one safe, size- and checksum-valid manifest record. M2 reloads
the indexed evaluations and rechecks complete trials, matched controls and hardware, provenance,
fallbacks, identical-checkpoint MTP pairing, one cross-profile candidate/reference pair, and
disjoint datasets. M3 reloads the checksum-bound calibration manifest, verifies separation and
provenance, requires finite tensor-scoped measurements, and verifies every targeted-sensitivity
ancestor. M6 reloads the bound artifact, quality comparison, and validation for every
measurement, recomputes the versioned complete objective, and requires a measured, validated,
monotonic parent/child gain. Complete-measurement construction also rejects non-authoritative
profile thresholds, an inconsistent validation pass label, core release metrics below their
active thresholds, nonzero kernel fallbacks, and a passing size overage without its governed
plan-bound exception. M7 rebuilds every Pareto point and frontier member from the bound
measurement set. The
compatibility matrix must bind that same candidate manifest, runtime checks, and validation. The
audit also reloads every checkpoint from the original compatibility request and re-hashes its
manifest, plan, runtime checks, and validation. The wheel must declare Python 3.11+, MIT, and all
runtime dependencies; the artifact, plan, recipe, and wheel must identify the same AXQuant
version. It reports M0 through M8 separately and returns `0` only when all nine milestones pass;
an alpha or pre-1.0 wheel, including one still carrying an Alpha distribution classifier, cannot
pass M8. An executed publication packages that exact authorizing result as `release_audit.json`
and refuses to overwrite a different existing audit.

Preview publication first. Add `--yes` only when the release should be uploaded; an executed
upload also requires the matching audit and its original request so the full M0–M8 proof can be
rerun from current evidence:

```bash
axquant publish \
  --model AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json \
  --release-audit release-audit.json \
  --release-audit-request examples/release-audit-request.yaml
```

Before publication, build the complete comparison index. BF16, uniform 4-bit, uniform 6-bit,
and the identical AXQuant MTP-off/on pair are mandatory. Mixed-precision, AWQ, and DWQ entries
may be unavailable, but they cannot be omitted and must state why:

```bash
axquant benchmark-index \
  --request examples/benchmark-evidence-request.yaml \
  --output benchmark-evidence-index.json
```

Build one benchmark index and validation report for each required profile, using distinct
evaluation datasets. Then bind them into the publication gate:

```bash
axquant validation-index \
  --request examples/release-validation-request.yaml \
  --output release-validation-index.json
```

Publication rejects a missing profile, a reused dataset, differing candidate/reference
identities, a failed validation, or a non-ready benchmark index.

Every prepared release includes `reproduction_recipe.yaml` with argument-array commands for
downloading the pinned source, converting it, checking both runtimes, and verifying every
regenerated Safetensors file. Prepared MTP layouts additionally checksum-bind the provenance and
runtime companion files required to reuse the transformed sidecar without applying the transform
again. After running those commands, verification can also be invoked directly:

```bash
axquant verify-reproduction \
  --recipe reproduction_recipe.yaml \
  --artifact regenerated-model \
  --output reproduction-verification.json
```

Build the M5 family matrix from checksum-bound artifact, AX Engine, MLX-LM, and validation
evidence:

```bash
axquant compatibility-matrix \
  --request examples/qwen36-compatibility-request.yaml \
  --output compatibility-matrix.json
```

The request declares the complete official dense catalog as verified at a timezone-qualified
timestamp. The command returns `1` and still writes the matrix when any declared official dense
Qwen 3.6 model is absent, uses inconsistent candidate evidence, or lacks a compatible
`agent-coding` or `general` validation profile. The checked-in example lists 27B as the only dense
size in the linked catalog; refresh `catalog_verified_at` and `required_dense_models` before every
release. FP8 is a representation of a parameter size, not a second model size.

