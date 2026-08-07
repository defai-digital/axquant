# Qwen 3.6 flagship certification

AXQuant’s first flagship policy is `qwen36-mtp-v2`. It applies to one exact source:

```text
Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

It does not certify existing development packs automatically and does not change any M0–M8
quality, size, MTP, protection-floor, or fallback threshold.

## Trust model

- `CheckpointKey` identifies source config, tokenizer, Safetensors index, and checkpoint members
  without including an absolute mount path.
- `CandidateKey` additionally binds policy, calibration, activation capture (or an explicit
  no-capture sentinel), sensitivity, semantic plan, converted manifest, and checkpoint bytes.
- A campaign freezes exactly one candidate, released toolkit wheel, runtime builds, three matched
  baselines, six disjoint dataset roles, `df-macbookpro-m5` host scope, role assignments, cycle budget, and
  durable storage proof.
- Formal performance, MTP, memory, and hardware-registry evidence is authorizing only when bound
  to `df-macbookpro-m5`. A different clean host proves reproduction and path neutrality, not performance.
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

## Operator sequence

1. Create all bound files under a durable campaign root. The repository-local disposable
   temporary-report directory is not a durable root. The campaign request, state transitions, raw
   evidence, reviews, no-go/publication records, and outputs must all remain beneath that exact
   non-symlinked root.
2. Run `campaign-overlap` once for each dataset against every other campaign dataset. Reports
   contain record digests and similarities, not private record ids or text.
3. Run `campaign-frontier` from `axquant.flagship-frontier-request.v1` to build a complete
   cheapest-failure-first `axquant.flagship-frontier.v1`; it retains failed candidates, rechecks
   every gate-evidence checksum, and derives formal eligibility rather than accepting it as a
   summary assertion.
4. Run `campaign-freeze`, then run `campaign-preflight` on `df-macbookpro-m5`. Preflight requires fresh
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
- Existing development packs keep their current names and status.
- A package containing `public-claim.json` or a flagship lifecycle registry cannot be published
  through an older request.
- Qwen3-Next/Coder artifacts produced before the v1.2.0 fused-expert classification fix remain
  `regeneration_required`; lifecycle impact scans model this class of invalidation explicitly.
