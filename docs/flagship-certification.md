# Qwen 3.6 flagship certification

AXQuant’s first flagship policy is `qwen36-mtp-v2`. It applies to one exact source:

```text
Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

It does not certify existing development packs automatically and does not change any M0–M8
quality, size, MTP, protection-floor, or fallback threshold.

## Current certification status

The exact Qwen 3.6 27B AXQ 6-bit **artifact edition v3** passed checkpoint Tier 1 on
2026-08-08 and MTP acceleration Tier 2 (scoped) on 2026-08-08. Public records:

- [Checkpoint Tier 1](certifications/qwen36-27b-axq6-tier1.md)
- [MTP acceleration Tier 2](certifications/qwen36-27b-axq6-tier2.md)

Sibling packs certified on the same host (`df-macbookpro-m5`) for **checkpoint Tier 1**
(and scoped Tier 2 where noted):

| Pack | Tier 1 | Tier 2 |
| --- | --- | --- |
| 27B AXQ 6-bit v3 | [Yes](certifications/qwen36-27b-axq6-tier1.md) | [Scoped yes](certifications/qwen36-27b-axq6-tier2.md) |
| 27B AXQ 4-bit (5.6 BPW) | [Yes](certifications/qwen36-27b-axq4-tier1.md) | [Scoped yes](certifications/qwen36-27b-axq4-tier2.md) |
| 35B-A3B AXQ 4-bit | [Yes](certifications/qwen36-35b-axq4-tier1.md) | [Scoped yes](certifications/qwen36-35b-axq4-tier2.md) |
| 35B-A3B AXQ 6-bit | [Yes](certifications/qwen36-35b-axq6-tier1.md) | [Scoped yes](certifications/qwen36-35b-axq6-tier2.md) |

| Claim | Status |
| --- | --- |
| Checkpoint size, matched-reference quality, conversion integrity | Certified (per pack record) |
| Safe/stable default AX Engine text route | Certified; product default remains direct fallback |
| MTP speculative-decode speedup and exactness (decode-heavy authorizing profiles) | **Certified (scoped)** for dense 27B packs on `df-macbookpro-m5` / AX Engine 6.14.0 and for MoE 35B-A3B packs on the same host / AX Engine 6.14.1 (MoE exact profile) |
| Short-answer / universal prompt acceleration | Not certified |
| Vision-language quality | Not certified |
| Full M0–M8 flagship publication campaign | Separate process; not implied by Tier 2 metric closure |


### Gemma-4 AXQ siblings

Hub packs ship fused **assistant-MTP** (`assistant/` + `ax_gemma4_assistant_mtp.json`) under the
Qwen-style `…-MLX-AXQ-*-MTP` names. Checkpoint **Tier 1** is certified for both AXQ 4-bit and
6-bit fused Hub heads on `df-macbookpro-m5` (2026-08-09). **Tier 2 is not certified** on any
Gemma pack.

| Pack | Tier 1 | Tier 2 |
| --- | --- | --- |
| 12B AXQ 4-bit (IT rebuild) | [Certified](certifications/gemma4-12b-axq4-tier1.md) | [Not Certified](certifications/gemma4-12b-axq4-tier1.md#tier-2-status) |
| 12B AXQ 6-bit (IT rebuild) | [Certified](certifications/gemma4-12b-axq6-tier1.md) | [Not Certified](certifications/gemma4-12b-axq6-tier1.md#tier-2-status) |
| 26B-A4B AXQ 4-bit | [Certified](certifications/gemma4-26b-a4b-axq4-tier1.md) | [Not Certified](certifications/gemma4-26b-a4b-axq4-tier1.md#tier-2-status) |
| 26B-A4B AXQ 6-bit | [Certified](certifications/gemma4-26b-a4b-axq6-tier1.md) | [Not Certified](certifications/gemma4-26b-a4b-axq6-tier1.md#tier-2-status) |
| 31B AXQ 4-bit | [Certified](certifications/gemma4-31b-axq4-tier1.md) | [Not Certified](certifications/gemma4-31b-axq4-tier1.md#tier-2-status) |
| 31B AXQ 6-bit | [Certified](certifications/gemma4-31b-axq6-tier1.md) | [Not Certified](certifications/gemma4-31b-axq6-tier1.md#tier-2-status) |

Tier 1 binds the **fused Hub head**: target weight digests match the quality-bound canonical
pack; assistant assets are attached without mutating target weights. The 12B packs were rebuilt
from `google/gemma-4-12b-it` after non-IT sources failed quality. Formal assistant-MTP A/B pilots
on `df-macbookpro-m5` / AX Engine 6.14.0 (complete exact-profile confidence gates) show weighted
and prompt-median speed can clear while **greedy outputs diverge** when drafts are accepted —
exactness is fail-closed, so Tier 2 stays unclaimed. Product default remains direct fallback.

### Qwen3-Coder-Next (non-MTP direct-decode)

Hybrid MoE coding checkpoint (`Qwen3NextForCausalLM`) with **no declared MTP**. Public
certificates are checkpoint **Tier 1 only** on `df-macbookpro-m5` (2026-08-10): size vs matched
uniform, quality retention on agent-coding + general, MLX-LM load. **Tier 2 is not applicable.**

| Pack | Tier 1 | Tier 2 |
| --- | --- | --- |
| Coder-Next AXQ 4-bit | [Certified](certifications/qwen3-coder-next-axq4-tier1.md) | N/A |
| Coder-Next AXQ 6-bit | [Certified](certifications/qwen3-coder-next-axq6-tier1.md) | N/A |

### Qwen3-VL 30B-A3B Instruct (non-MTP VL MoE)

Vision MoE Instruct checkpoint (`Qwen3VLMoeForConditionalGeneration`) with **no declared MTP**.
Public certificates are checkpoint **Tier 1 only** on `df-macbookpro-m5` (2026-08-11) with
**AX Engine 6.15.0** primary runtime and MLX-VLM vision smoke: size vs matched mlx-community
uniform, quality retention on agent-coding + general. **Tier 2 is not applicable.**

| Pack | Tier 1 | Tier 2 |
| --- | --- | --- |
| 30B-A3B Instruct AXQ 4-bit | [Certified](certifications/qwen3-vl-30b-axq4-tier1.md) | N/A |
| 30B-A3B Instruct AXQ 6-bit | [Certified](certifications/qwen3-vl-30b-axq6-tier1.md) | N/A |

## Future certification host

New public checkpoint Tier 1 and new product certifications run on
`df-macstudio-m2` (Mac Studio M2 Ultra, 192 GB, Ext4T). Do not start a new
certification campaign on `df-macbookpro-m5`.

Existing certificates in this document remain bound to the host they were
measured on (mostly `df-macbookpro-m5`). The flagship M0–M8 campaign schema
and `PublicClaimManifest` performance-scope literal are still frozen to
`df-macbookpro-m5` until that contract is versioned separately.

## Two-tier claim policy

AXQuant checkpoint certification is the first tier: it proves the bound artifact's size,
quality, conversion integrity, and standard-runtime compatibility. An MTP sidecar may be part of
that artifact, but its presence is not a performance claim.

AX Engine MTP acceleration is a separate second tier. It may be claimed only when the same
candidate has M5-bound evidence for greedy-stream exactness (100%), token-weighted decode speedup
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
gates and release-ready A/B bindings pass on `df-macbookpro-m5`.

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
- Existing development packs keep their current names and status. The only current exception is
  the exact Qwen 3.6 27B AXQ 6-bit v3 revision listed in the Tier 1 certificate.
- A package containing `public-claim.json` or a flagship lifecycle registry cannot be published
  through an older request.
- Qwen3-Next/Coder artifacts produced before the v1.2.0 fused-expert classification fix remain
  `regeneration_required`; lifecycle impact scans model this class of invalidation explicitly.
