# MTP gate mapping: AXQuant Tier 2 vs AX Engine MTP-S / MTP-P / MTP-D

Status: explanatory cross-reference. This page is **not** part of the normative
[certification specification v1.0](../contracts/certification-spec-v1.0.md); it
records how to read an AXQuant Tier 2 certificate next to AX Engine's MTP gate
vocabulary (internal decision ADR-033, 2026-09-18).

## Why this page exists

AX Engine and AXQuant both use the word "Tier 2" for MTP, but they mean
different things:

- **AXQuant Tier 2** — a *publisher* certification: a specific pack, on a
  specific host and AX Engine build, achieved measured MTP decode acceleration
  on named authorizing workloads, within the thresholds recorded in the
  certificate.
- **AX Engine "MTP Tier 2"** — a *runtime* gate programme that ADR-033 split
  into three independent gates, one of which (default promotion) is still
  pending for the current Qwen 3.8 27B pack.

A reader who sees "Tier 2 Certified" on one side and "MTP Tier 2 pending" on the
other is reading two different claims, not a contradiction.

## Gate mapping

| AX Engine gate (ADR-033) | What it decides | Is an AXQuant Tier 2 certificate evidence for it? |
| --- | --- | --- |
| **MTP-S** — safety | In-path exactness: under one verifier state, every accepted draft equals the token that verifier's own greedy decision would select | **No.** MTP-S is a runtime acceptance property owned by AX Engine. An AXQuant Tier 2 record measures MTP-off/on output identity across a benchmark dataset; it does not introspect per-acceptance verifier logits and cannot establish the in-path invariant |
| **MTP-P** — performance claim license | Licenses a scoped, reproducible public acceleration multiplier (weighted >= 1.20x, prompt-median >= 1.10x, two named authorizing workloads, short-answer negative control) | **Closest match, with two differences.** AXQuant Tier 2 uses the same 1.20x / 1.10x thresholds and the same authorizing-workload discipline. It is measured under the certificate's recorded execution profile, not necessarily the shipping default path, and it is bound to one engine build |
| **MTP-D** — default promotion | Whether MTP becomes the product default. Requires a separate decision, a release tag, and 100% greedy parity on the default product path | **No.** AXQuant certificates record `product_default_route = direct-fallback` and do not claim default promotion |

## What a Tier 2 certificate is evidence for

For the exact artifact revision, host, and AX Engine build bound in the record:

- measured token-weighted decode speedup and prompt-median speedup on the
  certificate's named authorizing workloads, against the recorded thresholds;
- MTP-off/on token identity over the harness scope recorded in the certificate
  (`measured_trials`, `max_tokens`, `ignore_eos`, seed, draft depth);
- reproducible provenance for the comparison: dataset digest, comparison digest,
  runtime environment, and toolchain versions.

## What a Tier 2 certificate is not evidence for

- AX Engine in-path safety (**MTP-S**) or AX Engine default promotion
  (**MTP-D**).
- Any host or engine build other than the bound ones. Each certificate states its
  own integrity rule: the result is invalid if weights, thresholds, host, or
  engine binary diverge from the bound digests. Certified records are historical;
  they are not re-certified for later AX Engine releases.
- Workloads or generation lengths outside the recorded scope — including
  short-answer/chat-length speedup, which the certificate harness explicitly
  does not claim.
- Sibling packs, other revisions, or other quantization product classes.

## Worked example: Qwen3.8-27B MLX AXQ 6-bit MTP

| Surface | Statement | Refers to |
| --- | --- | --- |
| AXQuant | Tier 2 certified, bound to AX Engine 6.16.1, host `df-macbookpro-m3`, authorizing profiles agent-coding and long-form general | Scoped measured acceleration for that pack and engine build |
| AX Engine | "MTP Tier 2 pending" | MTP-D not opened, and MTP-P not yet evidenced **on the default product path** |

Both statements can be true at once. The AXQuant record covers acceleration
measured under a certification arithmetic profile on a specific engine build; the
AX Engine record concerns the shipping default path on a later engine build.

## How to read the matrices

This page discharges the disclosure obligation in
[`docs/certifications/README.md`](README.md),
[`docs/releases/certification-matrix.md`](../releases/certification-matrix.md),
and the README headline matrix. Each certified Tier 2 cell names the AX Engine
build it is bound to, and each matrix repeats the scope rule. When a certificate
is superseded by re-certification on a newer engine build, add the new record
rather than editing the historical one.
