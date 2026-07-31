# ADR 0004: Measured Evidence and Converted-Artifact Equivalence

**Status:** Accepted  
**Date:** 2026-07-28

## Context

Release planning cannot use architecture priors as a substitute for forward measurements, and a
successful MLX-LM conversion does not itself prove that every protected or external component is
present. The first vertical slice exposed the latter failure mode when the portable conversion
omitted the source vision tower while still producing loadable language weights.

## Decision

AXQuant uses a deterministic tokenized calibration cache and a resumable isolated-module MLX
probe for per-tensor 4/6/8/BF16 measurements. Development-scale probes are explicitly
`measured_development`; release-quality `measured` evidence requires the declared sample, token,
domain, long-context, provenance, and calibration/evaluation separation gates.

Conversion remains atomic and adds a final staging inventory. Before the staging directory can be
renamed, AXQuant requires:

- exact total logical parameters relative to the plan;
- exact MTP logical parameters;
- exact protected-vision logical parameters;
- a valid inspected set of model Safetensors.

`axquant.artifact.v2` records the reconstructed total and main logical parameters, inspected
total/main/MTP/protected weight bytes, planner-estimated BPW, and authoritative measured total and
main BPW. Declared root MTP and protected-vision sidecars are checkpoint members.

The AX Engine A/B harness compares the same checkpoint, prompts, seed, runtime, settings, and
hardware with MTP disabled and enabled. Greedy output divergence, inactive MTP, trial failures, or
nonpositive speed evidence fail the release claim.

## Consequences

- A loadable but incomplete checkpoint never appears at the requested final path.
- Old `axquant.artifact.v1` manifests do not satisfy the new release contract and must be
  regenerated.
- Measured artifact BPW cannot be replaced by the planner proxy.
- Smoke probes and manual vertical slices remain useful without being mislabeled as release
  evidence.
- A failed MTP run is retained as evidence and cannot be summarized as an acceleration result.
