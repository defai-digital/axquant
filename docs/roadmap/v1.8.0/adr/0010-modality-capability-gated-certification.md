# ADR-0010: Capability-gated vision/audio certification

| Field | Value |
| --- | --- |
| Status | Accepted |
| Date | 2026-08-14 |
| Release | AXQuant 1.8.0 |
| Spec | [certification-spec-v1.0 §8](../../../certification-spec-v1.0.md) |
| Code | `src/axquant/modality_certification.py`, `schema/public_certification.py` |

## Context

Public Tier 1 certificates measured only text dual-suite quality while many packs ship
vision (and occasionally audio) weights as BF16-protected sidecars. Model cards report
“Vision present” without a machine-readable claim boundary, so readers can confuse
presence with quality certification. Gemma 4 text-path packs strip multimodal configs
entirely; Qwen3-VL binds MLX-VLM smoke without a vision retention suite.

## Decision

1. **If a modality is not supported** on the certified pack → status `not-applicable`
   (`supported=false`). Do not run vision/audio suites.
2. **If a modality is supported** → either:
   - run smoke → `smoke-certified`;
   - run quality suite → `quality-certified`; or
   - leave explicit `present-not-certified` (protect-only / no claim).
3. Text dual-suite never upgrades multimodal claim status.
4. Optional `modalities` block on Tier 1 JSON; omission means legacy-unstated, not a pass.

## Consequences

- Public certs can state multimodal policy without inventing false quality scores.
- Future vision/audio quality suites plug into the same statuses.
- Cards and indexes should prefer the `modalities` block over free-text notes alone.
