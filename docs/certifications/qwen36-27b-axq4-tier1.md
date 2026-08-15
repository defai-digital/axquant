# Qwen 3.6 27B AXQ 4-bit MTP (5.6 BPW) — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-08 (quality + size
budget for product class `5p6bpw`; see size note).

This certificate covers the exact weight set published as
[`AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP)
at Hub commit
[`f44a9eeebec0c488d0f42201c8763db770a1c0a8`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP/tree/f44a9eeebec0c488d0f42201c8763db770a1c0a8).

It certifies measured size against the matched uniform-4 reference under a
**`5p6bpw` size budget**, quality retention against that reference, conversion
integrity, and loadability on AX Engine / mlx-lm. It does **not** by itself
certify MTP acceleration (see [Tier 2](qwen36-27b-axq4-tier2.md)).

The companion [machine-readable record](qwen36-27b-axq4-tier1.json) contains the
exact hashes and unrounded values.

## Bound artifact

| Property | Value |
| --- | --- |
| Hub repo | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` |
| Immutable Hub commit | [`f44a9eeebec0c488d0f42201c8763db770a1c0a8`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP/tree/f44a9eeebec0c488d0f42201c8763db770a1c0a8) |
| Product class | `5p6bpw` mixed precision (marketing name: 4-bit AXQ) |
| Source | `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Candidate manifest SHA-256 | `fa58908d73a8f7046d9f8d86cff278326fc97632e71f0d6675cbc3e8b13439b7` |
| Plan SHA-256 | `c4532c4084d20508cc730a386f03a18802efa1542d747be7092fa70182d6a857` |
| Plan evidence | `architecture_prior` (not measured calibration evidence) |
| Planned effective BPW | `5.579999028841856` |
| Measured total BPW | `5.580079760185251` |
| Measured main BPW | `5.418315254285654` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Effective BPW | ≈ target `5.58` | `5.580080` measured total | Pass |
| Weight-size ratio vs uniform-4 | ≤ `1.25` (`5p6bpw` budget) | `1.206999` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.992647` | Pass |
| Reference/candidate scorer errors | `0 / 0` | `0 / 0` both profiles | Pass |

### Size note (read carefully)

This pack is **not** a pure uniform-4 artifact. Target class is `5p6bpw`
(mixed 4/8/16-bit). Against `mlx-community/Qwen3.6-27B-4bit` the weight ratio is
about **1.207×** (19.38 GiB candidate vs 16.05 GiB uniform-4). The strict
uniform-4 ≤1.10 gate used for pure-4bit product classes does **not** apply; the
scoreboard used `max_size_ratio=1.25` for this class.

Candidate weight bytes: `19,377,822,978`. Uniform-4 reference:
`16,054,541,599`.

### Quality suites

| Evaluation profile | Tasks | Reference | Candidate | Retention | Perplexity ratio | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 44 | `1.000000` | `1.000000` | `1.000000` | `0.987408` | `0 / 0` |
| Agent-coding | 76 | `0.894737` | `0.888158` | `0.992647` | `0.991979` | `0 / 0` |

Agent-coding JSON category delta was `-0.033` (one task-level miss on
`json-015`); aggregate retention still meets the ≥0.98 gate. Evaluation used
greedy generation, seed `20260728`, max generation length 64. Dataset SHA-256
values are pinned in the machine-readable record.

These are reproducible AXQuant development suites, not an independent
third-party benchmark.

## Plan evidence limitation

The embedded plan uses `evidence_kind: architecture_prior`. Formal scoreboard
**release-claim** eligibility therefore remains limited (scoreboard warning:
architecture_prior cannot support full release claims). Public Tier 1 for this
sibling pack is based on **measured quality, size, and runtime doctor**
evidence on the formal host, not on measured-plan provenance.

## Runtime scope

AX Engine doctor (`6.14.0`) and mlx-lm static compatibility both passed on
`df-macbookpro-m5`. Product **default** MTP remains direct fallback; MTP
acceleration is covered only by [Tier 2](qwen36-27b-axq4-tier2.md).

## Related certificates

- [Tier 2 MTP acceleration](qwen36-27b-axq4-tier2.md)
- Dense sibling 6-bit flagship: [Tier 1](qwen36-27b-axq6-tier1.md) /
  [Tier 2](qwen36-27b-axq6-tier2.md)

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision present sidecar=['vision.safetensors']; mlx-vlm smoke failed on df-macstudio-m2 (see evidence). Text Tier 1 unchanged. Evidence: docs/certifications/evidence/modality-recert-macstudio-m2/results/qwen36-27b-axq4-mtp.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
