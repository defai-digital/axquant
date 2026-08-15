# GPT-OSS 20B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-10.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit)
commit
[`a04eea371082f799f1c6aa76f6afceb615334627`](https://huggingface.co/AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit/tree/a04eea371082f799f1c6aa76f6afceb615334627).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `6bit` |
| Source (convert input) | `mlx-community/gpt-oss-20b-MXFP4-Q4@f356f2747216d7e98fee755df25987459fc19089` |
| Upstream lineage | OpenAI `gpt-oss-20b` (`model_type=gpt_oss`) |
| Candidate manifest SHA-256 | `cc501a7b093f40acf67374db5e8f46a292c39f3e93cddf47c4ecde44c737189e` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `6.000036707478516` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `6.0` | `6.000036707478516` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.55` | `1.403235` | Pass |
| General quality retention | ≥ `0.98` | `0.999944` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.024845` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `15,686,163,854` vs MXFP4-Q4 uniform reference
`11,178,569,159` → **1.403×** (expected: 6-bit pack vs denser 4-bit MXFP4 baseline).

### Quality suites

| Profile | Reference | Candidate | Retention |
| --- | ---: | ---: | ---: |
| Agent-coding (76) | `0.353070` | `0.361842` | `1.024845` |
| General (44) | `0.638182` | `0.638146` | `0.999944` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.
Quality is measured against the matched `mlx-community` MXFP4-Q4 pack (not BF16).

## Tier 2 status

**Not applicable.** GPT-OSS has no declared MTP weights; this certificate is
non-MTP direct-decode checkpoint Tier 1 only. No speculative-decode speedup
claim is authorized.

## Related

- Sibling 4-bit (certified recovery recipe): [gpt-oss-20b-axq4-tier1.md](gpt-oss-20b-axq4-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-20b-axq6-tier1.json](gpt-oss-20b-axq6-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `not-applicable` | `false` | vision not supported (no tower config and no sidecar weights) |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
