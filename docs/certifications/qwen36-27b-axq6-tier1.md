# Qwen 3.6 27B AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-08.

This certificate covers the exact v3 weight set published as
[`AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP)
and tagged `v3`. It certifies measured size, quality retention against the matched uniform-6
reference, conversion integrity, and the safe/stable default text-runtime route. It does **not**
certify MTP acceleration, vision-language quality, every possible prompt, or sibling revisions.

The companion [machine-readable record](qwen36-27b-axq6-tier1.json) contains the exact hashes and
unrounded values.

`v3` is the public artifact edition, not a schema migration. The embedded manifest intentionally
retains `schema_version: axquant.artifact.v2`; no `axquant.artifact.v3` schema is introduced by
this certification.

## Bound artifact

| Property | Value |
| --- | --- |
| Artifact edition | `v3` |
| Hub tag | `v3` |
| Immutable Hub commit | [`cdd13bf81cf21818a01cf59a31fc116ef84326bc`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP/tree/cdd13bf81cf21818a01cf59a31fc116ef84326bc) |
| Product class | `6bit` mixed precision |
| Source | `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Candidate manifest SHA-256 | `88269a83d0fd7bf70381745e1163f2b919e09798bcc92e352757d0f00e112c43` |
| Quantization profile | `agent-coding` |
| Plan evidence | measured |
| Planned effective BPW | `5.961609247665644` |
| Measured main-model BPW | `5.805848959887475` |
| Conversion coverage | 487/487 modules, zero fallback |
| Certification host | `df-macbookpro-m5` |

Publication removes local filesystem paths and changes the model card, but does not change any
tensor, quantization assignment, source revision, or measurement. The original and public plans
have the same assignment digest
`986c3d8c479d5b6f3d042b299ac83345bdbf9a982b3d8f474b3230b7dc3bf179`.
The previous development artifact remains immutable at the Hub `v2` tag.

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Effective BPW | ≤ `5.977` | `5.961609` | Pass |
| Weight-size ratio vs uniform-6 | ≤ `1.0` | `0.876234` | Pass |
| General quality retention | ≥ `0.99` | `1.011494` | Pass |
| Agent-coding quality retention | ≥ `0.99` | `1.007353` | Pass |
| Reference/candidate scorer errors | `0 / 0` | `0 / 0` in both profiles | Pass |
| Worst per-task delta | ≥ `-0.02` | `0.0` in both profiles | Pass |
| Default AX Engine route | MTP off, direct fallback | policy `4`, fallback `1` | Pass |

The v3 checkpoint contains `20,703,029,777` weight bytes versus `23,627,280,146` for the matched
uniform-6 reference: a `12.38%` reduction.

### Quality suites

| Evaluation profile | Tasks | Reference | Candidate | Retention | Perplexity ratio | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 44 | `0.988636` | `1.000000` | `1.011494` | `0.966136` | `0 / 0` |
| Agent-coding | 76 | `0.894737` | `0.901316` | `1.007353` | `0.969531` | `0 / 0` |

Agent-coding JSON-validity delta was `0.0`; syntax-validity delta was `+0.066667`. Evaluation used
greedy generation, seed `20260728`, maximum sequence length 2,048, and maximum generation length
64. Dataset SHA-256 values and task counts are pinned in the machine-readable record.

These are reproducible AXQuant development suites, not an independent third-party benchmark or a
fresh blinded holdout. Public claims must retain that scope and must not generalize the result to
all downstream applications.

## Stable default runtime scope

The runtime smoke used AX Engine 6.13.5 with MLX 0.32.0. Telemetry proved that the public default
does not silently enter the unreleased Qwen linear-MTP candidate route:

| Telemetry | Value |
| --- | ---: |
| `ax_mlx_mtp_model_policy` | `4` |
| `ax_mlx_mtp_model_policy_route_safe` | `0` |
| `ax_mlx_mtp_model_policy_active` | `0` |
| `ax_mlx_qwen_linear_mtp_certification_candidate` | `0` |
| `ax_mlx_qwen_linear_mtp_direct_fallback` | `1` |
| `ax_mtp_requested` | `0` |
| `ax_mtp_decode_steps` | `0` |
| `ax_mlx_direct_pipeline_steps` | `7` |

Here, “stable” means the bound standard text path passed conversion, model loading, inference, and
the fail-closed default-route gate on the certification machine. It is not a promise that every
application, context length, or third-party runtime is defect-free.

## Tier 2 status

Checkpoint Tier 1 does **not** by itself certify MTP acceleration. For this exact v3 artifact,
the separate MTP acceleration certificate is now published:

- [Tier 2 MTP acceleration certification](qwen36-27b-axq6-tier2.md)
- [Machine-readable Tier 2 record](qwen36-27b-axq6-tier2.json)

Tier 2 is **scoped**: decode-heavy authorizing profiles on `df-macbookpro-m5` under the formal
Qwen linear MTP exact contract. The product **default** route remains direct fallback (safe Tier 1
default); short-answer chat is not an authorizing acceleration claim.

## Reproducibility and integrity

- AXQuant commit: `726d9b9b69c5e8bc45a0186882074009ec31277d`
- AX Engine commit: `0a2ec300472f4829a52e6f25389425ff2004842c`
- Python: 3.13.14
- MLX: 0.32.0
- MLX-LM: 0.31.3
- Certification summary SHA-256:
  `80dd3099fb93b6f592b186be5ef3c82968e90e797750468b408a7621d46f461e`
- General scorecard SHA-256:
  `e9d93e8572769564ee04e886ec6cedd70c9c53dcb474879a09b3cf2d5f369290`
- Agent-coding scorecard SHA-256:
  `96b9a8a3cce12e7499a99c6a2be6f0f3c7820a61c6a74b4453c4071a09743485`

The certificate is valid only while the published v3 LFS object hashes match the weight-file
hashes in the machine-readable record. A weight, plan, tokenizer, source, threshold, or runtime
policy change requires impact review and, when material, recertification.
