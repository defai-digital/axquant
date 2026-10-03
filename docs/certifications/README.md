# AXQuant checkpoint certifications

This directory records public, revision-bound AXQuant certificates. A checkpoint-tier certificate
covers the exact artifact named by its record; it does not promote sibling models, other
revisions, or unscoped runtime claims.

The index table below is **generated** from the `*-tier1.json` / `*-tier2.json` records
(`public_index` fields plus verdicts) for packs with `public_index.listed = true`.
Do not edit the table by hand — update the certificate JSON and run
`python scripts/render_certification_docs.py --write`. CI enforces exact agreement
via `tests/test_documentation.py`.

**Complete inventory** (listed + unlisted records): [full-list.md](full-list.md).

**Host policy:** factory conversion, **all Tier 1**, and **all Tier 2**
certificates **must** be measured on `df-macstudio-m2` (Mac Studio M2 Ultra,
192 GB, Ext16TR0). Do not convert or certify on `df-macbookpro-m5` or
`df-macbookpro-m3`. Historical records keep the `host_id` they were measured
on; do not rewrite those files to the new host. Recertify on
`df-macstudio-m2` for a current-host claim. The flagship M0–M8 campaign
schema remains frozen to `df-macbookpro-m5` until that contract is versioned
separately.

**Multimodal (1.8.0):** Tier 1 text quality never implies vision/audio quality. Each
certificate may carry a capability-gated `modalities` block: unsupported modalities are
`not-applicable` (disabled); supported ones are `present-not-certified`, `smoke-certified`,
or `quality-certified` only with bound evidence. Spec:
[certification-spec-v1.0 §8](../contracts/certification-spec-v1.0.md).

<!-- BEGIN:AXQUANT_CERTIFICATION_MATRIX -->
| Checkpoint | Edition | Tier 1 (quality) | Tier 2 (MTP -- Scoped) |
| --- | --- | --- | --- |

**Tier 2 (MTP -- Scoped)** is a scoped MTP *acceleration* certification: token-weighted decode speedup >= 1.20x and prompt-median >= 1.10x on the certificate's named authorizing workloads, measured on the host and AX Engine build recorded in that certificate.

No certified Tier 2 row is present, so no engine binding applies. Certified records are historical and are not re-certified for later AX Engine releases.

A Tier 2 certificate is a scoped acceleration claim only. It is **not** the AX Engine MTP ship gate (MTP-S, in-path exactness), **not** AX Engine default promotion (MTP-D), and not a claim for hosts, engines, or workloads outside its recorded binding. See [MTP gate mapping](adr033-mapping.md) for what a Tier 2 record is and is not evidence for.
<!-- END:AXQUANT_CERTIFICATION_MATRIX -->

**Gemma 4:** the table preserves revision-bound checkpoint **Tier 1** records for historical AXQ
4-bit and 6-bit fused assistant-MTP revisions (12B / 26B-A4B / 31B). The six Hub heads rebuilt on
2026-08-30 for corrected Gemma/oMLX layout compatibility have different immutable revisions and
are not covered by those records. **Tier 2 (MTP acceleration) is not certified** on any current
Gemma head.

**Qwen3-Coder-Next:** hybrid MoE coding checkpoint with **no declared MTP**. Public certificates
are non-MTP direct-decode checkpoint Tier 1 only (size, matched uniform quality, MLX-LM load).
MXFP4 is certified on `df-macstudio-m2`; 4/6-bit remain on `df-macbookpro-m5`.

**GPT-OSS:** OpenAI MoE (`GptOssForCausalLM`) with **no declared MTP**. Converted from
`mlx-community` MXFP4-Q4 via `--allow-quantized` re-pack on `df-macbookpro-m5`. Public
**20B 4-bit** (manual attention-8 / expert-4 recovery), **20B 6-bit**, and **120B 6-bit**
(manual no-4-bit / attention-8 recipe) are checkpoint Tier 1 certified. **120B 4-bit is not
certified** (agent-coding retention best ~0.952 &lt; 0.98; further recert skipped).
See the unlisted [evaluation record](gpt-oss-120b-axq4-tier1.md).

**DeepSeek V4 Flash:** the older-source experimental **2-bit** pack is checkpoint
Tier 1 on `df-macstudio-m2`. **3-bit Flash is withdrawn.** Flash-0731 ship SKUs
are 2-bit, 4-bit g128, MXFP4, and 6-bit; 0731 dual-suite cert is not closed
(2-bit recert on AX Engine 7.1.5 still 0.633 &lt; 0.90).
**6-bit (and likely MXFP4 if it lands near 179 GB) cannot be certified on
192 GB** — listed as not-certified for factory-host memory. MTP acceleration
is not certified.

**Qwen3-VL 30B-A3B Instruct:** vision MoE Instruct with **no declared MTP**. Public
**4-bit** and **6-bit** packs are checkpoint Tier 1 on `df-macbookpro-m5` with AX Engine
**6.15.0** primary (generate-manifest + doctor) and MLX-VLM vision smoke.

Machine-readable companions:

- [27B 6-bit Tier 1 JSON](qwen36-27b-axq6-tier1.json)
- [27B 6-bit Tier 2 JSON](qwen36-27b-axq6-tier2.json)
- [27B 6-bit Tier 2 evidence package](evidence/qwen36-27b-axq6-tier2/)
- [27B 4-bit Tier 1 JSON](qwen36-27b-axq4-tier1.json)
- [27B 4-bit Tier 2 JSON](qwen36-27b-axq4-tier2.json)
- [35B 4-bit Tier 1 JSON](qwen36-35b-axq4-tier1.json)
- [35B 4-bit Tier 2 JSON](qwen36-35b-axq4-tier2.json)
- [35B 6-bit Tier 1 JSON](qwen36-35b-axq6-tier1.json)
- [35B 6-bit Tier 2 JSON](qwen36-35b-axq6-tier2.json)
- [12B 4-bit Tier 1 JSON](gemma4-12b-axq4-tier1.json)
- [12B 6-bit Tier 1 JSON](gemma4-12b-axq6-tier1.json)
- [26B-A4B 4-bit Tier 1 JSON](gemma4-26b-a4b-axq4-tier1.json)
- [26B-A4B 6-bit Tier 1 JSON](gemma4-26b-a4b-axq6-tier1.json)
- [31B 4-bit Tier 1 JSON](gemma4-31b-axq4-tier1.json)
- [31B 6-bit Tier 1 JSON](gemma4-31b-axq6-tier1.json)
- [Coder-Next MXFP4 Tier 1 JSON](qwen3-coder-next-axq-mxfp4-tier1.json)
- [Coder-Next 4-bit Tier 1 JSON](qwen3-coder-next-axq4-tier1.json)
- [Coder-Next 6-bit Tier 1 JSON](qwen3-coder-next-axq6-tier1.json)
- [Qwen3-VL 30B 4-bit Tier 1 JSON](qwen3-vl-30b-axq4-tier1.json)
- [Qwen3-VL 30B 6-bit Tier 1 JSON](qwen3-vl-30b-axq6-tier1.json)
- [GPT-OSS 20B 6-bit Tier 1 JSON](gpt-oss-20b-axq6-tier1.json)
- [GPT-OSS 20B 4-bit Tier 1 JSON](gpt-oss-20b-axq4-tier1.json)
- [GPT-OSS 120B 4-bit Tier 1 JSON](gpt-oss-120b-axq4-tier1.json)
- [GPT-OSS 120B 6-bit Tier 1 JSON](gpt-oss-120b-axq6-tier1.json)
- [DeepSeek V4 Flash 2-bit Tier 1 JSON](deepseek-v4-flash-axq2-tier1.json)
- [DeepSeek V4 Flash 3-bit Tier 1 JSON](deepseek-v4-flash-axq3-tier1.json)

See [flagship certification](../guides/flagship-certification.md) for the two-tier policy and claim
boundaries (default route vs formal acceleration route; decode-heavy vs short-answer).
