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
192 GB, Ext12T). Do not convert or certify on `df-macbookpro-m5` or
`df-macbookpro-m3`. Historical records keep the `host_id` they were measured
on; do not rewrite those files to the new host. Recertify on
`df-macstudio-m2` for a current-host claim. The flagship M0–M8 campaign
schema remains frozen to `df-macbookpro-m5` until that contract is versioned
separately.

**Multimodal (1.8.0):** Tier 1 text quality never implies vision/audio quality. Each
certificate may carry a capability-gated `modalities` block: unsupported modalities are
`not-applicable` (disabled); supported ones are `present-not-certified`, `smoke-certified`,
or `quality-certified` only with bound evidence. Spec:
[certification-spec-v1.0 §8](../certification-spec-v1.0.md).

<!-- BEGIN:AXQUANT_CERTIFICATION_MATRIX -->
| Checkpoint | Edition | Tier 1 (checkpoint) | Tier 2 (MTP acceleration) |
| --- | --- | --- | --- |
| [Qwen3.8-27B MLX AXQ 4-bit MTP](qwen38-27b-axq4-mtp-tier1.md) | main@`32f44846` | [Certified](qwen38-27b-axq4-mtp-tier1.md) | [Certified](qwen38-27b-axq4-mtp-tier2.md) |
| [Qwen3.8-27B MLX AXQ 6-bit MTP](qwen38-27b-axq6-mtp-tier1.md) | main@`a5a0b700` | [Certified](qwen38-27b-axq6-mtp-tier1.md) | [Certified](qwen38-27b-axq6-mtp-tier2.md) |
| [Qwen 3.6 27B MLX AXQ 4-bit MTP](qwen36-27b-axq4-tier1.md) | main@`f44a9eee` | [Certified](qwen36-27b-axq4-tier1.md) | [Certified](qwen36-27b-axq4-tier2.md) |
| [Qwen 3.6 27B MLX AXQ 6-bit MTP](qwen36-27b-axq6-tier1.md) | v3 | [Certified](qwen36-27b-axq6-tier1.md) | [Certified](qwen36-27b-axq6-tier2.md) |
| [Qwen 3.6 35B-A3B MLX AXQ 4-bit MTP](qwen36-35b-axq4-tier1.md) | main@`a549387d` | [Certified](qwen36-35b-axq4-tier1.md) | [Certified](qwen36-35b-axq4-tier2.md) |
| [Qwen 3.6 35B-A3B MLX AXQ 6-bit MTP](qwen36-35b-axq6-tier1.md) | main@`7b9ff47a` | [Certified](qwen36-35b-axq6-tier1.md) | [Certified](qwen36-35b-axq6-tier2.md) |
| [Qwen3-VL 30B-A3B Instruct MLX AXQ 4-bit](qwen3-vl-30b-axq4-tier1.md) | main@`ffcad97e` | [Certified](qwen3-vl-30b-axq4-tier1.md) | N/A (no MTP) |
| [Qwen3-VL 30B-A3B Instruct MLX AXQ 6-bit](qwen3-vl-30b-axq6-tier1.md) | main@`71f90ad5` | [Certified](qwen3-vl-30b-axq6-tier1.md) | N/A (no MTP) |
| [Holo3-35B-A3B MLX AXQ 4-bit](holo3-35b-axq4-tier1.md) | main@`7b225613` | [Certified](holo3-35b-axq4-tier1.md) | N/A (no MTP) |
| [Holo3-35B-A3B MLX AXQ 6-bit](holo3-35b-axq6-tier1.md) | main@`e6cc340b` | [Certified](holo3-35b-axq6-tier1.md) | N/A (no MTP) |
| [Ornith-1.0-35B MLX AXQ 4-bit](ornith-35b-axq4-tier1.md) | main@`d7416c66` | [Certified](ornith-35b-axq4-tier1.md) | N/A (no MTP) |
| [Ornith-1.0-35B MLX AXQ 6-bit](ornith-35b-axq6-tier1.md) | main@`37361076` | [Certified](ornith-35b-axq6-tier1.md) | N/A (no MTP) |
| [Holo-3.1-35B-A3B MLX AXQ MXFP4](holo31-35b-axq-mxfp4-tier1.md) | main@23aa374f | [Certified](holo31-35b-axq-mxfp4-tier1.md) | N/A (no MTP) |
| [GPT-OSS 20B MLX AXQ 4-bit](gpt-oss-20b-axq4-tier1.md) | main@`0c1806bf` | [Certified](gpt-oss-20b-axq4-tier1.md) | N/A (no MTP) |
| [GPT-OSS 20B MLX AXQ 6-bit](gpt-oss-20b-axq6-tier1.md) | main@`a04eea37` | [Certified](gpt-oss-20b-axq6-tier1.md) | N/A (no MTP) |
| [GPT-OSS 120B MLX AXQ 6-bit](gpt-oss-120b-axq6-tier1.md) | main@`50537a80` | [Certified](gpt-oss-120b-axq6-tier1.md) | N/A (no MTP) |
| [Qwen3.8-27B MLX AXQ MXFP4 MTP](qwen38-27b-axq-mxfp4-mtp-tier1.md) | main@`594de650` | [Certified](qwen38-27b-axq-mxfp4-mtp-tier1.md) | [Not Certified](qwen38-27b-axq-mxfp4-mtp-tier1.md#tier-2-status) |
| [Qwen3.8-27B MLX AXQ 8-bit MTP](qwen38-27b-axq8-mtp-tier1.md) | main@`7772fd6e` | [Certified](qwen38-27b-axq8-mtp-tier1.md) | [Not Certified](qwen38-27b-axq8-mtp-tier1.md#tier-2-status) |
| [DeepSeek V4 Flash MLX AXQ 2-bit MTP (exp.)](deepseek-v4-flash-axq2-tier1.md) | main@`e22b117a` | [Certified](deepseek-v4-flash-axq2-tier1.md) | [Not Certified](deepseek-v4-flash-axq2-tier1.md#tier-2-status) |
| [Gemma 4 12B MLX AXQ 4-bit](gemma4-12b-axq4-tier1.md) | main@`6d124af8` (IT rebuild) | [Certified](gemma4-12b-axq4-tier1.md) | [Not Certified](gemma4-12b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 12B MLX AXQ 6-bit](gemma4-12b-axq6-tier1.md) | main@`d0a1a932` (IT rebuild) | [Certified](gemma4-12b-axq6-tier1.md) | [Not Certified](gemma4-12b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 26B-A4B MLX AXQ 4-bit](gemma4-26b-a4b-axq4-tier1.md) | main@`85b0a78a` | [Certified](gemma4-26b-a4b-axq4-tier1.md) | [Not Certified](gemma4-26b-a4b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 26B-A4B MLX AXQ 6-bit](gemma4-26b-a4b-axq6-tier1.md) | main@`4a62bf66` | [Certified](gemma4-26b-a4b-axq6-tier1.md) | [Not Certified](gemma4-26b-a4b-axq6-tier1.md#tier-2-status) |
| [Gemma 4 31B MLX AXQ 4-bit](gemma4-31b-axq4-tier1.md) | main@`bc2de70b` | [Certified](gemma4-31b-axq4-tier1.md) | [Not Certified](gemma4-31b-axq4-tier1.md#tier-2-status) |
| [Gemma 4 31B MLX AXQ 6-bit](gemma4-31b-axq6-tier1.md) | main@`f024707a` | [Certified](gemma4-31b-axq6-tier1.md) | [Not Certified](gemma4-31b-axq6-tier1.md#tier-2-status) |
| [DeepSeek V4 Flash-0731 MLX AXQ 2-bit MTP (exp.)](deepseek-v4-flash-0731-axq2-tier1.md) | studio-eval@`408c0ab3` | [Not Certified](deepseek-v4-flash-0731-axq2-tier1.md) | [Not Certified](deepseek-v4-flash-0731-axq2-tier1.md#tier-2-status) |
| [DeepSeek V4 Flash-0731 MLX AXQ 4-bit MTP](deepseek-v4-flash-0731-axq4-tier1.md) | studio-local-g128 | [Not Certified](deepseek-v4-flash-0731-axq4-tier1.md) | [Not Certified](deepseek-v4-flash-0731-axq4-tier1.md#tier-2-status) |
| [DeepSeek V4 Flash-0731 MLX AXQ MXFP4](deepseek-v4-flash-0731-axq-mxfp4-tier1.md) | recipe-only | [Not Certified](deepseek-v4-flash-0731-axq-mxfp4-tier1.md) | [Not Certified](deepseek-v4-flash-0731-axq-mxfp4-tier1.md#tier-2-status) |
| [DeepSeek V4 Flash-0731 MLX AXQ 6-bit](deepseek-v4-flash-0731-axq6-tier1.md) | memory-blocked-192gb | [Not Certified](deepseek-v4-flash-0731-axq6-tier1.md) | [Not Certified](deepseek-v4-flash-0731-axq6-tier1.md#tier-2-status) |
<!-- END:AXQUANT_CERTIFICATION_MATRIX -->

**Gemma 4:** checkpoint **Tier 1** is certified for the AXQ 4-bit and 6-bit fused assistant-MTP
Hub packs (12B / 26B-A4B / 31B). **Tier 2 (MTP acceleration) is not certified** on any Gemma pack
while formal assistant-MTP exactness remains open on a released engine.

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
are 2-bit, 4-bit g128, MXFP4, and 6-bit; 0731 dual-suite cert is not closed.
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

See [flagship certification](../flagship-certification.md) for the two-tier policy and claim
boundaries (default route vs formal acceleration route; decode-heavy vs short-answer).
