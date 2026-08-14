# Qwen3.8-27B AXQ vision-language retention assessment

| Field | Value |
| --- | --- |
| Status | Evidence assessment; not a vision-language quality certificate |
| Assessment date | 2026-08-14 |
| Source checkpoint | [`Qwen/Qwen3.8-27B@1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |

## Technical summary

**Verdict:** the published Qwen3.8-27B AXQ 4-bit and 6-bit packs preserve the learned vision
tower at BF16, so AXQuant does not intentionally remove or quantize those vision weights. That
fact does **not** establish retention of the source model's end-to-end vision-language (VL)
quality. The language/reasoning trunk is quantized, no image or video quality suite was run, and
the certified Hub revisions omit two processor configuration files present in the source
checkpoint.

The available evidence therefore supports this narrower statement:

> Qwen3.8-27B AXQ preserves the vision weights at BF16 and passes its published text-quality
> gates. End-to-end image and video quality has not been evaluated or certified.

It does not support either “VL ability is lost” or “VL quality is unchanged.” A severe loss is
not implied by the weight layout, but it cannot be ruled out from the current evidence. For a
VL-sensitive deployment, use the BF16 source until matched testing is complete. If AXQ is needed
for memory reasons, the 6-bit pack is the conservative trial candidate, not a measured VL winner.

## The vision weights are retained at BF16 across all four editions

Every published Qwen3.8-27B AXQ edition contains a protected `vision.safetensors` sidecar with
the same recorded payload:

| Edition | Pinned Hub revision | Vision tensors | Parameters | Dtype | Sidecar SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| AXQ 4-bit | [`a8c56f94`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit/tree/a8c56f941eafc5d5078177ea03e173d84e30b977) | 333 | 460,730,096 | BF16 | `d0d927c4…ac68f` |
| AXQ 4-bit MTP | [`32f44846`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP/tree/32f448461caf4aedcc3c16a77a63b6a94bf0667c) | 333 | 460,730,096 | BF16 | `d0d927c4…ac68f` |
| AXQ 6-bit | [`edfedb5c`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit/tree/edfedb5c1976ffd796ebcecdbff5d1aba3b50f5b) | 333 | 460,730,096 | BF16 | `d0d927c4…ac68f` |
| AXQ 6-bit MTP | [`a5a0b700`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP/tree/a5a0b700ea7c5c529c66ca3005b79425ab2f7ea6) | 333 | 460,730,096 | BF16 | `d0d927c4…ac68f` |

The sidecar manifests bind the payload to the source model and revision. The matching sidecar
hash proves that the four AXQ editions carry the same protected vision payload. It does not prove
that the complete multimodal pipeline produces the same outputs: preprocessing, language-trunk
quantization, prompt construction, multimodal position encoding, and runtime execution still
affect the final answer.

## The quantized language trunk remains an end-to-end VL risk

Qwen3.8-27B is a unified VL model. Image features produced by the BF16 vision tower are consumed
by the language trunk, which performs multimodal reasoning and generates the answer. AXQuant's
optimization scope for these packs is `text-path`, so preserving the tower removes one direct
source of quantization error but not the error that can arise after visual features enter the
quantized trunk.

The non-MTP cards describe the following storage-weighted layouts:

| Pack | Measured BPW | 4-bit share | 6-bit share | 8-bit share | BF16 share |
| --- | ---: | ---: | ---: | ---: | ---: |
| [AXQ 4-bit](hub-cards/AX-Qwen3.8-27B-MLX-AXQ-4bit.md) | 5.0667 | 89.01% | — | 9.29% | 1.69% |
| [AXQ 6-bit](hub-cards/AX-Qwen3.8-27B-MLX-AXQ-6bit.md) | 6.0001 | 79.69% | 9.32% | 4.65% | 6.34% |

This makes the 6-bit edition the more conservative **precision choice** for an unvalidated VL
trial. It is not evidence that its VL score is higher. The published text results do not rank the
packs monotonically and are too narrow to substitute for image evaluation.

## Existing Tier 1 results are text evidence, not VL evidence

The exact 4-bit and 6-bit checkpoint certificates compare each candidate with the same pinned
BF16 source on two text-oriented suites:

| Pack | Agent-coding retention | Samples | General retention | Samples | VL samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| [AXQ 4-bit](certifications/qwen38-27b-axq4-tier1.md) | 1.0072 | 76 | 1.0000 | 44 | 0 |
| [AXQ 6-bit](certifications/qwen38-27b-axq6-tier1.md) | 0.9928 | 76 | 1.0000 | 44 | 0 |

These results show that neither exact checkpoint suffered a broad collapse on the measured text
tasks. The 4-bit value above 1.0 means it scored slightly higher on that finite evaluation set; it
does not establish that quantization improved the model or that 4-bit is preferable for VL.

The MTP siblings report the same two text-suite retention results and have separate, scoped MTP
acceleration certificates. Those MTP records do not add image or video quality evidence. All four
checkpoint certificates explicitly state that vision is BF16-protected and VLM quality is not
claimed.

## Published processor metadata does not match the source package

The immutable source revision contains both
[`preprocessor_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json)
and
[`video_preprocessor_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/video_preprocessor_config.json).
Neither file appears in any of the four pinned AXQ Hub trees listed above.

This omission does not prove that image input fails. [AX Engine 6.16.1's Qwen processor
path](https://github.com/defai-digital/ax-engine/blob/v6.16.1/crates/ax-engine-server/src/openai/chat_requests.rs#L3667-L3724)
can derive some visual geometry from `config.json` and use built-in defaults when processor
metadata is absent. However, the default path is not identical to the source contract: for
example, the source image processor declares a 65,536-pixel shortest edge, while the missing-file
fallback starts at 4,096 pixels. That can change the visual token budget for small inputs. The
source's separate video processor limits are also not bound into the AXQ package.

Consequently, the present artifact proves that the vision **weights** were preserved, but it does
not prove that the source image/video **preprocessing behavior** was preserved. Copying and
checksum-binding both processor files should precede a VL-quality campaign and any refreshed VL
release claim.

## Scope and definitions

This assessment covers the four immutable AXQ revisions in the first table and the exact source
revision shown at the top of the report.

- **Weight preservation** means the protected vision tensors remain BF16 and are present in the
  packaged checkpoint.
- **Runtime availability** means a loader consumes the vision sidecar and accepts the intended
  image or video request shape. A text-only loader can ignore the sidecar even when it is present.
- **VL quality retention** means matched source-versus-candidate evaluation on image or video
  tasks with the same processor, prompt template, decoding settings, and scoring rules.
- **Checkpoint Tier 1** here covers the published size, integrity, runtime-readiness, and specified
  text-quality gates. It is not a blanket certificate for every upstream capability.

The documented `mlx_lm.generate` route covers text/backbone inference and may ignore optional
sidecars. It must not be used as evidence of VL availability or quality. A VL-capable runtime
should be checked for image capability before evaluation, followed by a real image request rather
than only a load or doctor check.

## Methodology

This is an artifact-and-evidence audit; no new model inference was performed. The assessment:

1. compared the file lists of the source and four AXQ repositories at immutable Hub revisions;
2. inspected each AXQ protected-vision manifest for tensor count, parameter count, dtype, source
   identity, and output checksum;
3. reviewed the 4-bit and 6-bit precision distributions and optimization scope;
4. reviewed the checkpoint and MTP certificates to identify the evaluated datasets and claim
   boundaries; and
5. compared the source processor configuration with the Qwen visual preprocessing fallback in
   AX Engine 6.16.1.

The evidence is descriptive. Architecture and precision provide a reason to expect the AXQ packs
to retain useful VL capability, but only a matched experiment can estimate the magnitude and
distribution of any quality change.

## Limitations and uncertainty

- No same-pin BF16-versus-AXQ image, multi-image, video, OCR, grounding, or visual-reasoning
  comparison is available.
- Text retention on 120 total samples cannot be extrapolated into a VL retention percentage.
- Preserving BF16 weights does not validate tensor loading, media preprocessing, multimodal RoPE,
  prompt expansion, or the generated answer.
- The processor-file difference has a clear mechanism for changing inputs, but its score impact
  has not been measured.
- “Prefer 6-bit” is a risk-minimizing inference from its higher-precision allocation, not a
  benchmark result. Workload-specific behavior may be non-monotonic.
- MTP exactness and speed evidence is scoped to the published Tier 2 protocol and must not be
  extended to multimodal requests without a matched VL A/B run.

## Recommended next steps

1. **Repair the package contract.** Copy and checksum-bind the source image and video processor
   configuration files into newly pinned AXQ revisions. Re-run artifact integrity and image/video
   runtime smokes against the downloaded Hub bytes.
2. **Establish a same-pin baseline.** Compare the BF16 source with both AXQ precision classes on
   the same AX Engine version, processor files, chat template, media inputs, prompts, and
   deterministic decoding settings.
3. **Cover distinct failure modes.** Include general image understanding, OCR/documents and
   charts, visual mathematics/science, spatial reasoning or grounding, multi-image input, and
   video if video ability will be advertised.
4. **Report slices, not only an aggregate.** Publish per-suite scores, paired sample counts,
   failures by category, and uncertainty. Define acceptance thresholds before examining the
   candidate results so a strong aggregate cannot hide a critical OCR or spatial regression.
5. **Keep the public claim narrow until then.** Describe the current packs as text-quality
   certified with a BF16-protected vision tower, not as VL-quality certified.

## Further questions

- After restoring the processor files, do the BF16 and AXQ runtimes produce identical visual
  token grids for the same inputs?
- Are any VL regressions concentrated in OCR, small-image detail, spatial grounding, long visual
  reasoning, or video rather than general image description?
- Does 6-bit materially improve those slices over 4-bit, and is that improvement worth the memory
  increase?
- Do MTP-on and direct-decode paths remain output-exact for multimodal prompts under the supported
  runtime profile?
