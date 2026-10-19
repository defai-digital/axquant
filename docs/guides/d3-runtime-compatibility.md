# D3 runtime compatibility

Checked 2026-10-18 against `vllm-sr/d3` revision
`cf15549a78fc52598ddd1951f321c58372bbd64f` and local AXQ development artifacts.
This page covers end-to-end D3 decision inference, not just loading its Qwen
backbone.

D3 answers choice, yes/no, and score questions through `system_one(state=...,
questions=..., images=..., videos=...)`. It returns selected answers and
probabilities using a separate 255-class readout; it is not a free-form text
generation model. The source checkpoint declares `noncausal_full_attention`.
See the [upstream D3 model card](https://huggingface.co/vllm-sr/d3); the tested
source revision is `cf15549a78fc52598ddd1951f321c58372bbd64f`.

## Engine status

| Inference path | Upstream D3 support | Exact AXQ artifact result |
| --- | --- | --- |
| Transformers `AutoModel` with `trust_remote_code=True` | Upstream-documented `system_one` API | Not tested with the converted AXQ artifacts; upstream support does not establish their compatibility |
| vLLM System One endpoint | Upstream-documented for the source model; upstream says it requires current main/nightly builds | Not tested with the converted AXQ artifacts |
| MLX-VLM 0.7.0rc0 | No D3 `system_one` adapter | MXFP4 and MXFP8 each loaded and emitted two generic text tokens. Neither used the D3 readout or returned a D3 decision; the short `READY` prompt smoke did not follow the prompt. NVFP4 failed to load because its CUDA tensor names do not match the MLX model |
| AX Engine 7.6.4 | No D3 `system_one` path | MXFP4 and MXFP8 were rejected before inference because the generated model manifest has no `lm_head` tensor role. NVFP4 was rejected because it has no `token_embedding` role in the layout AX Engine expects |
| oMLX / MTPLX | No D3 support claim | The manifest-bound AXQuant compatibility report marks both unsupported: there is no audited `qwen3_5` export profile |

The MLX smoke only establishes that MLX-VLM can execute a short generic
backbone-generation path on the two MLX-format files. It does not establish
D3 decision, image, or video support. AX Engine did not load any of the three
artifacts. The NVFP4 artifact is a CUDA export and was not usable through the
tested Apple MLX paths.

## Publication status

The MXFP4, MXFP8, and NVFP4 D3 artifacts remain development-only and were not
uploaded. The requested publication gate requires D3 decision inference on
both MLX and AX Engine; neither engine passed that end-to-end gate. The output
of a generic backbone smoke is not D3 runtime or quality evidence.

A future support claim requires a D3-specific runtime path that consumes the
separate readout and preserves D3's attention and modality behavior, followed
by decision-result checks on the exact, hash-bound artifacts. Upstream
Transformers or vLLM support for the source model does not transfer to an
AXQ-converted checkpoint.
