# Nemotron AXQ MTP development packs

Verified publication snapshot: 2026-10-04. These are development checkpoints,
not quality or speed certificates. Immutable links below identify the exact
payload checked by the publisher.

## MLX checkpoints

| Source family | Physical format | Pack size | Immutable Hub snapshot |
| --- | --- | --- | --- |
| Lightning 30B-A3B | MXFP4 | 20.2 GB | [AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP4-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP4-MTP/tree/5ab1716c9885e9bce30c063d159f5da4e0bb8c6f) |
| Lightning 30B-A3B | MXFP8 | 35.6 GB | [AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP8-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP8-MTP/tree/8e6018d4c00ac09e150a67721d415a6af32400b1) |
| Super 120B-A12B | MXFP4 | 71.3 GB | [AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-MXFP4-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-MXFP4-MTP/tree/a34eec4ea82757a65207863c39a3233dc18c16fc) |
| Super 120B-A12B | MXFP8 | 131.0 GB | [AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-MXFP8-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-MXFP8-MTP/tree/a50a9dc169fedd1293c2752e684c70646a240960) |

All four backbones passed stock MLX-LM lazy loading and a deterministic,
eight-token generation smoke with MLX-LM 0.31.3 and MLX 0.32.1 on an Apple
M2 Ultra with 192 GB memory. This is bounded backbone execution evidence,
not a quality evaluation, performance certificate, or MTP execution test.

The checkpoint uses normal MLX-LM config, tokenizer, Safetensors shards and
weight index. The MLX physical MXFP4/MXFP8 modes are distinct from CUDA
NVFP4 and from a generic portable OCP interchange export.

```python
from mlx_lm import load, generate

model, tokenizer = load("AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP4-MTP")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2?"}],
    tokenize=False,
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=32))
```

## MTP packaging and consumer responsibility

Lightning carries 270 original MTP tensors and Super carries 1,040. The MLX
converter preserves their source tensor payloads in `mtp.safetensors`,
separate from the backbone index. `ax_nemotron_mtp_manifest.json` identifies
`nemotron_h_mtp_v1`, pins the NVIDIA source revision, and binds the sidecar
payload and tensor-name inventory by SHA256. It explicitly records
`runtime_compatibility: unverified`.

Loading the standard backbone does not activate the sidecar. MLX-LM,
AX Engine, MTPLX and oMLX each own Nemotron-H MTP discovery, interpretation,
and execution support. The pack does not reuse a Qwen or DeepSeek norm or
runtime contract. A `-MTP` name means trained MTP tensors are packaged;
it does not certify speculative decoding.

The static MLX fleet audit recognizes this neutral sidecar contract:

```bash
python scripts/audit_mtp_hub_fleet.py   --repo AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-MXFP4-MTP
```

This audit checks packaging metadata and the sidecar tensor-name header.
It does not execute MTP or establish quality. Publication separately checks
all local SHA256 inventory members, anonymous Hub file access, and large
weight-file hashes at the exact uploaded revision.

## CUDA checkpoints

| Source family | Physical format | Immutable Hub snapshot |
| --- | --- | --- |
| Lightning 30B-A3B | NVFP4A16 | [AX-Nemotron-3.5-Lightning-30B-A3B-CUDA-AXQ-NVFP4-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3.5-Lightning-30B-A3B-CUDA-AXQ-NVFP4-MTP/tree/b04ff0a889ec2278e71c5017cbd0a72ebc8fe3df) |
| Super 120B-A12B | NVFP4A16 | [AX-Nemotron-3-Super-120B-A12B-CUDA-AXQ-NVFP4-MTP](https://huggingface.co/AutomatosX/AX-Nemotron-3-Super-120B-A12B-CUDA-AXQ-NVFP4-MTP/tree/3fd44399dca208f16a047d42823c55e2b4346a5b) |

These native AXQuant RTN exports use compressed-tensors NVFP4 weight
serialization and BF16 activations. They use no AWQ or activation calibration.
Factory conversion used the CPU `numpy-reference` codec; that establishes
serialized-format evidence, not GPU execution or quality certification.

CUDA packs retain the original integrated `mtp.*` tensor layout in the main
Safetensors index. Source precision, shape and payload bytes were verified
for all 270 Lightning and 1,040 Super MTP tensors. CUDA consumers must load
and execute that original architecture; the MLX sidecar audit does not apply.

Lightning passed eight-token backbone and MTP generation smokes on RTX 5090
(vLLM 0.25.1) and Thor (vLLM 0.24.0). The test uses synchronous scheduling,
512 MiB KV/Mamba cache, eager execution, a 256-token context and one request.
The backbone uses Marlin and the preserved BF16 MTP experts use Triton.
Each MTP smoke recorded four draft tokens and four accepted draft tokens;
this tiny prompt does not establish a general acceptance rate or speedup.
NVFP4 and preservation regressions passed 94 tests on each GPU.
An earlier 64 MiB cache setting stalled request scheduling; disabling
asynchronous scheduling alone did not resolve it. These are bounded execution
checks, not quality, speed or long-context certification.

The updated CUDA revisions protect the complete integrated MTP namespace,
including virtual fused-expert projections used by consumers. Exact source
payloads remain unchanged. Super CUDA generation and MTP execution remain
unverified; its checkpoint is 85.2 GB and requires its own runtime validation.
The MLX MTP sidecars also remain unverified at execution time.

The [CUDA runtime smoke record](../reports/nemotron-cuda-runtime-smoke-2026-10-04.json)
binds each bounded run to its publication revision and manifest digest.
The probe requires positive draft-token counters when testing MTP. Run it
inside an externally timed container so a stalled worker cannot retain GPU
memory indefinitely. The original backbone runs predate the metadata fix;
the later MTP runs use the corrected Lightning revision above.

## Sources and licenses

- [Lightning BF16 source](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
  revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`, OpenMDW 1.1.
- [Super BF16 source](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16),
  revision `2dc98e2afe4face0e4ce40972a915c45368bd34a`, NVIDIA Nemotron Open Model License.

Source licenses and available notices accompany the derived checkpoints.
