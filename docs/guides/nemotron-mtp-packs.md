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

## Sources and licenses

- [Lightning BF16 source](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
  revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`, OpenMDW 1.1.
- [Super BF16 source](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16),
  revision `2dc98e2afe4face0e4ce40972a915c45368bd34a`, NVIDIA Nemotron Open Model License.

Source licenses and available notices accompany the derived checkpoints.
