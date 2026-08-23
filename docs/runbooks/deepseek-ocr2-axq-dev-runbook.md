# DeepSeek-OCR-2 — development AXQ 4/6-bit convert + Hugging Face publish

**Host:** `df-macstudio-m2` (factory convert + Ext4T)  
**Adapter:** `deepseek-ocr2-v1` (MLX-VLM `deepseekocr_2`)  
**Claims:** **development evidence only** — no certified OCR accuracy claims  

## Published packs (live)

| Pack | Hub repo | Language trunk | Hub revision |
| --- | --- | --- | --- |
| 4-bit | [`AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-4bit) | experts/attention/MLP **4-bit** | [`bf87c5d2d89bb9969f97a26c7766c30a23e11b7d`](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-4bit/tree/bf87c5d2d89bb9969f97a26c7766c30a23e11b7d) |
| 6-bit | [`AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-6bit) | experts/MLP **6-bit**, attention **8-bit** | [`c5087faa48039c4eb9ee014a24243458298cd763`](https://huggingface.co/AutomatosX/AX-DeepSeek-OCR-2-MLX-AXQ-6bit/tree/c5087faa48039c4eb9ee014a24243458298cd763) |

**Official source pin:** `deepseek-ai/DeepSeek-OCR-2` @
[`aaa02f3811945a91062062994c5c4a3f4c0af2b0`](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/tree/aaa02f3811945a91062062994c5c4a3f4c0af2b0)
(Apache-2.0).

**Convert input:** `mlx-community/DeepSeek-OCR-2-bf16` @
[`9946f9ac306378a3e6a86cad7d7f8be8e536f092`](https://huggingface.co/mlx-community/DeepSeek-OCR-2-bf16/tree/9946f9ac306378a3e6a86cad7d7f8be8e536f092)
(MLX-native layout; same weight lineage as official).

## Architecture notes

| Item | Detail |
| --- | --- |
| Model | DeepSeek-OCR-2 document VL (~3B MoE language + SAM/Qwen2 vision) |
| Convert backend | MLX-VLM (`deepseekocr_2`) |
| Vision | SAM + Qwen2 encoder + projector **BF16-protected** |
| Routers | `MoEGate` stays **BF16** (not `nn.Linear`; public quantize cannot pack it) |
| Total BPW | Dominated by vision BF16; product class refers to **language trunk** |

## Factory recipe (summary)

```bash
export HF_HOME=/path/to/huggingface-cache
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH=/path/to/axquant/src

# Download MLX BF16 with Xet
hf download mlx-community/DeepSeek-OCR-2-bf16 \
  --revision 9946f9ac306378a3e6a86cad7d7f8be8e536f092 \
  --local-dir $WORK/src-deepseek-ocr2-bf16

# 4-bit language trunk (product class 4bit)
axquant quantize --model $WORK/src-deepseek-ocr2-bf16 \
  --model-id deepseek-ai/DeepSeek-OCR-2 \
  --revision aaa02f3811945a91062062994c5c4a3f4c0af2b0 \
  --target-bpw 4.0 --ladder prior --profile general \
  --runtime-smoke none \
  --output $WORK/AX-DeepSeek-OCR-2-MLX-AXQ-4bit

# 6-bit language trunk: candidate bits must exclude 4 so experts land on 6
# (policy minimum rises ~8.3 because vision is BF16). Use staged plan API:
# architecture_prior_report(..., candidate_bits=(6,8,16))
# PlanRequest(target_bpw=8.4, candidate_bits=(6,8,16), allow_unmeasured=True)

# Smoke
python - <<'PY'
from mlx_vlm.utils import load_model
from pathlib import Path
for p in [...]:
    m = load_model(Path(p), lazy=False)
    assert sum(1 for _ in m.named_modules()) > 100
PY
```

Source prep strips torch remote-code (`modeling_*.py`, `auto_map`) and forces
`model_type=deepseekocr_2` so MLX-VLM convert does not require torch.

## Claim language

**Allowed:** development AXQ MLX packs; language trunk 4/6-bit; vision BF16-preserved.  
**Not allowed:** certified OCR accuracy, document-bench scores without measured evals, Tier 1/2 claims.

## Related

- Adapter: `deepseek-ocr2-v1` in `src/axquant/architectures/dense_family.py`
- Convert path: `_convert_deepseek_ocr2` in `src/axquant/multimodal_backend.py`
- Prep: `prepare_deepseek_ocr2_source` in `src/axquant/source_prep.py`
