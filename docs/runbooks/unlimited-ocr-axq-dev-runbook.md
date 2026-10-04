# Unlimited-OCR — development AXQ MX convert + Hugging Face publish

**Host:** factory Mac Studio (M2 Ultra, 192 GB)
**Adapter:** `unlimited-ocr-v1` (MLX-VLM `unlimited_ocr`)
**Claims:** **development evidence only** — no certified OCR accuracy claims

## Published packs (live)

| Pack | Hub repo | Language trunk | Hub revision |
| --- | --- | --- | --- |
| MXFP4 | [`AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4`](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4) | experts/attention/MLP **MXFP4** gs32, embed 8-bit | [`9776dd383d74`](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4/tree/9776dd383d74d204894b3c865f72491d396442ee) |
| MXFP8 | [`AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8`](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8) | experts/attention/MLP/embed **MXFP8** gs32 | [`da2260012416`](https://huggingface.co/AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8/tree/da2260012416f6c7a1bab56d518709c7bd43ed5c) |
| MXFP6 | ~~`AutomatosX/AX-Unlimited-OCR-3B-MoE-AXQ-MXFP6`~~ | E2M3 reference export | deleted 2026-10-03 with AXQ-051 (`29704fa1bf8b`) |

**Official source pin:** `baidu/Unlimited-OCR` @
`07dea832e22aefee32ad281d4b80551282e1c168` (MIT).

**Convert input:** `tokimoa/unlimited-ocr-mlx-bf16` @
`cd57bdd8d4efa47a2e49c2123550cbd698ff529a` (community MLX BF16 remaster).
AutomatosX byte-verified it against upstream: 592 direct + 6 conv-transpose
+ 33 packed-expert comparisons, zero differences. Reject v1 `deepseekocr`
remasters (mikoy92, vimalnakrani) — wrong lineage.

## Architecture notes

| Item | Detail |
| --- | --- |
| Model | Unlimited-OCR 3B MoE document VL (DeepSeek-V2 MoE + CLIP/SAM vision) |
| Convert backend | MLX-VLM (`unlimited_ocr`, `UnlimitedOCRProcessor`) |
| Vision | CLIP + SAM encoders + projector **BF16-protected** |
| Routers | `MoEGate` stays **BF16** (not `nn.Linear`; public quantize cannot pack it) |
| Total BPW | Dominated by vision BF16; product class refers to **language trunk** |

## Factory recipe (summary)

```bash
export HF_HOME=/path/to/huggingface-cache
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH=/path/to/axquant/src

# Download byte-verified MLX BF16 with Xet
hf download tokimoa/unlimited-ocr-mlx-bf16 \
  --revision cd57bdd8d4efa47a2e49c2123550cbd698ff529a \
  --local-dir $WORK/src-unlimited-ocr-bf16

# Inspect (adapter unlimited-ocr-v1, thin convertible, no cert track)
axquant inspect --model $WORK/src-unlimited-ocr-bf16 \
  --model-id baidu/Unlimited-OCR \
  --revision 07dea832e22aefee32ad281d4b80551282e1c168 \
  --output $WORK/work-uocr/inventory.json

# One plan-manual per lane; each plan lives in its own directory because
# axquant_source_binding.json sits beside the plan.
# MXFP4 recipe: trunk method mxfp4 (AXQ-047: no affine4 + remap), gs32,
# hardware supported_methods [affine, mxfp4, bf16].
# MXFP8 recipe: trunk + embedding affine8 gs32 (every 8-bit allocation is
# remapped, so floors must also be group32).
# MXFP4 / MXFP8 converts
axquant convert --model $WORK/src-unlimited-ocr-bf16 \
  --plan $WORK/work-uocr/mxfp4/plan.json --q-mode mxfp4 \
  --allow-unmeasured --ax-engine-manifest skip \
  --output $WORK/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4
axquant convert --model $WORK/src-unlimited-ocr-bf16 \
  --plan $WORK/work-uocr/mxfp8/plan.json --q-mode mxfp8 \
  --allow-unmeasured --ax-engine-manifest skip \
  --output $WORK/AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8

# MXFP6 lane retired (AXQ-051): no MLX or AX Engine runtime supports MXFP6
# inference, so `export-mxfp6`, the `axquant[mxfp6]` extra, and the
# `axquant.mxfp6-pack.v1` envelope were removed from the toolkit. The Hub
# pack was deleted the same day with the MXFP6 collection; no new MXFP6
# packs will be produced.

# Smoke
python - <<'PY'
from mlx_vlm.utils import load_model
for p in [...]:
    m = load_model(p, lazy=False)
    assert sum(1 for _ in m.named_modules()) > 100
PY

# Publish (needs AutomatosX write token: hf auth login --force)
bash scripts/publish_ocr_mx_20261003.sh
python scripts/sync_hf_collections.py --apply
```

Source prep strips torch remote-code (`modeling_*.py`, `auto_map`) and skips
`assets/` + `wheel/` bundles, so MLX-VLM convert does not require torch.

## Claim language

**Allowed:** development AXQ MLX packs; language trunk MXFP4/MXFP8; MXFP6
reference export (E2M3, not loadable); vision BF16-preserved.
**Not allowed:** certified OCR accuracy, document-bench scores without measured
evals, Tier 1/2 claims, MXFP6 loadability.

## Related

- Adapter: `unlimited-ocr-v1` in `src/axquant/architectures/dense_family.py`
- Convert path: `_convert_unlimited_ocr` in `src/axquant/multimodal_backend.py`
- Prep: `prepare_unlimited_ocr_source` in `src/axquant/source_prep.py`
