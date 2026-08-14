# Muse-Glimmer-30B — development AXQ 4/6-bit convert + Hugging Face publish

**Host:** `df-macstudio-m2` (factory convert + Ext4T)  
**Adapter:** `muse-glimmer-v1` (MLX-VLM `muse_glimmer`)  
**Claims:** **development evidence only** — no certified agentic/coding-bench scores  

## Published packs (live)

| Pack | Hub repo | Language trunk | Hub revision |
| --- | --- | --- | --- |
| 4-bit | [`AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit) | attention/MLP **4-bit** | [`bcfb0b748fc44487c1657fb6ae190592d515398b`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit/tree/bcfb0b748fc44487c1657fb6ae190592d515398b) |
| 6-bit | [`AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit) | attention/MLP **6-bit** | [`f1cfad2d2aa2fb0572786d63f7420fdb4321bed5`](https://huggingface.co/AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit/tree/f1cfad2d2aa2fb0572786d63f7420fdb4321bed5) (repo cleaned of stale shards @ `367745bd…`) |

**Source pin:** `meta-models/Muse-Glimmer-30B` @
[`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`](https://huggingface.co/meta-models/Muse-Glimmer-30B/tree/a4e59da52a7bc87ae7251dd5545c0dd437c44b68)
(Apache-2.0).

## Architecture notes

| Item | Detail |
| --- | --- |
| Layout | Dense multimodal: `model_type=muse_glimmer`, 52 text layers (6656 hidden), vision tower 50 layers |
| Convert backend | MLX-VLM (`muse_glimmer`) — requires mlx-vlm with `models.muse_glimmer` |
| Vision | `vision_tower` + `vision_adapter` + `vision_projection` **BF16-protected** |
| Tensor renames | HF `model.language_model.*` → MLX `language_model.model.*`; `model.vision_*` → bare `vision_*` |
| Total BPW | Includes large BF16 vision; product class is the **language trunk** |

## Factory recipe (summary)

```bash
export HF_HOME=/path/to/huggingface-cache
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH=/path/to/axquant/src

hf download meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --local-dir $WORK/src-muse-glimmer-30b

# 4-bit language trunk
axquant quantize --model $WORK/src-muse-glimmer-30b \
  --model-id meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --target-bpw 4.0 --ladder prior --profile general \
  --runtime-smoke none \
  --output $WORK/AX-Muse-Glimmer-30B-MLX-AXQ-4bit

# 6-bit language trunk: force attention/MLP to 6 after architecture-prior plan
# (auto grid may keep attention at 16 when vision floors raise total BPW)

# Smoke
python - <<'PY'
from mlx_vlm.utils import load_model
from pathlib import Path
for p in [...]:
    m = load_model(Path(p), lazy=False)
    assert sum(1 for _ in m.named_modules()) > 100
PY
```

## Claim language

**Allowed:** development AXQ MLX packs; language trunk 4/6-bit; vision BF16-preserved.  
**Not allowed:** certified agentic/coding scores, VLM quality claims, Tier 1/2.

## Related

- Adapter: `muse-glimmer-v1` in `src/axquant/architectures/dense_family.py`
- Convert: `_convert_muse_glimmer` in `src/axquant/multimodal_backend.py`
- Tensor aliases: `module_paths._mlx_wrapper_tensor_aliases` (Muse vision prefixes)
