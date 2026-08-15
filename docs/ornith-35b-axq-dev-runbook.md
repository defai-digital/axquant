# Ornith-1.0-35B — development AXQ 4/6-bit convert + Hugging Face publish

**Host:** `df-macstudio-m2` (factory convert + Ext4T)  
**Adapter:** `qwen35-moe-v1` (Qwen3.5-class 35B-A3B MoE / fine-tunes)  
**Claims:** **4-bit and 6-bit checkpoint Tier 1 certified** on `df-macstudio-m2`. Not the Qwen 3.6 certificate family.  
**Goal:** Build AutomatosX AXQ 4-bit and 6-bit MLX packs from Ornith BF16, publish, and record Tier 1 evidence.

## Published packs (live)

| Pack | Hub repo | Product class | Measured main BPW | Hub revision | Cert |
| --- | --- | --- | --- | --- | --- |
| 4-bit | [`AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit) | `4bit` | 4.880062 | [`d7416c665cd8ae6e5fbebc3f17bd547b78cf11fc`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit/tree/d7416c665cd8ae6e5fbebc3f17bd547b78cf11fc) | **Tier 1 certified** ([cert](certifications/ornith-35b-axq4-tier1.md)) |
| 6-bit | [`AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit) | `6bit` | 6.000062 | [`37361076641d7b7487d1b5ce1b68243ffbdbffe0`](https://huggingface.co/AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit/tree/37361076641d7b7487d1b5ce1b68243ffbdbffe0) | **Tier 1 certified** ([cert](certifications/ornith-35b-axq6-tier1.md)) |

**Source pin:** `deepreinforce-ai/Ornith-1.0-35B` @
[`5df2ed3f675c7beaa490328cc70bb573b65fb660`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/tree/5df2ed3f675c7beaa490328cc70bb573b65fb660)
(MIT; twin of `ornith-ai/Ornith-1.0-35B`).

## Why this path works

| Item | Detail |
| --- | --- |
| Source | `deepreinforce-ai/Ornith-1.0-35B` (MIT) |
| Layout | `model_type=qwen3_5_moe`, 40 layers / 256 experts / 8 active / hidden 2048 (35B-A3B signature) |
| Expert packing | Ornith ships **per-expert** `experts.<i>.{gate,up,down}_proj`; official Qwen packs `experts.gate_up_proj` / `experts.down_proj`. Convert-time prep (`prepare_qwen_moe_packed_experts_source`) restacks before MLX-LM convert |
| Convert path | Fused-expert MLX packing after prep; adapter `qwen35-moe-v1` |
| Vision | Present; **BF16-protected** (`vision.safetensors`) — text path only; no VLM quality claim |
| MTP | Config may declare `mtp_num_hidden_layers` without MTP weights — inspect clears config-only MTP; **omit `-MTP`** from Hub names |
| Cert | Secondary family; **checkpoint Tier 1** on `df-macstudio-m2` — do **not** cite Qwen 3.6 certificates |

Do **not** rename the source path to include `Qwen3.6` to force `qwen36-v1`. That mislabels the product family.

---

## 0. Preconditions

```bash
export PYTHONPATH=/path/to/axquant/src   # tree with qwen35-moe-v1 + expert pack prep
# Ext4T + Xet high performance (factory policy)
export HF_HOME=/path/to/huggingface-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=$HF_HOME/xet
unset HF_HUB_ENABLE_HF_TRANSFER

hf auth whoami
```

Pin source revision:

```bash
export ORNITH_ID=deepreinforce-ai/Ornith-1.0-35B
export ORNITH_REV=5df2ed3f675c7beaa490328cc70bb573b65fb660
export WORK=/path/to/axquant-work/ornith-35b-axq-dev
mkdir -p "$WORK/logs"
```

Disk order-of-magnitude: source BF16 ≈ 70 GB; packed staging ≈ 70 GB; AXQ 4-bit ≈ 20 GB; AXQ 6-bit ≈ 25 GB.

---

## 1. Download with Xet HP

```bash
hf download "$ORNITH_ID" --revision "$ORNITH_REV" \
  --local-dir "$WORK/src-ornith-35b"
# Confirm HF_XET_HIGH_PERFORMANCE=1 on the download PID (ps eww -p <pid>)
```

---

## 2. Pack per-expert tensors → MLX contract

```bash
python - <<'PY'
from pathlib import Path
from axquant.source_prep import (
    needs_qwen_moe_unpacked_expert_prep,
    prepare_qwen_moe_packed_experts_source,
)
import shutil
src = Path("$WORK/src-ornith-35b")
assert needs_qwen_moe_unpacked_expert_prep(src)
out = prepare_qwen_moe_packed_experts_source(src, work_dir=Path("$WORK/prep-work"))
packed = Path("$WORK/src-ornith-35b-packed")
if packed.exists():
    shutil.rmtree(packed)
shutil.move(str(out), str(packed))
print("packed", packed)
PY
```

---

## 3. Inspect + convert (architecture-prior)

```bash
axquant inspect --model "$WORK/src-ornith-35b-packed" \
  --model-id "$ORNITH_ID" --revision "$ORNITH_REV" \
  --output "$WORK/inventory-packed.json"
# Expect: adapter qwen35-moe-v1, convertible, mtp_declared false, vision true

axquant quantize --model "$WORK/src-ornith-35b-packed" \
  --model-id "$ORNITH_ID" --revision "$ORNITH_REV" \
  --target-bpw 4.0 --ladder prior --profile agent-coding \
  --runtime-smoke none \
  --output "$WORK/AX-Ornith-1.0-35B-MLX-AXQ-4bit"

axquant quantize --model "$WORK/src-ornith-35b-packed" \
  --model-id "$ORNITH_ID" --revision "$ORNITH_REV" \
  --target-bpw 6.0 --ladder prior --profile agent-coding \
  --runtime-smoke none \
  --output "$WORK/AX-Ornith-1.0-35B-MLX-AXQ-6bit"
```

---

## 4. Runtime smoke (MLX-LM Python API)

`axquant runtime-check --runtime mlx-lm` needs a `mlx_lm.generate` console script on `PATH`.
Prefer the library API on factory hosts:

```bash
python - <<'PY'
from mlx_lm import load, generate
for path, tag in [
    ("$WORK/AX-Ornith-1.0-35B-MLX-AXQ-4bit", "AXQ4"),
    ("$WORK/AX-Ornith-1.0-35B-MLX-AXQ-6bit", "AXQ6"),
]:
    model, tokenizer = load(path)
    prompt = f"Say exactly: {tag} OK"
    text = generate(model, tokenizer, prompt=prompt, max_tokens=48)
    print(tag, "len", len(text), text[:200])
    assert text.strip()
PY
```

Do not publish if load or generation fails.

---

## 5. Model card + Hub upload

Cards must state: source id@revision, MIT attribution, checkpoint Tier 1 scope,
vision BF16-preserved, no Qwen 3.6 certificate family claim, no MTP claim.

```bash
hf repos create AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit --type model || true
hf repos create AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit --type model || true
hf upload AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit \
  "$WORK/AX-Ornith-1.0-35B-MLX-AXQ-4bit" . \
  --commit-message "AXQ 4-bit development pack from Ornith-1.0-35B@${ORNITH_REV}"
hf upload AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit \
  "$WORK/AX-Ornith-1.0-35B-MLX-AXQ-6bit" . \
  --commit-message "AXQ 6-bit development pack from Ornith-1.0-35B@${ORNITH_REV}"
```

---

## 6. Claim language

**Allowed**

- “AXQuant checkpoint Tier 1 certified AXQ 4-bit / 6-bit MLX pack of Ornith-1.0-35B”
- “Converted with architecture-prior plan; vision preserved at BF16”
- “Quality vs matched uniform MLX convert on `df-macstudio-m2`”

**Not allowed**

- “Qwen 3.6 certified parity”
- MTP acceleration claims
- Vision / VLM quality claims

---

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| `inspect-only` / no adapter | Tree lacks `qwen35-moe-v1`; do not spoof `Qwen3.6` in the path |
| `parameters not in model` / experts.* | Source is unpacked; run expert pack prep first |
| `declares MTP but … no MTP tensor allocations` | Config-only MTP; inspect should clear `mtp_declared` when weights are missing |
| `executable not found: mlx_lm.generate` | Use the Python `mlx_lm.load` / `generate` smoke above |
| OOM / disk full | Convert only on Ext4T; archive other packs to NAS first |

---

## Related

- Support policy: `qwen35-moe-v1` in `src/axquant/support_policy.py`
- Expert pack prep: `prepare_qwen_moe_packed_experts_source` in `src/axquant/source_prep.py`
- Primary MoE cert track (different product): Qwen 3.6 35B-A3B packs under `qwen36-v1`
- Factory Ext4T / Xet: `AGENTS.md` and `scripts/setup-ext4t-hf.sh`
