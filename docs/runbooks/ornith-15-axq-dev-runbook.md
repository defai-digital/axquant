# Ornith-1.5 — development AXQ MXFP4 / 6-bit convert + Hugging Face publish

**Host:** `df-macstudio-m2` (factory convert)  
**Adapters:** `qwen35-dense-v1` (9B) and `qwen35-moe-v1` (35B-A3B, 397B)  
**Claims:** development evidence until a later checkpoint Tier 1 campaign. Not the Qwen 3.6 certificate family.  
**Goal:** Build AutomatosX AXQ MXFP4 and 6-bit MLX packs from Ornith-1.5 BF16 and publish.

Official MLX BF16 sources live in
[ornith-ai/ornith-15-mlx](https://huggingface.co/collections/ornith-ai/ornith-15-mlx).
AXQuant converts from the original Hugging Face BF16 checkpoints, not from those MLX BF16 trees.

## Packs

| Size | Adapter | Source | MXFP4 Hub | 6-bit Hub |
| --- | --- | --- | --- | --- |
| 9B dense | `qwen35-dense-v1` | `ornith-ai/Ornith-1.5-9B` | `AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP` | `AutomatosX/AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP` |
| 35B-A3B MoE | `qwen35-moe-v1` | `ornith-ai/Ornith-1.5-35B-A3B` | `AutomatosX/AX-Ornith-1.5-35B-A3B-MLX-AXQ-MXFP4-MTP` | `AutomatosX/AX-Ornith-1.5-35B-A3B-MLX-AXQ-6bit-MTP` |
| 397B MoE | `qwen35-moe-v1` | `ornith-ai/Ornith-1.5-397B` | `AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-MXFP4-MTP` | `AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP` |

Hub names include `-MTP` because convert extracts a protected MTP sidecar. Do not claim MTP acceleration until a Tier 2 A/B record exists.

## Why this path works

| Item | Detail |
| --- | --- |
| 9B | `model_type=qwen3_5`, 32 layers / hidden 4096 / intermediate 12288. Dense hybrid GDN+attention. |
| 35B-A3B | `model_type=qwen3_5_moe`, 40 layers / 256 experts / 8 active / hidden 2048. Same signature as Ornith-1.0-35B. |
| 397B | `model_type=qwen3_5_moe`, 60 layers / 512 experts / 10 active / hidden 4096 / moe intermediate 1024. |
| Expert packing | Language experts ship packed (`mlp.experts.gate_up_proj` / `down_proj`). MTP experts may be per-expert and stay in the sidecar. |
| Vision | Present; **BF16-protected**. Text path only; no VLM quality claim. |
| License | MIT |

Do **not** rename a source path to include `Qwen3.6` to force `qwen36-v1`.

---

## Factory one-shot

```bash
export PYTHONPATH=/path/to/axquant/src
export HF_HOME=/path/to/huggingface-cache
export HF_XET_HIGH_PERFORMANCE=1
export ORNITH15_WORK=/path/to/axquant-work/ornith-15
export ORNITH15_MODELS=/path/to/models

# 9B then 35B. 397B is ~807 GB BF16 — convert one pack at a time.
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 9b --pack mxfp4 all
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 9b --pack axq6 all
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 35b --pack mxfp4 all
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 35b --pack axq6 all
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 397b --pack mxfp4 all
PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 397b --pack axq6 all
```

Disk order-of-magnitude: 9B BF16 ≈ 19 GB; 35B ≈ 72 GB; 397B ≈ 807 GB. MXFP4 is roughly 4.3–5.6 BPW plus BF16 vision; 6-bit is higher.

397B convert on the 192 GB factory Studio (`df-macstudio-m2`) cannot use stock
`mlx_lm.convert`: that path holds the full quantized model (~216 GB MXFP4) and
SIGKILLs while writing shard 2. The factory script sets
`AXQUANT_STREAMING_CONVERT=1` so convert quantizes one leaf module, writes it
into a 5 GiB shard, and donates the weights. Do not retry stock convert on this
host. Unset `AXQUANT_STREAMING_CONVERT=0` only on a ≥512 GB Mac that can hold
the quantized model resident.

Skip generate smoke with `--skip-smoke` even after a successful convert: 397B
MXFP4 cannot load resident on 192 GB. Publish still copies the pack.

---

## Manual inspect + convert

```bash
axquant inspect --model "$SRC" --model-id "$ID" --revision "$REV" --output inventory.json
# 9B: adapter qwen35-dense-v1, convertible, vision true
# 35B/397B: adapter qwen35-moe-v1, convertible, vision true

axquant plan-manual --inventory inventory.json \
  --recipe examples/ornith-15-9b-axq-mxfp4-v0.1.yaml \
  --output plan.json

axquant convert --model "$SRC" --plan plan.json --output "$OUT" \
  --q-mode mxfp4 --allow-unmeasured --ax-engine-manifest skip
```

6-bit uses `--q-mode affine` and the matching `*-axq6-v0.1.yaml` recipe.

---

## Claim language

**Allowed**

- “AXQuant development AXQ MXFP4 / 6-bit MLX pack of Ornith-1.5-{9B,35B-A3B,397B}”
- “Converted with a manual recipe; vision preserved at BF16”
- “MTP weights extracted as a protected sidecar; no MTP acceleration claim”

**Not allowed**

- “Qwen 3.6 certified parity”
- MTP acceleration claims
- Vision / VLM quality claims
- Checkpoint Tier 1 until a bound quality campaign exists

## Related

- Support policy: `qwen35-dense-v1` and `qwen35-moe-v1` in `src/axquant/support_policy.py`
- Ornith-1.0 35B (certified 4/6-bit): [ornith-35b-axq-dev-runbook.md](ornith-35b-axq-dev-runbook.md)
