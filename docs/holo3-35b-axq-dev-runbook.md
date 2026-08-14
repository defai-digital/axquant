# Holo3-35B-A3B — development AXQ 4/6-bit convert + Hugging Face publish

**Host:** `df-macstudio-m2` (factory convert + Ext4T)  
**Adapter:** `qwen35-moe-v1` (Qwen3.5-class 35B-A3B MoE / fine-tunes)  
**Claims:** **4-bit and 6-bit checkpoint Tier 1 certified** on `df-macstudio-m2`. Not the Qwen 3.6 certificate family.  
**Goal:** Build AutomatosX AXQ 4-bit and 6-bit MLX packs from Holo3 BF16, publish, and record Tier 1 evidence.

## Published packs (live)

| Pack | Hub repo | Product class | Measured total BPW | Hub revision | Cert |
| --- | --- | --- | --- | --- | --- |
| 4-bit | [`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit) | `4bit` (attn-6 / expert-4) | 5.665439 | [`7b2256130cd55ea6b7489817a9a00c46e9874403`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit/tree/7b2256130cd55ea6b7489817a9a00c46e9874403) | **Tier 1 certified** ([cert](certifications/holo3-35b-axq4-tier1.md)) |
| 6-bit | [`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit) | `6bit` | 7.006493 | [`e6cc340b04bfcec57544e462ec756e48dd248cf9`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit/tree/e6cc340b04bfcec57544e462ec756e48dd248cf9) | **Tier 1 certified** ([cert](certifications/holo3-35b-axq6-tier1.md)) |

**Source pin:** `Hcompany/Holo3-35B-A3B` @
[`208d5ae3a03f99d561f32ab5e606f73397a390ea`](https://huggingface.co/Hcompany/Holo3-35B-A3B/tree/208d5ae3a03f99d561f32ab5e606f73397a390ea)
(Apache-2.0; fine-tune of `Qwen/Qwen3.5-35B-A3B`).

**MTP product status (2026-08-14):** Holo3 **`-MTP` Hub SKUs withdrawn**. Grafted/adapted MTP did not reach viable acceleration; product is **direct-decode 4/6-bit only**. Do not publish or claim Holo3 MTP acceleration.

## Why this path works

| Item | Detail |
| --- | --- |
| Source | `Hcompany/Holo3-35B-A3B` (Apache-2.0) |
| Layout | `model_type=qwen3_5_moe`, architecture `Qwen3_5MoeForConditionalGeneration`; text 40 layers / 256 experts / 8 active / hidden 2048 (35B-A3B signature) |
| Expert packing | Official **packed** `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` — no per-expert restack prep |
| Convert path | Fused-expert MLX packing; adapter `qwen35-moe-v1` |
| Vision | `model.visual.*` present; **BF16-protected** — language path optimized; no GUI-agent / VLM quality claim |
| MTP | Config may declare `mtp_num_hidden_layers` without MTP weights — inspect clears config-only MTP; **omit `-MTP`** from Hub names |
| Cert | Secondary development family — do **not** cite Qwen 3.6 certificates |

Do **not** rename the source path to include `Qwen3.6` to force `qwen36-v1`. That mislabels the product family.

---

## 0. Preconditions

```bash
export PYTHONPATH=/path/to/axquant/src   # tree with qwen35-moe-v1
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
export HOLO_ID=Hcompany/Holo3-35B-A3B
export HOLO_REV=208d5ae3a03f99d561f32ab5e606f73397a390ea
export WORK=/path/to/axquant-work/holo3-35b-axq-dev
mkdir -p "$WORK/logs"
```

Disk order-of-magnitude: source BF16 ≈ 70 GB; AXQ 4-bit ≈ 20 GB; AXQ 6-bit ≈ 25 GB.

---

## 1. Download with Xet HP

```bash
hf download "$HOLO_ID" --revision "$HOLO_REV" \
  --local-dir "$WORK/src-holo3-35b"
# Confirm HF_XET_HIGH_PERFORMANCE=1 on the download PID (ps eww -p <pid>)
```

---

## 2. Inspect

```bash
axquant inspect \
  --model "$WORK/src-holo3-35b" \
  --model-id "$HOLO_ID" \
  --revision "$HOLO_REV" \
  --output "$WORK/inventory.json"
# Expect: adapter qwen35-moe-v1, convertible, vision_present=true, mtp_declared cleared if config-only
```

---

## 3. Convert 4-bit and 6-bit

```bash
axquant quantize \
  --model "$WORK/src-holo3-35b" \
  --model-id "$HOLO_ID" \
  --revision "$HOLO_REV" \
  --target-bpw 4.0 --ladder prior --profile agent-coding \
  --runtime-smoke none \
  --output "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-4bit"

axquant quantize \
  --model "$WORK/src-holo3-35b" \
  --model-id "$HOLO_ID" \
  --revision "$HOLO_REV" \
  --target-bpw 6.0 --ladder prior --profile agent-coding \
  --runtime-smoke none \
  --output "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-6bit"
```

---

## 4. Runtime smoke

```bash
axquant runtime-check --model "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-4bit" --runtime mlx-lm
axquant runtime-check --model "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-6bit" --runtime mlx-lm
```

---

## 5. Hub publish (AutomatosX)

```bash
hf repo create AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit --type model --exist-ok
hf upload AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit \
  "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-4bit" . --commit-message "AXQ 4-bit development pack from Holo3-35B-A3B@$HOLO_REV"

hf repo create AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit --type model --exist-ok
hf upload AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit \
  "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-6bit" . --commit-message "AXQ 6-bit development pack from Holo3-35B-A3B@$HOLO_REV"
```

## AX Engine runtime

Holo3 AXQ packs use the **same native `qwen3_5` MoE graph** as Qwen 3.6 35B-A3B.
AX Engine 6.15.0 loads them after `generate-manifest` (Hub packs include
`model-manifest.json`).

```bash
# Local artifact serve (mlx-preview)
ax-engine-bench generate-manifest /path/to/AX-Holo3-35B-A3B-MLX-AXQ-4bit --validate
ax-engine serve /path/to/AX-Holo3-35B-A3B-MLX-AXQ-4bit --mlx --port 31418

# Product id resolves to holo3-35b (not qwen3.6) from path name
# Preset (after ax-engine Holo3 support patch): --preset holo3-35b
# Download aliases: ax-holo3-35b / ax-holo3-35b-4bit / ax-holo3-35b-6bit
```

**Verified on `df-macstudio-m2`:** load + vision sidecar + `/v1/chat/completions`
and `/v1/completions` smoke for both 4-bit and 6-bit packs.

Text-path only: GUI / vision quality is **not** claimed.

## Checkpoint Tier 1 (2026-08-14, `df-macstudio-m2`)

| Gate | 4-bit (attn-6 / expert-4 recovery) | 6-bit |
| --- | ---: | ---: |
| Size ratio vs uniform | 1.1469 ≤ 1.15 **pass** | 1.0135 ≤ 1.15 **pass** |
| General retention | 1.0488 **pass** | 1.000 **pass** |
| Agent-coding retention | 1.0069 **pass** | 1.007 **pass** |
| Verdict | **certified** | **certified** |

Uniform references: `mlx_lm convert -q --q-bits {4,6} --q-group-size 64` from the same BF16 pin.
Suites: development-agent-coding (76) + development-general (44), seed `20260728`, max gen 64.

First architecture_prior 4-bit pack failed agent-coding (0.9793; long_context 0.875).
Certified 4-bit uses [`examples/holo3-35b-axq4-agent-v0.1.yaml`](../examples/holo3-35b-axq4-agent-v0.1.yaml)
(attention 6-bit / experts 4-bit). Attention-8 cleared quality but size 1.162 &gt; 1.15.

## MTP product withdrawn (2026-08-14)

Holo3 **`-MTP` Hub SKUs and public cert rows are removed**. Grafted/adapted MTP
never reached viable speculative acceleration (accept stayed low; speedup ≪ 1.2×).
Product surface is **direct-decode 4/6-bit only**.

Engineering notes (non-product) remain under
`docs/certifications/evidence/holo3-35b-axq6-mtp-*` for internal history only —
do **not** republish as a customer SKU without a new Tier 2 pass.

## Claim language

**Allowed:** checkpoint Tier 1 for 4/6-bit; measured size/quality vs matched uniform; AX Engine/MLX-LM text smoke; vision BF16-preserved; source pin.  
**Not allowed:** GUI/VLM quality claims; Holo3 MTP acceleration/speedup; equating Holo3 to official Qwen 3.6 35B certificates; claiming live `-MTP` Hub packs.

## Related

- Adapter: `qwen35-moe-v1` in `src/axquant/architectures/qwen36.py`
- Graft tooling: `src/axquant/grafted_mtp.py` (`prepare-grafted-mtp`, `compose-grafted-mtp`)
- Same family path as Ornith: [docs/ornith-35b-axq-dev-runbook.md](ornith-35b-axq-dev-runbook.md)
- AX Engine: preset `holo3-35b`, model-id inference, download aliases in `ax-engine`
