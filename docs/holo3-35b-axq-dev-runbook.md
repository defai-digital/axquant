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
| 4-bit-MTP | [`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP) | `4bit` + grafted MTP | 5.665439 (main) | [`c048f577843225ac0545be5674b4d68b9a51dcf0`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP/tree/c048f577843225ac0545be5674b4d68b9a51dcf0) | **Tier 1 certified** ([cert](certifications/holo3-35b-axq4-mtp-tier1.md)); Tier 2 **not certified** |
| 6-bit-MTP | [`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP) | `6bit` + grafted MTP | 7.006493 (main) | [`f474549461817cafb73909847af43af2431d4a0d`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP/tree/f474549461817cafb73909847af43af2431d4a0d) | **Tier 1 certified** ([cert](certifications/holo3-35b-axq6-mtp-tier1.md)); Tier 2 **not certified** |

**Source pin:** `Hcompany/Holo3-35B-A3B` @
[`208d5ae3a03f99d561f32ab5e606f73397a390ea`](https://huggingface.co/Hcompany/Holo3-35B-A3B/tree/208d5ae3a03f99d561f32ab5e606f73397a390ea)
(Apache-2.0; fine-tune of `Qwen/Qwen3.5-35B-A3B`).

**MTP donor pin (grafted packs only):** `Qwen/Qwen3.5-35B-A3B` @
[`59d61f3ce65a6d9863b86d2e96597125219dc754`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754)
— not co-trained on Holo3.

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
and `/v1/completions` smoke for both 4-bit and 6-bit packs; MLX load smoke for
both `-MTP` packs with `mtp.safetensors` present.

Text-path only: GUI / vision quality is **not** claimed.

## Grafted MTP packs (`-MTP`)

Holo3 BF16 declares `mtp_num_hidden_layers` but ships **no** `mtp.*` weights.
`-MTP` SKUs attach a **grafted** BF16 head from the parent Qwen3.5 checkpoint:

```bash
export WORK_MTP=/path/to/axquant-work/holo3-35b-mtp-axq
export DONOR_ID=Qwen/Qwen3.5-35B-A3B
export DONOR_REV=59d61f3ce65a6d9863b86d2e96597125219dc754
export HOLO_ID=Hcompany/Holo3-35B-A3B
export HOLO_REV=208d5ae3a03f99d561f32ab5e606f73397a390ea

# 1) Pack donor MTP (785 unpacked experts → 19 packed tensors)
axquant prepare-grafted-mtp \
  --donor "$DONOR_DIR" \
  --donor-model-id "$DONOR_ID" --donor-revision "$DONOR_REV" \
  --trunk-model-id "$HOLO_ID" --trunk-revision "$HOLO_REV" \
  --output "$WORK_MTP/mtp-graft"

# 2) Attach onto certified non-MTP trunk without mutating main digests
axquant compose-grafted-mtp \
  --model-dir "$WORK/AX-Holo3-35B-A3B-MLX-AXQ-6bit" \
  --mtp-dir "$WORK_MTP/mtp-graft" \
  --output "$WORK_MTP/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP"
```

Pack includes `mtp.safetensors`, `axquant_mtp_sidecar_manifest.json`, and
`axquant_mtp_graft.json` (honest donor/trunk provenance).

## Checkpoint Tier 1 (2026-08-14, `df-macstudio-m2`)

| Gate | 4-bit (attn-6 / expert-4 recovery) | 6-bit | 4-bit-MTP | 6-bit-MTP |
| --- | ---: | ---: | ---: | ---: |
| Size ratio vs uniform | 1.1469 ≤ 1.15 **pass** | 1.0135 ≤ 1.15 **pass** | same trunk **pass** | same trunk **pass** |
| General retention | 1.0488 **pass** | 1.000 **pass** | same trunk **pass** | same trunk **pass** |
| Agent-coding retention | 1.0069 **pass** | 1.007 **pass** | same trunk **pass** | same trunk **pass** |
| Verdict | **certified** | **certified** | **certified** | **certified** |

Uniform references: `mlx_lm convert -q --q-bits {4,6} --q-group-size 64` from the same BF16 pin.
Suites: development-agent-coding (76) + development-general (44), seed `20260728`, max gen 64.

First architecture_prior 4-bit pack failed agent-coding (0.9793; long_context 0.875).
Certified 4-bit uses [`examples/holo3-35b-axq4-agent-v0.1.yaml`](../examples/holo3-35b-axq4-agent-v0.1.yaml)
(attention 6-bit / experts 4-bit). Attention-8 cleared quality but size 1.162 &gt; 1.15.

## Tier 2 MTP acceleration

**Not certified** for either `-MTP` SKU. Decision after two probes on
`df-macstudio-m2` / AX Engine 6.15.0:

| Probe | Exactness | Accept rate | Speedup | Verdict |
| --- | --- | ---: | ---: | --- |
| Soft `mtp-diagnose` kill-switch matrix | pass | **0%** | ~0.43–0.45× | fail |
| Qwen **MoE exact** profile A/B (same env family as Qwen 35B Tier 2) | pass | **0%** (0/128) | **~0.50×** | fail |

Exactness passes because the engine **verifies and falls back**; drafts never
accept, so MTP only adds overhead. That is a **graft limit** (parent Qwen3.5
head vs Holo3 fine-tune trunk), not a missing env flag.

**Do not invest further in runtime-only Tier 2 for this graft.** Paths that
could still work later: train/adapt an MTP head on Holo3. Until then, product
default remains **direct decode**; keep non-MTP SKUs as primary.

Probe harness: [`scripts/run_holo3_35b_mtp_tier2_probe.py`](../scripts/run_holo3_35b_mtp_tier2_probe.py)  
Evidence: [`docs/certifications/evidence/holo3-35b-axq6-mtp-tier2/`](certifications/evidence/holo3-35b-axq6-mtp-tier2/)

## Holo3-aligned MTP (best practices + tooling)

**Do not chase 1.20× while accept≈0.** Order of work:

1. **Online accept_rate** (AX Engine A/B / MoE-exact probe) — ground truth  
2. **Offline teacher-forced top-1** (trunk hidden + MTP head vs trunk greedy) — train metric  
3. **Stage-1 adapt** — freeze trunk + freeze MTP transformer; train `mtp.fc` + pre_fc norms + `mtp.norm`  
4. **Stage-2** — unfreeze `mtp.layers.0` if stage-1 stalls  
5. **Speedup sweep / formal Tier 2** only after accept is viable (≥~0.5 heuristic)

| Command | Purpose |
| --- | --- |
| `axquant mtp-align-evaluate --report probe_decision.json` | Ladder decision from existing probe |
| `axquant mtp-align-teacher-force --model PACK --prompts P.jsonl` | Offline top-1 baseline |
| `axquant mtp-align-prepare-data --model PACK --prompts P.jsonl --output data.jsonl` | Self-distill labels |
| `axquant mtp-align-adapt-fc --model PACK --data data.jsonl --init-mtp PACK/mtp.safetensors --output mtp-adapted/` | Stage-1 train |
| `axquant compose-grafted-mtp --model-dir TRUNK --mtp-dir mtp-adapted --output PACK-adapted` | Ship new sidecar without mutating main weights |
| `scripts/run_holo3_mtp_align_campaign.py` | Measure → decide → print next commands |

Provenance for adapted heads uses `graft_kind: holo3-adapted-mtp-v1` (not “co-trained Holo3 MTP” unless trunk was jointly trained).

**Product:** keep non-MTP SKUs primary until formal Tier 2 passes on an adapted head.

### Multi-day factory campaign (stage-1 adapt)

Order (do not skip to Tier 2):

1. **Labels** — `mtp-align-prepare-data` (writes `data.jsonl` + `.features.safetensors`)
2. **Stage-1 adapt** — `mtp-align-adapt-fc` (fc + pre_fc norms + mtp.norm only)
3. **Offline top-1** — `mtp-align-teacher-force` before/after
4. **Compose** — `compose-grafted-mtp` onto certified non-MTP trunk (main digests unchanged)
5. **Online accept** — MoE-exact medium probe / `run_holo3_35b_mtp_tier2_probe.py`
6. **Only then** speedup / formal Tier 2

One-shot orchestrator on `df-macstudio-m2`:

```bash
python scripts/run_holo3_mtp_adapt_campaign.py \
  --pack /Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP \
  --trunk /Volumes/Ext4T/axquant/work/holo3-35b-axq-dev/AX-Holo3-35B-A3B-MLX-AXQ-6bit \
  --work /Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/align-campaign-v2
```

If stage-1 does not raise offline top-1 after a non-trivial step budget, fail closed and escalate to stage-2 (full `mtp.layers.0`) or stop — do not claim acceleration.

**Stage-1 factory result (2026-08-14, `df-macstudio-m2`):** 384 labels, 400 adapt steps, loss 14.5→3.14; offline top-1 **0.0→0.25**; online accept **0→~0.023**; speedup still ~0.51×. Compose preserved main digests. Evidence: [`docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage1/`](certifications/evidence/holo3-35b-axq6-mtp-adapt-stage1/).

**Stage-2 factory result (full-layer, 300 steps from stage-1 init):** offline top-1 **0.21875** (not better than stage-1); online accept still **~0.023**; speedup **~0.52×**. Provenance `holo3-adapted-mtp-full-v1`. Evidence: [`docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage2/`](certifications/evidence/holo3-35b-axq6-mtp-adapt-stage2/). CLI: `axquant mtp-align-adapt-full`.

**Tier 2 remains not certified.** On this label budget, stage-1 is the best offline checkpoint; stage-2 did not lift accept.

**V3 scaled stage-1 (2026-08-14):** 40 prompts, **1024** labels, **1200** steps from stage-1 init. Offline top-1 (48-pos) **0.208→0.25**; online accept **~0.023→~0.045**; speedup still **~0.51×**. Evidence: [`docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-v3/`](certifications/evidence/holo3-35b-axq6-mtp-adapt-v3/). Script: `scripts/run_holo3_mtp_adapt_v3_campaign.py`.

Next levers: still more data / better optim / architecture-matched labels, or product stop (keep non-MTP primary).

## Claim language

**Allowed:** checkpoint Tier 1 for 4/6-bit and 4/6-bit-MTP; measured size/quality vs matched uniform on the language trunk; AX Engine/MLX-LM text smoke; vision BF16-preserved; source pin; disclosure that MTP is grafted from parent Qwen3.5.  
**Not allowed:** GUI/VLM quality claims; Tier 2 MTP acceleration/speedup; co-trained Holo3 MTP; equating Holo3 to official Qwen 3.6 35B certificates.

## Related

- Adapter: `qwen35-moe-v1` in `src/axquant/architectures/qwen36.py`
- Graft tooling: `src/axquant/grafted_mtp.py` (`prepare-grafted-mtp`, `compose-grafted-mtp`)
- Same family path as Ornith: [docs/ornith-35b-axq-dev-runbook.md](ornith-35b-axq-dev-runbook.md)
- AX Engine: preset `holo3-35b`, model-id inference, download aliases in `ax-engine`
