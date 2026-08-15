# GPT-OSS remake runbook — OpenAI native source → AXQ 4/6-bit → Hub overwrite

**Host:** `df-macstudio-m2` (recommended; 120B needs large free disk + long CPU convert).
Historical GPT-OSS Tier 1 records were measured on `df-macbookpro-m5`.  
**Toolkit:** AXQuant ≥ 1.6.2 (current release **1.7.0**) with `mlx-lm` that implements `model_type=gpt_oss`  
**Goal:** Rebuild four AutomatosX packs from **official OpenAI native weights**, then overwrite the public Hub repos.

| Pack | Hub repo | Product class | Default plan |
| --- | --- | --- | --- |
| 20B 4-bit | `AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit` | `4bit` | manual [`examples/gpt-oss-20b-axq4-agent-v0.1.yaml`](../examples/gpt-oss-20b-axq4-agent-v0.1.yaml) |
| 20B 6-bit | `AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit` | `6bit` | try `architecture_prior` BPW 6.0 first; fallback manual [`examples/gpt-oss-20b-axq6-agent-v0.1.yaml`](../examples/gpt-oss-20b-axq6-agent-v0.1.yaml) |
| 120B 4-bit | `AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit` | `4bit` | manual [`examples/gpt-oss-120b-axq4-agent-v0.1.yaml`](../examples/gpt-oss-120b-axq4-agent-v0.1.yaml) (**create/republish only if quality gates pass**) |
| 120B 6-bit | `AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit` | `6bit` | manual [`examples/gpt-oss-120b-axq6-agent-v0.1.yaml`](../examples/gpt-oss-120b-axq6-agent-v0.1.yaml) |

## Why this remake

| Bad previous path | Correct remake path |
| --- | --- |
| Convert input = `mlx-community/gpt-oss-*-MXFP4-Q4` | Convert input = `openai/gpt-oss-*` |
| Community pack already quantized attention/embed to 4-bit affine | OpenAI keeps **attention / router / embed / lm_head at BF16** |
| Experts MXFP4 + non-experts Q4 → dequant → AXQ = **double-quant on sensitive layers** | Experts only are native MXFP4 → dequant → affine; non-experts quantized **once** from BF16 |

OpenAI ships **no pure-BF16 expert dump**. Experts are **post-trained MXFP4**. `--allow-quantized` remains required and is the supported GPT-OSS path (`support_policy` / `gpt-oss-v1` adapter).

Do **not** treat community “BF16” upcasts as more authoritative than `openai/gpt-oss-*` unless you deliberately want a third-party dequant lineage.

---

## 0. Preconditions

```bash
source /path/to/user/code/axquant/.venv/bin/activate
cd /path/to/user/code/axquant

# mlx-lm must know gpt_oss
python -c "import mlx_lm.models.gpt_oss as m; print('gpt_oss OK', m.__file__)"

# Hugging Face write token (overwrite AutomatosX/*)
huggingface-cli whoami
# export HF_TOKEN=...   # if needed

# Disk (order-of-magnitude; measure before 120B)
#   OpenAI 20B source  ~15–25 GB
#   OpenAI 120B source ~60–80 GB
#   Outputs: 20B×2 ~30 GB; 120B 4-bit ~70 GB; 120B 6-bit ~96 GB
#   Plus working space for dequant peaks — prefer ≥400 GB free on Ext4T
df -h /path/to/ext-storage
```

Pin sources (refresh if you intentionally move forward):

```bash
# Resolved at runbook authoring time; re-check before production:
#   openai/gpt-oss-20b  → pin with:
python - <<'PY'
from huggingface_hub import model_info
for mid in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
    print(mid, model_info(mid).sha)
PY
```

Record the printed SHAs as:

```bash
export OSS20_REV="<40-char-sha>"   # openai/gpt-oss-20b
export OSS120_REV="<40-char-sha>"  # openai/gpt-oss-120b
export SEED=20260728
export MAX_TOKENS=64
export HOST_ID=df-macstudio-m2
export AXQ_ROOT=/path/to/user/code/axquant
export WORK=/path/to/ext-storage/axquant/work/gpt-oss-openai-native-remake
export PUB=/path/to/ext-storage/axquant/axq-publish
export CERT=/path/to/ext-storage/axquant-certification/gpt-oss-openai-native-remake
mkdir -p "$WORK" "$PUB" "$CERT"/{inventories,plans,quality,logs,size}
```

Optional but strongly recommended for 120B re-pack (Metal timeouts historically):

```bash
export AXQUANT_FORCE_CPU=1
```

---

## 1. Materialize pinned OpenAI sources

Prefer local Hub cache under Ext4T (see `scripts/setup-ext4t-hf.sh` if HF home is not already pointed at the volume).

```bash
# Download / ensure cache (immutable revision)
huggingface-cli download openai/gpt-oss-20b  --revision "$OSS20_REV"
huggingface-cli download openai/gpt-oss-120b --revision "$OSS120_REV"

# Resolve snapshot paths (for inspect/convert --model)
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for mid, rev, env in [
    ("openai/gpt-oss-20b", os.environ["OSS20_REV"], "OSS20_DIR"),
    ("openai/gpt-oss-120b", os.environ["OSS120_REV"], "OSS120_DIR"),
]:
    p = snapshot_download(mid, revision=rev, local_files_only=True)
    print(f"export {env}={p}")
PY
# eval the printed exports, or set manually:
# export OSS20_DIR=...
# export OSS120_DIR=...
```

Sanity: official config must show **MXFP4 only on MoE**, not full Q4 attention.

```bash
python - <<'PY'
import json, os
from pathlib import Path
for label, d in [("20b", os.environ["OSS20_DIR"]), ("120b", os.environ["OSS120_DIR"])]:
    cfg = json.loads(Path(d, "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    print(label, "quant_method=", qc.get("quant_method"),
          "modules_to_not_convert=", qc.get("modules_to_not_convert"))
    assert qc.get("quant_method") == "mxfp4"
    assert any("self_attn" in m for m in qc.get("modules_to_not_convert", []))
print("OK: OpenAI native mixed MXFP4 experts + BF16 non-experts")
PY
```

---

## 2. Inventory (required for manual plans)

```bash
axquant inspect \
  --model "$OSS20_DIR" \
  --model-id openai/gpt-oss-20b \
  --revision "$OSS20_REV" \
  --allow-quantized \
  --output "$CERT/inventories/gpt-oss-20b.inventory.json"

axquant inspect \
  --model "$OSS120_DIR" \
  --model-id openai/gpt-oss-120b \
  --revision "$OSS120_REV" \
  --allow-quantized \
  --output "$CERT/inventories/gpt-oss-120b.inventory.json"
```

Expect:

- `quantized_source: true`
- expert `*_blocks` / MXFP4 bodies marked quantizable under `--allow-quantized`
- attention / embedding present as BF16-class weights (not pre-affine-4)

---

## 3. Plan

### 3a. Manual plans (recommended for remake consistency)

```bash
# 20B 4-bit
axquant plan-manual \
  --inventory "$CERT/inventories/gpt-oss-20b.inventory.json" \
  --recipe "$AXQ_ROOT/examples/gpt-oss-20b-axq4-agent-v0.1.yaml" \
  --output "$CERT/plans/gpt-oss-20b-axq4.plan.json" \
  --markdown-output "$CERT/plans/gpt-oss-20b-axq4.plan.md"

# 20B 6-bit (manual fallback; skip if using prior convert in §4b)
axquant plan-manual \
  --inventory "$CERT/inventories/gpt-oss-20b.inventory.json" \
  --recipe "$AXQ_ROOT/examples/gpt-oss-20b-axq6-agent-v0.1.yaml" \
  --output "$CERT/plans/gpt-oss-20b-axq6.plan.json" \
  --markdown-output "$CERT/plans/gpt-oss-20b-axq6.plan.md"

# 120B 4-bit
axquant plan-manual \
  --inventory "$CERT/inventories/gpt-oss-120b.inventory.json" \
  --recipe "$AXQ_ROOT/examples/gpt-oss-120b-axq4-agent-v0.1.yaml" \
  --output "$CERT/plans/gpt-oss-120b-axq4.plan.json" \
  --markdown-output "$CERT/plans/gpt-oss-120b-axq4.plan.md"

# 120B 6-bit
axquant plan-manual \
  --inventory "$CERT/inventories/gpt-oss-120b.inventory.json" \
  --recipe "$AXQ_ROOT/examples/gpt-oss-120b-axq6-agent-v0.1.yaml" \
  --output "$CERT/plans/gpt-oss-120b-axq6.plan.json" \
  --markdown-output "$CERT/plans/gpt-oss-120b-axq6.plan.md"
```

Check planned BPW in the markdown summaries before convert:

| Pack | Recipe target BPW | Soft product window | Size gate vs mlx-community MXFP4-Q4 |
| --- | ---: | --- | ---: |
| 20B 4-bit | 5.2 | ~4.8–5.2 product class | ≤ 1.20× |
| 20B 6-bit | 6.0 prior / 6.2 manual | ~6.0±0.35 (manual may land higher) | ≤ 1.55× |
| 120B 4-bit | 5.1 | mixed 4-bit class | ≤ 1.20× |
| 120B 6-bit | 6.6 | product name 6-bit; measured ~6.58 historically | ≤ 1.55× |

---

## 4. Convert

Manual / unmeasured plans require `--allow-unmeasured`. Outputs go to **new** directories (never mutate Hub clones in place until upload).

### 4a. 20B 4-bit (manual)

```bash
export OUT20_4="$PUB/AX-gpt-oss-20b-MLX-AXQ-4bit-openai-native"
rm -rf "$OUT20_4"
axquant convert \
  --model "$OSS20_DIR" \
  --revision "$OSS20_REV" \
  --plan "$CERT/plans/gpt-oss-20b-axq4.plan.json" \
  --allow-unmeasured \
  --ax-engine-manifest if-available \
  --output "$OUT20_4" \
  2>&1 | tee "$CERT/logs/convert-20b-axq4.log"
```

### 4b. 20B 6-bit (try prior first)

```bash
export OUT20_6="$PUB/AX-gpt-oss-20b-MLX-AXQ-6bit-openai-native"
rm -rf "$OUT20_6"

# Preferred: architecture_prior @ 6.0 (simple door)
axquant quantize \
  --model "$OSS20_DIR" \
  --model-id openai/gpt-oss-20b \
  --revision "$OSS20_REV" \
  --target-bpw 6.0 \
  --profile agent-coding \
  --allow-quantized \
  --runtime-smoke mlx-lm \
  --output "$OUT20_6" \
  2>&1 | tee "$CERT/logs/convert-20b-axq6-prior.log"

# If quality later fails, re-convert with manual plan instead:
# rm -rf "$OUT20_6"
# axquant convert --model "$OSS20_DIR" --revision "$OSS20_REV" \
#   --plan "$CERT/plans/gpt-oss-20b-axq6.plan.json" \
#   --allow-unmeasured --output "$OUT20_6"
```

### 4c. 120B 4-bit and 6-bit (manual; force CPU if needed)

```bash
export AXQUANT_FORCE_CPU=1
export OUT120_4="$PUB/AX-gpt-oss-120b-MLX-AXQ-4bit-openai-native"
export OUT120_6="$PUB/AX-gpt-oss-120b-MLX-AXQ-6bit-openai-native"

rm -rf "$OUT120_4"
axquant convert \
  --model "$OSS120_DIR" \
  --revision "$OSS120_REV" \
  --plan "$CERT/plans/gpt-oss-120b-axq4.plan.json" \
  --allow-unmeasured \
  --ax-engine-manifest if-available \
  --output "$OUT120_4" \
  2>&1 | tee "$CERT/logs/convert-120b-axq4.log"

rm -rf "$OUT120_6"
axquant convert \
  --model "$OSS120_DIR" \
  --revision "$OSS120_REV" \
  --plan "$CERT/plans/gpt-oss-120b-axq6.plan.json" \
  --allow-unmeasured \
  --ax-engine-manifest if-available \
  --output "$OUT120_6" \
  2>&1 | tee "$CERT/logs/convert-120b-axq6.log"
```

Post-convert checks:

```bash
for d in "$OUT20_4" "$OUT20_6" "$OUT120_4" "$OUT120_6"; do
  echo "==== $d ===="
  test -f "$d/axquant_manifest.json"
  test -f "$d/axquant_plan.json"
  ls "$d"/model*.safetensors | wc -l
  python - <<PY
import json
from pathlib import Path
m=json.loads(Path("$d/axquant_manifest.json").read_text())
src=m.get("source_model") or m.get("source") or {}
print("source", src)
print("bpw", m.get("measured_total_bpw") or m.get("measured_main_bpw"))
print("weight_bytes", m.get("weight_file_size_bytes"))
PY
done
```

Confirm manifest source is **`openai/gpt-oss-*`** with the pinned revision — not `mlx-community/*`.

---

## 5. Runtime smoke

```bash
axquant runtime-check --model "$OUT20_4" --runtime mlx-lm 2>&1 | tee "$CERT/logs/smoke-20b-4.log"
axquant runtime-check --model "$OUT20_6" --runtime mlx-lm 2>&1 | tee "$CERT/logs/smoke-20b-6.log"
# 120B smokes are long / memory heavy:
axquant runtime-check --model "$OUT120_4" --runtime mlx-lm 2>&1 | tee "$CERT/logs/smoke-120b-4.log"
axquant runtime-check --model "$OUT120_6" --runtime mlx-lm 2>&1 | tee "$CERT/logs/smoke-120b-6.log"
```

---

## 6. Quality evaluation (dual baseline)

### 6.1 Datasets (same as prior certs)

```bash
export DS_AGENT=/path/to/ext-storage/axquant-certification/qwen36-27b-axq6-v1/datasets/development-agent-coding/dataset.jsonl
export DS_GENERAL=/path/to/ext-storage/axquant-certification/qwen36-27b-axq6-v1/datasets/development-general/dataset.jsonl
test -f "$DS_AGENT" && test -f "$DS_GENERAL"
```

### 6.2 Baselines

| Role | Model | Purpose |
| --- | --- | --- |
| **Primary quality reference (recommended for remake)** | OpenAI native `$OSS20_DIR` / `$OSS120_DIR` | “How close to official mixed-precision” |
| **Size / continuity reference (optional)** | `mlx-community/gpt-oss-*-MXFP4-Q4` @ old pinned revs | Compare download size vs prior Hub narrative |

```bash
# --- OpenAI native references ---
axquant evaluate-quality \
  --model "$OSS20_DIR" --model-id openai/gpt-oss-20b --revision "$OSS20_REV" \
  --dataset "$DS_AGENT" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
  --output "$CERT/quality/ref-openai-20b-agent.json"

axquant evaluate-quality \
  --model "$OSS20_DIR" --model-id openai/gpt-oss-20b --revision "$OSS20_REV" \
  --dataset "$DS_GENERAL" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
  --output "$CERT/quality/ref-openai-20b-general.json"

# 120B refs are expensive; run once and reuse
axquant evaluate-quality \
  --model "$OSS120_DIR" --model-id openai/gpt-oss-120b --revision "$OSS120_REV" \
  --dataset "$DS_AGENT" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
  --output "$CERT/quality/ref-openai-120b-agent.json"

axquant evaluate-quality \
  --model "$OSS120_DIR" --model-id openai/gpt-oss-120b --revision "$OSS120_REV" \
  --dataset "$DS_GENERAL" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
  --output "$CERT/quality/ref-openai-120b-general.json"
```

### 6.3 Candidates + compare

```bash
eval_pair () {
  local name="$1" model="$2" ref_agent="$3" ref_gen="$4"
  axquant evaluate-quality --model "$model" \
    --dataset "$DS_AGENT" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
    --output "$CERT/quality/${name}-agent.json"
  axquant evaluate-quality --model "$model" \
    --dataset "$DS_GENERAL" --seed "$SEED" --max-tokens "$MAX_TOKENS" \
    --output "$CERT/quality/${name}-general.json"
  axquant compare-quality --reference "$ref_agent" --candidate "$CERT/quality/${name}-agent.json" \
    --output "$CERT/quality/${name}-agent-compare.json"
  axquant compare-quality --reference "$ref_gen" --candidate "$CERT/quality/${name}-general.json" \
    --output "$CERT/quality/${name}-general-compare.json"
  python - <<PY
import json
from pathlib import Path
for suite in ("agent","general"):
  c=json.loads(Path("$CERT/quality/${name}-"+suite+"-compare.json").read_text())
  # field names vary slightly by version; print whole summary keys
  print("$name", suite, {k:c.get(k) for k in c if "retent" in k.lower() or "score" in k.lower() or k in ("verdict","aggregate")})
  print("  keys", sorted(c)[:20])
PY
}

eval_pair 20b-axq4 "$OUT20_4" \
  "$CERT/quality/ref-openai-20b-agent.json" "$CERT/quality/ref-openai-20b-general.json"
eval_pair 20b-axq6 "$OUT20_6" \
  "$CERT/quality/ref-openai-20b-agent.json" "$CERT/quality/ref-openai-20b-general.json"
eval_pair 120b-axq4 "$OUT120_4" \
  "$CERT/quality/ref-openai-120b-agent.json" "$CERT/quality/ref-openai-120b-general.json"
eval_pair 120b-axq6 "$OUT120_6" \
  "$CERT/quality/ref-openai-120b-agent.json" "$CERT/quality/ref-openai-120b-general.json"
```

**Publish gates (checkpoint Tier 1 style):**

- general retention ≥ **0.98**
- agent-coding retention ≥ **0.98**
- MLX-LM load/smoke pass
- size ratio vs **mlx-community MXFP4-Q4** still applied for product packaging continuity (next section)

If 120B 4-bit still fails agent-coding, **do not publish** that class; keep certified 6-bit only (see `docs/known-issues.md`).

---

## 7. Size evidence vs mlx-community MXFP4-Q4

Keep the historical size baseline so packs remain comparable to community Q4 downloads:

```bash
export MX20_REV=f356f2747216d7e98fee755df25987459fc19089
export MX120_REV=bce781bef0f2fc85ed4e575af74054f5aad73ddd
huggingface-cli download mlx-community/gpt-oss-20b-MXFP4-Q4  --revision "$MX20_REV"
huggingface-cli download mlx-community/gpt-oss-120b-MXFP4-Q4 --revision "$MX120_REV"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def weight_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in model_dir.glob("model*.safetensors"))

pairs = [
    ("20b-4", os.environ["OUT20_4"], "mlx-community/gpt-oss-20b-MXFP4-Q4", os.environ["MX20_REV"], 1.20),
    ("20b-6", os.environ["OUT20_6"], "mlx-community/gpt-oss-20b-MXFP4-Q4", os.environ["MX20_REV"], 1.55),
    ("120b-4", os.environ["OUT120_4"], "mlx-community/gpt-oss-120b-MXFP4-Q4", os.environ["MX120_REV"], 1.20),
    ("120b-6", os.environ["OUT120_6"], "mlx-community/gpt-oss-120b-MXFP4-Q4", os.environ["MX120_REV"], 1.55),
]
for name, cand, ref_id, rev, limit in pairs:
    ref = Path(snapshot_download(ref_id, revision=rev, local_files_only=True))
    cb, rb = weight_bytes(Path(cand)), weight_bytes(ref)
    ratio = cb / rb
    ok = ratio <= limit + 1e-9
    print(f"{name}: candidate={cb} ref={rb} ratio={ratio:.6f} limit={limit} {'PASS' if ok else 'FAIL'}")
PY
```

---

## 8. Model card + provenance sanitization

Use development card prep. Until new certificates are written and digests bound, **skip auto-binding old certs** (they still name `mlx-community` sources and old digests):

```bash
prep_card () {
  local dir="$1" repo="$2" klass="$3"
  python "$AXQ_ROOT/scripts/prepare_development_model_card.py" \
    --artifact "$dir" \
    --repo-id "$repo" \
    --product-class "$klass" \
    --no-public-certification
}

prep_card "$OUT20_4"  AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit  4bit
prep_card "$OUT20_6"  AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit  6bit
# only if quality+size passed:
prep_card "$OUT120_4" AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit 4bit
prep_card "$OUT120_6" AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit 6bit
```

Edit each `README.md` front-matter / body so claims match reality:

- Source: `openai/gpt-oss-*@<sha>`
- Path: `--allow-quantized` dequant (experts MXFP4) → affine re-pack
- Quality measured vs OpenAI native (state suite sizes, seed, host, AXQuant version)
- Size vs mlx-community MXFP4-Q4 (optional continuity)
- Explicit: **not** BF16 experts; **not** AX Engine Tier 2 / MTP

---

## 9. Publish / overwrite Hub `main`

GPT-OSS is a **thin** family: there is no full flagship release-audit door for these packs. Overwrite with Hub upload of the prepared artifact directories (same as prior AutomatosX development/cert packs).

### 9.1 Dry-run inventory

```bash
upload_list () {
  local dir="$1"
  find "$dir" -type f ! -name '.DS_Store' | sed "s|^$dir/||" | sort
}
upload_list "$OUT20_4" | tee "$CERT/logs/files-20b-4.txt"
```

Ensure LFS-tracked safetensors; `huggingface_hub` handles this via `hf upload`.

### 9.2 Overwrite existing repos (creates new `main` commit)

```bash
# Requires write access to AutomatosX org
publish_one () {
  local dir="$1" repo="$2" tag="$3"
  echo "PUBLISH $repo from $dir"
  # --commit-message documents the remake lineage
  hf upload "$repo" "$dir" . \
    --repo-type model \
    --commit-message "Rebuild from openai native MXFP4 mixed source ($tag); replace community MXFP4-Q4 re-pack lineage."
}

publish_one "$OUT20_4" AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit  "20b-axq4"
publish_one "$OUT20_6" AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit  "20b-axq6"
publish_one "$OUT120_6" AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit "120b-axq6"

# 120B 4-bit: only if gates passed; repo may need create if previously deleted
# hf repo create AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit --type model  # if missing
# publish_one "$OUT120_4" AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit "120b-axq4"
```

### 9.3 Post-publish pin

```bash
python - <<'PY'
from huggingface_hub import model_info
for mid in [
  "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit",
  "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit",
  "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit",
  # "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit",
]:
    info = model_info(mid)
    print(mid, info.sha)
PY
```

Download the published revision into a clean directory and re-run a short `mlx_lm.generate` smoke against the **Hub commit**, not only local pre-upload paths.

---

## 10. Update AXQuant certificates (after Hub commits exist)

For each pack that passes gates:

1. Write/update `docs/certifications/gpt-oss-*-tier1.json` with:
   - new Hub commit
   - `source_model_id: openai/gpt-oss-*`
   - `source_revision: <OSS*_REV>`
   - quality numbers vs OpenAI native
   - size vs mlx-community (if still used)
   - measured BPW + manifest SHA-256
2. Regenerate markdown matrices:

```bash
python scripts/render_certification_docs.py --write
```

3. Adjust `docs/known-issues.md` / `docs/certifications/README.md` GPT-OSS blurb:
   - source lineage is OpenAI native, not community Q4 re-pack
4. Commit docs in the axquant git repo (separate from Hub weight upload).

Until certificates are updated, keep `--no-public-certification` cards so README does not claim an obsolete digest.

---

## 11. Operator decision tree

```text
Convert from openai/gpt-oss-* + --allow-quantized
        │
        ├─ smoke fail ────────────────────────► fix mlx-lm / layout; do not publish
        │
        ├─ size > gate vs MXFP4-Q4 ───────────► tighten recipe BPW / bits; re-plan
        │
        ├─ quality vs OpenAI native < 0.98 ───► raise attention / protect capacity layers
        │                                        (do not switch back to community Q4 source)
        │
        └─ all gates pass ────────────────────► card → hf upload overwrite → cert JSON → render docs
```

### Do / do not

| Do | Do not |
| --- | --- |
| Pin OpenAI revisions | Convert from `mlx-community/*-MXFP4-Q4` as source |
| Use `--allow-quantized` for native MXFP4 experts | Claim pure-BF16 expert provenance |
| Keep attention protected (8-bit) on 4-bit packs | Overwrite Hub before quality+size gates |
| Dual-report size vs community Q4 if product needs continuity | Publish 120B 4-bit if agent-coding still fails |
| Force CPU on 120B if Metal times out | Treat this path as AX Engine Tier 2 cert |

---

## 12. Quick command index

```bash
# inventory
axquant inspect --model $OSS20_DIR --model-id openai/gpt-oss-20b --revision $OSS20_REV --allow-quantized -o inv.json

# plan
axquant plan-manual --inventory inv.json --recipe examples/gpt-oss-20b-axq4-agent-v0.1.yaml -o plan.json

# convert
AXQUANT_FORCE_CPU=1 axquant convert --model $OSS20_DIR --revision $OSS20_REV --plan plan.json --allow-unmeasured -o $OUT

# quality
axquant evaluate-quality --model $OUT --dataset $DS_AGENT --seed 20260728 --max-tokens 64 -o cand.json
axquant compare-quality --reference ref.json --candidate cand.json -o cmp.json

# card + upload
python scripts/prepare_development_model_card.py --artifact $OUT --repo-id AutomatosX/... --product-class 4bit --no-public-certification
hf upload AutomatosX/... $OUT . --repo-type model --commit-message "openai-native remake"
```

---

## 13. Expected outcomes after remake

| Pack | Prior community-Q4 lineage | Expected after OpenAI native remake |
| --- | --- | --- |
| 20B 4-bit | Certified with attn-8 manual | Should remain certifiable; likely stronger vs true source |
| 20B 6-bit | Certified prior 6.0 | Should remain easy; prior may suffice |
| 120B 6-bit | Certified manual 6.6 / attn-8 | Should remain certifiable; re-run gates |
| 120B 4-bit | Failed agent-coding 0.952; not published | **Maybe** clears 0.98 after removing double-quant; if not, leave unpublished |

This runbook does not guarantee 120B 4-bit certification — it removes a known source defect so the remaining failure, if any, is honest recipe/budget limitation.
