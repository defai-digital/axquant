#!/bin/bash
# Factory-only: plan + convert Flash-0731 2-bit attn6-affine v0.3, then T1 v-extract.
# Host: df-macstudio-m2 + Ext12T. Does not publish. Does not claim certification.
set -euo pipefail
export PATH="/Users/devop/.local/bin:$PATH"
export HF_HOME=/Volumes/Ext12T/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=$HF_HOME/xet
unset HF_HUB_ENABLE_HF_TRANSFER
export AX_ENGINE_2BIT_EXPERIMENTAL=1

AXQ=/Users/devop/code/axquant/.venv/bin/axquant
PY=/Users/devop/code/axquant/.venv/bin/python
ROOT=/Users/devop/code/axquant
SRC=/Volumes/Ext12T/models/DeepSeek-V4-Flash-0731
REV=7872f01b1d1fe23eabc4c98b48bffcef5a386062
INV=/Volumes/Ext12T/axquant/work/flash-0731/inventory.json
WORK=/Volumes/Ext12T/axquant/work/flash-0731-2bit-attn6-affine
OUT=/Volumes/Ext12T/models/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-attn6-affine
RECIPE=$ROOT/examples/deepseek-v4-experimental-2bit-attn6-affine-v0.3.yaml
LOG=$WORK/logs
mkdir -p "$LOG" "$WORK/plans"

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 1
fi

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) plan 2bit-attn6-affine ====="
"$AXQ" plan-manual \
  --inventory "$INV" \
  --recipe "$RECIPE" \
  --output "$WORK/plans/2bit-attn6-affine.json" \
  --markdown-output "$WORK/plans/2bit-attn6-affine.md"
python3 - <<PY
import json
from collections import Counter
p = json.load(open("$WORK/plans/2bit-attn6-affine.json"))
c = Counter((a["role"], a["bits"], a["method"]) for a in p["assignments"])
print("effective_bpw", p.get("effective_bpw"))
print("target_class", p.get("target_class"))
print("role_bits")
for key, n in c.most_common():
    print(" ", key, n)
PY

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) convert 2bit-attn6-affine ====="
if [ -f "$OUT/config.json" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  echo "reuse $OUT"
else
  rm -rf "$OUT"
  "$AXQ" convert \
    --model "$SRC" \
    --revision "$REV" \
    --plan "$WORK/plans/2bit-attn6-affine.json" \
    --allow-unmeasured \
    --ax-engine-manifest skip \
    --output "$OUT"
fi
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) convert done ====="
du -sh "$OUT" || true

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) T1 v-extract ====="
export DSV4_AXQ2="$OUT"
export DSV4_AXQ2_MODEL_ID="AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP"
export AX_ENGINE_SERVER=/Users/devop/opt/ax-engine-80f2a3e6/ax-engine-server
export AX_ENGINE_GENERATE_MANIFEST=/Users/devop/opt/ax-engine-7.1.5/ax-engine-bench
export AX_ENGINE_VERSION=80f2a3e6-attn6-affine
export DSV4_QA_PROTOCOL=v-extract
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
"$PY" scripts/run_deepseek_v4_0731_axq2_axengine.py all
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) T1 done ====="
