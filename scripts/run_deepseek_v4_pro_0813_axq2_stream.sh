#!/usr/bin/env bash
# Stream-convert DeepSeek-V4-Pro-0813 → AXQ 2-bit MTP on df-macstudio-m2.
# Does not mlx_lm.load the 1.78 TB mixed FP4+FP8 snapshot.
# Hobby pack: will not be certified. SSD layer-stack paging required.
set -euo pipefail

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi

REV="${DSV4_PRO_REV:-72e1d3230f6c080a530b0a1d46f8eb4602340597}"
MODEL_ID="${DSV4_PRO_MODEL_ID:-deepseek-ai/DeepSeek-V4-Pro-0813}"
export HF_HOME="${HF_HOME:-/Volumes/Ext16TR0/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

SNAP="${DSV4_PRO_SNAP:-$HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813/snapshots/$REV}"
ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext16TR0/axquant-ssd-stream}"
WORK="${DSV4_PRO_AXQ2_WORK:-/Volumes/Ext16TR0/axquant/work/deepseek-v4-pro-0813-axq2}"
OUT="${DSV4_PRO_AXQ2_OUT:-/Volumes/Ext16TR0/models/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AXQUANT_FORCE_CPU="${AXQUANT_FORCE_CPU:-0}"
export PYTHONUNBUFFERED=1
export AX_ENGINE_2BIT_EXPERIMENTAL=1
export TMPDIR="${AXQUANT_TMPDIR:-/Volumes/Ext16TR0/axquant/tmp}"
mkdir -p "$WORK" "$(dirname "$OUT")" "$TMPDIR" "$WORK/logs"

echo "[dsv4-pro-0813] host=$host"
echo "[dsv4-pro-0813] source=$SNAP"
echo "[dsv4-pro-0813] python=$PY root=$ROOT out=$OUT"
echo "[dsv4-pro-0813] HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"

if [[ ! -f "$SNAP/config.json" ]]; then
  echo "missing Pro-0813 snapshot at $SNAP (wait for Xet download)" >&2
  exit 2
fi
if [[ ! -d "$ROOT/src/axquant" ]]; then
  echo "missing stream-convert tree at $ROOT" >&2
  exit 2
fi
# Confirm this PID actually has Xet HP (not only the parent script).
if ! python3 -c "import os,sys; sys.exit(0 if os.environ.get('HF_XET_HIGH_PERFORMANCE')=='1' else 1)"; then
  echo "HF_XET_HIGH_PERFORMANCE is not 1 on this PID" >&2
  exit 2
fi

if [[ ! -f "$WORK/inventory.json" ]]; then
  echo "[dsv4-pro-0813] inspect"
  "$PY" -m axquant inspect \
    --model "$SNAP" \
    --model-id "$MODEL_ID" \
    --revision "$REV" \
    --allow-quantized \
    --output "$WORK/inventory.json"
fi

if [[ ! -f "$WORK/plan.json" ]]; then
  echo "[dsv4-pro-0813] plan-manual"
  "$PY" -m axquant plan-manual \
    --inventory "$WORK/inventory.json" \
    --recipe "$ROOT/examples/deepseek-v4-pro-0813-experimental-2bit-v0.1.yaml" \
    --output "$WORK/plan.json"
fi

if [[ -d "$OUT" ]]; then
  echo "output already exists: $OUT" >&2
  exit 2
fi

echo "[dsv4-pro-0813] convert stream 2-bit + MTP sidecar + expert-stream required"
"$PY" -m axquant convert \
  --model "$SNAP" \
  --revision "$REV" \
  --plan "$WORK/plan.json" \
  --allow-unmeasured \
  --expert-stream required \
  --ax-engine-manifest skip \
  --output "$OUT"

echo "[dsv4-pro-0813] done $OUT"
du -sh "$OUT" || true
ls "$OUT/mtp.safetensors" "$OUT/ax_expert_stream.json" "$OUT/axquant_manifest.json"
