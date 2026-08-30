#!/usr/bin/env bash
# Stream-convert MiniMax-M3 BF16 → AXQ 2-bit on df-macstudio-m2.
# After SC1 Pro-0813 cleanup. Hobby pack: will not certify. Stream required.
# Default Hub name has no -MTP (config num_mtp_modules is not packaged MTP).
set -euo pipefail

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi

MODEL_ID="${MINIMAX_M3_MODEL_ID:-MiniMaxAI/MiniMax-M3}"
REV="${MINIMAX_M3_REV:-}"
export HF_HOME="${HF_HOME:-/Volumes/Ext16TR0/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext16TR0/axquant-ssd-stream}"
WORK="${MINIMAX_M3_AXQ2_WORK:-/Volumes/Ext16TR0/axquant/work/minimax-m3-axq2}"
OUT="${MINIMAX_M3_AXQ2_OUT:-/Volumes/Ext16TR0/models/AX-MiniMax-M3-MLX-AXQ-2bit}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export AX_ENGINE_2BIT_EXPERIMENTAL=1
export TMPDIR="${AXQUANT_TMPDIR:-/Volumes/Ext16TR0/axquant/tmp}"
mkdir -p "$WORK" "$(dirname "$OUT")" "$TMPDIR" "$WORK/logs"

echo "[minimax-m3-axq2] host=$host python=$PY root=$ROOT out=$OUT"
if ! python3 -c "import os,sys; sys.exit(0 if os.environ.get('HF_XET_HIGH_PERFORMANCE')=='1' else 1)"; then
  echo "HF_XET_HIGH_PERFORMANCE is not 1 on this PID" >&2
  exit 2
fi

echo "[minimax-m3-axq2] public MLX language-path generate gate"
"$PY" -c "from axquant.mlx_generate_gate import require_minimax_m3_generate_path; require_minimax_m3_generate_path()"

if [[ -z "$REV" ]]; then
  echo "set MINIMAX_M3_REV to the inspect-pinned SHA" >&2
  exit 2
fi
SNAP="${MINIMAX_M3_SNAP:-$HF_HOME/hub/models--MiniMaxAI--MiniMax-M3/snapshots/$REV}"
if [[ ! -f "$SNAP/config.json" ]]; then
  echo "missing MiniMax-M3 BF16 snapshot at $SNAP" >&2
  exit 2
fi

if [[ ! -f "$WORK/inventory.json" ]]; then
  echo "[minimax-m3-axq2] inspect"
  "$PY" -m axquant inspect \
    --model "$SNAP" \
    --model-id "$MODEL_ID" \
    --revision "$REV" \
    --allow-quantized \
    --output "$WORK/inventory.json"
fi

if [[ ! -f "$WORK/plan.json" ]]; then
  echo "[minimax-m3-axq2] plan-manual"
  "$PY" -m axquant plan-manual \
    --inventory "$WORK/inventory.json" \
    --recipe "$ROOT/examples/minimax-m3-experimental-2bit-v0.1.yaml" \
    --output "$WORK/plan.json"
fi

if [[ -d "$OUT" ]]; then
  echo "output already exists: $OUT" >&2
  exit 2
fi

echo "[minimax-m3-axq2] convert stream 2-bit --expert-stream required"
"$PY" -m axquant convert \
  --model "$SNAP" \
  --revision "$REV" \
  --plan "$WORK/plan.json" \
  --allow-unmeasured \
  --expert-stream required \
  --ax-engine-manifest skip \
  --output "$OUT"

echo "[minimax-m3-axq2] done $OUT"
ls "$OUT/ax_expert_stream.json" "$OUT/axquant_manifest.json"
