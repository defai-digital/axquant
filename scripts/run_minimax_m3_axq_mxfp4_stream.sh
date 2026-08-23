#!/usr/bin/env bash
# Stream-convert MiniMax-M3 BF16 → AXQ MXFP4 on df-macstudio-m2 after 2-bit cleanup.
set -euo pipefail

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi

MODEL_ID="${MINIMAX_M3_MODEL_ID:-MiniMaxAI/MiniMax-M3}"
REV="${MINIMAX_M3_REV:-}"
export HF_HOME="${HF_HOME:-/Volumes/Ext12T/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext12T/axquant-ssd-stream}"
WORK="${MINIMAX_M3_MXFP4_WORK:-/Volumes/Ext12T/axquant/work/minimax-m3-mxfp4}"
OUT="${MINIMAX_M3_MXFP4_OUT:-/Volumes/Ext12T/models/AX-MiniMax-M3-MLX-AXQ-MXFP4}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export AX_ENGINE_2BIT_EXPERIMENTAL=1
export TMPDIR="${AXQUANT_TMPDIR:-/Volumes/Ext12T/axquant/tmp}"
mkdir -p "$WORK" "$(dirname "$OUT")" "$TMPDIR" "$WORK/logs"

echo "[minimax-m3-mxfp4] host=$host out=$OUT"
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
  "$PY" -m axquant inspect \
    --model "$SNAP" --model-id "$MODEL_ID" --revision "$REV" \
    --allow-quantized --output "$WORK/inventory.json"
fi
if [[ ! -f "$WORK/plan.json" ]]; then
  "$PY" -m axquant plan-manual \
    --inventory "$WORK/inventory.json" \
    --recipe "$ROOT/examples/minimax-m3-experimental-mxfp4-v0.1.yaml" \
    --output "$WORK/plan.json"
fi
if [[ -d "$OUT" ]]; then
  echo "output already exists: $OUT" >&2
  exit 2
fi

"$PY" -m axquant convert \
  --model "$SNAP" --revision "$REV" --plan "$WORK/plan.json" \
  --allow-unmeasured --expert-stream required --q-mode mxfp4 \
  --ax-engine-manifest skip --output "$OUT"

ls "$OUT/ax_expert_stream.json" "$OUT/axquant_manifest.json"
