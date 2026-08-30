#!/usr/bin/env bash
# Stream-convert Kimi-K3 native MXFP4 → AXQ 2-bit on df-macstudio-m2.
# After MiniMax cleanup. 2-bit only. Hobby pack: will not certify.
set -euo pipefail

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi

MODEL_ID="${KIMI_K3_MODEL_ID:-moonshotai/Kimi-K3}"
REV="${KIMI_K3_REV:-}"
export HF_HOME="${HF_HOME:-/Volumes/Ext16TR0/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext16TR0/axquant-ssd-stream}"
WORK="${KIMI_K3_AXQ2_WORK:-/Volumes/Ext16TR0/axquant/work/kimi-k3-axq2}"
OUT="${KIMI_K3_AXQ2_OUT:-/Volumes/Ext16TR0/models/AX-Kimi-K3-MLX-AXQ-2bit}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export AX_ENGINE_2BIT_EXPERIMENTAL=1
export TMPDIR="${AXQUANT_TMPDIR:-/Volumes/Ext16TR0/axquant/tmp}"
mkdir -p "$WORK" "$(dirname "$OUT")" "$TMPDIR" "$WORK/logs"

echo "[kimi-k3-axq2] host=$host out=$OUT"
"$PY" -c "from axquant.mlx_generate_gate import require_kimi_k3_kda_path; require_kimi_k3_kda_path()"

if [[ -z "$REV" ]]; then
  echo "set KIMI_K3_REV to the inspect-pinned SHA" >&2
  exit 2
fi
SNAP="${KIMI_K3_SNAP:-$HF_HOME/hub/models--moonshotai--Kimi-K3/snapshots/$REV}"
if [[ ! -f "$SNAP/config.json" ]]; then
  echo "missing Kimi-K3 snapshot at $SNAP" >&2
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
    --recipe "$ROOT/examples/kimi-k3-experimental-2bit-v0.1.yaml" \
    --output "$WORK/plan.json"
fi
if [[ -d "$OUT" ]]; then
  echo "output already exists: $OUT" >&2
  exit 2
fi

"$PY" -m axquant convert \
  --model "$SNAP" --revision "$REV" --plan "$WORK/plan.json" \
  --allow-unmeasured --expert-stream required \
  --ax-engine-manifest skip --output "$OUT"

ls "$OUT/ax_expert_stream.json" "$OUT/axquant_manifest.json"
