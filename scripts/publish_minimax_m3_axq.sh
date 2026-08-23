#!/usr/bin/env bash
# Hobby publish for MiniMax M3 2-bit or MXFP4. Host: df-macstudio-m2.
set -euo pipefail
host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory publish must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi
SKU="${1:-2bit}"
export HF_HOME="${HF_HOME:-/Volumes/Ext12T/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
REV="${MINIMAX_M3_REV:?set MINIMAX_M3_REV}"
SNAP="${MINIMAX_M3_SNAP:-$HF_HOME/hub/models--MiniMaxAI--MiniMax-M3/snapshots/$REV}"
ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext12T/axquant-ssd-stream}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
HF="${AXQUANT_HF:-/Users/devop/code/axquant/.venv/bin/hf}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$SKU" == "mxfp4" ]]; then
  OUT="${MINIMAX_M3_MXFP4_OUT:-/Volumes/Ext12T/models/AX-MiniMax-M3-MLX-AXQ-MXFP4}"
  REPO="${MINIMAX_M3_MXFP4_REPO:-AutomatosX/AX-MiniMax-M3-MLX-AXQ-MXFP4}"
  CARD="$ROOT/docs/hub-cards/AX-MiniMax-M3-MLX-AXQ-MXFP4.md"
  WORK="${MINIMAX_M3_MXFP4_WORK:-/Volumes/Ext12T/axquant/work/minimax-m3-mxfp4}"
else
  OUT="${MINIMAX_M3_AXQ2_OUT:-/Volumes/Ext12T/models/AX-MiniMax-M3-MLX-AXQ-2bit}"
  REPO="${MINIMAX_M3_AXQ2_REPO:-AutomatosX/AX-MiniMax-M3-MLX-AXQ-2bit}"
  CARD="$ROOT/docs/hub-cards/AX-MiniMax-M3-MLX-AXQ-2bit.md"
  WORK="${MINIMAX_M3_AXQ2_WORK:-/Volumes/Ext12T/axquant/work/minimax-m3-axq2}"
fi
if [[ ! -f "$OUT/axquant_manifest.json" || ! -f "$OUT/ax_expert_stream.json" ]]; then
  echo "convert output incomplete: $OUT" >&2
  exit 2
fi
if [[ -f "$OUT/mtp.safetensors" ]]; then
  echo "MiniMax pack unexpectedly contains mtp.safetensors; HNC forbids -MTP-less name" >&2
  exit 2
fi
"$PY" -m axquant.hobby_pack --output "$OUT" --source "$SNAP" --git-dir "$ROOT" --card "$CARD"
# Batched commits: a single hf upload of Super-class shards times out on Hub.
"$PY" -m axquant.hobby_upload --repo "$REPO" --folder "$OUT"
if ! "$PY" -m axquant.hobby_upload --repo "$REPO" --check-repo; then
  echo "Hub did not list weights; keeping Ext12T copies" >&2
  exit 2
fi
SNAP_ROOT="${HF_HOME}/hub/models--MiniMaxAI--MiniMax-M3"
du -sh "$OUT" "$SNAP_ROOT" "$WORK" 2>/dev/null || true
rm -rf "$OUT" "$WORK"
# 2-bit is followed by MXFP4 on the same BF16 snapshot; keep it unless
# this is the MXFP4 publish (or an operator forces delete).
if [[ "$SKU" == "mxfp4" && "${MINIMAX_M3_KEEP_SNAP:-0}" != "1" ]]; then
  rm -rf "$SNAP_ROOT"
fi
df -h /Volumes/Ext12T | tail -1
