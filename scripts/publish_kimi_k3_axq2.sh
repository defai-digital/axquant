#!/usr/bin/env bash
# Hobby publish for Kimi K3 2-bit. Host: df-macstudio-m2.
set -euo pipefail
host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory publish must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/Volumes/Ext16TR0/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
REV="${KIMI_K3_REV:?set KIMI_K3_REV}"
SNAP="${KIMI_K3_SNAP:-$HF_HOME/hub/models--moonshotai--Kimi-K3/snapshots/$REV}"
ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext16TR0/axquant-ssd-stream}"
OUT="${KIMI_K3_AXQ2_OUT:-/Volumes/Ext16TR0/models/AX-Kimi-K3-MLX-AXQ-2bit}"
REPO="${KIMI_K3_AXQ2_REPO:-AutomatosX/AX-Kimi-K3-MLX-AXQ-2bit}"
CARD="$ROOT/docs/hub-cards/AX-Kimi-K3-MLX-AXQ-2bit.md"
WORK="${KIMI_K3_AXQ2_WORK:-/Volumes/Ext16TR0/axquant/work/kimi-k3-axq2}"
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
HF="${AXQUANT_HF:-/Users/devop/code/axquant/.venv/bin/hf}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$OUT/axquant_manifest.json" || ! -f "$OUT/ax_expert_stream.json" ]]; then
  echo "convert output incomplete: $OUT" >&2
  exit 2
fi
"$PY" -m axquant.hobby_pack --output "$OUT" --source "$SNAP" --git-dir "$ROOT" --card "$CARD"
# Batched commits: a single hf upload of Super-class shards times out on Hub.
"$PY" -m axquant.hobby_upload --repo "$REPO" --folder "$OUT" --batch-size 8 --skip-existing
if ! "$PY" -m axquant.hobby_upload --repo "$REPO" --check-repo; then
  echo "Hub did not list weights; keeping Ext16TR0 copies" >&2
  exit 2
fi
SNAP_ROOT="${HF_HOME}/hub/models--moonshotai--Kimi-K3"
du -sh "$OUT" "$SNAP_ROOT" "$WORK" 2>/dev/null || true
rm -rf "$OUT" "$WORK"
if [[ "${KIMI_K3_KEEP_SNAP:-0}" != "1" ]]; then
  rm -rf "$SNAP_ROOT"
fi
df -h /Volumes/Ext16TR0 | tail -1
