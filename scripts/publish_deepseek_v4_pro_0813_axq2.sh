#!/usr/bin/env bash
# After stream convert finishes: license + git SHA + privacy scan, then hf upload.
# Host: df-macstudio-m2. Does not certify. Cleanup only after Hub lists weights.
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

REV="${DSV4_PRO_REV:-72e1d3230f6c080a530b0a1d46f8eb4602340597}"
SNAP="${DSV4_PRO_SNAP:-$HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813/snapshots/$REV}"
OUT="${DSV4_PRO_AXQ2_OUT:-/Volumes/Ext16TR0/models/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP}"
REPO="${DSV4_PRO_AXQ2_REPO:-AutomatosX/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP}"
ROOT="${AXQUANT_SSD_STREAM:-/Volumes/Ext16TR0/axquant-ssd-stream}"
CARD="${DSV4_PRO_AXQ2_CARD:-$ROOT/docs/hub-cards/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP.md}"
if [[ ! -f "$CARD" ]]; then
  CARD=/Users/devop/code/axquant/docs/hub-cards/AX-DeepSeek-V4-Pro-0813-MLX-AXQ-2bit-MTP.md
fi
PY="${AXQUANT_PYTHON:-/Users/devop/code/axquant/.venv/bin/python}"
HF="${AXQUANT_HF:-/Users/devop/code/axquant/.venv/bin/hf}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$OUT/axquant_manifest.json" || ! -f "$OUT/ax_expert_stream.json" || ! -f "$OUT/mtp.safetensors" ]]; then
  echo "convert output incomplete: $OUT" >&2
  ls -l "$OUT" 2>/dev/null | head >&2 || true
  exit 2
fi

echo "[dsv4-pro-0813] hobby-pack prepare (license, git SHA, privacy)"
"$PY" -m axquant.hobby_pack \
  --output "$OUT" \
  --source "$SNAP" \
  --git-dir "$ROOT" \
  --card "$CARD"

echo "upload $REPO from $OUT (Xet HP=$HF_XET_HIGH_PERFORMANCE)"
"$HF" upload "$REPO" "$OUT" . --repo-type model \
  --commit-message "AXQ experimental Super-class stream pack (hobby, not certified)"

echo "confirm Hub lists weights before Ext16TR0 cleanup"
if ! "$HF" repo files "$REPO" --repo-type model | grep -E 'safetensors|axquant_manifest.json'; then
  echo "Hub did not list weights; keeping Ext16TR0 copies (SC-R6)" >&2
  exit 2
fi

WORK="${DSV4_PRO_AXQ2_WORK:-/Volumes/Ext16TR0/axquant/work/deepseek-v4-pro-0813-axq2}"
SNAP_ROOT="${HF_HOME}/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813"
echo "cleanup Ext16TR0 after Hub confirm (PRD SC-R6)"
du -sh "$OUT" "$SNAP_ROOT" "$WORK" 2>/dev/null || true
rm -rf "$OUT" "$WORK"
if [[ "${DSV4_PRO_KEEP_SNAP:-0}" != "1" ]]; then
  rm -rf "$SNAP_ROOT"
fi
df -h /Volumes/Ext16TR0 | tail -1
