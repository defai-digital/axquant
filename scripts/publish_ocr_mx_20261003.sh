#!/usr/bin/env bash
# Publish the six OCR MX development packs (DeepSeek-OCR-2 + Unlimited-OCR,
# MXFP4 / MXFP8 MLX packs and MXFP6 reference artifacts). Host: df-macstudio-m2.
# Executed 2026-10-03; the two MXFP6 packs were deleted the same day with
# AXQ-051, so re-running fails fast on the surviving repos. Kept as history.
#
# Prerequisites: `hf auth login` with an AutomatosX write token. The script
# never takes a token argument; it uses the `hf` CLI auth state.
set -euo pipefail

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory publish must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/Volumes/Ext16TR0/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
unset HF_TOKEN_PATH || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF="${AXQUANT_HF:-$REPO_ROOT/.venv/bin/hf}"
WORK="${OCR_MX_WORK:-/Volumes/Ext16TR0/models/ocr-mx-20261003}"

if ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "Hub auth invalid; run \`hf auth login --force\` first" >&2
  exit 2
fi

PACKS=(
  "AX-DeepSeek-OCR-2-MLX-AXQ-MXFP4"
  "AX-DeepSeek-OCR-2-MLX-AXQ-MXFP8"
  "AX-DeepSeek-OCR-2-AXQ-MXFP6"
  "AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4"
  "AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8"
  "AX-Unlimited-OCR-3B-MoE-AXQ-MXFP6"
)

for pack in "${PACKS[@]}"; do
  dir="$WORK/$pack"
  if [[ ! -f "$dir/README.md" ]]; then
    echo "missing model card: $dir/README.md" >&2
    exit 2
  fi
  if [[ "$pack" == *"-MXFP6" ]]; then
    if [[ ! -f "$dir/axquant_mxfp6.json" ]]; then
      echo "missing MXFP6 manifest: $dir" >&2
      exit 2
    fi
    if [[ -f "$dir/config.json" ]]; then
      echo "MXFP6 reference must not ship config.json: $dir" >&2
      exit 2
    fi
  else
    for req in axquant_manifest.json config.json tokenizer.json; do
      if [[ ! -f "$dir/$req" ]]; then
        echo "incomplete MLX pack $dir: missing $req" >&2
        exit 2
      fi
    done
  fi
done

for pack in "${PACKS[@]}"; do
  repo="AutomatosX/$pack"
  # create fails when the repo already exists: never overwrite in place.
  "$HF" repos create "$repo" --type model
  (cd "$WORK/$pack" && "$HF" upload "$repo" . --include "*.safetensors" \
    --include "*.json" --include "*.jinja" --include "*.txt" --include "README.md")
  echo "published $repo"
done
