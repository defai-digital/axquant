#!/usr/bin/env bash
# After the in-flight convert exits, retry every Flash-Next pack with the
# PLE shard↔shards binding fix. Does not certify. Survives SSH drop.
set -euo pipefail
host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory convert must run on df-macstudio-m2; observed $host" >&2
  exit 2
fi
EXT="${FLASH_NEXT_EXT:-/Volumes/Ext16TR0}"
export HF_HOME="${HF_HOME:-$EXT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export AXQUANT_FORCE_CPU=1
export PYTHONPATH="${FLASH_NEXT_ROOT:-/Users/akiralam/code/axquant}/src"
export PYTHONUNBUFFERED=1
export FLASH_NEXT_EXT="$EXT"
export FLASH_NEXT_VENV="${FLASH_NEXT_VENV:-$EXT/axquant-venv}"
export FLASH_NEXT_WORK="${FLASH_NEXT_WORK:-$EXT/axquant/work/qwen38-flash-next}"
ROOT="${FLASH_NEXT_ROOT:-/Users/akiralam/code/axquant}"
PY="${FLASH_NEXT_VENV}/bin/python"
LOG="${FLASH_NEXT_WORK}/logs/retry-remaining.log"
mkdir -p "${FLASH_NEXT_WORK}/logs"
exec >>"$LOG" 2>&1
echo "[$(date -Iseconds)] waiting for in-flight axquant convert or driver"
while ps -ax -o command= | awk '
  /python -m axquant convert/ && $0 !~ /awk/ { found=1 }
  /run_qwen38_flash_next_axq.py/ && $0 !~ /retry_qwen38_flash_next/ && $0 !~ /awk/ { found=1 }
  END { exit !found }
'; do
  sleep 30
done
echo "[$(date -Iseconds)] converting remaining Flash-Next packs"
cd "$ROOT"
exec "$PY" "$ROOT/scripts/run_qwen38_flash_next_axq.py" remaining
