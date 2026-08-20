#!/bin/bash
# Factory-only: OptiQ-2bit on the same v-extract protocol as AXQ T1.
# Does not convert. Does not copy mlx-optiq. Does not claim certification.
set -euo pipefail
export PATH="/Users/devop/.local/bin:$PATH"
export HF_HOME=/Volumes/Ext12T/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=$HF_HOME/xet
unset HF_HUB_ENABLE_HF_TRANSFER
export DSV4_QA_PROTOCOL=v-extract
export DSV4_FORCE_EVAL=1
export DSV4_OPTIQ_VS_AXQ_WORK=/Volumes/Ext12T/axquant-certification/deepseek-v4-0731-optiq-v-extract
export PYTHONPATH=/Users/devop/code/axquant/src

ROOT=/Users/devop/code/axquant
PY=$ROOT/.venv/bin/python

host=$(hostname -s)
if [[ "$host" != "df-macstudio-m2" && "$host" != "devopsmacstudio" ]]; then
  echo "factory eval must run on df-macstudio-m2; observed $host" >&2
  exit 1
fi

mkdir -p "$DSV4_OPTIQ_VS_AXQ_WORK"
cd "$ROOT"
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) market v-extract OptiQ ====="
"$PY" scripts/run_deepseek_v4_0731_optiq_vs_axq2.py market-v-extract
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) market v-extract done ====="
