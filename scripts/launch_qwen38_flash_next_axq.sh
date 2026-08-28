#!/usr/bin/env bash
# Detached factory driver for Qwen3.8-Flash-Next AXQ convert + Hub publish.
# Survives SSH drop (screen + nohup). Does not certify.
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
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
unset HF_HUB_DISABLE_XET || true

ROOT="${FLASH_NEXT_ROOT:-/Users/akiralam/code/axquant}"
VENV="${FLASH_NEXT_VENV:-$EXT/axquant-venv}"
WORK="${FLASH_NEXT_WORK:-$EXT/axquant/work/qwen38-flash-next}"
SESSION="${FLASH_NEXT_SESSION:-qwen38-flash-next-axq}"
mkdir -p "$WORK/logs" "$HF_HOME/xet" "$HF_HOME/hub"

if [[ "${HF_XET_HIGH_PERFORMANCE}" != "1" ]]; then
  echo "HF_XET_HIGH_PERFORMANCE is not 1 on this PID" >&2
  exit 2
fi

LOG="$WORK/logs/pipeline.$(date +%Y%m%dT%H%M%S).log"
WRAPPER="$WORK/logs/run-detached.sh"
UV="${UV:-$HOME/.local/bin/uv}"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE="$HF_XET_CACHE"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"
unset HF_HUB_ENABLE_HF_TRANSFER || true
unset HF_HUB_DISABLE_XET || true
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export FLASH_NEXT_EXT="$EXT"
export FLASH_NEXT_VENV="$VENV"
export FLASH_NEXT_WORK="$WORK"
cd "$ROOT"
UV="$UV"
if [[ ! -x "\$UV" ]]; then
  UV="\$(command -v uv)"
fi
echo "[\$(date -Iseconds)] start host=\$(hostname -s) HF_XET_HIGH_PERFORMANCE=\$HF_XET_HIGH_PERFORMANCE uv=\$UV"
python3 -c "import os,sys; sys.exit(0 if os.environ.get('HF_XET_HIGH_PERFORMANCE')=='1' else 1)"
"\$UV" python install 3.12
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import sys" 2>/dev/null; then
  "\$UV" venv "$VENV" --python 3.12
fi
"\$UV" pip install --python "$VENV/bin/python" -e '$ROOT[mlx]' --upgrade hf-xet
"\$UV" pip install --python "$VENV/bin/python" "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"
"$VENV/bin/python" -c "import hf_xet,os,sys; assert os.environ.get('HF_XET_HIGH_PERFORMANCE')=='1'; print('hf-xet', hf_xet.__file__)"
exec "$VENV/bin/python" "$ROOT/scripts/run_qwen38_flash_next_axq.py" all
EOF
chmod +x "$WRAPPER"

if screen -list 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session $SESSION already running" >&2
  screen -list
  exit 0
fi

# Prefer screen so the job can be reattached after SSH drop.
if command -v screen >/dev/null 2>&1; then
  screen -dmS "$SESSION" bash -lc "exec >>'$LOG' 2>&1; exec '$WRAPPER'"
  echo "started screen session $SESSION"
  echo "log: $LOG"
  echo "reattach: screen -r $SESSION"
else
  nohup bash "$WRAPPER" >>"$LOG" 2>&1 </dev/null &
  echo "started nohup pid $!"
  echo "log: $LOG"
fi

sleep 2
if ! pgrep -f "run_qwen38_flash_next_axq.py" >/dev/null 2>&1; then
  echo "driver process not visible yet; check $LOG" >&2
fi
ps eww -p "$(pgrep -f run_qwen38_flash_next_axq.py | head -1)" 2>/dev/null | tr ' ' '\n' | grep HF_XET || true
