#!/usr/bin/env bash
# Launch MiniMax-M3 Tier 1 cert so SSH disconnect does not kill it.
# Primary: nohup. Optional: screen -r minimax-cert if screen is used.
set -euo pipefail

ROOT="${AXQUANT_ROOT:-$HOME/code/axquant}"
WORK="${MINIMAX_M3_WORK:-$HOME/axquant-certification/minimax-m3-tier1}"
LOG="$WORK/logs/run-cert.log"
PIDFILE="$WORK/logs/run-cert.pid"
PY="$ROOT/.venv/bin/python"

mkdir -p "$WORK/logs"

if [[ ! -x "$PY" ]]; then
  echo "missing venv python: $PY" >&2
  exit 2
fi
if [[ ! -f "$ROOT/scripts/run_minimax_m3_tier1.py" ]]; then
  echo "missing cert runner in $ROOT" >&2
  exit 2
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid $(cat "$PIDFILE")"
  echo "log: tail -f $LOG"
  exit 0
fi

export PATH="$ROOT/.venv/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$ROOT/src"
export AX_ENGINE_2BIT_EXPERIMENTAL=1
export AX_STREAM_EXPERT_LAYERS="${AX_STREAM_EXPERT_LAYERS:-64}"
export AX_ENGINE_ROOT="${AX_ENGINE_ROOT:-$HOME/opt/ax-engine-minimax}"
export AX_ENGINE_RELEASE="${AX_ENGINE_RELEASE:-v7.2.0}"
export MINIMAX_M3_FORCE_QUALITY="${MINIMAX_M3_FORCE_QUALITY:-1}"

cd "$ROOT"
nohup bash -c "
  set +e
  echo \"[cert] \$(date) both packs AX Engine native\" >>\"$LOG\"
  \"$PY\" -u scripts/run_minimax_m3_tier1.py both >>\"$LOG\" 2>&1
  status=\$?
  if [[ \$status -eq 0 ]]; then echo DONE >>\"$LOG\"; else echo FAILED >>\"$LOG\"; fi
  exit \$status
" </dev/null >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
sleep 1
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "started pid $(cat "$PIDFILE")"
  echo "log:  tail -f $LOG"
  echo "wait: wait \$(cat $PIDFILE)"
else
  echo "process exited immediately; last log lines:" >&2
  tail -30 "$LOG" >&2
  exit 1
fi
