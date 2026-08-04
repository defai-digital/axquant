#!/usr/bin/env bash
# Standard Ext4T + Hugging Face layout for axquant factory Macs (M3, M5, …).
#
# Canonical layout (identical on every machine):
#
#   /Volumes/Ext4T/
#     huggingface/          # real HF cache (hub, xet, datasets, tokens)
#     models/               # local LLM factory checkpoints
#     axquant/              # publish / smokes / work / logs
#     logs/
#
#   ~/.cache/huggingface  ->  /Volumes/Ext4T/huggingface
#   ~/models              ->  ~/.cache/huggingface/hub   (optional convenience)
#
#   # optional but recommended in ~/.zshrc:
#   export HF_HOME=/Volumes/Ext4T/huggingface
#   export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
#
# Usage:
#   bash scripts/setup-ext4t-hf.sh --layout          # create dirs only
#   bash scripts/setup-ext4t-hf.sh --relink           # point HF home at Ext4T
#   bash scripts/setup-ext4t-hf.sh --shell-env        # append HF_HOME to ~/.zshrc
#   bash scripts/setup-ext4t-hf.sh --sync-from-nas    # rsync hub/xet from NAS
#   bash scripts/setup-ext4t-hf.sh --sync-from-host HOST  # rsync hub/xet via ssh HOST
#   bash scripts/setup-ext4t-hf.sh --status
#   bash scripts/setup-ext4t-hf.sh --verify
#   bash scripts/setup-ext4t-hf.sh --all              # layout + shell-env + relink
#
# Safe: never deletes source data. Relink backs up existing ~/.cache/huggingface.
set -euo pipefail

EXT_ROOT="${EXT_ROOT:-/Volumes/Ext4T}"
HF_DST="${EXT_ROOT}/huggingface"
MODELS_DST="${EXT_ROOT}/models"
AXQ_DST="${EXT_ROOT}/axquant"
LOG_DIR="${EXT_ROOT}/logs/setup"
NAS_MODELS="${NAS_MODELS:-/Volumes/data-models/models}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-${HOME}/.cache/huggingface}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ZSHRC="${ZSHRC:-${HOME}/.zshrc}"
MARKER_BEGIN="# >>> axquant-ext4t-hf >>>"
MARKER_END="# <<< axquant-ext4t-hf <<<"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

require_ext4t() {
  [[ -d "$EXT_ROOT" ]] || die "$EXT_ROOT not mounted — plug in / mount the Ext4T volume first"
}

layout() {
  require_ext4t
  mkdir -p \
    "$HF_DST"/{hub,xet,datasets} \
    "$MODELS_DST" \
    "$AXQ_DST"/{models,axq-publish,logs,work,smokes} \
    "$LOG_DIR" \
    "${EXT_ROOT}/logs/migration"

  if [[ ! -f "${EXT_ROOT}/README-LAYOUT.txt" ]]; then
    cat >"${EXT_ROOT}/README-LAYOUT.txt" <<'EOF'
Ext4T primary layout (shared standard: M3 / M5 / studio)
========================================================

/Volumes/Ext4T/
  huggingface/   Hugging Face cache — linked from ~/.cache/huggingface
  models/        Local LLM factory checkpoints
  axquant/       AXQuant factory work (publish, smokes, logs)
  logs/          setup + migration logs

Shell (optional):
  export HF_HOME=/Volumes/Ext4T/huggingface
  export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

Each machine keeps its own local copy on its Ext4T volume.
EOF
  fi

  ln -sfn "$HF_DST" "${EXT_ROOT}/data-models-hf" 2>/dev/null || true
  ln -sfn "$MODELS_DST" "${EXT_ROOT}/llm-models" 2>/dev/null || true

  log "layout ready under $EXT_ROOT"
  ls -la "$EXT_ROOT"
}

copy_hf_meta() {
  # Tokens / small files: prefer existing local HF dir if it is a real directory
  local src="$HF_HOME_LOCAL"
  if [[ -L "$src" ]]; then
    return 0
  fi
  [[ -d "$src" ]] || return 0
  for f in token stored_tokens .agent_harnesses.json .check_for_update_done CACHEDIR.TAG; do
    if [[ -e "$src/$f" && ! -e "$HF_DST/$f" ]]; then
      cp -a "$src/$f" "$HF_DST/$f"
      log "copied HF meta $f"
    fi
  done
  if [[ -d "$src/datasets" && ! -L "$src/datasets" ]]; then
    rsync -aH --partial --exclude '.DS_Store' "$src/datasets/" "$HF_DST/datasets/" 2>/dev/null || \
      rsync -aH --exclude '.DS_Store' "$src/datasets/" "$HF_DST/datasets/" || true
  fi
}

relink() {
  require_ext4t
  layout
  copy_hf_meta
  mkdir -p "${HOME}/.cache"

  local n
  n="$(find "$HF_DST/hub" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
  # Allow empty hub for brand-new machines (downloads will land on Ext4T)
  log "hub entries on Ext4T: ${n:-0}"

  if [[ -L "$HF_HOME_LOCAL" ]]; then
    local cur
    cur="$(readlink "$HF_HOME_LOCAL")"
    if [[ "$cur" == "$HF_DST" ]]; then
      log "HF already linked: $HF_HOME_LOCAL -> $cur"
    else
      log "replacing HF symlink $cur -> $HF_DST"
      rm "$HF_HOME_LOCAL"
      ln -sfn "$HF_DST" "$HF_HOME_LOCAL"
    fi
  elif [[ -d "$HF_HOME_LOCAL" ]]; then
    local backup="${HF_HOME_LOCAL}.pre-ext4t-${STAMP}"
    log "backing up $HF_HOME_LOCAL -> $backup"
    mv "$HF_HOME_LOCAL" "$backup"
    ln -sfn "$HF_DST" "$HF_HOME_LOCAL"
    log "linked $HF_HOME_LOCAL -> $HF_DST"
  else
    ln -sfn "$HF_DST" "$HF_HOME_LOCAL"
    log "created $HF_HOME_LOCAL -> $HF_DST"
  fi

  # Convenience: ~/models -> hub (same pattern as factory machines)
  if [[ ! -e "${HOME}/models" ]] || [[ -L "${HOME}/models" ]]; then
    ln -sfn "${HF_HOME_LOCAL}/hub" "${HOME}/models"
    log "~/models -> ${HF_HOME_LOCAL}/hub"
  else
    log "WARN: ~/models exists and is not a symlink; left unchanged"
  fi

  {
    echo "relinked_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(scutil --get ComputerName 2>/dev/null || hostname)"
    echo "hf_home_local=$HF_HOME_LOCAL"
    echo "hf_dst=$HF_DST"
    echo "hub_entries=${n:-0}"
  } | tee "${LOG_DIR}/hf-relink-${STAMP}.txt"
  cp "${LOG_DIR}/hf-relink-${STAMP}.txt" "${LOG_DIR}/hf-relink-latest.txt"
}

shell_env() {
  local block
  block=$(cat <<EOF
${MARKER_BEGIN}
# Local Ext4T Hugging Face cache (shared standard: M3 / M5)
export HF_HOME="/Volumes/Ext4T/huggingface"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_CACHE="\$HF_HOME/hub"
${MARKER_END}
EOF
)
  mkdir -p "$(dirname "$ZSHRC")"
  touch "$ZSHRC"
  if grep -qF "$MARKER_BEGIN" "$ZSHRC" 2>/dev/null; then
    # Replace existing block
    local tmp
    tmp="$(mktemp)"
    awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
      $0 == b {skip=1; next}
      $0 == e {skip=0; next}
      !skip {print}
    ' "$ZSHRC" >"$tmp"
    cat "$tmp" >"$ZSHRC"
    rm -f "$tmp"
  fi
  printf '\n%s\n' "$block" >>"$ZSHRC"
  log "wrote HF_HOME block to $ZSHRC"
  echo "Open a new shell or: source $ZSHRC"
}

rsync_open() {
  # macOS openrsync-compatible flags
  rsync -aH --partial --progress --stats --exclude '.DS_Store' "$@"
}

sync_from_nas() {
  require_ext4t
  layout
  [[ -d "$NAS_MODELS/hub" ]] || die "NAS hub missing: $NAS_MODELS/hub (mount data-models first)"
  local logf="${LOG_DIR}/rsync-hf-hub-nas-${STAMP}.log"
  log "sync NAS hub -> $HF_DST/hub (log $logf)"
  rsync_open "$NAS_MODELS/hub/" "$HF_DST/hub/" | tee -a "$logf"
  if [[ -d "$NAS_MODELS/xet" ]]; then
    logf="${LOG_DIR}/rsync-hf-xet-nas-${STAMP}.log"
    log "sync NAS xet -> $HF_DST/xet"
    rsync_open "$NAS_MODELS/xet/" "$HF_DST/xet/" | tee -a "$logf"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/hf-sync-complete.txt"
  log "NAS HF sync complete"
}

sync_from_host() {
  local host="${1:-}"
  [[ -n "$host" ]] || die "usage: --sync-from-host HOST"
  require_ext4t
  layout
  local logf="${LOG_DIR}/rsync-hf-hub-from-${host}-${STAMP}.log"
  log "sync HF hub from ${host}:/Volumes/Ext4T/huggingface/hub/ -> $HF_DST/hub/"
  # Pull via remote rsync over ssh (openrsync on both sides)
  rsync -aH --partial --progress --stats --exclude '.DS_Store' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "${host}:/Volumes/Ext4T/huggingface/hub/" "$HF_DST/hub/" | tee -a "$logf"
  logf="${LOG_DIR}/rsync-hf-xet-from-${host}-${STAMP}.log"
  rsync -aH --partial --progress --stats --exclude '.DS_Store' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "${host}:/Volumes/Ext4T/huggingface/xet/" "$HF_DST/xet/" | tee -a "$logf" || true
  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/hf-sync-complete.txt"
  log "host HF sync complete from $host"
}

show_status() {
  echo "=== host ==="
  scutil --get ComputerName 2>/dev/null || true
  hostname
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  echo
  echo "=== Ext4T ==="
  df -h "$EXT_ROOT" 2>/dev/null || echo "not mounted"
  ls -la "$EXT_ROOT" 2>/dev/null || true
  echo
  echo "=== HF link ==="
  if [[ -L "$HF_HOME_LOCAL" ]]; then
    echo "$HF_HOME_LOCAL -> $(readlink "$HF_HOME_LOCAL")"
  elif [[ -d "$HF_HOME_LOCAL" ]]; then
    echo "$HF_HOME_LOCAL is a real directory"
    ls -la "$HF_HOME_LOCAL" | head -12
  else
    echo "$HF_HOME_LOCAL missing"
  fi
  echo "HF_HOME=${HF_HOME:-}"
  echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-}"
  echo
  echo "=== hub entries ==="
  if [[ -d "$HF_DST/hub" ]]; then
    find "$HF_DST/hub" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' '
  else
    echo 0
  fi
  echo
  echo "=== shell marker ==="
  if [[ -f "$ZSHRC" ]] && grep -qF "$MARKER_BEGIN" "$ZSHRC"; then
    echo "present in $ZSHRC"
  else
    echo "missing in $ZSHRC"
  fi
  echo
  echo "=== latest relink ==="
  cat "${LOG_DIR}/hf-relink-latest.txt" 2>/dev/null || echo "(none)"
}

verify() {
  local ok=1
  require_ext4t
  if [[ -L "$HF_HOME_LOCAL" && "$(readlink "$HF_HOME_LOCAL")" == "$HF_DST" ]]; then
    echo "OK HF symlink"
  else
    echo "FAIL HF symlink (want $HF_HOME_LOCAL -> $HF_DST)"
    ok=0
  fi
  if [[ -d "$HF_DST/hub" && -d "$HF_DST/xet" ]]; then
    echo "OK HF dst dirs"
  else
    echo "FAIL HF dst dirs"; ok=0
  fi
  if [[ -e "$HF_HOME_LOCAL/hub" ]]; then
    echo "OK hub resolves via HF home"
  else
    echo "FAIL hub path"; ok=0
  fi
  if [[ -n "${HF_HOME:-}" && "$HF_HOME" == "$HF_DST" ]]; then
    echo "OK HF_HOME env"
  else
    echo "WARN HF_HOME not set in this shell (check ~/.zshrc / new shell)"
  fi
  if [[ "$ok" -eq 1 ]]; then
    echo "VERIFY_PASS"
    return 0
  fi
  echo "VERIFY_FAIL"
  return 1
}

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    -h|--help|"") usage; exit 0 ;;
    --layout) layout ;;
    --relink) relink ;;
    --shell-env) shell_env ;;
    --sync-from-nas) sync_from_nas ;;
    --sync-from-host) shift; sync_from_host "${1:-}" ;;
    --status) show_status ;;
    --verify) verify ;;
    --all)
      layout
      shell_env
      relink
      verify || true
      log "layout+env+relink done. Seed hub with --sync-from-nas or --sync-from-host when ready."
      ;;
    *) die "unknown: $cmd (try --help)" ;;
  esac
}

main "$@"
