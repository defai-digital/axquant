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

validate_configuration() {
  for path in "$EXT_ROOT" "$NAS_MODELS" "$HF_HOME_LOCAL" "$ZSHRC"; do
    [[ "$path" == /* && "$path" != *"//"* && "$path" != */ ]] || {
      die "path must be absolute and canonical: $path"
    }
    [[ "/$path/" != *"/../"* && "/$path/" != *"/./"* ]] || {
      die "path must not contain dot components: $path"
    }
  done
  [[ "$EXT_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$EXT_ROOT" != "/" && "$EXT_ROOT" != "/Volumes" ]] || {
    die "unsafe Ext4T root: $EXT_ROOT"
  }
  for path in "$HF_HOME_LOCAL" "$ZSHRC"; do
    [[ "$path" == "$HOME/"* && "$path" != "$HOME" ]] || {
      die "$path must be a child of HOME"
    }
  done
}

require_ext4t() {
  [[ -d "$EXT_ROOT" && ! -L "$EXT_ROOT" ]] || {
    die "$EXT_ROOT is not a real mounted directory — plug in / mount Ext4T first"
  }
  local mounted_at
  mounted_at="$(df -P "$EXT_ROOT" 2>/dev/null | awk 'NR == 2 {print $NF}')"
  [[ "$mounted_at" == "$EXT_ROOT" ]] || {
    die "$EXT_ROOT is not a mounted volume (df reports ${mounted_at:-unknown})"
  }
}

ensure_real_dir() {
  local path="$1"
  [[ ! -L "$path" ]] || die "managed directory must not be a symlink: $path"
  [[ ! -e "$path" || -d "$path" ]] || die "managed path is not a directory: $path"
  mkdir -p "$path"
}

ensure_home_dir() {
  local path="$1"
  [[ "$path" == "$HOME" || "$path" == "$HOME/"* ]] || die "path is outside HOME: $path"
  local relative="${path#"$HOME"}"
  relative="${relative#/}"
  local current="$HOME"
  local -a components=()
  local component
  IFS='/' read -r -a components <<<"$relative"
  for component in "${components[@]}"; do
    [[ -n "$component" ]] || continue
    current="$current/$component"
    ensure_real_dir "$current"
  done
}

safe_output_file() {
  local path="$1"
  [[ ! -L "$path" ]] || die "refusing symlinked output file: $path"
  [[ ! -e "$path" || -f "$path" ]] || die "output path is not a regular file: $path"
}

ensure_convenience_link() {
  local target="$1"
  local link="$2"
  if [[ -L "$link" ]]; then
    [[ "$(readlink "$link")" == "$target" ]] || die "$link points somewhere unexpected"
  elif [[ -e "$link" ]]; then
    die "$link exists and is not a symlink"
  else
    ln -s "$target" "$link"
  fi
}

layout() {
  require_ext4t
  local path
  for path in \
    "$HF_DST" "$HF_DST/hub" "$HF_DST/xet" "$HF_DST/datasets" "$MODELS_DST" \
    "$AXQ_DST" "$AXQ_DST/models" "$AXQ_DST/axq-publish" "$AXQ_DST/logs" \
    "$AXQ_DST/work" "$AXQ_DST/smokes" "${EXT_ROOT}/logs" "$LOG_DIR" \
    "${EXT_ROOT}/logs/migration"; do
    ensure_real_dir "$path"
  done

  safe_output_file "${EXT_ROOT}/README-LAYOUT.txt"
  if [[ ! -e "${EXT_ROOT}/README-LAYOUT.txt" ]]; then
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

  ensure_convenience_link "$HF_DST" "${EXT_ROOT}/data-models-hf"
  ensure_convenience_link "$MODELS_DST" "${EXT_ROOT}/llm-models"

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
    if ! rsync -aH --partial --exclude '.DS_Store' \
      "$src/datasets/" "$HF_DST/datasets/" 2>/dev/null; then
      rsync -aH --exclude '.DS_Store' "$src/datasets/" "$HF_DST/datasets/" \
        || die "local datasets cache could not be copied"
    fi
  fi
}

relink() {
  require_ext4t
  layout
  copy_hf_meta
  ensure_home_dir "$(dirname "$HF_HOME_LOCAL")"

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
      ln -s "$HF_DST" "$HF_HOME_LOCAL"
    fi
  elif [[ -d "$HF_HOME_LOCAL" ]]; then
    local backup="${HF_HOME_LOCAL}.pre-ext4t-${STAMP}"
    [[ ! -e "$backup" && ! -L "$backup" ]] || die "backup path already exists: $backup"
    log "backing up $HF_HOME_LOCAL -> $backup"
    mv "$HF_HOME_LOCAL" "$backup"
    ln -s "$HF_DST" "$HF_HOME_LOCAL"
    log "linked $HF_HOME_LOCAL -> $HF_DST"
  elif [[ ! -e "$HF_HOME_LOCAL" ]]; then
    ln -s "$HF_DST" "$HF_HOME_LOCAL"
    log "created $HF_HOME_LOCAL -> $HF_DST"
  else
    die "$HF_HOME_LOCAL exists and is not a directory or symlink"
  fi

  # Convenience: ~/models -> hub (same pattern as factory machines)
  if [[ -L "${HOME}/models" ]]; then
    if [[ "$(readlink "${HOME}/models")" == "${HF_HOME_LOCAL}/hub" ]]; then
      log "${HOME}/models already points to ${HF_HOME_LOCAL}/hub"
    else
      log "WARN: ${HOME}/models is an unrelated symlink; left unchanged"
    fi
  elif [[ ! -e "${HOME}/models" ]]; then
    ln -s "${HF_HOME_LOCAL}/hub" "${HOME}/models"
    log "${HOME}/models -> ${HF_HOME_LOCAL}/hub"
  else
    log "WARN: ~/models exists and is not a symlink; left unchanged"
  fi

  safe_output_file "${LOG_DIR}/hf-relink-${STAMP}.txt"
  safe_output_file "${LOG_DIR}/hf-relink-latest.txt"
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
export HF_HOME="${EXT_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_CACHE="\$HF_HOME/hub"
${MARKER_END}
EOF
)
  ensure_home_dir "$(dirname "$ZSHRC")"
  [[ ! -L "$ZSHRC" ]] || die "refusing to rewrite symlinked shell configuration: $ZSHRC"
  [[ ! -e "$ZSHRC" || -f "$ZSHRC" ]] || die "shell configuration is not a file: $ZSHRC"
  touch "$ZSHRC"
  local begin_count end_count begin_line end_line
  begin_count="$(awk -v marker="$MARKER_BEGIN" '$0 == marker {n++} END {print n + 0}' "$ZSHRC")"
  end_count="$(awk -v marker="$MARKER_END" '$0 == marker {n++} END {print n + 0}' "$ZSHRC")"
  if [[ "$begin_count" -ne "$end_count" || "$begin_count" -gt 1 ]]; then
    die "refusing to rewrite malformed or duplicate HF_HOME marker block in $ZSHRC"
  fi
  local tmp
  tmp="$(mktemp "${ZSHRC}.axquant.XXXXXX")"
  if [[ "$begin_count" -eq 1 ]]; then
    begin_line="$(awk -v marker="$MARKER_BEGIN" '$0 == marker {print NR}' "$ZSHRC")"
    end_line="$(awk -v marker="$MARKER_END" '$0 == marker {print NR}' "$ZSHRC")"
    [[ "$begin_line" -lt "$end_line" ]] || {
      rm -f "$tmp"
      die "refusing to rewrite out-of-order HF_HOME markers in $ZSHRC"
    }
    awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
      $0 == b {skip=1; next}
      $0 == e {skip=0; next}
      !skip {print}
    ' "$ZSHRC" >"$tmp"
  else
    cp "$ZSHRC" "$tmp"
  fi
  printf '%s\n' "$block" >>"$tmp"
  chmod "$(stat -f '%Lp' "$ZSHRC")" "$tmp"
  mv "$tmp" "$ZSHRC"
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
  local nas_mount
  nas_mount="$(df -P "$NAS_MODELS" 2>/dev/null | awk 'NR == 2 {print $NF}')"
  [[ -n "$nas_mount" && "$nas_mount" != "/" ]] || {
    die "NAS model cache is not on a dedicated mounted filesystem"
  }
  [[ ! -L "$NAS_MODELS/hub" ]] || die "NAS hub must not be a symlink"
  local logf="${LOG_DIR}/rsync-hf-hub-nas-${STAMP}.log"
  safe_output_file "$logf"
  log "sync NAS hub -> $HF_DST/hub (log $logf)"
  rsync_open "$NAS_MODELS/hub/" "$HF_DST/hub/" | tee -a "$logf"
  if [[ -d "$NAS_MODELS/xet" ]]; then
    [[ ! -L "$NAS_MODELS/xet" ]] || die "NAS xet cache must not be a symlink"
    logf="${LOG_DIR}/rsync-hf-xet-nas-${STAMP}.log"
    safe_output_file "$logf"
    log "sync NAS xet -> $HF_DST/xet"
    rsync_open "$NAS_MODELS/xet/" "$HF_DST/xet/" | tee -a "$logf"
  fi
  safe_output_file "${LOG_DIR}/hf-sync-complete.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/hf-sync-complete.txt"
  log "NAS HF sync complete"
}

sync_from_host() {
  local host="${1:-}"
  [[ -n "$host" ]] || die "usage: --sync-from-host HOST"
  [[ "$host" != -* && "$host" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
    die "invalid sync host: $host"
  }
  require_ext4t
  layout
  local logf="${LOG_DIR}/rsync-hf-hub-from-${host}-${STAMP}.log"
  safe_output_file "$logf"
  log "sync HF hub from ${host}:/Volumes/Ext4T/huggingface/hub/ -> $HF_DST/hub/"
  # Pull via remote rsync over ssh (openrsync on both sides)
  rsync -aH --partial --progress --stats --exclude '.DS_Store' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "${host}:/Volumes/Ext4T/huggingface/hub/" "$HF_DST/hub/" | tee -a "$logf"
  logf="${LOG_DIR}/rsync-hf-xet-from-${host}-${STAMP}.log"
  safe_output_file "$logf"
  rsync -aH --partial --progress --stats --exclude '.DS_Store' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "${host}:/Volumes/Ext4T/huggingface/xet/" "$HF_DST/xet/" | tee -a "$logf"
  safe_output_file "${LOG_DIR}/hf-sync-complete.txt"
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
    find "$HF_HOME_LOCAL" -mindepth 1 -maxdepth 1 -print | sed -n '1,12p'
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
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
  local cmd="${1:-}"
  validate_configuration
  case "$cmd" in
    -h|--help|"") [[ $# -le 1 ]] || die "unexpected arguments"; usage; exit 0 ;;
    --layout) [[ $# -eq 1 ]] || die "--layout takes no arguments"; layout ;;
    --relink) [[ $# -eq 1 ]] || die "--relink takes no arguments"; relink ;;
    --shell-env) [[ $# -eq 1 ]] || die "--shell-env takes no arguments"; shell_env ;;
    --sync-from-nas) [[ $# -eq 1 ]] || die "--sync-from-nas takes no arguments"; sync_from_nas ;;
    --sync-from-host)
      [[ $# -eq 2 ]] || die "--sync-from-host requires exactly one host"
      sync_from_host "$2"
      ;;
    --status) [[ $# -eq 1 ]] || die "--status takes no arguments"; show_status ;;
    --verify) [[ $# -eq 1 ]] || die "--verify takes no arguments"; verify ;;
    --all)
      [[ $# -eq 1 ]] || die "--all takes no arguments"
      layout
      shell_env
      relink
      verify
      log "layout+env+relink done. Seed hub with --sync-from-nas or --sync-from-host when ready."
      ;;
    *) die "unknown: $cmd (try --help)" ;;
  esac
}

main "$@"
