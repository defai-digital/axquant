#!/usr/bin/env bash
# Migrate Hugging Face cache + Ext4T2 LLM models onto the new Ext4T volume.
#
# Safe defaults:
#   - never deletes sources
#   - never touches Photos libraries on Ext4T2
#   - rsync is restartable (re-run until COMPLETE)
#   - HF relink is gated until hub sync has completed once
#
# Usage:
#   bash scripts/migrate-to-ext4t.sh              # full pipeline (sync + relink when ready)
#   bash scripts/migrate-to-ext4t.sh --sync-only  # rsync only
#   bash scripts/migrate-to-ext4t.sh --relink-only
#   bash scripts/migrate-to-ext4t.sh --status
#   bash scripts/migrate-to-ext4t.sh --verify
set -euo pipefail

EXT_ROOT="${EXT_ROOT:-/Volumes/Ext4T}"
EXT_OLD="${EXT_OLD:-/Volumes/Ext4T2}"
NAS_MODELS="${NAS_MODELS:-/Volumes/data-models/models}"
HF_HOME_LOCAL="${HF_HOME_LOCAL:-${HOME}/.cache/huggingface}"
LOG_DIR="${EXT_ROOT}/logs/migration"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

HF_DST="${EXT_ROOT}/huggingface"
MODELS_DST="${EXT_ROOT}/models"

# Ext4T2 top-level entries that are NOT LLM model dirs
EXCLUDE_OLD_NAMES=(
  "PhotosLibrary.photoslibrary"
  "2nd.photoslibrary"
  "data-models"
  "models"
  ".DS_Store"
  ".fseventsd"
  ".Spotlight-V100"
  ".TemporaryItems"
  ".Trashes"
  ".DocumentRevisions-V100"
  ".VolumeIcon.icns"
)

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

die() { log "ERROR: $*"; exit 1; }

require_mounts() {
  [[ -d "$EXT_ROOT" ]] || die "$EXT_ROOT not mounted"
  mkdir -p "$HF_DST/hub" "$HF_DST/xet" "$HF_DST/datasets" "$MODELS_DST" "$LOG_DIR"
  mkdir -p "$EXT_ROOT/axquant"/{models,axq-publish,logs,work,smokes}
}

is_excluded_old() {
  local name="$1"
  local e
  for e in "${EXCLUDE_OLD_NAMES[@]}"; do
    [[ "$name" == "$e" ]] && return 0
  done
  return 1
}

rsync_flags() {
  # openrsync (macOS stock) is ~2.6.9-compatible: no --info=*, no --partial-dir
  echo -aH --partial --progress --stats
}

sync_hf_from_nas() {
  [[ -d "$NAS_MODELS/hub" ]] || die "NAS hub missing: $NAS_MODELS/hub (is data-models mounted?)"
  local logf="${LOG_DIR}/rsync-hf-hub-${STAMP}.log"
  log "sync HF hub: $NAS_MODELS/hub/ -> $HF_DST/hub/"
  # shellcheck disable=SC2046
  rsync $(rsync_flags) \
    --exclude '.DS_Store' \
    "$NAS_MODELS/hub/" "$HF_DST/hub/" | tee -a "$logf"
  log "sync HF xet: $NAS_MODELS/xet/ -> $HF_DST/xet/"
  logf="${LOG_DIR}/rsync-hf-xet-${STAMP}.log"
  # shellcheck disable=SC2046
  rsync $(rsync_flags) \
    --exclude '.DS_Store' \
    "$NAS_MODELS/xet/" "$HF_DST/xet/" | tee -a "$logf"

  # Local-only HF metadata / small caches
  if [[ -d "${HF_HOME_LOCAL}/datasets" && ! -L "${HF_HOME_LOCAL}/datasets" ]]; then
    log "sync local datasets -> $HF_DST/datasets/"
    # shellcheck disable=SC2046
    rsync $(rsync_flags) "${HF_HOME_LOCAL}/datasets/" "$HF_DST/datasets/" \
      | tee -a "${LOG_DIR}/rsync-hf-datasets-${STAMP}.log"
  fi
  for f in token stored_tokens .agent_harnesses.json .check_for_update_done CACHEDIR.TAG; do
    if [[ -e "${HF_HOME_LOCAL}/$f" && ! -e "${HF_DST}/$f" ]]; then
      cp -a "${HF_HOME_LOCAL}/$f" "${HF_DST}/$f"
      log "copied HF meta $f"
    fi
  done

  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/hf-sync-complete.txt"
  log "HF NAS sync complete -> ${LOG_DIR}/hf-sync-complete.txt"
}

sync_llm_from_ext4t2() {
  [[ -d "$EXT_OLD" ]] || die "$EXT_OLD not mounted"
  local name src logf
  local count=0
  for src in "$EXT_OLD"/*; do
    [[ -e "$src" ]] || continue
    name="$(basename "$src")"
    is_excluded_old "$name" && continue
    [[ -L "$src" ]] && { log "skip symlink $name"; continue; }
    [[ -d "$src" ]] || continue
    count=$((count + 1))
    logf="${LOG_DIR}/rsync-llm-${name}-${STAMP}.log"
    log "sync LLM [$count]: $name"
    # shellcheck disable=SC2046
    rsync $(rsync_flags) \
      --exclude '.DS_Store' \
      "$src/" "$MODELS_DST/$name/" | tee -a "$logf"
  done
  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/llm-sync-complete.txt"
  log "LLM Ext4T2 sync complete ($count dirs) -> ${LOG_DIR}/llm-sync-complete.txt"
}

relink_hf() {
  require_mounts
  [[ -d "$HF_DST/hub" ]] || die "missing $HF_DST/hub — run sync first"
  # Require at least one hub model entry so we do not point at an empty cache by accident
  local n
  n="$(find "$HF_DST/hub" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
  if [[ "${n:-0}" -lt 5 ]]; then
    die "HF hub on Ext4T looks empty ($n entries). Finish sync before --relink-only."
  fi

  # Ensure tokens exist on destination
  for f in token stored_tokens; do
    if [[ -e "${HF_HOME_LOCAL}/$f" && ! -e "${HF_DST}/$f" ]]; then
      cp -a "${HF_HOME_LOCAL}/$f" "${HF_DST}/$f"
    fi
  done

  mkdir -p "${HOME}/.cache"
  local backup=""
  if [[ -L "$HF_HOME_LOCAL" ]]; then
    local cur
    cur="$(readlink "$HF_HOME_LOCAL")"
    if [[ "$cur" == "$HF_DST" ]]; then
      log "HF already linked: $HF_HOME_LOCAL -> $cur"
      return 0
    fi
    log "replacing existing HF symlink $cur -> $HF_DST"
    rm "$HF_HOME_LOCAL"
  elif [[ -d "$HF_HOME_LOCAL" ]]; then
    backup="${HF_HOME_LOCAL}.pre-ext4t-${STAMP}"
    log "backing up $HF_HOME_LOCAL -> $backup"
    mv "$HF_HOME_LOCAL" "$backup"
  fi

  ln -sfn "$HF_DST" "$HF_HOME_LOCAL"
  log "linked $HF_HOME_LOCAL -> $HF_DST"

  # Convenience: ~/models already points at ~/.cache/huggingface/hub for many tools
  if [[ -L "${HOME}/models" ]]; then
    log "~/models -> $(readlink "${HOME}/models")"
  fi

  # Ext4T-side convenience links (mirror old Ext4T2 pattern, but local)
  ln -sfn "$HF_DST" "${EXT_ROOT}/data-models-hf"
  ln -sfn "$MODELS_DST" "${EXT_ROOT}/llm-models"

  {
    echo "relinked_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hf_home_local=$HF_HOME_LOCAL"
    echo "hf_dst=$HF_DST"
    echo "backup=${backup:-none}"
    echo "hub_entries=$n"
  } >"${LOG_DIR}/hf-relink.txt"
  log "relink recorded -> ${LOG_DIR}/hf-relink.txt"
}

show_status() {
  echo "=== mounts ==="
  df -h "$EXT_ROOT" 2>/dev/null || echo "Ext4T not mounted"
  df -h "$EXT_OLD" 2>/dev/null || echo "Ext4T2 not mounted"
  df -h /Volumes/data-models 2>/dev/null || echo "data-models not mounted"
  echo
  echo "=== Ext4T layout ==="
  ls -la "$EXT_ROOT" 2>/dev/null || true
  echo
  echo "=== HF link ==="
  if [[ -L "$HF_HOME_LOCAL" ]]; then
    echo "$HF_HOME_LOCAL -> $(readlink "$HF_HOME_LOCAL")"
  elif [[ -d "$HF_HOME_LOCAL" ]]; then
    echo "$HF_HOME_LOCAL is a real directory"
    ls -la "$HF_HOME_LOCAL" | head -20
  else
    echo "$HF_HOME_LOCAL missing"
  fi
  echo
  echo "=== sync markers ==="
  for f in hf-sync-complete.txt llm-sync-complete.txt hf-relink.txt; do
    if [[ -f "${LOG_DIR}/$f" ]]; then
      echo "$f: $(cat "${LOG_DIR}/$f")"
    else
      echo "$f: (pending)"
    fi
  done
  echo
  echo "=== rough sizes (may be slow) ==="
  du -sh "$HF_DST/hub" "$HF_DST/xet" "$MODELS_DST" 2>/dev/null || true
}

verify_layout() {
  require_mounts
  local ok=1
  echo "=== verify ==="
  if [[ -L "$HF_HOME_LOCAL" && "$(readlink "$HF_HOME_LOCAL")" == "$HF_DST" ]]; then
    echo "OK HF symlink"
  else
    echo "FAIL HF symlink (want $HF_HOME_LOCAL -> $HF_DST)"
    ok=0
  fi
  if [[ -d "$HF_DST/hub" ]]; then
    local n
    n="$(find "$HF_DST/hub" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
    echo "OK hub entries=$n"
    [[ "$n" -ge 5 ]] || { echo "FAIL hub too sparse"; ok=0; }
  else
    echo "FAIL hub missing"; ok=0
  fi
  if [[ -d "$MODELS_DST" ]]; then
    local m
    m="$(find "$MODELS_DST" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    echo "OK local LLM model dirs=$m"
  fi
  # Spot-check: hub should resolve through the symlink
  if [[ -e "$HF_HOME_LOCAL/hub" ]]; then
    echo "OK $HF_HOME_LOCAL/hub resolves"
  else
    echo "FAIL hub path via HF home"; ok=0
  fi
  if [[ "$ok" -eq 1 ]]; then
    echo "VERIFY_PASS"
    return 0
  fi
  echo "VERIFY_FAIL"
  return 1
}

main() {
  local mode="${1:-all}"
  case "$mode" in
    -h|--help) usage; exit 0 ;;
    --status) show_status; exit 0 ;;
    --verify) verify_layout; exit 0 ;;
    --sync-only)
      require_mounts
      sync_hf_from_nas
      sync_llm_from_ext4t2
      ;;
    --sync-hf)
      require_mounts
      sync_hf_from_nas
      ;;
    --sync-llm)
      require_mounts
      sync_llm_from_ext4t2
      ;;
    --relink-only)
      relink_hf
      ;;
    all|"")
      require_mounts
      log "starting full migration onto $EXT_ROOT"
      sync_hf_from_nas
      sync_llm_from_ext4t2
      relink_hf
      verify_layout
      log "done"
      ;;
    *)
      die "unknown mode: $mode (try --help)"
      ;;
  esac
}

main "${1:-all}"
