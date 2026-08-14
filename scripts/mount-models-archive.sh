#!/usr/bin/env bash
# Map DF-NAS01 home/models as the factory model archive.
#
#   SMB:  //devop@DF-NAS01/home   (already mounted as /Volumes/home)
#   Data: /Volumes/home/models    (empty archive root today)
#   Names:
#     /Volumes/models-archive           preferred (needs one-time sudo ln)
#     /Volumes/Ext4T/models-archive     always-writable fallback (no sudo)
#
# Policy: do not delete factory checkpoints to free Ext4T. Move or rsync
# them here, then remove only the Ext4T copy after the NAS copy verifies.
#
# Usage:
#   bash scripts/mount-models-archive.sh          # ensure links
#   bash scripts/mount-models-archive.sh --status
set -euo pipefail

NAS_HOME="${NAS_HOME:-/Volumes/home}"
NAS_MODELS="${NAS_MODELS:-$NAS_HOME/models}"
VOL_LINK="${VOL_LINK:-/Volumes/models-archive}"
EXT_LINK="${EXT_LINK:-/Volumes/Ext4T/models-archive}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

ensure_nas() {
  mount | grep -q 'DF-NAS01.*on /Volumes/home' || {
    die "$NAS_HOME is not the DF-NAS01 SMB home share — mount it in Finder first"
  }
  [[ -d "$NAS_MODELS" ]] || die "missing $NAS_MODELS"
}

link_path() {
  local dest="$1" src="$2"
  if [[ -L "$dest" ]]; then
    local cur
    cur="$(readlink "$dest")"
    if [[ "$cur" == "$src" ]]; then
      log "ok $dest -> $src"
      return 0
    fi
    rm "$dest"
  elif [[ -e "$dest" ]]; then
    die "$dest exists and is not a symlink"
  fi
  ln -sfn "$src" "$dest"
  log "linked $dest -> $src"
}

status() {
  echo "NAS_HOME=$NAS_HOME"
  mount | grep DF-NAS01 || echo "DF-NAS01 not mounted"
  df -h "$NAS_HOME" 2>/dev/null | tail -1 || true
  echo "NAS_MODELS=$NAS_MODELS"
  ls -ld "$NAS_MODELS" 2>/dev/null || echo "missing models dir"
  echo "VOL_LINK=$VOL_LINK"
  ls -ld "$VOL_LINK" 2>/dev/null || echo "missing (run with sudo: sudo ln -sfn $NAS_MODELS $VOL_LINK)"
  echo "EXT_LINK=$EXT_LINK"
  ls -ld "$EXT_LINK" 2>/dev/null || echo "missing"
}

ensure() {
  ensure_nas
  link_path "$EXT_LINK" "$NAS_MODELS"
  if ln -sfn "$NAS_MODELS" "$VOL_LINK" 2>/dev/null; then
    log "ok $VOL_LINK -> $NAS_MODELS"
  elif sudo -n ln -sfn "$NAS_MODELS" "$VOL_LINK" 2>/dev/null; then
    log "ok $VOL_LINK -> $NAS_MODELS (sudo)"
  else
    log "WARN cannot create $VOL_LINK (need: sudo ln -sfn $NAS_MODELS $VOL_LINK)"
    log "use $EXT_LINK until that one-time sudo is done"
  fi
  status
}

case "${1:-ensure}" in
  --status | status) status ;;
  --ensure | ensure | "") ensure ;;
  *) die "unknown arg: $1" ;;
esac
