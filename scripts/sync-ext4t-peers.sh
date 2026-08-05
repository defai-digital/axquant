#!/usr/bin/env bash
# shellcheck disable=SC2029
# Bidirectional union-sync of Ext4T between factory Macs (M3 <-> M5).
#
# Goal: both machines end with the same content under:
#   /Volumes/Ext4T/{huggingface,models,axquant}
#
# Strategy (safe union — no deletes):
#   1) push local -> peer
#   2) pull peer  -> local
# Re-run anytime; rsync is restartable.
#
# Usage (run on either machine that can SSH to the peer):
#   bash scripts/sync-ext4t-peers.sh --peer mbp-m5
#   bash scripts/sync-ext4t-peers.sh --peer mbp-m5 --status
#   bash scripts/sync-ext4t-peers.sh --peer mbp-m5 --models-only
#   bash scripts/sync-ext4t-peers.sh --peer mbp-m5 --hf-only
set -euo pipefail

EXT_ROOT="${EXT_ROOT:-/Volumes/Ext4T}"
PEER="${PEER:-mbp-m5}"
PEER_EXT="${PEER_EXT:-/Volumes/Ext4T}"
LOG_DIR="${EXT_ROOT}/logs/peer-sync"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60}"
declare -a SSH_ARGS
read -r -a SSH_ARGS <<<"$SSH_OPTS"
RSYNC_SSH="ssh"
# ${arr[@]+...} guard: expanding an empty array (SSH_OPTS="") crashes
# bash 3.2 under set -u
for ssh_arg in ${SSH_ARGS[@]+"${SSH_ARGS[@]}"}; do
  printf -v RSYNC_SSH '%s %q' "$RSYNC_SSH" "$ssh_arg"
done

# Trees to keep identical (relative to Ext4T)
TREES=(huggingface models axquant)

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

validate_configuration() {
  [[ "$PEER" != -* && "$PEER" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
    die "invalid peer name: $PEER"
  }
  for path in "$EXT_ROOT" "$PEER_EXT"; do
    [[ "$path" == /* && "$path" =~ ^/[A-Za-z0-9._/-]+$ && "$path" != */ ]] || {
      die "sync roots must be simple absolute paths: $path"
    }
    [[ "$path" != "/" && "$path" != "/Volumes" ]] || {
      die "refusing to use a broad sync root: $path"
    }
    [[ "/$path/" != *"/../"* && "/$path/" != *"/./"* && "$path" != *"//"* ]] || {
      die "sync roots must be canonical: $path"
    }
  done
}

ensure_real_dir() {
  local path="$1"
  [[ ! -L "$path" ]] || die "managed directory must not be a symlink: $path"
  [[ ! -e "$path" || -d "$path" ]] || die "managed path is not a directory: $path"
  mkdir -p "$path"
}

safe_output_file() {
  local path="$1"
  [[ ! -L "$path" ]] || die "refusing symlinked output file: $path"
  [[ ! -e "$path" || -f "$path" ]] || die "output path is not a regular file: $path"
}

rsync_base() {
  # openrsync-compatible; no --delete (union sync)
  rsync -aH --partial --progress --stats \
    --exclude '.DS_Store' \
    --exclude '.rsync-partial' \
    -e "$RSYNC_SSH" \
    "$@"
}

require() {
  [[ -d "$EXT_ROOT" && ! -L "$EXT_ROOT" ]] || die "$EXT_ROOT is not a real mounted directory"
  local mounted_at
  mounted_at="$(df -P "$EXT_ROOT" 2>/dev/null | awk 'NR == 2 {print $NF}')"
  [[ "$mounted_at" == "$EXT_ROOT" ]] || {
    die "$EXT_ROOT is not a mounted volume (df reports ${mounted_at:-unknown})"
  }
  if ! ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" bash -s -- "$PEER_EXT" <<'REMOTE'
set -eu
peer_ext="$1"
test -d "$peer_ext"
test ! -L "$peer_ext"
mounted_at="$(df -P "$peer_ext" | awk 'NR == 2 {print $NF}')"
test "$mounted_at" = "$peer_ext"
REMOTE
  then
    die "peer $PEER is missing a real exact mount at $PEER_EXT"
  fi
  ensure_real_dir "${EXT_ROOT}/logs"
  ensure_real_dir "$LOG_DIR"
}

ensure_tree_roots() {
  local tree="$1"
  ensure_real_dir "${EXT_ROOT}/${tree}"
  if ! ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" bash -s -- "$PEER_EXT" "$tree" <<'REMOTE'
set -eu
peer_ext="$1"
tree="$2"
path="$peer_ext/$tree"
test ! -L "$path"
if test -e "$path"; then
  test -d "$path"
else
  mkdir "$path"
fi
REMOTE
  then
    die "peer tree is missing, symlinked, or not a directory: ${PEER_EXT}/${tree}"
  fi
}

push_tree() {
  local tree="$1"
  local src="${EXT_ROOT}/${tree}/"
  local dst="${PEER}:${PEER_EXT}/${tree}/"
  log "PUSH $tree -> $PEER"
  local logf="${LOG_DIR}/push-${tree}-${STAMP}.log"
  safe_output_file "$logf"
  rsync_base --ignore-existing "$src" "$dst" | tee -a "$logf"
  log "PUSH done $tree"
}

pull_tree() {
  local tree="$1"
  local src="${PEER}:${PEER_EXT}/${tree}/"
  local dst="${EXT_ROOT}/${tree}/"
  log "PULL $tree <- $PEER"
  local logf="${LOG_DIR}/pull-${tree}-${STAMP}.log"
  safe_output_file "$logf"
  rsync_base --ignore-existing "$src" "$dst" | tee -a "$logf"
  log "PULL done $tree"
}

sync_tree() {
  local tree="$1"
  ensure_tree_roots "$tree"
  push_tree "$tree"
  pull_tree "$tree"
  local differences
  # Content-only verify: the union copy passes use --ignore-existing, which
  # never aligns times/perms on pre-existing files, so attribute-only itemize
  # lines (leading '.') would make this fail forever on byte-identical trees
  # (e.g. the same model downloaded on both hosts). Content differences
  # ('>', '<', 'c', 'h') and '*deleting' stay fatal.
  differences="$(
    rsync -rlHnc --delete --itemize-changes \
      --exclude '.DS_Store' \
      --exclude '.rsync-partial' \
      -e "$RSYNC_SSH" \
      "${EXT_ROOT}/${tree}/" "${PEER}:${PEER_EXT}/${tree}/" \
      | grep -Ev '^\.' || true
  )"
  [[ -z "$differences" ]] || {
    echo "$differences" >&2
    die "$tree differs between local and peer after union sync; resolve conflicts explicitly"
  }
}

ensure_local_link() {
  local target="$1"
  local link="$2"
  if [[ -L "$link" ]]; then
    [[ "$(readlink "$link")" == "$target" ]] || {
      die "$link points somewhere unexpected"
    }
  elif [[ -e "$link" ]]; then
    die "$link exists and is not a symlink"
  else
    ln -s "$target" "$link"
  fi
}

ensure_peer_link() {
  local target="$1"
  local link="$2"
  ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" \
    "if [ -L '$link' ]; then [ \"\$(readlink '$link')\" = '$target' ]; elif [ -e '$link' ]; then exit 1; else ln -s '$target' '$link'; fi" \
    || die "peer convenience link is missing or unsafe: $link"
}

sync_docs() {
  # Copy only missing root docs. Conflicts fail closed instead of silently
  # choosing one machine's version.
  for f in README-LAYOUT.txt HF-STANDARD.txt; do
    local local_kind="missing"
    if [[ -L "${EXT_ROOT}/$f" ]]; then
      die "root document must not be a symlink: ${EXT_ROOT}/$f"
    elif [[ -f "${EXT_ROOT}/$f" ]]; then
      local_kind="file"
    elif [[ -e "${EXT_ROOT}/$f" ]]; then
      die "root document path is not a regular file: ${EXT_ROOT}/$f"
    fi
    local remote_kind
    remote_kind="$(
      ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" bash -s -- "$PEER_EXT" "$f" <<'REMOTE'
path="$1/$2"
if test -L "$path"; then
  echo symlink
elif test -f "$path"; then
  echo file
elif test -e "$path"; then
  echo other
else
  echo missing
fi
REMOTE
    )"
    [[ "$remote_kind" != "symlink" && "$remote_kind" != "other" ]] || {
      die "peer root document path is unsafe: ${PEER_EXT}/$f ($remote_kind)"
    }
    if [[ "$local_kind" == "file" ]]; then
      rsync -aH --ignore-existing -e "$RSYNC_SSH" \
        "${EXT_ROOT}/$f" "${PEER}:${PEER_EXT}/$f"
    fi
    if [[ "$remote_kind" == "file" ]]; then
      rsync -aH --ignore-existing -e "$RSYNC_SSH" \
        "${PEER}:${PEER_EXT}/$f" "${EXT_ROOT}/$f"
    fi
    if [[ -f "${EXT_ROOT}/$f" ]]; then
      local local_sha peer_sha
      local_sha="$(shasum -a 256 "${EXT_ROOT}/$f" | awk '{print $1}')"
      peer_sha="$(
        ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" "shasum -a 256 '${PEER_EXT}/$f' | awk '{print \$1}'"
      )"
      [[ "$local_sha" == "$peer_sha" ]] || {
        die "root document differs between local and peer: $f"
      }
    fi
  done
  ensure_peer_link "${PEER_EXT}/huggingface" "${PEER_EXT}/data-models-hf"
  ensure_peer_link "${PEER_EXT}/models" "${PEER_EXT}/llm-models"
  ensure_local_link "${EXT_ROOT}/huggingface" "${EXT_ROOT}/data-models-hf"
  ensure_local_link "${EXT_ROOT}/models" "${EXT_ROOT}/llm-models"
}

show_status() {
  echo "=== local $(scutil --get ComputerName 2>/dev/null || hostname) ==="
  df -h "$EXT_ROOT" | tail -1
  for t in "${TREES[@]}"; do
    if [[ -d "${EXT_ROOT}/$t" ]]; then
      printf '  %-12s %s  entries=%s\n' "$t" "$(du -sh "${EXT_ROOT}/$t" 2>/dev/null | cut -f1)" \
        "$(find "${EXT_ROOT}/$t" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
    else
      printf '  %-12s MISSING\n' "$t"
    fi
  done
  echo
  echo "=== peer $PEER ==="
  ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" bash -s -- "$PEER_EXT" <<'REMOTE'
peer_ext="$1"
df -h "$peer_ext" | tail -1
for t in huggingface models axquant; do
  if [ -d "$peer_ext/$t" ]; then
    printf '  %-12s %s  entries=%s\n' "$t" \
      "$(du -sh "$peer_ext/$t" 2>/dev/null | cut -f1)" \
      "$(find "$peer_ext/$t" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
  else
    printf '  %-12s MISSING\n' "$t"
  fi
done
REMOTE
  echo
  echo "=== recent peer-sync logs ==="
  find "$LOG_DIR" -mindepth 1 -maxdepth 1 -type f -print 2>/dev/null | sed -n '1,15p'
  if [[ -f "${LOG_DIR}/last-complete.txt" ]]; then
    echo "last-complete: $(cat "${LOG_DIR}/last-complete.txt")"
  else
    echo "last-complete: (none)"
  fi
}

compare_counts() {
  log "compare top-level entry counts"
  local mismatch=0
  local -a trees=("$@")
  [[ "${#trees[@]}" -gt 0 ]] || trees=("${TREES[@]}")
  for t in "${trees[@]}"; do
    local l r
    if [[ ! -d "${EXT_ROOT}/$t" || -L "${EXT_ROOT}/$t" ]]; then
      echo "DIFF $t missing locally"
      mismatch=1
      continue
    fi
    if ! ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" \
      "test -d '${PEER_EXT}/$t' && test ! -L '${PEER_EXT}/$t'"; then
      echo "DIFF $t missing on peer"
      mismatch=1
      continue
    fi
    l="$(find "${EXT_ROOT}/$t" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
    r="$(ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$PEER" "find '${PEER_EXT}/$t' -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' '")"
    if [[ "$l" == "$r" ]]; then
      echo "OK  $t local=$l peer=$r"
    else
      echo "DIFF $t local=$l peer=$r"
      mismatch=1
    fi
  done
  [[ "$mismatch" -eq 0 ]] || return 1
}

run_all() {
  require
  log "peer-sync start local=$(hostname) peer=$PEER stamp=$STAMP"
  # Models first (largest unique delta on M3), then HF, then axquant
  for t in models huggingface axquant; do
    sync_tree "$t"
  done
  sync_docs
  compare_counts
  safe_output_file "${LOG_DIR}/last-complete.txt"
  safe_output_file "${LOG_DIR}/complete-${STAMP}.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${LOG_DIR}/last-complete.txt"
  {
    echo "completed_at=$(cat "${LOG_DIR}/last-complete.txt")"
    echo "peer=$PEER"
    echo "stamp=$STAMP"
  } | tee "${LOG_DIR}/complete-${STAMP}.txt"
  log "peer-sync complete"
}

main() {
  local mode="all"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --peer)
        [[ $# -ge 2 && -n "$2" ]] || die "--peer requires a host"
        PEER="$2"
        shift 2
        ;;
      --status) mode=status; shift ;;
      --models-only) mode=models; shift ;;
      --hf-only) mode=hf; shift ;;
      --compare) mode=compare; shift ;;
      -h|--help)
        sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
      *) die "unknown arg: $1" ;;
    esac
  done

  validate_configuration
  case "$mode" in
    status) require; show_status ;;
    compare) require; compare_counts ;;
    models) require; sync_tree models; sync_docs; compare_counts models ;;
    hf) require; sync_tree huggingface; compare_counts huggingface ;;
    all) run_all ;;
  esac
}

main "$@"
