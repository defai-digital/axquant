#!/usr/bin/env bash
# Sync Ext4T Hugging Face cache across factory Macs (union, no deletes).
#
# Canonical layout on every machine:
#   /Volumes/Ext4T/huggingface/{hub,xet,datasets,...}
#   ~/.cache/huggingface -> /Volumes/Ext4T/huggingface
#   HF_HOME=/Volumes/Ext4T/huggingface
#
# Peers (SSH host aliases):
#   mbp-m5          (AKMBPM5MAX, user akiralam)
#   macstudio-m2u   (M2 Ultra, user devop) — HF path may have been hf-data; this
#                   script standardizes to huggingface
#
# Usage (run from a machine that can SSH to peers, usually M3):
#   bash scripts/sync-ext4t-hf-fleet.sh --status
#   bash scripts/sync-ext4t-hf-fleet.sh --prepare-m2u
#   bash scripts/sync-ext4t-hf-fleet.sh --push-all          # local HF -> all peers
#   bash scripts/sync-ext4t-hf-fleet.sh --pull-all          # all peers -> local
#   bash scripts/sync-ext4t-hf-fleet.sh --sync-all          # push then pull (union)
#   bash scripts/sync-ext4t-hf-fleet.sh --peer mbp-m5 --push
set -euo pipefail

EXT_ROOT="${EXT_ROOT:-/Volumes/Ext4T}"
HF_LOCAL="${EXT_ROOT}/huggingface"
LOG_DIR="${EXT_ROOT}/logs/hf-fleet"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60}"

# Default fleet (override with PEERS="host1 host2")
PEERS_DEFAULT=(mbp-m5 macstudio-m2u)
PEERS_SPEC="${PEERS-}"
declare -a PEERS=()
if [[ -n "$PEERS_SPEC" ]]; then
  read -r -a PEERS <<<"$PEERS_SPEC"
else
  PEERS=("${PEERS_DEFAULT[@]}")
fi
declare -a SSH_ARGS=()
read -r -a SSH_ARGS <<<"$SSH_OPTS"
RSYNC_SSH="ssh"
# ${arr[@]+...} guard: expanding an empty array (SSH_OPTS="") crashes
# bash 3.2 under set -u
for ssh_arg in ${SSH_ARGS[@]+"${SSH_ARGS[@]}"}; do
  printf -v RSYNC_SSH '%s %q' "$RSYNC_SSH" "$ssh_arg"
done

# Per-host remote Ext4T HF path (after prepare, all use huggingface)
remote_hf_path() {
  # Always target the standard path; prepare-m2u creates it.
  echo "/Volumes/Ext4T/huggingface"
}

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

validate_peer() {
  local peer="$1"
  [[ "$peer" != -* && "$peer" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
    die "invalid peer name: $peer"
  }
}

validate_configuration() {
  [[ "$EXT_ROOT" == /* && "$EXT_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$EXT_ROOT" != */ ]] || {
    die "EXT_ROOT must be a simple canonical absolute path: $EXT_ROOT"
  }
  [[ "$EXT_ROOT" != "/" && "$EXT_ROOT" != "/Volumes" ]] || {
    die "refusing broad Ext4T root: $EXT_ROOT"
  }
  [[ "/$EXT_ROOT/" != *"/../"* && "/$EXT_ROOT/" != *"/./"* && "$EXT_ROOT" != *"//"* ]] || {
    die "EXT_ROOT must not contain dot or empty components: $EXT_ROOT"
  }
  [[ "${#PEERS[@]}" -gt 0 ]] || die "peer list must not be empty"
  local peer
  for peer in "${PEERS[@]}"; do
    validate_peer "$peer"
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

require_local() {
  [[ -d "$EXT_ROOT" ]] || die "$EXT_ROOT not mounted"
  [[ ! -L "$EXT_ROOT" ]] || die "$EXT_ROOT must not be a symlink"
  local mounted_at
  mounted_at="$(df -P "$EXT_ROOT" 2>/dev/null | awk 'NR == 2 {print $NF}')"
  [[ "$mounted_at" == "$EXT_ROOT" ]] || {
    die "$EXT_ROOT is not an exact mounted volume (df reports ${mounted_at:-unknown})"
  }
  [[ -d "$HF_LOCAL" && ! -L "$HF_LOCAL" ]] || {
    die "missing or symlinked $HF_LOCAL — run setup-ext4t-hf.sh --layout first"
  }
  local path
  for path in "$HF_LOCAL/hub" "$HF_LOCAL/xet" "$HF_LOCAL/datasets"; do
    ensure_real_dir "$path"
  done
  ensure_real_dir "${EXT_ROOT}/logs"
  ensure_real_dir "$LOG_DIR"
}

ssh_peer() {
  local peer="$1"
  shift
  # Arguments are fixed command tokens in this script; peer names are validated.
  # shellcheck disable=SC2029
  ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$peer" "$@"
}

require_peer_ext4t() {
  local peer="$1"
  if ! ssh_peer "$peer" bash -s <<'REMOTE'
set -eu
ext=/Volumes/Ext4T
test -d "$ext"
test ! -L "$ext"
mounted_at="$(df -P "$ext" | awk 'NR == 2 {print $NF}')"
test "$mounted_at" = "$ext"
REMOTE
  then
    die "$peer is missing a real exact mount at /Volumes/Ext4T"
  fi
}

ensure_peer_hf() {
  local peer="$1"
  local mode="${2:-require}"
  require_peer_ext4t "$peer"
  if ! ssh_peer "$peer" bash -s -- "$mode" <<'REMOTE'
set -eu
mode="$1"
hf=/Volumes/Ext4T/huggingface
if test -L "$hf" || { test -e "$hf" && test ! -d "$hf"; }; then
  exit 1
fi
if test ! -d "$hf"; then
  test "$mode" = create
  mkdir "$hf"
fi
for name in hub xet datasets; do
  path="$hf/$name"
  test ! -L "$path"
  if test -e "$path"; then
    test -d "$path"
  else
    test "$mode" = create
    mkdir "$path"
  fi
done
REMOTE
  then
    die "$peer has an unsafe Hugging Face cache path"
  fi
}

rsync_to() {
  local src="$1" dst="$2" logf="$3"
  safe_output_file "$logf"
  rsync -aH --partial --progress --stats --ignore-existing \
    --exclude '.DS_Store' \
    --exclude '.rsync-partial' \
    -e "$RSYNC_SSH" \
    "$src" "$dst" | tee -a "$logf"
}

verify_transfer() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local differences
  # Content-only verify: --ignore-existing transfers never align times/perms
  # on files present on both sides, so attribute-only itemize lines (leading
  # '.') must not fail the verify; content lines ('>', '<', 'c', 'h',
  # '*deleting') stay fatal.
  differences="$(
    rsync -rlHnc --itemize-changes \
      --exclude '.DS_Store' \
      --exclude '.rsync-partial' \
      -e "$RSYNC_SSH" \
      "$src" "$dst" \
      | grep -Ev '^\.' || true
  )"
  [[ -z "$differences" ]] || {
    echo "$differences" >&2
    die "$label left missing or conflicting source entries"
  }
}

verify_peer() {
  local peer="$1"
  local remote
  remote="$(remote_hf_path "$peer")"
  local differences
  # Content-only comparison: attribute-only itemize lines (leading '.') are
  # expected after --ignore-existing union passes and must not fail the
  # verify; content lines and '*deleting' stay fatal.
  differences="$(
    rsync -rlHnc --delete --itemize-changes \
      --exclude '.DS_Store' \
      --exclude '.rsync-partial' \
      -e "$RSYNC_SSH" \
      "${HF_LOCAL}/" "${peer}:${remote}/" \
      | grep -Ev '^\.' || true
  )"
  [[ -z "$differences" ]] || {
    echo "$differences" >&2
    die "HF cache differs between local and $peer after union sync; resolve conflicts explicitly"
  }
}

# --- M2U: migrate hf-data -> huggingface and align links ---
prepare_m2u() {
  local peer="macstudio-m2u"
  validate_peer "$peer"
  log "prepare $peer Ext4T HF standard path"
  ssh_peer "$peer" bash -s <<'REMOTE'
set -euo pipefail
EXT=/Volumes/Ext4T
HF="$EXT/huggingface"
OLD="$EXT/hf-data"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ensure_real_dir() {
  local path="$1"
  [[ ! -L "$path" ]] || die "managed directory must not be a symlink: $path"
  [[ ! -e "$path" || -d "$path" ]] || die "managed path is not a directory: $path"
  mkdir -p "$path"
}

ensure_link() {
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

merge_and_backup() {
  local source="$1"
  local destination="$2"
  local stamp backup differences
  rsync -aH --partial --ignore-existing --exclude '.DS_Store' "$source/" "$destination/"
  # Content-only comparison: --ignore-existing leaves attribute differences on
  # files present on both sides; only content itemize lines are conflicts.
  differences="$(
    rsync -rlHnc --itemize-changes --exclude '.DS_Store' "$source/" "$destination/" \
      | grep -Ev '^\.' || true
  )"
  [[ -z "$differences" ]] || {
    echo "$differences" >&2
    die "content conflict while merging $source into $destination"
  }
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="${source}.pre-standard-${stamp}"
  [[ ! -e "$backup" && ! -L "$backup" ]] || die "backup path already exists: $backup"
  mv "$source" "$backup"
  echo "preserved migrated source at $backup"
}

[[ -d "$EXT" && ! -L "$EXT" ]] || die "$EXT is not a real directory"
mounted_at="$(df -P "$EXT" 2>/dev/null | awk 'NR == 2 {print $NF}')"
[[ "$mounted_at" == "$EXT" ]] || {
  die "$EXT is not an exact mount point (df reports ${mounted_at:-unknown})"
}

# Prefer the existing real hf-data tree when the canonical path is missing.
[[ ! -L "$HF" ]] || die "$HF must not be a symlink"
if [[ ! -e "$HF" && -d "$OLD" && ! -L "$OLD" ]]; then
  echo "rename hf-data -> huggingface"
  mv "$OLD" "$HF"
elif [[ -e "$HF" && ! -d "$HF" ]]; then
  die "$HF exists and is not a directory"
else
  ensure_real_dir "$HF"
fi

if [[ -d "$OLD" && ! -L "$OLD" ]]; then
  echo "merge hf-data into huggingface (union)"
  merge_and_backup "$OLD" "$HF"
elif [[ -L "$OLD" ]]; then
  [[ "$(readlink "$OLD")" == "$HF" ]] || die "$OLD points somewhere unexpected"
elif [[ -e "$OLD" ]]; then
  die "$OLD exists and is not a directory or symlink"
fi

for path in "$HF/hub" "$HF/xet" "$HF/datasets" "$EXT/logs" "$EXT/logs/hf-fleet" \
  "$EXT/models" "$EXT/axquant"; do
  ensure_real_dir "$path"
done
ensure_link "$HF" "$OLD"
ensure_link "$HF" "$EXT/data-models-hf"

# Relink user HF home (devop)
HF_HOME_LOCAL="${HOME}/.cache/huggingface"
ensure_real_dir "${HOME}/.cache"
if [[ -L "$HF_HOME_LOCAL" ]]; then
  [[ "$(readlink "$HF_HOME_LOCAL")" == "$HF" ]] || {
    die "$HF_HOME_LOCAL points somewhere unexpected"
  }
  echo "HF already linked to $HF"
elif [[ -d "$HF_HOME_LOCAL" ]]; then
  echo "merge existing local HF cache into $HF"
  merge_and_backup "$HF_HOME_LOCAL" "$HF"
  ln -s "$HF" "$HF_HOME_LOCAL"
elif [[ -e "$HF_HOME_LOCAL" ]]; then
  die "$HF_HOME_LOCAL exists and is not a directory or symlink"
else
  ln -s "$HF" "$HF_HOME_LOCAL"
fi

# Shell env block
ZSHRC="${HOME}/.zshrc"
MARKER_BEGIN="# >>> axquant-ext4t-hf >>>"
MARKER_END="# <<< axquant-ext4t-hf <<<"
[[ ! -L "$ZSHRC" ]] || die "refusing to rewrite symlinked shell configuration: $ZSHRC"
[[ ! -e "$ZSHRC" || -f "$ZSHRC" ]] || die "shell configuration is not a file: $ZSHRC"
touch "$ZSHRC"
begin_count="$(awk -v marker="$MARKER_BEGIN" '$0 == marker {n++} END {print n + 0}' "$ZSHRC")"
end_count="$(awk -v marker="$MARKER_END" '$0 == marker {n++} END {print n + 0}' "$ZSHRC")"
[[ "$begin_count" -eq "$end_count" && "$begin_count" -le 1 ]] || {
  die "malformed or duplicate HF_HOME marker block in $ZSHRC"
}
tmp="$(mktemp "${ZSHRC}.axquant.XXXXXX")"
if [[ "$begin_count" -eq 1 ]]; then
  begin_line="$(awk -v marker="$MARKER_BEGIN" '$0 == marker {print NR}' "$ZSHRC")"
  end_line="$(awk -v marker="$MARKER_END" '$0 == marker {print NR}' "$ZSHRC")"
  [[ "$begin_line" -lt "$end_line" ]] || {
    rm -f "$tmp"
    die "out-of-order HF_HOME marker block in $ZSHRC"
  }
  awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0==b{skip=1;next} $0==e{skip=0;next} !skip{print}
  ' "$ZSHRC" >"$tmp"
else
  cp "$ZSHRC" "$tmp"
fi
# a config without a trailing newline would glue the begin marker onto the
# user's last line and break both that line and future marker matching
[[ ! -s "$tmp" ]] || [[ -z "$(tail -c1 "$tmp")" ]] || echo >>"$tmp"
cat >>"$tmp" <<'EOF'
# >>> axquant-ext4t-hf >>>
# Local Ext4T Hugging Face cache (shared standard: M2U / M3 / M5)
export HF_HOME="/Volumes/Ext4T/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE="$HF_HOME/xet"
unset HF_HUB_ENABLE_HF_TRANSFER
# <<< axquant-ext4t-hf <<<
EOF
chmod "$(stat -f '%Lp' "$ZSHRC")" "$tmp"
mv "$tmp" "$ZSHRC"

echo "HF_LINK=$(readlink "$HF_HOME_LOCAL")"
echo "HUB_ENTRIES=$(find "$HF/hub" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
df -h "$EXT" | tail -1
ls -la "$EXT" | sed -n '1,20p'
REMOTE
  log "prepare-m2u done"
}

push_peer() {
  local peer="$1"
  validate_peer "$peer"
  local remote
  remote="$(remote_hf_path "$peer")"
  require_local
  ensure_peer_hf "$peer" create
  log "PUSH HF -> $peer:$remote"
  local logf="${LOG_DIR}/push-${peer//\//_}-${STAMP}.log"
  rsync_to "${HF_LOCAL}/" "${peer}:${remote}/" "$logf"
  verify_transfer "${HF_LOCAL}/" "${peer}:${remote}/" "push to $peer"
  log "PUSH done $peer"
}

pull_peer() {
  local peer="$1"
  validate_peer "$peer"
  local remote
  remote="$(remote_hf_path "$peer")"
  require_local
  ensure_peer_hf "$peer"
  log "PULL HF <- $peer:$remote"
  local logf="${LOG_DIR}/pull-${peer//\//_}-${STAMP}.log"
  rsync_to "${peer}:${remote}/" "${HF_LOCAL}/" "$logf"
  verify_transfer "${peer}:${remote}/" "${HF_LOCAL}/" "pull from $peer"
  log "PULL done $peer"
}

push_all() {
  require_local
  local p
  for p in "${PEERS[@]}"; do
    push_peer "$p"
  done
  safe_output_file "${LOG_DIR}/push-all-complete-${STAMP}.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "${LOG_DIR}/push-all-complete-${STAMP}.txt"
}

pull_all() {
  require_local
  local p
  for p in "${PEERS[@]}"; do
    pull_peer "$p"
  done
  safe_output_file "${LOG_DIR}/pull-all-complete-${STAMP}.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "${LOG_DIR}/pull-all-complete-${STAMP}.txt"
}

sync_all() {
  require_local
  log "fleet union sync start peers=${PEERS[*]}"
  # Pull first to absorb unique peer content, then push full union out
  pull_all
  push_all
  local p
  for p in "${PEERS[@]}"; do
    verify_peer "$p"
  done
  safe_output_file "${LOG_DIR}/sync-all-complete-${STAMP}.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "${LOG_DIR}/sync-all-complete-${STAMP}.txt"
  show_status
  log "fleet union sync complete"
}

show_status() {
  echo "=== local $(scutil --get ComputerName 2>/dev/null || hostname) ==="
  df -h "$EXT_ROOT" 2>/dev/null | tail -1 || echo "Ext4T missing"
  echo "HF_LOCAL=$HF_LOCAL"
  if [[ -L "${HOME}/.cache/huggingface" ]]; then
    echo "HF link: $(readlink "${HOME}/.cache/huggingface")"
  fi
  if [[ -d "$HF_LOCAL/hub" ]]; then
    echo "hub entries: $(find "$HF_LOCAL/hub" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
    du -sh "$HF_LOCAL" "$HF_LOCAL/hub" "$HF_LOCAL/xet" 2>/dev/null || true
  fi
  local p
  for p in "${PEERS[@]}"; do
    validate_peer "$p"
    echo
    echo "=== peer $p ==="
    # the probe timeout must come first: OpenSSH keeps the FIRST value of a
    # repeated option, so a ConnectTimeout inside SSH_ARGS would override it
    if ! ssh -o ConnectTimeout=8 ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} "$p" true 2>/dev/null; then
      echo "UNREACHABLE"
      continue
    fi
    ssh_peer "$p" bash -s <<'REMOTE'
df -h /Volumes/Ext4T 2>/dev/null | tail -1 || echo "Ext4T missing"
if [ -L "$HOME/.cache/huggingface" ]; then
  echo "HF link: $(readlink "$HOME/.cache/huggingface")"
fi
for path in /Volumes/Ext4T/huggingface /Volumes/Ext4T/hf-data; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    if [ -L "$path" ]; then
      echo "$path -> $(readlink "$path")"
    else
      echo "$path (real) hub=$(find "$path/hub" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
      du -sh "$path" 2>/dev/null || true
    fi
  fi
done
REMOTE
  done
}

main() {
  local mode="" peer=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --status) mode=status; shift ;;
      --prepare-m2u) mode=prepare-m2u; shift ;;
      --push-all) mode=push-all; shift ;;
      --pull-all) mode=pull-all; shift ;;
      --sync-all) mode=sync-all; shift ;;
      --peer)
        [[ $# -ge 2 && -n "$2" ]] || die "--peer requires a host"
        peer="$2"
        shift 2
        ;;
      --push) mode=push; shift ;;
      --pull) mode=pull; shift ;;
      --peers)
        [[ $# -ge 2 && -n "$2" ]] || die "--peers requires a quoted host list"
        read -r -a PEERS <<<"$2"
        shift 2
        ;;
      -h|--help)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
      *) die "unknown: $1" ;;
    esac
  done

  validate_configuration
  [[ -z "$peer" ]] || validate_peer "$peer"
  case "${mode:-status}" in
    status) show_status ;;
    prepare-m2u) prepare_m2u ;;
    push-all) push_all ;;
    pull-all) pull_all ;;
    sync-all) sync_all ;;
    push)
      [[ -n "$peer" ]] || die "--push needs --peer HOST"
      push_peer "$peer"
      ;;
    pull)
      [[ -n "$peer" ]] || die "--pull needs --peer HOST"
      pull_peer "$peer"
      ;;
    *) die "no mode" ;;
  esac
}

main "$@"
