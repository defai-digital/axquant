#!/usr/bin/env bash
# Run on macstudio-m2u (or: ssh macstudio-m2u 'bash -s' < this file)
# Completes Ext4T factory layout after any interrupted smoke migration.
set -euo pipefail

EXT_ROOT="/Volumes/Ext4T"
AXQ="${EXT_ROOT}/axquant"
ART="${HOME}/axquant-artifacts"
MODELS="${HOME}/models"
SMOKE_NAMES=(qwen36-27b-dev-smoke qwen36-35b-a3b-dev-smoke)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

safe_remove_tree() {
  local target="$1"
  local allowed_parent="$2"
  [[ "$(dirname "$target")" == "$allowed_parent" ]] || {
    die "refusing to remove anything except a direct child of $allowed_parent: $target"
  }
  [[ -d "$target" ]] || die "refusing to remove a non-directory: $target"
  [[ ! -L "$target" ]] || die "refusing to recursively remove symlink: $target"
  rm -rf -- "${target:?}"
}

ensure_real_dir() {
  local path="$1"
  [[ ! -L "$path" ]] || die "managed directory must not be a symlink: $path"
  [[ ! -e "$path" || -d "$path" ]] || die "managed path is not a directory: $path"
  mkdir -p "$path"
}

sync_and_verify() {
  local source="$1"
  local destination="$2"
  [[ -d "$source" && ! -L "$source" ]] || die "source must be a real directory: $source"
  ensure_real_dir "$destination"
  # Preserve existing destination files. A same-name content conflict is reported by
  # the checksum pass below and leaves both source and destination in place.
  rsync -aH --ignore-existing --exclude '.DS_Store' "$source/" "$destination/"
  local differences
  # Content-only verify: --ignore-existing never aligns times/perms on files
  # already present, so attribute-only itemize lines (leading '.') must not
  # block the migration. Content differences itemize as '>', '<', 'c', 'h',
  # or '*deleting' and stay fatal.
  differences="$(
    rsync -rlHnc --itemize-changes --exclude '.DS_Store' "$source/" "$destination/" \
      | grep -Ev '^\.' || true
  )"
  [[ -z "$differences" ]] || {
    echo "$differences" >&2
    die "checksum verification failed: $source -> $destination"
  }
}

if [[ ! -d "$EXT_ROOT" ]]; then
  die "$EXT_ROOT not mounted"
fi
mounted_at="$(df -P "$EXT_ROOT" 2>/dev/null | awk 'NR == 2 {print $NF}')"
[[ "$mounted_at" == "$EXT_ROOT" ]] || {
  die "$EXT_ROOT is not a mounted volume (df reports ${mounted_at:-unknown})"
}

ensure_real_dir "$AXQ"
for path in models axq-publish logs work smokes; do
  ensure_real_dir "$AXQ/$path"
done

# HF cache must stay on Ext4T before this script can claim the layout is complete.
[[ -d "$EXT_ROOT/huggingface" && ! -L "$EXT_ROOT/huggingface" ]] || {
  die "$EXT_ROOT/huggingface must be a real directory"
}
[[ -L "${HOME}/.cache/huggingface" ]] || {
  die "${HOME}/.cache/huggingface is not a symlink"
}
[[ "$(readlink "${HOME}/.cache/huggingface")" == "$EXT_ROOT/huggingface" ]] || {
  die "${HOME}/.cache/huggingface points somewhere unexpected"
}
echo "HF cache -> $(readlink "${HOME}/.cache/huggingface")"

# Ensure artifact symlinks for publish/logs. This script replaces children of ART,
# so following a symlinked ART root would make its deletion boundary meaningless.
ensure_real_dir "$ART"

# Migrate a live directory by moving it aside first: writers landing between
# the checksum verify and the delete would otherwise be erased silently, while
# after the rename late writers fail loudly. A leftover *.migrating dir from an
# interrupted run is merged before anything else.
migrate_live_dir() {
  local source="$1"
  local destination="$2"
  local staging="${source}.migrating"
  if [[ -d "$staging" && ! -L "$staging" ]]; then
    sync_and_verify "$staging" "$destination"
    safe_remove_tree "$staging" "$(dirname "$staging")"
  fi
  if [[ -d "$source" && ! -L "$source" ]]; then
    [[ ! -e "$staging" && ! -L "$staging" ]] || {
      die "unmergeable migration staging path exists: $staging"
    }
    mv "$source" "$staging"
    sync_and_verify "$staging" "$destination"
    safe_remove_tree "$staging" "$(dirname "$staging")"
  fi
}

migrate_live_dir "$ART/axq-publish" "$AXQ/axq-publish"
if [[ -L "$ART/axq-publish" ]]; then
  [[ "$(readlink "$ART/axq-publish")" == "$AXQ/axq-publish" ]] || {
    die "$ART/axq-publish points somewhere unexpected"
  }
else
  ln -s "$AXQ/axq-publish" "$ART/axq-publish"
fi

migrate_live_dir "$ART/logs" "$AXQ/logs"
if [[ -L "$ART/logs" ]]; then
  [[ "$(readlink "$ART/logs")" == "$AXQ/logs" ]] || {
    die "$ART/logs points somewhere unexpected"
  }
else
  ln -s "$AXQ/logs" "$ART/logs"
fi

# Models: real dirs -> Ext4T + symlink. A symlinked ~/models is commonly the HF
# hub convenience link and must never be traversed as a migration source.
if [[ -L "$MODELS" ]]; then
  echo "skip model migration: $MODELS is a symlink -> $(readlink "$MODELS")"
elif [[ -d "$MODELS" ]]; then
  for d in "$MODELS"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    if [[ -L "$MODELS/$name" ]]; then
      [[ "$(readlink "$MODELS/$name")" == "$AXQ/models/$name" ]] || {
        die "model symlink $name points somewhere unexpected"
      }
      echo "OK model symlink $name"
      continue
    fi
    dest="$AXQ/models/$name"
    echo "migrate model $name"
    sync_and_verify "$MODELS/$name" "$dest"
    safe_remove_tree "$MODELS/$name" "$MODELS"
    ln -s "$dest" "$MODELS/$name"
  done
fi

# Smokes: migrate only the two named smoke artifacts. Unknown artifact directories
# are intentionally left untouched instead of being reclassified as smokes.
if [[ -d "$ART" ]]; then
  for name in "${SMOKE_NAMES[@]}"; do
    [[ -e "$ART/$name" || -L "$ART/$name" ]] || continue
    if [[ -L "$ART/$name" ]]; then
      [[ "$(readlink "$ART/$name")" == "$AXQ/smokes/$name" ]] || {
        die "smoke symlink $name points somewhere unexpected"
      }
      echo "OK smoke symlink $name"
      continue
    fi
    [[ -d "$ART/$name" ]] || die "smoke artifact is not a directory: $ART/$name"
    dest="$AXQ/smokes/$name"
    echo "migrate smoke $name"
    sync_and_verify "$ART/$name" "$dest"
    src_n="$(find "$ART/$name" -name '*.safetensors' | wc -l | tr -d ' ')"
    dst_n="$(find "$dest" -name '*.safetensors' | wc -l | tr -d ' ')"
    src_sz="$(du -sm "$ART/$name" | cut -f1)"
    dst_sz="$(du -sm "$dest" | cut -f1)"
    echo "  tensors $src_n->$dst_n MB $src_sz->$dst_sz"
    if [[ "$dst_n" != "$src_n" ]] || [[ "$dst_sz" -lt $((src_sz * 90 / 100)) ]]; then
      echo "ERROR integrity $name" >&2
      exit 1
    fi
    safe_remove_tree "$ART/$name" "$ART"
    ln -s "$dest" "$ART/$name"
  done
fi

[[ ! -L "$AXQ/layout-complete.txt" ]] || die "layout completion marker must not be a symlink"
[[ ! -e "$AXQ/layout-complete.txt" || -f "$AXQ/layout-complete.txt" ]] || {
  die "layout completion marker is not a regular file"
}
date -u +%Y-%m-%dT%H:%M:%SZ >"$AXQ/layout-complete.txt"
echo "=== df ==="
df -h /System/Volumes/Data "$EXT_ROOT" | cat
echo "=== artifacts ==="
ls -la "$ART"
echo "COMPLETE $(cat "$AXQ/layout-complete.txt")"
