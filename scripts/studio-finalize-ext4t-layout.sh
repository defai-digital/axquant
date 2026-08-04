#!/usr/bin/env bash
# Run on macstudio-m2u (or: ssh macstudio-m2u 'bash -s' < this file)
# Completes Ext4T factory layout after any interrupted smoke migration.
set -euo pipefail

EXT_ROOT="/Volumes/Ext4T"
AXQ="${EXT_ROOT}/axquant"
ART="${HOME}/axquant-artifacts"
MODELS="${HOME}/models"

if [[ ! -d "$EXT_ROOT" ]]; then
  echo "ERROR: $EXT_ROOT not mounted" >&2
  exit 1
fi

mkdir -p "$AXQ/models" "$AXQ/axq-publish" "$AXQ/logs" "$AXQ/work" "$AXQ/smokes"

# HF cache must stay on Ext4T
if [[ ! -L "${HOME}/.cache/huggingface" ]]; then
  echo "WARN: ~/.cache/huggingface is not a symlink to Ext4T" >&2
else
  echo "HF cache -> $(readlink "${HOME}/.cache/huggingface")"
fi

# Ensure artifact symlinks for publish/logs
mkdir -p "$ART"
if [[ -d "$ART/axq-publish" && ! -L "$ART/axq-publish" ]]; then
  rsync -aH --delete "$ART/axq-publish/" "$AXQ/axq-publish/"
  rm -rf "$ART/axq-publish"
fi
ln -sfn "$AXQ/axq-publish" "$ART/axq-publish"

if [[ -d "$ART/logs" && ! -L "$ART/logs" ]]; then
  rsync -aH "$ART/logs/" "$AXQ/logs/"
  rm -rf "$ART/logs"
fi
ln -sfn "$AXQ/logs" "$ART/logs"

# Models: real dirs -> Ext4T + symlink
if [[ -d "$MODELS" ]]; then
  for d in "$MODELS"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    if [[ -L "$MODELS/$name" ]]; then
      echo "OK model symlink $name"
      continue
    fi
    dest="$AXQ/models/$name"
    echo "migrate model $name"
    mkdir -p "$dest"
    rsync -aH --delete "$MODELS/$name/" "$dest/"
    rm -rf "$MODELS/$name"
    ln -s "$dest" "$MODELS/$name"
  done
fi

# Smokes: discard incomplete Ext4T copy, migrate remaining real dirs
for name in qwen36-27b-dev-smoke qwen36-35b-a3b-dev-smoke; do
  partial="$AXQ/smokes/$name"
  if [[ -d "$partial" && ! -L "$ART/$name" ]]; then
    n="$(find "$partial" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
    sz="$(du -sm "$partial" 2>/dev/null | cut -f1)"
    if [[ "${n:-0}" -lt 4 || "${sz:-0}" -lt 10000 ]]; then
      echo "remove incomplete Ext4T $name (tensors=$n MB=$sz)"
      rm -rf "$partial"
    fi
  fi
done

if [[ -d "$ART" ]]; then
  for d in "$ART"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    case "$name" in
      axq-publish|logs) continue ;;
    esac
    if [[ -L "$ART/$name" ]]; then
      echo "OK smoke symlink $name"
      continue
    fi
    dest="$AXQ/smokes/$name"
    echo "migrate smoke $name"
    rm -rf "$dest"
    mkdir -p "$dest"
    rsync -aH --delete "$ART/$name/" "$dest/"
    src_n="$(find "$ART/$name" -name '*.safetensors' | wc -l | tr -d ' ')"
    dst_n="$(find "$dest" -name '*.safetensors' | wc -l | tr -d ' ')"
    src_sz="$(du -sm "$ART/$name" | cut -f1)"
    dst_sz="$(du -sm "$dest" | cut -f1)"
    echo "  tensors $src_n->$dst_n MB $src_sz->$dst_sz"
    if [[ "$dst_n" != "$src_n" ]] || [[ "$dst_sz" -lt $((src_sz * 90 / 100)) ]]; then
      echo "ERROR integrity $name" >&2
      exit 1
    fi
    rm -rf "$ART/$name"
    ln -s "$dest" "$ART/$name"
  done
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"$AXQ/layout-complete.txt"
echo "=== df ==="
df -h /System/Volumes/Data "$EXT_ROOT" | cat
echo "=== artifacts ==="
ls -la "$ART"
echo "COMPLETE $(cat "$AXQ/layout-complete.txt")"
