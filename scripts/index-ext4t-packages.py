#!/usr/bin/env python3
"""Index Ext4T package directories for cross-host dedupe + layout planning.

For each package dir under models/ and axquant/{models,axq-publish,work,smokes,...}
emits a JSON line with:
  - path, size, file_count
  - manifest_sha256: sha256 of sorted "relpath\\tsize" lines (content identity without full file hashing)
  - top_files: largest files (relpath, size) for human review

Usage:
  python3 scripts/index-ext4t-packages.py --root /Volumes/Ext4T --host local > index.jsonl
  python3 scripts/index-ext4t-packages.py --root /Volumes/Ext4T --host mbp-m5 --trees models,axquant
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat as stat_mod
import sys
import time
from pathlib import Path


SKIP_DIR_NAMES = {
    ".git",
    ".cache",
    ".Spotlight-V100",
    ".fseventsd",
    ".TemporaryItems",
    ".Trashes",
    ".DocumentRevisions-V100",
    "__pycache__",
}


def dir_size_and_manifest(root: Path) -> tuple[int, int, str, list[tuple[int, str]]]:
    """Return total_bytes, file_count, manifest_sha256, top_files[(size, relpath)]."""
    lines: list[str] = []
    top: list[tuple[int, str]] = []
    total = 0
    nfiles = 0
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # prune noisy / huge non-content dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            if name == ".DS_Store":
                continue
            fp = base / name
            try:
                st = fp.stat()
            except OSError:
                continue
            if not fp.is_file():
                continue
            rel = fp.relative_to(root).as_posix()
            size = int(st.st_size)
            total += size
            nfiles += 1
            lines.append(f"{rel}\t{size}")
            top.append((size, rel))
    lines.sort()
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8", errors="surrogateescape"))
        h.update(b"\n")
    top.sort(reverse=True)
    return total, nfiles, h.hexdigest(), top[:8]


def package_roots(ext_root: Path, trees: list[str]) -> list[tuple[str, Path]]:
    """Yield (category, package_path)."""
    out: list[tuple[str, Path]] = []
    for tree in trees:
        base = ext_root / tree
        if not base.is_dir():
            continue
        if tree == "models":
            for child in sorted(base.iterdir()):
                if child.name.startswith("."):
                    continue
                # is_dir() follows symlinks, so this covers real model dirs
                # and symlinked ones alike
                if child.is_dir():
                    out.append(("models", child))
        elif tree == "axquant":
            # packages live one level down under known buckets; also index nested work/*
            for bucket in (
                "models",
                "axq-publish",
                "work",
                "smokes",
                "certification",
                "logs",
                "scripts",
            ):
                bdir = base / bucket
                if not bdir.is_dir():
                    continue
                # work may have a candidates grouping folder
                for child in sorted(bdir.iterdir()):
                    if child.name.startswith(".") or child.name == ".DS_Store":
                        continue
                    if not child.is_dir():
                        # small files at bucket root (README etc.) — skip
                        continue
                    # If this is a grouping dir (only subdirs, no weight files), expand one level
                    if bucket == "work" and _looks_like_group(child):
                        for gchild in sorted(child.iterdir()):
                            if gchild.is_dir() and not gchild.name.startswith("."):
                                out.append((f"axquant/work/{child.name}", gchild))
                        continue
                    out.append((f"axquant/{bucket}", child))
        elif tree == "huggingface":
            # only top-level hub blobs inventory is huge; record top-level dirs only as size rollups
            for child in sorted(base.iterdir()):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    out.append(("huggingface", child))
    return out


def _looks_like_group(path: Path) -> bool:
    """Heuristic: group folder if it has subdirs and almost no large weight files at this level."""
    subdirs = 0
    big_files = 0
    try:
        for p in path.iterdir():
            if p.name.startswith("."):
                continue
            if p.is_dir():
                subdirs += 1
            elif p.is_file() and p.stat().st_size > 50 * 1024 * 1024:
                big_files += 1
    except OSError:
        return False
    return subdirs >= 2 and big_files == 0


def index_root(ext_root: Path, host: str, trees: list[str], skip_hf_deep: bool) -> list[dict]:
    results: list[dict] = []
    packages = package_roots(ext_root, trees)
    total = len(packages)
    for i, (category, path) in enumerate(packages, 1):
        # huggingface top-level: size-only via du-like walk but still manifest (hub is large)
        if category == "huggingface" and skip_hf_deep:
            # cheap: only total size + file count, empty manifest marker
            total_b, nfiles = _cheap_size(path)
            rec = {
                "host": host,
                "category": category,
                "name": path.name,
                "path": str(path),
                "size_bytes": total_b,
                "file_count": nfiles,
                "manifest_sha256": f"cheap:{total_b}:{nfiles}",
                "top_files": [],
                "fingerprint_mode": "cheap",
            }
        else:
            t0 = time.time()
            total_b, nfiles, msh, top = dir_size_and_manifest(path)
            rec = {
                "host": host,
                "category": category,
                "name": path.name,
                "path": str(path),
                "size_bytes": total_b,
                "file_count": nfiles,
                "manifest_sha256": msh,
                "top_files": [{"size": s, "relpath": r} for s, r in top],
                "fingerprint_mode": "manifest",
                "index_seconds": round(time.time() - t0, 3),
            }
        results.append(rec)
        print(
            f"[{i}/{total}] {host} {category}/{path.name} "
            f"{total_b/1e9:.2f}G files={nfiles} fp={rec['manifest_sha256'][:12]}",
            file=sys.stderr,
            flush=True,
        )
    return results


def _cheap_size(root: Path) -> tuple[int, int]:
    total = 0
    n = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name == ".DS_Store":
                continue
            fp = Path(dirpath) / name
            # skip symlinks: HF snapshots/ links into blobs/, which would
            # double-count every blob
            try:
                st = fp.lstat()
            except OSError:
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                continue
            total += st.st_size
            n += 1
    return total, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/Ext4T")
    ap.add_argument("--host", required=True, help="logical host label")
    ap.add_argument(
        "--trees",
        default="models,axquant",
        help="comma-separated under Ext4T (default: models,axquant)",
    )
    ap.add_argument(
        "--include-huggingface",
        action="store_true",
        help="also index huggingface top-level dirs (cheap mode)",
    )
    ap.add_argument("-o", "--output", help="write JSONL here (default stdout)")
    args = ap.parse_args()

    trees = [t.strip() for t in args.trees.split(",") if t.strip()]
    if args.include_huggingface and "huggingface" not in trees:
        trees.append("huggingface")

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} missing", file=sys.stderr)
        return 2

    recs = index_root(root, args.host, trees, skip_hf_deep=True)
    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        meta = {
            "type": "index_meta",
            "host": args.host,
            "root": str(root),
            "trees": trees,
            "package_count": len(recs),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out_f.write(json.dumps(meta) + "\n")
        for r in recs:
            out_f.write(json.dumps(r, sort_keys=True) + "\n")
    finally:
        if args.output:
            out_f.close()
    print(f"indexed {len(recs)} packages on {args.host}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
