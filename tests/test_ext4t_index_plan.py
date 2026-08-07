"""Tests for Ext4T package indexing fingerprints and delete planning safety."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(name: str, filename: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


indexer = _load("axquant_index_ext4t", "index-ext4t-packages.py")
planner = _load("axquant_plan_ext4t", "plan-ext4t-from-index.py")


def _write_pkg(root: Path, name: str, files: dict[str, bytes]) -> Path:
    pkg = root / name
    for rel, data in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return pkg


def test_content_fingerprint_distinguishes_same_layout_different_bytes(
    tmp_path: Path,
) -> None:
    a = _write_pkg(
        tmp_path,
        "model-a",
        {"model.safetensors": b"WEIGHTS-A", "config.json": b'{"n":1}'},
    )
    b = _write_pkg(
        tmp_path,
        "model-b",
        {"model.safetensors": b"WEIGHTS-B", "config.json": b'{"n":1}'},
    )
    # same relative paths and sizes (WEIGHTS-A/B are both 9 bytes)
    assert (a / "model.safetensors").stat().st_size == (b / "model.safetensors").stat().st_size

    total_a, n_a, fp_a, _, mode_a, unread_a = indexer.dir_size_and_manifest(a)
    total_b, n_b, fp_b, _, mode_b, unread_b = indexer.dir_size_and_manifest(b)

    assert mode_a == mode_b == "content"
    assert unread_a == unread_b == []
    assert total_a == total_b
    assert n_a == n_b == 2
    assert fp_a != fp_b


def test_path_size_fingerprint_collides_when_layout_matches(tmp_path: Path) -> None:
    a = _write_pkg(tmp_path, "model-a", {"w.bin": b"AAAAAAAA"})
    b = _write_pkg(tmp_path, "model-b", {"w.bin": b"BBBBBBBB"})
    _, _, fp_a, _, mode_a, _ = indexer.dir_size_and_manifest(a, content_hash=False)
    _, _, fp_b, _, mode_b, _ = indexer.dir_size_and_manifest(b, content_hash=False)
    assert mode_a == mode_b == "manifest"
    assert fp_a == fp_b  # documents the weak mode collision


def test_unreadable_file_marks_incomplete(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, "model", {"ok.bin": b"data", "bad.bin": b"secret"})
    bad = pkg / "bad.bin"
    # Make the file unreadable (best-effort; skip if platform ignores mode)
    bad.chmod(0o000)

    try:
        if os.access(bad, os.R_OK):
            pytest.skip("filesystem ignores chmod unreadable bit")
        _, _, fp, _, mode, unreadable = indexer.dir_size_and_manifest(pkg)
    finally:
        with contextlib.suppress(OSError):
            bad.chmod(0o644)

    assert mode == "incomplete"
    assert fp.startswith("incomplete:")
    assert "bad.bin" in unreadable


def test_unreadable_directory_marks_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = _write_pkg(tmp_path, "model", {"ok.bin": b"data"})
    blocked = pkg / "blocked"
    blocked.mkdir()

    def walk_with_blocked_directory(
        root: Path,
        *,
        followlinks: bool,
        onerror,
    ):
        assert followlinks is False
        assert onerror is not None
        yield os.fspath(root), ["blocked"], ["ok.bin"]
        onerror(PermissionError(13, "permission denied", os.fspath(blocked)))

    monkeypatch.setattr(indexer.os, "walk", walk_with_blocked_directory)

    _, _, fp, _, mode, unreadable = indexer.dir_size_and_manifest(pkg)

    assert mode == "incomplete"
    assert fp.startswith("incomplete:")
    assert "blocked" in unreadable


def test_planner_deletes_only_content_safe_same_host_dups(tmp_path: Path) -> None:
    # Two same-host instances with identical content-safe fingerprints
    content_fp = "a" * 64
    index = tmp_path / "idx.jsonl"
    recs = [
        {
            "host": "m3",
            "category": "models",
            "name": "Foo-BF16",
            "path": "/Volumes/Ext4T/models/Foo-BF16",
            "size_bytes": 1000,
            "file_count": 2,
            "manifest_sha256": content_fp,
            "fingerprint_mode": "content",
        },
        {
            "host": "m3",
            "category": "axquant/work",
            "name": "Foo-BF16-copy",
            "path": "/Volumes/Ext4T/axquant/work/Foo-BF16-copy",
            "size_bytes": 1000,
            "file_count": 2,
            "manifest_sha256": content_fp,
            "fingerprint_mode": "content",
        },
    ]
    index.write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "plan.md"
    plan_json = tmp_path / "plan.json"
    _run_planner(index, out, plan_json)

    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    deletes = [m for m in plan["local_moves"] if m["action"] == "delete_duplicate"]
    assert len(deletes) == 1
    assert deletes[0]["host"] == "m3"
    assert deletes[0]["path"] == "/Volumes/Ext4T/axquant/work/Foo-BF16-copy"
    assert deletes[0]["keep"] == "/Volumes/Ext4T/models/Foo-BF16"


def test_planner_keeps_top_level_models_not_axquant_models_suffix(
    tmp_path: Path,
) -> None:
    """Nested axquant/models/X must not win keep via endswith('/models/X').

    preferred_location maps both models and axquant/models categories to
    final models/{name}. A loose suffix match would treat the nested path as
    already-final and schedule delete_duplicate on the real top-level copy.
    """
    content_fp = "c" * 64
    index = tmp_path / "idx.jsonl"
    recs = [
        {
            "host": "m3",
            "category": "models",
            "name": "Bar-BF16",
            "path": "/Volumes/Ext4T/models/Bar-BF16",
            "size_bytes": 2000,
            "file_count": 2,
            "manifest_sha256": content_fp,
            "fingerprint_mode": "content",
        },
        {
            "host": "m3",
            "category": "axquant/models",
            "name": "Bar-BF16",
            "path": "/Volumes/Ext4T/axquant/models/Bar-BF16",
            "size_bytes": 2000,
            "file_count": 2,
            "manifest_sha256": content_fp,
            "fingerprint_mode": "content",
        },
    ]
    index.write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "plan.md"
    plan_json = tmp_path / "plan.json"
    _run_planner(index, out, plan_json)

    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    deletes = [m for m in plan["local_moves"] if m["action"] == "delete_duplicate"]
    assert len(deletes) == 1
    assert deletes[0]["keep"] == "/Volumes/Ext4T/models/Bar-BF16"
    assert deletes[0]["path"] == "/Volumes/Ext4T/axquant/models/Bar-BF16"
    assert planner.is_at_final_path("/Volumes/Ext4T/models/Bar-BF16", "models/Bar-BF16")
    assert not planner.is_at_final_path(
        "/Volumes/Ext4T/axquant/models/Bar-BF16",
        "models/Bar-BF16",
    )


def _run_planner(index: Path, out: Path, plan_json: Path) -> None:
    old = sys.argv
    try:
        sys.argv = [
            "plan-ext4t-from-index.py",
            "--index",
            str(index),
            "-o",
            str(out),
            "--json-out",
            str(plan_json),
        ]
        assert planner.main() == 0
    finally:
        sys.argv = old


def test_planner_never_deletes_on_manifest_or_cheap_or_incomplete(
    tmp_path: Path,
) -> None:
    weak_fp = "b" * 64
    rows = [
        # path-size collision on same host — must NOT delete
        {
            "host": "m3",
            "category": "models",
            "name": "Model-A",
            "path": "/Volumes/Ext4T/models/Model-A",
            "size_bytes": 2000,
            "file_count": 3,
            "manifest_sha256": weak_fp,
            "fingerprint_mode": "manifest",
        },
        {
            "host": "m3",
            "category": "axquant/work",
            "name": "Model-B",
            "path": "/Volumes/Ext4T/axquant/work/Model-B",
            "size_bytes": 2000,
            "file_count": 3,
            "manifest_sha256": weak_fp,
            "fingerprint_mode": "manifest",
        },
        # incomplete — dropped from load, or if present must not delete
        {
            "host": "m3",
            "category": "models",
            "name": "Broken",
            "path": "/Volumes/Ext4T/models/Broken",
            "size_bytes": 500,
            "file_count": 1,
            "manifest_sha256": "incomplete:deadbeef",
            "fingerprint_mode": "incomplete",
        },
        {
            "host": "m3",
            "category": "axquant/work",
            "name": "Broken-copy",
            "path": "/Volumes/Ext4T/axquant/work/Broken-copy",
            "size_bytes": 500,
            "file_count": 1,
            "manifest_sha256": "incomplete:deadbeef",
            "fingerprint_mode": "incomplete",
        },
        # cheap — skipped entirely
        {
            "host": "m3",
            "category": "huggingface",
            "name": "hub-cache",
            "path": "/Volumes/Ext4T/huggingface/hub-cache",
            "size_bytes": 9_000_000_000,
            "file_count": 100,
            "manifest_sha256": "cheap:9000000000:100",
            "fingerprint_mode": "cheap",
        },
        {
            "host": "m3",
            "category": "huggingface",
            "name": "hub-cache-2",
            "path": "/Volumes/Ext4T/huggingface/hub-cache-2",
            "size_bytes": 9_000_000_000,
            "file_count": 100,
            "manifest_sha256": "cheap:9000000000:100",
            "fingerprint_mode": "cheap",
        },
    ]
    index = tmp_path / "idx.jsonl"
    index.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "plan.md"
    plan_json = tmp_path / "plan.json"
    _run_planner(index, out, plan_json)

    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    deletes = [m for m in plan["local_moves"] if m["action"] == "delete_duplicate"]
    assert deletes == []
    # incomplete + cheap dropped; only the two path-size records remain, isolated
    assert plan["totals"]["unique_packages"] == 2
    assert plan["totals"]["reclaimable_bytes"] == 0
    assert plan["totals"]["unverified_fingerprints"] == 2


def test_content_identity_key_isolates_weak_modes() -> None:
    content = {
        "host": "h",
        "path": "/a",
        "manifest_sha256": "abc",
        "fingerprint_mode": "content",
    }
    manifest = {
        "host": "h",
        "path": "/a",
        "manifest_sha256": "abc",
        "fingerprint_mode": "manifest",
    }
    incomplete = {
        "host": "h",
        "path": "/a",
        "manifest_sha256": "incomplete:abc",
        "fingerprint_mode": "incomplete",
    }
    assert planner.fingerprint_is_content_safe(content)
    assert not planner.fingerprint_is_content_safe(manifest)
    assert not planner.fingerprint_is_content_safe(incomplete)
    assert not planner.fingerprint_is_content_safe(
        {"manifest_sha256": "x", "fingerprint_mode": "cheap"}
    )
    # legacy missing mode is unsafe
    assert not planner.fingerprint_is_content_safe({"manifest_sha256": "x"})
    assert planner.content_identity_key(content) == "abc"
    assert planner.content_identity_key(manifest).startswith("unverified:")
    assert planner.content_identity_key(manifest) != planner.content_identity_key(
        {**manifest, "path": "/b"}
    )


def test_index_root_emits_content_mode(tmp_path: Path) -> None:
    root = tmp_path / "Ext4T"
    models = root / "models"
    _write_pkg(models, "Tiny-Model", {"w.bin": b"hello", "cfg.json": b"{}"})
    recs = indexer.index_root(root, "local", ["models"], skip_hf_deep=True)
    assert len(recs) == 1
    assert recs[0]["fingerprint_mode"] == "content"
    assert recs[0]["unreadable_files"] == []
    assert len(recs[0]["manifest_sha256"]) == 64
