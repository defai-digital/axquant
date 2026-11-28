from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.hf_api import RepoFile

from axquant import publisher
from axquant.errors import PublishingError

REPO = "AutomatosX/AX-Test-MLX-AXQ-MXFP4"
PARENT = "a" * 40
NEW = "b" * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _repo_file(name: str, data: bytes, *, lfs: bool) -> RepoFile:
    payload: dict[str, Any] = {"path": name, "size": len(data), "oid": _blob(data)}
    if lfs:
        payload["lfs"] = {"oid": _sha256(data), "size": len(data), "pointerSize": 134}
    return RepoFile(**payload)


class FakeHub:
    def __init__(self, tree: dict[str, tuple[bytes, bool]], *, head: str = PARENT) -> None:
        self.trees = {head: tree}
        self.head = head
        self.commits: list[dict[str, Any]] = []
        self.corrupt: str | None = None

    def model_info(self, *, repo_id: str) -> SimpleNamespace:
        return SimpleNamespace(sha=self.head)

    def list_repo_tree(self, **kwargs: Any) -> list[RepoFile]:
        tree = self.trees[kwargs["revision"]]
        return [_repo_file(name, data, lfs=lfs) for name, (data, lfs) in tree.items()]

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self.commits.append(kwargs)
        tree = dict(self.trees[kwargs["parent_commit"]])
        for operation in kwargs["operations"]:
            if isinstance(operation, CommitOperationDelete):
                tree.pop(operation.path_in_repo)
            else:
                assert isinstance(operation, CommitOperationAdd)
                data = Path(str(operation.path_or_fileobj)).read_bytes()
                if operation.path_in_repo == self.corrupt:
                    data = data + b"!"
                tree[operation.path_in_repo] = (data, operation.path_in_repo.endswith(".bin"))
        self.trees[NEW] = tree
        self.head = NEW
        return SimpleNamespace(oid=NEW)


@pytest.fixture
def pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "pack"
    directory.mkdir()
    files = {
        "config.json": b'{"a": 1}\n',
        "model.bin": b"new weights",
        "README.md": b"card\n",
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / "axquant_manifest.json").write_text("{}\n")
    (directory / "axquant_publication_smoke.json").write_text("{}\n")
    manifest = SimpleNamespace(
        files=[
            SimpleNamespace(path=name, size_bytes=len(data), sha256=_sha256(data))
            for name, data in files.items()
        ]
    )
    monkeypatch.setattr(publisher, "load_artifact_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        publisher,
        "require_publication_preconditions",
        lambda *, model_dir, repo_id: Path(model_dir).resolve(),
    )
    return directory


def _live_tree() -> dict[str, tuple[bytes, bool]]:
    return {
        ".gitattributes": (b"*.bin filter=lfs\n", False),
        "config.json": (b'{"a": 1}\n', False),
        "model-old.bin": (b"old weights", True),
        "README.md": (b"old card\n", False),
    }


def test_preview_classifies_changes_and_deletes_stale_files(pack: Path) -> None:
    hub = FakeHub(_live_tree())

    plan = publisher.republish_development_pack(
        model_dir=pack, repo_id=REPO, expected_parent=PARENT, api=hub
    )

    assert plan.unchanged == ("config.json",)
    assert plan.replaced == ("README.md",)
    assert plan.added == ("axquant_manifest.json", "axquant_publication_smoke.json", "model.bin")
    assert plan.deleted == ("model-old.bin",)
    assert plan.commit is None
    assert hub.commits == []


def test_execute_commits_on_parent_and_verifies_tree(pack: Path) -> None:
    hub = FakeHub(_live_tree())

    result = publisher.republish_development_pack(
        model_dir=pack, repo_id=REPO, expected_parent=PARENT, execute=True, api=hub
    )

    assert result.commit == NEW
    (commit,) = hub.commits
    assert commit["parent_commit"] == PARENT
    assert set(hub.trees[NEW]) == {
        ".gitattributes",
        "config.json",
        "README.md",
        "model.bin",
        "axquant_manifest.json",
        "axquant_publication_smoke.json",
    }


def test_refuses_when_main_moved(pack: Path) -> None:
    hub = FakeHub(_live_tree(), head="c" * 40)
    hub.trees[PARENT] = _live_tree()

    with pytest.raises(PublishingError, match="not the expected parent"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent=PARENT, api=hub
        )


def test_refuses_mutable_parent(pack: Path) -> None:
    with pytest.raises(PublishingError, match="40-hex"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent="main", api=FakeHub(_live_tree())
        )


def test_refuses_file_not_bound_by_manifest(pack: Path) -> None:
    (pack / "stray.json").write_text("{}\n")

    with pytest.raises(PublishingError, match=r"not bound by axquant_manifest\.json: stray\.json"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent=PARENT, api=FakeHub(_live_tree())
        )


def test_refuses_bytes_changed_after_manifest(pack: Path) -> None:
    (pack / "model.bin").write_bytes(b"tampered!!!")

    with pytest.raises(PublishingError, match=r"digest mismatch: model\.bin"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent=PARENT, api=FakeHub(_live_tree())
        )


def test_fails_closed_when_uploaded_bytes_differ(pack: Path) -> None:
    hub = FakeHub(_live_tree())
    hub.corrupt = "model.bin"

    with pytest.raises(PublishingError, match="differ from the pack"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent=PARENT, execute=True, api=hub
        )


def test_refuses_noop_republication(pack: Path) -> None:
    tree = {
        path.name: (path.read_bytes(), path.name == "model.bin")
        for path in pack.iterdir()
        if path.is_file()
    }

    with pytest.raises(PublishingError, match="nothing to republish"):
        publisher.republish_development_pack(
            model_dir=pack, repo_id=REPO, expected_parent=PARENT, execute=True, api=FakeHub(tree)
        )
