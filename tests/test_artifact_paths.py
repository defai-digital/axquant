from __future__ import annotations

from pathlib import Path

import pytest

from axquant.artifact_paths import (
    artifact_member_path,
    artifact_tree_files,
    artifact_tree_symlinks,
)


def test_artifact_member_path_requires_canonical_relative_path(tmp_path: Path) -> None:
    for value in ("", ".", "../file", "/absolute", "nested//file", "nested/./file", "a\\b"):
        with pytest.raises(ValueError, match="unsafe artifact path"):
            artifact_member_path(tmp_path, value)
    assert artifact_member_path(tmp_path, "nested/file") == tmp_path / "nested/file"


def test_artifact_tree_rejects_root_and_member_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "file").write_text("data", encoding="utf-8")
    root_link = tmp_path / "artifact-link"
    root_link.symlink_to(artifact, target_is_directory=True)
    assert artifact_tree_symlinks(root_link) == ["."]
    with pytest.raises(ValueError, match="root is a symlink"):
        artifact_tree_files(root_link)

    external = tmp_path / "external"
    external.write_text("outside", encoding="utf-8")
    (artifact / "linked").symlink_to(external)
    assert artifact_tree_symlinks(artifact) == ["linked"]
    with pytest.raises(ValueError, match="contains symlinks"):
        artifact_tree_files(artifact)
