from __future__ import annotations

from pathlib import Path


def artifact_tree_symlinks(directory: Path) -> list[str]:
    """Return every symlink below an artifact root without following it."""

    if directory.is_symlink():
        return ["."]
    root = directory.resolve()
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()
    )


def artifact_tree_files(directory: Path) -> list[Path]:
    """List regular artifact files, rejecting trees that contain symlinks."""

    if directory.is_symlink():
        raise ValueError("artifact root is a symlink")
    root = directory.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {directory}")
    symlinks = artifact_tree_symlinks(root)
    if symlinks:
        raise ValueError(f"artifact tree contains symlinks: {symlinks}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact file escapes its root: {path}") from exc
        files.append(path)
    return sorted(files)


def artifact_member_path(directory: Path, value: str) -> Path:
    """Resolve a safe, symlink-free relative path below an artifact root."""

    if directory.is_symlink():
        raise ValueError("artifact root is a symlink")
    root = directory.resolve()
    relative = Path(value)
    normalized_parts = value.split("/")
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized_parts)
    ):
        raise ValueError(f"unsafe artifact path: {value}")
    path = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {value}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes its root: {value}") from exc
    return path
