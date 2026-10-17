from __future__ import annotations

import tomllib
from pathlib import Path

from axquant import __version__

_ROOT = Path(__file__).resolve().parents[1]


def test_source_version_and_readme_are_consistent() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    version = project["version"]

    assert __version__ == version
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    # README may phrase toolkit version vs certified Hub release carefully.
    assert f"`{version}`" in readme
    assert version in readme


def test_official_toolkit_release_keeps_model_evidence_separate() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    classifiers = set(project["classifiers"])
    assert "Development Status :: 3 - Alpha" not in classifiers
    assert "Development Status :: 4 - Beta" not in classifiers
    # Official distribution does not certify every format/model/runtime pair.
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "official toolkit release" in readme
    assert "not a prerelease" in readme
    assert "does not extend this frozen certification contract" in readme
    assert "development evidence" in readme


def test_release_notes_exist_for_current_version() -> None:
    """The release workflow loads docs/releases/<version>.md and fails closed if empty.

    Mirror that check here so a missing notes file breaks CI before a tag is pushed.
    """
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    notes_path = _ROOT / "docs" / "releases" / f"{version}.md"
    assert notes_path.is_file(), f"missing curated release notes: {notes_path.relative_to(_ROOT)}"
    assert notes_path.read_text(encoding="utf-8").strip(), (
        f"curated release notes are empty: {notes_path.relative_to(_ROOT)}"
    )


def test_source_distribution_excludes_local_runtime_evidence() -> None:
    """A common CLI output beside pyproject.toml must not be uploaded to PyPI."""
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    excluded = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/.internal" in excluded
    assert "/credentials.json" in excluded
    assert "/runtime_check.json" in excluded
    assert "/.venv*" in excluded
    included = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert "/src" in included
    assert not any(path.startswith("/.") or path in {"*", "/", "/*"} for path in included)
    assert "runtime_check.json" in (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
