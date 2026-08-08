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


def test_v1_toolkit_status_is_honest_about_host_certification() -> None:
    """Toolkit readiness is Beta until formal host/candidate release-audit is green.

    Production/Stable would over-claim a certified Hub model release while MTP speed
    and size-floor evidence remain open.
    """
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    classifiers = set(project["classifiers"])
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")

    # Beta status is appropriate for the whole 1.x line: no certified Hub model
    # release has happened yet regardless of minor/patch version, so a bump
    # (1.0.x -> 1.1.0) must not silently keep claiming Beta without this test
    # still verifying the README's honesty language, and any version outside
    # that line forces an explicit reconsideration of the classifier below.
    # v1.1.0 reconsideration: GPTQ + end-to-end AWQ add capability but do not
    # change the gating fact — MTP speed and size-floor evidence remain open,
    # so Beta still holds.
    major = project["version"].split(".")[0]
    if major == "1":
        assert "Development Status :: 4 - Beta" in classifiers
        assert "Development Status :: 5 - Production/Stable" not in classifiers
        assert "Development Status :: 3 - Alpha" not in classifiers
        assert "Beta" in readme
        assert "no" in readme.lower() and "certified" in readme.lower()
    else:
        assert "Development Status :: 3 - Alpha" in classifiers


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
