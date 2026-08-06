"""Structural checks for shipped GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE = _ROOT / ".github" / "workflows" / "release.yml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"


def test_release_workflow_gates_pypi_on_enable_variable() -> None:
    """PyPI publish must not run (and red every tag) until operator-configured.

    Trusted Publishing lives on pypi.org, not in this repo. Without a gate the
    `pypi` job fails every release with invalid-publisher even when the GitHub
    Release assets already uploaded successfully.
    """
    text = _RELEASE.read_text(encoding="utf-8")
    assert "ENABLE_PYPI_PUBLISH" in text
    assert "vars.ENABLE_PYPI_PUBLISH" in text
    # Job-level gate (not a no-op step condition alone).
    assert "if: ${{ vars.ENABLE_PYPI_PUBLISH == 'true' }}" in text
    assert "pypa/gh-action-pypi-publish" in text


def test_release_workflow_uses_exact_version_tag_as_title() -> None:
    """Created and rerun releases must display only their immutable version tag."""
    text = _RELEASE.read_text(encoding="utf-8")
    assert "gh release create" in text
    assert "gh release edit" in text
    assert text.count('--title "${GITHUB_REF_NAME}"') == 2


def test_ci_workflow_runs_non_mlx_and_mlx_jobs() -> None:
    text = _CI.read_text(encoding="utf-8")
    assert "python-compatibility" in text
    assert 'pip install -e ".[dev]"' in text
    assert 'pip install -e ".[dev,mlx]"' in text
    assert 'pytest -m "not integration"' in text
