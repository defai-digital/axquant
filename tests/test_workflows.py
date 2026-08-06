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
    # Titles bind to RELEASE_TAG (tag push or dispatch input), never a branch name.
    assert text.count('--title "${RELEASE_TAG}"') == 2


def test_release_workflow_dispatch_binds_tag_input() -> None:
    """Manual Release runs must take an explicit tag and check out that ref.

    workflow_dispatch without a tag input leaves GITHUB_REF_NAME as the branch
    selected in the UI (usually main), which fails version asserts and would
    title a release incorrectly.
    """
    text = _RELEASE.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "inputs:" in text
    assert "tag:" in text
    assert "RELEASE_TAG:" in text
    assert "github.event_name == 'workflow_dispatch' && inputs.tag" in text
    assert "ref: ${{ env.RELEASE_TAG }}" in text
    # Version check and notes extraction must use RELEASE_TAG (not branch ref_name).
    assert 'os.environ["RELEASE_TAG"]' in text
    assert 'os.environ["GITHUB_REF_NAME"]' not in text
    assert "${GITHUB_REF_NAME}" not in text


def test_ci_workflow_runs_non_mlx_and_mlx_jobs() -> None:
    """Ubuntu owns the non-MLX contract; macOS owns MLX execution."""
    text = _CI.read_text(encoding="utf-8")
    assert "python-compatibility" in text
    assert 'pip install -e ".[dev]"' in text
    assert 'pip install -e ".[dev,mlx]"' in text
    assert 'pytest -m "not integration"' in text
    # Display names make the split obvious in the Actions UI.
    assert "name: lint (non-MLX)" in text
    assert "name: non-MLX (Python ${{ matrix.python-version }})" in text
    assert "name: MLX (macOS)" in text
    # Ubuntu non-MLX jobs must not install the mlx extra.
    ubuntu_block = text.split("python-compatibility:")[1].split("test:")[0]
    assert '".[dev,mlx]"' not in ubuntu_block
    assert "runs-on: ubuntu-latest" in ubuntu_block
    # macOS job must install mlx and run on Apple Silicon runners.
    mlx_block = text.split("test:")[1]
    assert '".[dev,mlx]"' in mlx_block
    assert "runs-on: macos-14" in mlx_block


def test_ci_workflow_asserts_install_surfaces() -> None:
    """Fail closed if a job accidentally gets the wrong optional backend set."""
    text = _CI.read_text(encoding="utf-8")
    assert "Assert MLX is not on the non-MLX surface" in text
    assert "Assert MLX backend is importable" in text
    # Explicit package names the non-MLX surface must not import.
    for package in ("mlx", "mlx_lm", "mlx_audio", "mlx_vlm"):
        assert f'"{package}"' in text or f"'{package}'" in text
    assert "import mlx.core" in text


def test_ci_workflow_is_retriggerable_and_cancels_superseded_runs() -> None:
    """Tip health needs manual re-runs and cancel-in-progress for obsolete SHAs."""
    text = _CI.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "concurrency:" in text
    assert "cancel-in-progress: true" in text


def test_release_workflow_documents_distribution_channels() -> None:
    """PyPI is canonical; GitHub Packages is not used for pip wheels."""
    text = _RELEASE.read_text(encoding="utf-8")
    assert "Distribution summary" in text
    assert "pypi.org/project/axquant" in text
    assert "GitHub Packages" in text
    assert "unused for Python" in text


def test_project_urls_point_at_pypi_and_github() -> None:
    """PEP 621 project.urls must advertise the real install + source homes."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'Homepage = "https://pypi.org/project/axquant/"' in text
    assert 'Repository = "https://github.com/defai-digital/axquant"' in text
    assert "Issues =" in text
    assert "Changelog =" in text
