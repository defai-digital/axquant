from pathlib import Path

from axquant import versioning


def test_executable_distribution_version_for_homebrew_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "Cellar" / "ax-engine" / "6.11.1" / "bin" / "ax-engine-bench"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(versioning.shutil, "which", lambda _: str(executable))

    assert versioning.executable_distribution_version("ax-engine-bench", "ax-engine") == "6.11.1"


def test_executable_distribution_version_accepts_standalone_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "runtimes" / "6.12.1" / "bin" / "ax-engine-bench"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(versioning.shutil, "which", lambda _: str(executable))

    assert versioning.executable_distribution_version("ax-engine-bench", "ax-engine") == "6.12.1"
    assert versioning.standalone_executable_version(executable) == "6.12.1"


def test_executable_distribution_version_is_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(versioning.shutil, "which", lambda _: None)

    assert versioning.executable_distribution_version("ax-engine-bench", "ax-engine") is None
