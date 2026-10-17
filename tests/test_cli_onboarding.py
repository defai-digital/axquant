"""Portable discovery and fail-fast conversion before model access."""

from pathlib import Path

import pytest
import structlog

from axquant import __version__
from axquant.cli import main
from axquant.conversion_formats import MLX_FORMATS, format_ladder
from axquant.errors import ArtifactError, PlanningError
from axquant.quantize import quick_convert
from axquant.simple_convert import simple_convert


@pytest.fixture(autouse=True)
def reset_cli_logging():
    # CLI configuration binds pytest's captured stderr. Do not leave a closed
    # capture stream in the global logger factory for later library tests.
    yield
    structlog.reset_defaults()


def test_installed_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == f"axquant {__version__}"


@pytest.mark.parametrize("backend", ["all", "mlx", "cuda"])
def test_format_catalog(backend: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["formats", "--backend", backend]) == 0
    output = capsys.readouterr().out
    assert "not runtime compatibility certificates" in output
    assert "does not check this host" in output
    if backend != "cuda":
        for name in MLX_FORMATS:
            assert f"| {name} | {format_ladder(name).default_target_bpw:.1f} |" in output
        assert "MTPLX" in output
        assert "oMLX" in output
    else:
        assert "## MLX" not in output
    if backend != "mlx":
        assert "--q-mode nvfp4" in output
        assert "--allow-unmeasured" in output
        assert "vLLM" in output
    else:
        assert "## CUDA" not in output


def _forbid_model_access(*args, **kwargs):
    pytest.fail("invalid options must not resolve, download or inspect a model")


@pytest.mark.parametrize("api", ["simple", "quick"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"ladder": "measured-lite"}, "requires measured sensitivity"),
        ({"ladder": "measured-full", "recipe": "missing"}, "non-prior --ladder"),
        ({"kv_cache": "invalid"}, "off\\|prior"),
        ({"target_bpw": float("nan")}, "finite"),
        ({"target_bpw": float("inf")}, "finite"),
        ({"target_bpw": 0}, "finite"),
        ({"target_bpw": 17}, "finite"),
        ({"recipe": "missing", "target_bpw": 4.8}, "cannot be combined"),
        ({"format": "mxfp8", "recipe": "missing"}, "cannot be combined"),
        ({"format": "nvfp4"}, "unknown MLX format"),
    ],
)
def test_invalid_options_never_access_source(
    api: str,
    options: dict,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("axquant.inspector.resolve_model_dir", _forbid_model_access)
    monkeypatch.setattr("axquant.quantize.inspect_model", _forbid_model_access)
    with pytest.raises(PlanningError, match=message):
        if api == "simple":
            simple_convert("org/model", allow_download=True, **options)
        else:
            quick_convert(
                model="org/model", output=tmp_path / "output", allow_download=True, **options
            )


@pytest.mark.parametrize("kind", ["file", "directory", "broken-symlink"])
def test_existing_output_prevents_download(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("axquant.inspector.resolve_model_dir", _forbid_model_access)
    output = tmp_path / "output"
    if kind == "file":
        output.write_text("keep me", encoding="utf-8")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(ArtifactError, match="already exists"):
        simple_convert("org/model", output=output, allow_download=True)
    if kind == "file":
        assert output.read_text() == "keep me"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()
