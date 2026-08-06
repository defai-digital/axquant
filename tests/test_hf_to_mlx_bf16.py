from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_helper() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "hf_to_mlx_bf16.py"
    spec = importlib.util.spec_from_file_location("axquant_hf_to_mlx_bf16", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def test_hub_bf16_conversion_is_revision_pinned_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "candidate"
    work = tmp_path / "work"
    revision = "a" * 40
    observed: dict[str, object] = {}

    def fake_prepare(hf_id: str, source_revision: str, work_dir: Path, prepared: Path) -> Path:
        observed.update(
            {
                "hf_id": hf_id,
                "revision": source_revision,
                "work": work_dir,
                "prepared": prepared,
            }
        )
        return source

    def fake_convert(command: list[str]) -> None:
        staging = Path(command[command.index("--mlx-path") + 1])
        assert staging != output
        staging.mkdir()
        (staging / "config.json").write_text("{}", encoding="utf-8")
        (staging / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(helper, "prepare_hf_dir", fake_prepare)
    monkeypatch.setattr(helper.subprocess, "check_call", fake_convert)

    helper.main(
        [
            "--hf-id",
            "org/model",
            "--revision",
            revision,
            "--mlx-path",
            str(output),
            "--work",
            str(work),
        ]
    )

    assert observed["revision"] == revision
    assert output.is_dir()
    assert not list(tmp_path.glob(".candidate.*"))
    provenance = json.loads((output / "axquant_source.json").read_text(encoding="utf-8"))
    assert provenance == {
        "schema_version": "axquant.source-conversion.v1",
        "source_model": "org/model",
        "source_revision": revision,
        "dtype": "bfloat16",
        "key_remap_applied": False,
    }


def test_qwen3_asr_bf16_conversion_uses_public_mlx_audio_stt_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr"}),
        encoding="utf-8",
    )
    output = tmp_path / "candidate"
    observed: list[str] = []

    monkeypatch.setattr(
        helper,
        "prepare_hf_dir",
        lambda hf_id, revision, work, prepared: source,
    )

    def fake_convert(command: list[str]) -> None:
        observed.extend(command)
        staging = Path(command[command.index("--mlx-path") + 1])
        staging.mkdir()
        (staging / "config.json").write_text("{}", encoding="utf-8")
        (staging / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(helper.subprocess, "check_call", fake_convert)

    helper.main(
        [
            "--hf-id",
            "Qwen/Qwen3-ASR-1.7B",
            "--revision",
            "c" * 40,
            "--mlx-path",
            str(output),
            "--work",
            str(tmp_path / "work"),
        ]
    )

    assert observed[:3] == [helper.sys.executable, "-m", "mlx_audio.convert"]
    assert observed[observed.index("--model-domain") + 1] == "stt"
    assert observed[observed.index("--dtype") + 1] == "bfloat16"
    provenance = json.loads((output / "axquant_source.json").read_text(encoding="utf-8"))
    assert provenance["key_remap_applied"] is True


def test_hub_bf16_conversion_rejects_mutable_revision_and_existing_output(
    tmp_path: Path,
) -> None:
    base = [
        "--hf-id",
        "org/model",
        "--mlx-path",
        str(tmp_path / "candidate"),
        "--work",
        str(tmp_path / "work"),
    ]
    with pytest.raises(SystemExit, match="immutable 40-character"):
        helper.main([*base, "--revision", "main"])

    output = tmp_path / "candidate"
    output.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        helper.main([*base, "--revision", "b" * 40])
