from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = _ROOT / "scripts" / "run_qwen38_flash_next_axq.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_qwen38_flash_next_axq", _DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leftover_staging_dirs_match_convert_tempfile_prefix(tmp_path: Path) -> None:
    driver = _load_driver()
    pack = tmp_path / "AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP"
    leftover = tmp_path / ".AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP.1i2aws3b"
    leftover.mkdir()
    (leftover / "artifact").mkdir()
    other = tmp_path / ".unrelated.dir"
    other.mkdir()
    found = driver.leftover_staging_dirs(pack)
    assert found == [leftover]
    driver.remove_leftover_staging(pack)
    assert not leftover.exists()
    assert other.is_dir()


def test_convert_exit_was_signaled_covers_unix_and_shell_codes() -> None:
    driver = _load_driver()
    assert driver.convert_exit_was_signaled(-15)
    assert driver.convert_exit_was_signaled(-9)
    assert driver.convert_exit_was_signaled(143)
    assert driver.convert_exit_was_signaled(137)
    assert not driver.convert_exit_was_signaled(0)
    assert not driver.convert_exit_was_signaled(2)


def test_find_inflight_flash_next_converts_ignores_other_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _load_driver()
    listing = "\n".join(
        [
            "  11 /usr/bin/ssh um-macstudio-m2",
            "  22 /Volumes/Ext16TR0/axquant-venv/bin/python -m axquant convert "
            + "--model /Volumes/Ext16TR0/models/Qwen3.8-Flash-Next "
            + "--output /Volumes/Ext16TR0/models/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP",
            "  33 /Volumes/Ext16TR0/axquant-venv/bin/python -m axquant convert "
            + "--model /data/Qwen3.6-27B --output /data/out",
            "  44 /bin/bash scripts/retry_qwen38_flash_next_packs.sh",
        ]
    )
    monkeypatch.setattr(
        driver.subprocess,
        "check_output",
        lambda *args, **kwargs: listing,
    )
    found = driver.find_inflight_flash_next_converts(pid_self=1)
    assert [pid for pid, _command in found] == [22]
    with pytest.raises(SystemExit, match="already running \\(pid 22\\)"):
        driver.refuse_inflight_convert()


def test_remaining_does_not_swallow_signaled_convert() -> None:
    driver = _load_driver()
    assert issubclass(driver.ConvertInterrupted, SystemExit)
    with pytest.raises(driver.ConvertInterrupted):
        raise driver.ConvertInterrupted("interrupted (-15)")


@pytest.mark.parametrize("stage", ["convert", "publish"])
def test_factory_rejects_unchecked_existing_pack_before_reuse_or_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from axquant.errors import ArtifactError

    driver = _load_driver()
    monkeypatch.setattr(driver, "require_factory_host", lambda hostname: None)
    monkeypatch.setattr(driver, "refuse_inflight_convert", lambda: None)
    monkeypatch.setattr(driver, "MODELS", tmp_path / "models")
    monkeypatch.setattr(driver, "WORK", tmp_path / "work")
    driver.WORK.mkdir()
    (driver.WORK / "inventory.json").write_text("{}")
    pack = driver.pack_dir("mxfp4")
    pack.mkdir(parents=True)
    (pack / "axquant_manifest.json").write_text("{}")
    (pack / "preserved.txt").write_text("existing pack")
    monkeypatch.setattr(
        driver, "run", lambda *args, **kwargs: pytest.fail("must not run or upload")
    )
    with pytest.raises(ArtifactError, match="no runtime compatibility record"):
        getattr(driver, "cmd_" + stage)("mxfp4")
    assert (pack / "preserved.txt").read_text() == "existing pack"
