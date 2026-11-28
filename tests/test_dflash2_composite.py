"""AXQ-067 DF-5: DFlash2 composite target + drafter packs (local pilot)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axquant import dflash2_target
from axquant.dflash2_composite import compose_dflash2_composite, composite_issues
from axquant.errors import ArtifactError
from axquant.naming import (
    DFLASH2_COMPANION_NAME,
    dflash2_composite_name,
    require_dflash2_suffix_matches_packaging,
    require_mtp_suffix_matches_packaging,
)
from axquant.schema.dflash2 import Dflash2Companion
from tests.test_dflash2_target import REVISION, _declare, _write_pack

CONFIG = b'{"block_size": 8}\n'
WEIGHTS = b"drafter weights"
README = b"---\nlicense: apache-2.0\nbase_model:\n  - Qwen/Qwen3.8-27B\n---\n# Drafter\n"


@pytest.fixture
def fake_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    real = dflash2_target.admitted_targets()
    first = real.targets[0].model_copy(
        update={
            "drafter_config_sha256": hashlib.sha256(CONFIG).hexdigest(),
            "drafter_weights_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
        }
    )
    mirror = real.model_copy(update={"targets": [first, *real.targets[1:]]})
    monkeypatch.setattr(dflash2_target, "admitted_targets", lambda: mirror)
    monkeypatch.setattr("axquant.dflash2_composite.admitted_targets", lambda: mirror)


def _drafter(root: Path, *, weights: bytes = WEIGHTS, readme: bytes = README) -> Path:
    root.mkdir()
    (root / "config.json").write_bytes(CONFIG)
    (root / "model.safetensors").write_bytes(weights)
    (root / "README.md").write_bytes(readme)
    return root


def _target(tmp_path: Path) -> Path:
    pack = _write_pack(tmp_path / "target")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    return pack


@pytest.mark.usefixtures("fake_mirror")
def test_composite_keeps_target_bytes_and_bundles_pinned_drafter(tmp_path: Path) -> None:
    target = _target(tmp_path)
    output = tmp_path / "composite"

    companion = compose_dflash2_composite(
        target_dir=target, drafter_dir=_drafter(tmp_path / "drafter"), output_dir=output
    )

    for path in target.iterdir():
        assert (output / path.name).read_bytes() == path.read_bytes()
    assert (output / "dflash2" / "model.safetensors").read_bytes() == WEIGHTS
    assert "apache-2.0" in (output / "dflash2" / "NOTICE").read_text()
    assert companion.drafter_license == "apache-2.0"
    assert companion.source_model_id == "Qwen/Qwen3.8-27B"
    recorded = Dflash2Companion.model_validate(
        json.loads((output / DFLASH2_COMPANION_NAME).read_text())
    )
    assert {item.path for item in recorded.files} == {
        "dflash2/NOTICE",
        "dflash2/README.md",
        "dflash2/config.json",
        "dflash2/model.safetensors",
    }
    assert composite_issues(output) == []
    assert not list(tmp_path.glob(".composite.staging-*"))


@pytest.mark.usefixtures("fake_mirror")
def test_composite_refuses_drafter_digest_mismatch(tmp_path: Path) -> None:
    target = _target(tmp_path)
    drafter = _drafter(tmp_path / "drafter", weights=b"requantized drafter")

    with pytest.raises(ArtifactError, match="does not match the admitted pin"):
        compose_dflash2_composite(
            target_dir=target, drafter_dir=drafter, output_dir=tmp_path / "composite"
        )
    assert not (tmp_path / "composite").exists()


@pytest.mark.usefixtures("fake_mirror")
def test_composite_requires_a_declared_license(tmp_path: Path) -> None:
    drafter = _drafter(tmp_path / "drafter", readme=b"# no front matter\n")

    with pytest.raises(ArtifactError, match="declares no license"):
        compose_dflash2_composite(
            target_dir=_target(tmp_path), drafter_dir=drafter, output_dir=tmp_path / "out"
        )


@pytest.mark.usefixtures("fake_mirror")
def test_composite_refuses_existing_output_and_non_targets(tmp_path: Path) -> None:
    drafter = _drafter(tmp_path / "drafter")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ArtifactError, match="already exists"):
        compose_dflash2_composite(
            target_dir=_target(tmp_path), drafter_dir=drafter, output_dir=existing
        )

    fine_tune = _write_pack(tmp_path / "fine-tune")
    _declare(fine_tune, model_id="AutomatosX/Swift-1.5-Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ArtifactError, match="declared DFlash2 target"):
        compose_dflash2_composite(
            target_dir=fine_tune, drafter_dir=drafter, output_dir=tmp_path / "out"
        )


@pytest.mark.usefixtures("fake_mirror")
def test_composite_issues_detect_later_changes(tmp_path: Path) -> None:
    output = tmp_path / "composite"
    compose_dflash2_composite(
        target_dir=_target(tmp_path), drafter_dir=_drafter(tmp_path / "drafter"), output_dir=output
    )
    weights = output / "dflash2" / "model.safetensors"
    weights.unlink()
    weights.write_bytes(b"swapped")
    (output / "dflash2" / "extra.bin").write_bytes(b"x")

    issues = composite_issues(output)

    assert "drafter file digest mismatch: dflash2/model.safetensors" in issues
    assert "drafter directory differs from the companion record: dflash2/extra.bin" in issues


def test_dflash2_naming_contract() -> None:
    leaf = "AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP"
    composite = dflash2_composite_name(leaf)
    assert composite == "AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP-DFlash2"
    with pytest.raises(ArtifactError):
        dflash2_composite_name(composite)
    with pytest.raises(ArtifactError):
        dflash2_composite_name("AX-Qwen3.8-27B-CUDA-AXQ-NVFP4-W4A16-MTP")

    bundled = ["mtp.safetensors", "dflash2/model.safetensors", DFLASH2_COMPANION_NAME]
    require_dflash2_suffix_matches_packaging(composite, filenames=bundled)
    require_mtp_suffix_matches_packaging(composite, filenames=bundled)
    with pytest.raises(ArtifactError, match="must end with -DFlash2"):
        require_dflash2_suffix_matches_packaging(leaf, filenames=bundled)
    with pytest.raises(ArtifactError, match="bundles no DFlash2 drafter"):
        require_dflash2_suffix_matches_packaging(composite, filenames=["mtp.safetensors"])
