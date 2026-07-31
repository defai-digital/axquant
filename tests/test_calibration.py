from __future__ import annotations

from pathlib import Path

import pytest

from axquant.calibration import prepare_calibration
from axquant.errors import ArtifactError
from axquant.schema import ModelIdentity, ProfileName


def test_calibration_manifest_is_reusable_for_identical_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
    output = tmp_path / "cache"
    first = prepare_calibration(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        dataset=dataset,
        output_dir=output,
        profile=ProfileName.AGENT_CODING,
        domains=["code", "tool-use"],
        sequence_length=1024,
        random_seed=7,
        separation_attested=True,
    )
    second = prepare_calibration(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        dataset=dataset,
        output_dir=output,
        profile=ProfileName.AGENT_CODING,
        domains=["code", "tool-use"],
        sequence_length=1024,
        random_seed=7,
        separation_attested=True,
    )
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.samples == 2


def test_calibration_rejects_non_object_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="JSON object"):
        prepare_calibration(
            model=ModelIdentity(model_id="org/model"),
            dataset=dataset,
            output_dir=tmp_path / "cache",
            profile=ProfileName.GENERAL,
            domains=["general"],
            sequence_length=128,
            random_seed=0,
            separation_attested=False,
        )
