from __future__ import annotations

from pathlib import Path

import pytest

from axquant.calibration import (
    calibration_manifest_matches,
    calibration_manifest_sha256,
    prepare_calibration,
)
from axquant.errors import ArtifactError
from axquant.schema import ModelIdentity, ProfileName
from axquant.serde import file_sha256


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
        tokenizer_revision="tokenizer-revision",
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
        tokenizer_revision="tokenizer-revision",
        separation_attested=True,
    )
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.samples == 2
    assert first.tokenizer_revision == "tokenizer-revision"

    manifest_path = output / "calibration_manifest.json"
    assert calibration_manifest_matches(
        manifest_path,
        first,
        calibration_manifest_sha256(first),
    )
    assert calibration_manifest_matches(manifest_path, first, file_sha256(manifest_path))


def test_calibration_defaults_tokenizer_revision_to_model_revision(tmp_path: Path) -> None:
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text('{"text":"one"}\n', encoding="utf-8")

    manifest = prepare_calibration(
        model=ModelIdentity(model_id="org/model", revision="model-revision"),
        dataset=dataset,
        output_dir=tmp_path / "cache",
        profile=ProfileName.GENERAL,
        domains=["general"],
        sequence_length=128,
        random_seed=0,
        separation_attested=False,
    )

    assert manifest.tokenizer_revision == "model-revision"


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
