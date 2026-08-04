from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from axquant.errors import ArtifactError
from axquant.schema import Inventory, ModelIdentity, TensorRole, TensorSpec
from axquant.serde import load_model, read_data, stable_sha256, write_data


def _tensor() -> TensorSpec:
    return TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(10, 1),
        dtype="BF16",
        parameters=10,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )


def _inventory(**overrides: object) -> Inventory:
    values: dict[str, object] = {
        "model": ModelIdentity(model_id="org/model", revision="abc"),
        "tensors": [_tensor()],
        "total_parameters": 10,
        "quantizable_parameters": 10,
        "mtp_present": False,
        "quantized_source": False,
        "source_files": ["model.safetensors"],
        "config_sha256": "a" * 64,
    }
    values.update(overrides)
    return Inventory.model_validate(values)


def test_stable_sha256_ignores_created_at() -> None:
    # Two artifacts with identical semantic content but different creation
    # timestamps must hash identically, or resume/lineage checks that compare
    # a freshly rebuilt object against a persisted one raise false mismatches.
    early = _inventory(created_at=datetime(2020, 1, 1, tzinfo=UTC))
    late = _inventory(created_at=datetime(2024, 6, 15, tzinfo=UTC))
    assert stable_sha256(early) == stable_sha256(late)


def test_stable_sha256_still_reflects_real_content_changes() -> None:
    baseline = _inventory()
    changed = _inventory(total_parameters=999)
    assert stable_sha256(baseline) != stable_sha256(changed)


def test_stable_sha256_strips_created_at_at_any_nesting_depth() -> None:
    nested_early = {"outer": {"created_at": "2020-01-01T00:00:00Z", "value": 1}}
    nested_late = {"outer": {"created_at": "2099-12-31T00:00:00Z", "value": 1}}
    assert stable_sha256(nested_early) == stable_sha256(nested_late)

    nested_changed = {"outer": {"created_at": "2020-01-01T00:00:00Z", "value": 2}}
    assert stable_sha256(nested_early) != stable_sha256(nested_changed)


def test_load_model_accepts_an_artifact_with_its_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    write_data(path, _inventory())
    loaded = load_model(path, Inventory)
    assert loaded.schema_version == _inventory().schema_version


def test_load_model_rejects_an_artifact_missing_schema_version(tmp_path: Path) -> None:
    # `schema_version` defaults for construction convenience in-repo, but a
    # persisted artifact that omits it entirely must not silently validate
    # as "current schema" -- that's the one guarantee AGENTS.md's "every
    # artifact carries a schema_version" promises.
    payload = json.loads(_inventory().model_dump_json())
    del payload["schema_version"]
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="missing required 'schema_version'"):
        load_model(path, Inventory)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_write_data_rejects_non_finite_raw_values(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ArtifactError, match="non-finite"):
        write_data(tmp_path / f"artifact{suffix}", {"metric": float("inf")})


def test_stable_sha256_rejects_non_finite_raw_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        stable_sha256({"metric": float("nan")})


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("artifact.json", '{"metric": NaN}'),
        ("artifact.yaml", "metric: .inf\n"),
    ],
)
def test_read_data_rejects_non_finite_raw_values(
    tmp_path: Path,
    name: str,
    payload: str,
) -> None:
    path = tmp_path / name
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ArtifactError, match="non-finite"):
        read_data(path)
