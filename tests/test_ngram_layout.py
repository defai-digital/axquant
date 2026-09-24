from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from axquant.errors import ArtifactError
from axquant.module_paths import is_ngram_shard_key
from axquant.ngram_layout import (
    INDEX_FILENAME,
    NGRAM_LAYOUT_SHARDED,
    NGRAM_LAYOUT_STANDALONE,
    NGRAM_RELAYOUT_MANIFEST_FILENAME,
    NGRAM_RELAYOUT_SCHEMA,
    NGRAM_TABLE_FILENAME,
    RUNTIME_CONTRACT_FILENAME,
    _pack_tensor_digests,
    relayout_ngram_table,
)

_PLE = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shards"


def _ngram_name(shard: int) -> str:
    return f"{_PLE}.{shard}.weight"


def _tensor(values: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, values, dtype=np.float32) + np.arange(
        int(np.prod(shape)) if shape else 1
    ).reshape(shape).astype(np.float32)


def _build_pack(
    directory: Path,
    *,
    runtime_contract: dict | None = None,
    index_weights: dict[str, str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, np.ndarray]] = {
        "model-00001-of-00003.safetensors": {
            _ngram_name(0): _tensor(1, (4, 8)),
            _ngram_name(1): _tensor(2, (4, 8)),
            "model.layers.0.self_attn.q_proj.weight": _tensor(3, (8, 8)),
        },
        "model-00002-of-00003.safetensors": {
            _ngram_name(2): _tensor(4, (4, 8)),
        },
        "model-00003-of-00003.safetensors": {
            "model.layers.1.mlp.down_proj.weight": _tensor(5, (8, 8)),
            "model.norm.weight": _tensor(6, (8,)),
        },
    }
    for name, tensors in shards.items():
        save_file(tensors, directory / name)
    save_file(
        {
            "mtp.fc.weight": _tensor(7, (8, 8)),
        },
        directory / "mtp.safetensors",
    )
    weights = index_weights
    if weights is None:
        weights = {tensor: shard for shard, tensors in shards.items() for tensor in tensors}
    weights = dict(weights)
    weights["mtp.fc.weight"] = "mtp.safetensors"
    total_size = int(
        sum(tensor.nbytes for tensors in shards.values() for tensor in tensors.values())
    )
    total_size += int(np.float32(0).nbytes * 8 * 8)  # mtp sidecar payload
    (directory / INDEX_FILENAME).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weights}),
        encoding="utf-8",
    )
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    contract = (
        runtime_contract
        if runtime_contract is not None
        else {
            "arch_id": "qwen3-next-mtp",
            "mtp_depth_max": 1,
            "mtp_tensor_count": 1,
        }
    )
    (directory / RUNTIME_CONTRACT_FILENAME).write_text(json.dumps(contract), encoding="utf-8")
    return directory


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_relayout_moves_ngram_keys_to_standalone_table(tmp_path: Path) -> None:
    source = _build_pack(tmp_path / "pack")
    output = tmp_path / "pack-mtplx"

    report = relayout_ngram_table(source, output)

    assert not report.dry_run
    assert report.moved_tensor_count == 3
    assert report.removed_shard_files == ("model-00002-of-00003.safetensors",)
    assert report.rebuilt_shard_files == ("model-00001-of-00003.safetensors",)
    assert report.copied_file_count == 3  # config.json, model-00003, mtp.safetensors
    assert report.total_size_before is not None
    assert report.total_size_after == report.total_size_before - 3 * (4 * 8 * 4)

    moved_names = {_ngram_name(index) for index in range(3)}
    table = load_file(output / NGRAM_TABLE_FILENAME)
    assert set(table) == moved_names
    for name in moved_names:
        assert table[name].dtype == np.float32
        assert table[name].shape == (4, 8)

    digests = _pack_tensor_digests(source)
    variant_digests = _pack_tensor_digests(output)
    assert variant_digests == digests

    index = json.loads((output / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == set(digests) - moved_names
    assert index["metadata"]["total_size"] == report.total_size_after
    assert not any(
        shard == "model-00002-of-00003.safetensors" for shard in index["weight_map"].values()
    )

    rebuilt = load_file(output / "model-00001-of-00003.safetensors")
    assert set(rebuilt) == {"model.layers.0.self_attn.q_proj.weight"}
    assert not (output / "model-00002-of-00003.safetensors").exists()
    assert _file_digest(output / "model-00003-of-00003.safetensors") == _file_digest(
        source / "model-00003-of-00003.safetensors"
    )
    assert _file_digest(output / "mtp.safetensors") == _file_digest(source / "mtp.safetensors")
    assert _file_digest(output / "config.json") == _file_digest(source / "config.json")

    contract = json.loads((output / RUNTIME_CONTRACT_FILENAME).read_text(encoding="utf-8"))
    assert contract["ngram_layout"] == NGRAM_LAYOUT_STANDALONE
    assert contract["arch_id"] == "qwen3-next-mtp"
    source_contract = json.loads((source / RUNTIME_CONTRACT_FILENAME).read_text(encoding="utf-8"))
    assert "ngram_layout" not in source_contract

    manifest = json.loads((output / NGRAM_RELAYOUT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == NGRAM_RELAYOUT_SCHEMA
    assert manifest["accounting"]["moved"] == 3
    assert manifest["accounting"]["lost"] == 0
    assert manifest["accounting"]["duplicated"] == 0
    assert manifest["ngram_layout"]["variant"] == NGRAM_LAYOUT_STANDALONE
    assert manifest["exactness_baseline"]["status"] == "unverified"
    assert manifest["exactness_baseline"]["public_release_blocker"] is True
    assert set(manifest["moved_tensors"]) == moved_names
    assert set(manifest["output_files"]) == {path.name for path in output.iterdir()} - {
        NGRAM_RELAYOUT_MANIFEST_FILENAME
    }

    # The source pack is untouched, including its runtime contract.
    assert not (source / NGRAM_TABLE_FILENAME).exists()
    assert (
        json.loads((source / INDEX_FILENAME).read_text(encoding="utf-8"))["weight_map"][
            _ngram_name(0)
        ]
        == "model-00001-of-00003.safetensors"
    )


def test_relayout_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = _build_pack(tmp_path / "pack")
    output = tmp_path / "pack-mtplx"

    report = relayout_ngram_table(source, output, dry_run=True)

    assert report.dry_run
    assert report.moved_tensor_count == 3
    assert not output.exists()
    assert not (source / NGRAM_TABLE_FILENAME).exists()


def test_relayout_rejects_plain_pack_without_ngram_keys(tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    save_file({"model.norm.weight": _tensor(1, (8,))}, source / "model.safetensors")
    (source / INDEX_FILENAME).write_text(
        json.dumps({"weight_map": {"model.norm.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    (source / RUNTIME_CONTRACT_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactError, match="no sharded n-gram keys"):
        relayout_ngram_table(source, tmp_path / "variant")


def test_relayout_rejects_existing_output_and_nested_output(tmp_path: Path) -> None:
    source = _build_pack(tmp_path / "pack")
    existing = tmp_path / "variant"
    existing.mkdir()

    with pytest.raises(ArtifactError, match="already exists"):
        relayout_ngram_table(source, existing)

    with pytest.raises(ArtifactError, match="outside the source pack"):
        relayout_ngram_table(source, source / "nested")


def test_relayout_requires_runtime_contract(tmp_path: Path) -> None:
    source = _build_pack(tmp_path / "pack")
    (source / RUNTIME_CONTRACT_FILENAME).unlink()

    with pytest.raises(ArtifactError, match=r"mtplx_runtime\.json missing"):
        relayout_ngram_table(source, tmp_path / "variant")


@pytest.mark.parametrize(
    ("explicit_layout", "match"),
    [
        ("standalone-table", "already standalone-table"),
        ("bogus-layout", "unknown explicit ngram_layout"),
    ],
)
def test_relayout_fails_closed_on_explicit_layout_values(
    tmp_path: Path, explicit_layout: str, match: str
) -> None:
    source = _build_pack(
        tmp_path / "pack",
        runtime_contract={"ngram_layout": explicit_layout},
    )

    with pytest.raises(ArtifactError, match=match):
        relayout_ngram_table(source, tmp_path / "variant")


def test_relayout_accepts_explicit_sharded_index_value(tmp_path: Path) -> None:
    source = _build_pack(
        tmp_path / "pack",
        runtime_contract={"ngram_layout": NGRAM_LAYOUT_SHARDED},
    )
    output = tmp_path / "variant"

    relayout_ngram_table(source, output)

    contract = json.loads((output / RUNTIME_CONTRACT_FILENAME).read_text(encoding="utf-8"))
    assert contract["ngram_layout"] == NGRAM_LAYOUT_STANDALONE


def test_relayout_gates_rerun_against_variant(tmp_path: Path) -> None:
    source = _build_pack(tmp_path / "pack")
    output = tmp_path / "variant"
    relayout_ngram_table(source, output)

    with pytest.raises(ArtifactError, match="already exists in"):
        relayout_ngram_table(output, tmp_path / "variant-2")


def test_is_ngram_shard_key_alias_coverage() -> None:
    assert is_ngram_shard_key(f"{_PLE}.0.weight")
    assert is_ngram_shard_key("model.ngram_embedding.shard_3.weight")
    assert not is_ngram_shard_key("model.layers.0.self_attn.q_proj.weight")
    assert not is_ngram_shard_key("model.embed_tokens.weight")
