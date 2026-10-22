"""Qwen 3.5/3.6/3.8 MTP head: oMLX export and the static verdicts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from axquant.errors import ArtifactError
from axquant.ngram_layout import _parse_safetensors_file
from axquant.runtime_compatibility import inspect_runtime_export
from axquant.runtime_export import QWEN35_HEAD_SHARD, export_runtime_pack
from axquant.serde import file_sha256

_HEAD_NORMS = {
    "mtp.layers.0.input_layernorm.weight": -0.1,
    "mtp.layers.0.post_attention_layernorm.weight": 0.87,
    "mtp.layers.0.self_attn.q_norm.weight": 0.77,
    "mtp.layers.0.self_attn.k_norm.weight": 0.74,
    "mtp.norm.weight": 1.9,
    "mtp.pre_fc_norm_embedding.weight": -0.7,
    "mtp.pre_fc_norm_hidden.weight": -0.5,
}


def _pack(
    directory: Path,
    *,
    trunk_norm_mean: float = 1.0,
    declare_trunk: bool = False,
    head_layout: str = "raw_hf_delta",
) -> Path:
    directory.mkdir()
    trunk = {
        "language_model.model.layers.0.input_layernorm.weight": np.full(
            8, trunk_norm_mean, dtype=np.float32
        ),
        "language_model.model.layers.0.self_attn.q_proj.weight": np.arange(
            16, dtype=np.float32
        ).reshape(4, 4),
        "language_model.model.norm.weight": np.full(8, 2.6, dtype=np.float32),
    }
    save_file(trunk, directory / "model-00001-of-00001.safetensors")
    head = {name: np.full(8, mean, dtype=np.float32) for name, mean in _HEAD_NORMS.items()}
    head["mtp.fc.weight"] = np.arange(32, dtype=np.float32).reshape(4, 8)
    head["mtp.layers.0.self_attn.q_proj.weight"] = np.arange(16, dtype=np.float32).reshape(4, 4)
    save_file(head, directory / "mtp.safetensors", metadata={"format": "mlx"})
    save_file(
        {"vision.patch.weight": np.zeros((2, 2), dtype=np.float32)},
        directory / "vision.safetensors",
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": dict.fromkeys(trunk, "model-00001-of-00001.safetensors"),
            }
        )
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1},
                "quantization": {"bits": 4, "group_size": 64},
            }
        )
    )
    runtime = {
        "schema_version": "axquant.mtp-runtime.v1",
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 1,
        "mtp_norm_layout": head_layout,
    }
    if declare_trunk:
        runtime["trunk_norm_layout"] = "mlx_multiplier"
    (directory / "mtplx_runtime.json").write_text(json.dumps(runtime))
    return directory


def test_raw_head_needs_an_export_for_omlx_and_none_for_mtplx(tmp_path: Path) -> None:
    report = inspect_runtime_export(_pack(tmp_path / "pack"))
    omlx, mtplx = report["targets"]["omlx"], report["targets"]["mtplx"]
    assert omlx["profile"] == "qwen35-mtp-v1" and omlx["status"] == "requires-export"
    assert any("weight index" in action for action in omlx["required_actions"])
    assert any("RMSNorm gammas" in action for action in omlx["required_actions"])
    assert any("vision" in note for note in omlx["modality_constraints"])
    assert mtplx["profile"] == "qwen35-mtp-native-v1" and mtplx["status"] == "static-compatible"


def test_export_shifts_only_the_head_norms_and_indexes_the_head(tmp_path: Path) -> None:
    source = _pack(tmp_path / "pack", declare_trunk=True)
    output = tmp_path / "omlx"
    export_runtime_pack(source, output, target="omlx")

    trunk = "model-00001-of-00001.safetensors"
    assert file_sha256(output / trunk) == file_sha256(source / trunk)  # trunk bytes untouched
    assert not (output / "mtp.safetensors").exists()
    assert not (output / "vision.safetensors").exists()

    head = load_file(output / QWEN35_HEAD_SHARD)
    original = load_file(source / "mtp.safetensors")
    assert {f"language_model.{name}" for name in original} == set(head)
    for name, tensor in original.items():
        exported = head[f"language_model.{name}"]
        if name in _HEAD_NORMS:
            np.testing.assert_allclose(exported, tensor + 1.0, rtol=0, atol=1e-6)
        else:
            np.testing.assert_array_equal(exported, tensor)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert all(
        index["weight_map"][f"language_model.{name}"] == QWEN35_HEAD_SHARD for name in original
    )
    layout = _parse_safetensors_file(output / trunk)
    expected_total = sum(t.byte_count for t in layout.tensors.values()) + sum(
        t.byte_count for t in _parse_safetensors_file(output / QWEN35_HEAD_SHARD).tensors.values()
    )
    assert index["metadata"]["total_size"] == expected_total

    runtime = json.loads((output / "mtplx_runtime.json").read_text())
    assert runtime["mtp_norm_layout"] == "mlx_multiplier"
    assert runtime["runtime_verified"] is False
    manifest = json.loads((output / "axquant_runtime_export.json").read_text())
    assert any("vision" in note for note in manifest["modality_constraints"])
    assert inspect_runtime_export(output)["targets"]["omlx"]["status"] == "static-compatible"


def test_trunk_declaration_is_required_and_checked_against_the_data(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="--source-trunk-norm-layout mlx_multiplier"):
        export_runtime_pack(_pack(tmp_path / "a"), tmp_path / "out-a", target="omlx")
    zero_centered = _pack(tmp_path / "b", trunk_norm_mean=0.0)
    with pytest.raises(ArtifactError, match="looks zero-centered"):
        export_runtime_pack(
            zero_centered,
            tmp_path / "out-b",
            target="omlx",
            source_trunk_norm_layout="mlx_multiplier",
        )
    assert not (tmp_path / "out-b").exists()


def test_declared_trunk_option_unlocks_the_export(tmp_path: Path) -> None:
    export_runtime_pack(
        _pack(tmp_path / "pack"),
        tmp_path / "out",
        target="omlx",
        source_trunk_norm_layout="mlx_multiplier",
    )
    manifest = json.loads((tmp_path / "out" / "axquant_runtime_export.json").read_text())
    assert manifest["source_trunk_norm_layout"] == "mlx_multiplier"


def test_a_head_that_is_not_declared_raw_delta_is_refused(tmp_path: Path) -> None:
    source = _pack(tmp_path / "pack", declare_trunk=True, head_layout="mlx_multiplier")
    with pytest.raises(ArtifactError, match="raw_hf_delta"):
        export_runtime_pack(source, tmp_path / "out", target="omlx")


def test_mtplx_needs_no_export_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = _pack(tmp_path / "pack", declare_trunk=True)
    with pytest.raises(ArtifactError, match="no export is needed"):
        export_runtime_pack(source, tmp_path / "mtplx", target="mtplx")
    export_runtime_pack(source, tmp_path / "dry", target="omlx", dry_run=True)
    assert not (tmp_path / "dry").exists()
